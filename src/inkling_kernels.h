// CUDA kernels of the Inkling 2 (inkling2_loop) runtime: depth mixing, loop controllers, short convolutions,
// relative-position attention with sinks, MoE routing / activation / combination.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ik {

// y[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w (w null: no gain), rows of n; y may alias x
void rms_rows(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s);
// per-head RMS norm without gain, in place: x[t*ts + h*hd + i]
void head_rms(float* x, int hd, int nh, int ts, int T, float eps, cudaStream_t s);
// per head: x *= scale[h][:] (scale may be null), L2 norm clipped to thr, then (tau) times
// 1 + alpha*log(max(1, (*posd + t + 1)/floor))
void clip_heads(float* x, int ts, const float* scale, int nh, int hd, int T, float thr, bool tau, const int* posd,
                float alpha, float floor_, cudaStream_t s);

// attention over depth: for each token, the n states (rows of E) are RMS-normed (no gain) and mixed with
// softmax(query_j . state / sqrt(E)) weights; out[j] (j < nq) receives the mix of query j
struct MixSrc {
  const float* p[8];
  int n;
};
struct MixDst {
  float* p[4];
};
void mix(const MixSrc& st, const float* query, int nq, int E, int T, const MixDst& out, cudaStream_t s);

// loop controller pieces: y[t] = [rms(a[t]), rms(b[t])] (rows of 2E); out = cur - tanh(f)*cur + tanh(g)*msg
void concat_rms(const float* a, const float* b, float* y, int E, int T, cudaStream_t s);
void ctrl_combine(float* out, const float* cur, const float* msg, const float* g, const float* f, int64_t n,
                  cudaStream_t s);

// causal short convolution with residual over T tokens of C channels (row stride C):
// y = (x + sum_j w[c][j] x_{t-3+j} + [dyn: sum_lag x_{t-lag} dyn[t][head(c)*4 + lag]]) * inv
// hist [3][C] holds the previous 3 inputs (oldest first) and is updated afterwards
void conv(const float* x, int T, int C, const float* w, const float* dyn, int dyn_ts, int ch_per_head, float* hist,
          float* y, float inv, cudaStream_t s);

// KV cache (f16, rows of nkv*hd per position; ring > 0: position p lives in row p % ring)
void kv_store(const float* k, int ks, const float* v, int vs, __half* kc, __half* vc, int rowlen, const int* posd,
              int T, int ring, cudaStream_t s);
// attention of T queries at positions pos0.. against cached keys [lo, pos], lo = window ? pos - window + 1 : 0,
// logit = q.k/sqrt(hd) + (dist < extent ? r . table[dist] : 0), plus a per-query sink logit (value 0).
// q [T][qs] (heads of hd), r [T][nh*16], sink [T][nh], table [extent][16]; out [T][nh*hd]
// positions come from device memory (*posd = position of the first query; CUDA-graph friendly); max_keys bounds the
// keys of a query tile (it sizes the key splits)
size_t attn_work_bytes(int T, int nh, int nkv, int max_keys);
void attn(const __half* kc, const __half* vc, int ring, const float* q, int qs, const float* r, const float* sink,
          const float* table, int extent, int window, float* out, const int* posd, int max_keys, int T, int nh, int nkv,
          int hd, void* work, size_t work_bytes, cudaStream_t s);

// o *= sigmoid(g) (n elements); y = a + b
void mul_sigmoid(float* o, const float* g, int64_t n, cudaStream_t s);
void add(float* y, const float* a, const float* b, int64_t n, cudaStream_t s);
void softcap(float* x, int64_t n, float cap, cudaStream_t s);

// MoE: router (sigmoid scores, selection bias, top-k, normalized weights times scale)
void moe_route(const float* logits, const float* bias, int n_exp, int k, float scale, int T, int* ids, float* w,
               cudaStream_t s);
// SITU activation of rows [gate | up] (2N each): h = b*tanh(g/b)*sigmoid(g) * lb*tanh(u/lb), g and u first
// multiplied by rscale[eid[a]] when given
void situ(const float* gu, int A, int N, float beta, float lbeta, const int* eid, const float* rscale, float* h,
          cudaStream_t s);
// out[t][n] = sum_j coef[t*k + j] * y[(map ? map[t*k + j] : t*k + j)][n]
void moe_combine(const float* y, const int* map, const float* coef, int T, int k, int N, float* out, cudaStream_t s);
// coef[a] = w[a] * escale[ids[a]]
void moe_coef(const float* w, const int* ids, const float* escale, int A, float* coef, cudaStream_t s);
// out[a] = x[rows[a]] (rows of K)
void gather_rows(const float* x, const int* rows, int A, int K, float* out, cudaStream_t s);

}  // namespace ik
