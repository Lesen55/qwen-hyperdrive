// CUDA kernels of qwen35-engine (host-side API).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define CUDA_CHECK(x)                                                                     \
  do {                                                                                    \
    cudaError_t e_ = (x);                                                                 \
    if (e_ != cudaSuccess) QE_FATAL("CUDA error %s at %s", cudaGetErrorString(e_), #x);  \
  } while (0)

namespace qk {

constexpr int kMaxB = 8;  // max tokens per "small batch" (decode / speculative verify) call

// x[B][K] f32 -> xq[B][K] int8 + xd[B][K/32] scales (per 32 block, symmetric).
void quantize_q8(const float* x, int8_t* xq, float* xd, int K, int B, cudaStream_t s);

// y[b*ldy + r] (+)= dot(W[r,:], x[b,:]) for r in [0,N), b in [0,B).  W is a GGUF-quantized matrix
// whose rows are `row_bytes` apart.  For F32/F16/BF16 weights xf (f32 activations) is used,
// otherwise xq/xd from quantize_q8.
void matvec(uint32_t type, const void* W, size_t row_bytes, int K, int N, const int8_t* xq,
            const float* xd, const float* xf, int B, float* y, int ldy, bool accumulate,
            cudaStream_t s);

// Prefill GEMM on quantized weights with int8 tensor cores: y[t*ldy + n] (+)= W[n,:] . x[t,:]
// for t < T (any T), x quantized by quantize_q8_256.  Not for IQ1_M / float weight types.
void quantize_q8_256(const float* x, int8_t* xq, float* xd, int K, int T, cudaStream_t s);
// Tensor-core small-batch matvec for IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS (B <= kMaxB tokens,
// K % 256 == 0); activations from quantize_q8_256.
bool matvec_tq_supported(uint32_t type);
int matvec_tq_min_batch(uint32_t type, int K);  // smallest B where matvec_tq beats qk::matvec
// gs = activation scale group: 256 (quantize_q8_256) or 32 (quantize_q8, the dp4a kernels' format)
void matvec_tq(uint32_t type, const void* W, size_t rb, int K, int N, const int8_t* xq, const float* xd, int B,
               float* y, int ldy, bool acc, cudaStream_t s, int gs = 256);
bool mmq_supported(uint32_t type);
void set_num_sms(int n);  // device SM count (launch heuristics)
void matvec_rows(uint32_t type, const void* W, size_t rb, int K, const int* rows, int N, const int8_t* xq,
                 const float* xd, float* y, cudaStream_t s);
// decode-once int8 GEMM for long prefill chunks (types: mmq_supported)
size_t gq_buffer_bytes(int K, int rows);
void gq_decode(uint32_t type, const void* W, size_t rb, int K, int N, void* buf, cudaStream_t s);
bool gq_needs_xsum(uint32_t type);
bool gq_supported(uint32_t type);
void gq_xsum(const int8_t* xq, int K, int T, int16_t* xs, cudaStream_t s);
void gq_gemm(uint32_t type, const void* buf, int K, int N, const int8_t* xq, const float* xd256,
             const int16_t* xs, int T, float* y, int ldy, bool accumulate, cudaStream_t s);
// gs = activation scale group: 32 (quantize_q8, more accurate) or 256 (quantize_q8_256, faster)
void mmq(uint32_t type, const void* W, size_t row_bytes, int K, int N, const int8_t* xq, const float* xd,
         int T, float* y, int ldy, bool accumulate, int gs, cudaStream_t s);

// dst[r*K + k] = dequant(W)[r,k] as f16, for r in [0,N).
void dequant_f16(uint32_t type, const void* W, size_t row_bytes, int K, int N, __half* dst,
                 cudaStream_t s);

// dst[t*ld_dst + c] (+)= src[t*ld_src + c] for t < rows, c < cols (f16 -> f32)
// sets *ovf = 1 if any source value is not finite
void h2f_accum(const __half* src, int ld_src, int rows, int cols, float* dst, int ld_dst, bool acc,
               int* ovf, cudaStream_t s);
// f32 -> f16 conversion (activations for GEMM).
void f32_to_f16(const float* x, __half* y, int64_t n, cudaStream_t s);

// y[t][i] = x[t][i] / rms(x[t]) * w[i]  (rows of n).  y may alias x.  If y16 != null also writes f16.
void add_rmsnorm_q8(float* x, const float* r, const float* w, float* y, int n, int rows, float eps, int8_t* xq,
                    float* xd, cudaStream_t s);
void silu_mul_q8(const float* g, const float* u, float* h, int64_t n, int8_t* xq, float* xd, cudaStream_t s);
void rmsnorm(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s,
             __half* y16 = nullptr);
// In-place per-head RMS norm on strided heads: x[t*tstride + h*hstride + i], i<hd.
void rmsnorm_heads(float* x, const float* w, int hd, int nh, int hstride, int tokens, int tstride,
                   float eps, cudaStream_t s);
// NeoX RoPE on the first rot_dims dims of each head (in place).
void rope_neox(float* x, const int* pos, int hd, int nh, int hstride, int tokens, int tstride,
               int rot_dims, float theta_base, cudaStream_t s);

void add_inplace(float* y, const float* x, int64_t n, cudaStream_t s);
void copy_f32_zc(float* dst, const float* host_src, int64_t n, cudaStream_t s);  // host_src: pinned
void iota_i32(int* dst, int p0, int n, cudaStream_t s);
// h[t][i] = silu(g[t][i]) * u[t][i]
void silu_mul(const float* g, const float* u, float* h, int64_t n, cudaStream_t s,
              __half* h16 = nullptr);
// attn out gating: o[t][h*hd+i] *= sigmoid(qg[t][h*2hd + hd + i])
bool qk_norm_rope(float* q, const float* qw, int qhs, int qts, int nhq, float* k, const float* kw, int khs, int kts,
                  int nhk, const int* pos, int hd, int tokens, int rot, float base, float eps, cudaStream_t s);
void matvec_float_pair(uint32_t type, const void* W1, const void* W2, size_t rb, int K, int N, const float* x, int B,
                       float* y1, float* y2, int ldy, cudaStream_t s);
void gate_sigmoid(float* o, const float* qg, int hd, int nh, int tokens, cudaStream_t s,
                  __half* o16 = nullptr, int8_t* xq = nullptr, float* xd = nullptr);

// ---- KV cache (full attention layers) -------------------------------------------------------
enum KVType : int { KV_F16 = 0, KV_Q8 = 1, KV_Q4 = 2, KV_Q3 = 3 };
// KV records are grouped in blocks of KV_BLOCK positions.  `bs` (block stride, bytes) below is the
// distance between consecutive blocks of the same layer, so all layers' blocks can be interleaved
// in one pool (block-major) and the unused tail of the pool can be lent to weights.
constexpr int KV_BLOCK_SHIFT = 11, KV_BLOCK = 1 << KV_BLOCK_SHIFT;
// Paged KV: physical block of logical block lb is btab[lb] (btab null = identity); `bs` = bytes
// between consecutive physical blocks of one layer.
struct KVMap {
  long long bs;
  const int* btab;
};
size_t kv_bytes_per_token_head(int kvtype, int hd);
// Store K (after norm+rope) and V for tokens at positions pos[t] into cache.
void kv_store(int kvtype, void* kcache, void* vcache, const float* k, const float* v,
              const int* pos, int nkv, int hd, int tokens, int k_tstride, int v_tstride, KVMap bs,
              cudaStream_t s);
// Decode/verify attention for `tokens` (<= kMaxB) consecutive queries at positions
// pos0..pos0+tokens-1, causal against cache positions [0, pos0+t].
// q: [t][nh][q_hstride] (uses first hd), out: [t][nh*hd]
void attn_decode(int kvtype, const void* kcache, const void* vcache, const float* q, int q_hstride,
                 int q_tstride, float* out, int nh, int nkv, int hd, int pos0, int tokens,
                 float scale, float* work, size_t work_bytes, KVMap bs, cudaStream_t s);
size_t attn_decode_work_bytes(int max_ctx, int nh, int hd, int tokens);
size_t attn_decode_work_bytes_auto(int max_ctx, int nh, int nkv, int hd, int tokens);

// Prefill attention (tensor cores, flash-attention style): queries at positions pos0..pos0+T-1
// (all T tokens of this chunk already stored in the cache), causal.  q f32 [T][nh][q_hstride].
// work: f16 [T][nh][hd] + f32 [T][nh][hd] scratch (attn_prefill_work_bytes).
void attn_prefill(int kvtype, const void* kcache, const void* vcache, const float* q,
                  int q_hstride, int q_tstride, float* out, int nh, int nkv, int hd, int pos0,
                  int T, float scale, void* work, KVMap bs, cudaStream_t s);
size_t attn_prefill_work_bytes(int T, int nh, int hd);

// ---- Gated DeltaNet ---------------------------------------------------------------------------
// conv1d (kernel 4) + SiLU over T tokens.  conv_state [3][C] holds the previous 3 inputs (oldest
// first) and is updated.  If hist != null the pre-call state and all inputs are kept there for
// rollback (hist: [(3+T)][C]).
// snap (optional): conv state after token snap_t is also written there (checkpoints)
void ssm_conv(const float* x, int x_tstride, float* conv_state, const float* w, float* y, int C,
              int T, float* hist, cudaStream_t s, int snap_t = -1, float* snap = nullptr);
// rebuild conv_state from hist after accepting n_acc tokens
void ssm_conv_rollback(float* conv_state, const float* hist, int C, int n_acc, cudaStream_t s);

// Per-token pre-processing: l2-normalize q,k heads (in place in qkv), compute
// g = softplus(alpha + dt_bias) * a, beta = sigmoid(beta_raw).
void gdn_prep(float* qkv, int qkv_tstride, const float* alpha_raw, const float* beta_raw,
              int ab_tstride, const float* dt_bias, const float* a, float* g, float* beta,
              int nk, int nv, int dk, int T, float eps, cudaStream_t s);
// Recurrence over T tokens: state_out = f(state_in, tokens) (state_out may equal state_in).
// State layout [nv][dv][dk] (row j = column j of S); out [T][nv*dv] (may be null).
struct GdnBatch {  // per-layer pointers for the batched rollback
  const float* qkv;
  const float* g;
  const float* beta;
  float* state;
  float* conv_state;
  const float* conv_hist;
};
void gdn_rollback_batched(const GdnBatch* tab, int nl, int qts, int nk, int nv, int dk, int n_keep, int C, bool conv,
                          cudaStream_t s, bool half = false);
void gdn_recurrence(const float* qkv, int qkv_tstride, const float* g, const float* beta,
                    const float* state_in, float* state_out, float* out, int nk, int nv, int dk, int T,
                    cudaStream_t s, int snap_t = -1, float* snap = nullptr, bool half = false);
// out[t][h*dv+i] = rmsnorm(out_h) * w[i] * silu(z[t][h*dv+i])
void gdn_gated_norm(float* o, const float* z, int z_tstride, const float* w, int nv, int dv,
                    int T, float eps, cudaStream_t s, __half* o16 = nullptr, int8_t* xq = nullptr,
                    float* xd = nullptr);

// ---- sampling ------------------------------------------------------------------------------------
// argmax over rows: idx[r] = argmax(logits[r*n ...])
void argmax_rows(const float* logits, int n, int rows, int* idx, cudaStream_t s);
size_t topk_work_bytes(int rows);
void topk_rows(const float* logits, int V, int rows, int K, float* out_val, int* out_idx, void* work, cudaStream_t s);
// argmax of one row and its softmax probability
void argmax_prob(const float* logits, int n, int* idx, float* prob, cudaStream_t s);
// top-k candidates per row (k <= 64): writes ids/vals sorted desc.
void topk_rows(const float* logits, int n, int rows, int k, int* ids, float* vals, float* work,
               cudaStream_t s);
size_t topk_work_bytes(int n, int rows);

// RoPE frequency table (YaRN): angle of pair i = pos * freq[i], cos / sin scaled by mscale
void set_rope_table(const double* freq, int half, float mscale);
// sensenova_loop residual streams (2-stream mHC router / merge / leave)
void mhc_route(const float* a, const float* b, const float* w, const float* bias, int d, int T, float* branch, float* pk,
               cudaStream_t s);
void mhc_merge(float* a, float* b, const float* x, const float* pk, int d, int T, cudaStream_t s);
void mhc_leave(float* y, const float* a, const float* b, int64_t n, cudaStream_t s);
// ---- MoE ---------------------------------------------------------------------------------------
// y[a*N + r] = (W_e[r,:] . x[a / xdiv]) * escale[eid[a]] for assignments a < A, expert e = ids[a] (ids null: e = a,
// experts gathered into slots); W_e = W + e * estride.  Activations from quantize_q8 (K % 32 == 0).
void moe_matvec(uint32_t type, const void* W, size_t estride, size_t rb, int K, int N, int A, const int* ids,
                const int* eid, const float* escale, const int8_t* xq, const float* xd, int xdiv, float* y,
                cudaStream_t s, const uint8_t* const* wptr = nullptr);  // wptr: per-assignment expert pointers
// grouped MMQ (MoE prefill): group e multiplies W + e*estride by x rows [goff[e], goff[e+1]) into the same y rows;
// x from quantize_q8 (per 32); max_rows bounds the rows of a group
void mmq_grouped(uint32_t type, const void* W, size_t estride, size_t rb, int K, int N, const int* goff, int ngroups,
                 int max_rows, const int8_t* xq, const float* xd, float* y, int ldy, cudaStream_t s);
// dst slot a <- the nbytes of expert ids[a] (src + ids[a]*estride; src may be mapped host memory)
void gather_experts(const void* src, size_t estride, size_t nbytes, const int* ids, int A, void* dst, cudaStream_t s);
// the same with a VRAM cache: expert e is read in place from eptr[e] (non-null: cached) or copied to dst slot a; ptr[a]
// receives the expert's address, cnt[e] (optional) counts its uses
void gather_experts_cached(const void* src, size_t estride, const int* ids, int A, const uint8_t* const* eptr, void* dst,
                           const uint8_t** ptr, int* cnt, cudaStream_t s);
}  // namespace qk
