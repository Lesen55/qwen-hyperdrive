// CUDA kernels of the Inkling 2 (inkling2_loop) runtime.
#include "inkling_kernels.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "common.h"
#include "kernels.h"

namespace ik {
namespace {

inline int cdiv(int64_t a, int64_t b) { return (int)((a + b - 1) / b); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// block-wide sum of NV values (blockDim.x multiple of 32, <= 1024); every thread gets the result
template <int NV>
__device__ __forceinline__ void block_sum(float (&v)[NV]) {
  __shared__ float red[NV][32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
  for (int i = 0; i < NV; i++) {
    v[i] = warp_sum(v[i]);
    if (lane == 0) red[i][w] = v[i];
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < NV; i++) {
    float x = lane < nw ? red[i][lane] : 0.f;
    v[i] = warp_sum(x);
  }
  __syncthreads();
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + __expf(-x)); }

// ---------------------------------------------------------------------------------------------
__global__ void k_rms_rows(const float* __restrict__ x, const float* __restrict__ w, float* y, int n, float eps) {
  const float* xr = x + (size_t)blockIdx.x * n;
  float* yr = y + (size_t)blockIdx.x * n;
  float v[1] = {0.f};
  for (int i = threadIdx.x; i < n; i += blockDim.x) v[0] += xr[i] * xr[i];
  block_sum<1>(v);
  const float sc = rsqrtf(v[0] / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * sc * (w ? w[i] : 1.f);
}

// one warp per (token, head)
__global__ void k_head_rms(float* x, int hd, int nh, int ts, int T, float eps) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= T * nh) return;
  float* p = x + (size_t)(wid / nh) * ts + (size_t)(wid % nh) * hd;
  float s = 0.f;
  for (int i = lane; i < hd; i += 32) s += p[i] * p[i];
  s = warp_sum(s);
  const float sc = rsqrtf(s / hd + eps);
  for (int i = lane; i < hd; i += 32) p[i] *= sc;
}

__global__ void k_clip_heads(float* x, int ts, const float* __restrict__ scale, int nh, int hd, int T, float thr,
                             int tau, const int* __restrict__ posd, float alpha, float floor_) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= T * nh) return;
  const int t = wid / nh, h = wid % nh;
  float* p = x + (size_t)t * ts + (size_t)h * hd;
  float s = 0.f;
  for (int i = lane; i < hd; i += 32) {
    const float v = p[i] * (scale ? scale[h * hd + i] : 1.f);
    p[i] = v;
    s += v * v;
  }
  s = warp_sum(s);
  const float nrm = sqrtf(s) + 1e-6f;
  float f = fminf(fmaxf(thr / nrm, 0.f), 1.f);
  if (tau) f *= 1.f + alpha * logf(fmaxf((float)(*posd + t + 1) / floor_, 1.f));
  for (int i = lane; i < hd; i += 32) p[i] *= f;
}

// ---------------------------------------------------------------------------------------------
// depth mixing: one block per token
__global__ void __launch_bounds__(256) k_mix(MixSrc st, const float* __restrict__ query, int nq, int E, MixDst out) {
  const int t = blockIdx.x;
  __shared__ float inv[8], sc[8][4];
  for (int s = 0; s < st.n; s++) {
    const float* xr = st.p[s] + (size_t)t * E;
    float v[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
      const float x = xr[i];
      v[0] += x * x;
#pragma unroll
      for (int j = 0; j < 4; j++)
        if (j < nq) v[1 + j] += x * query[(size_t)j * E + i];
    }
    block_sum<5>(v);
    if (threadIdx.x == 0) {
      const float r = rsqrtf(v[0] / E + 1e-6f);
      inv[s] = r;
      for (int j = 0; j < nq; j++) sc[s][j] = v[1 + j] * r * rsqrtf((float)E);
    }
  }
  __syncthreads();
  __shared__ float p[8][4];
  if (threadIdx.x < nq) {
    const int j = threadIdx.x;
    float m = -FLT_MAX;
    for (int s = 0; s < st.n; s++) m = fmaxf(m, sc[s][j]);
    float z = 0.f;
    for (int s = 0; s < st.n; s++) z += __expf(sc[s][j] - m);
    for (int s = 0; s < st.n; s++) p[s][j] = __expf(sc[s][j] - m) / z * inv[s];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < st.n; s++) {
      const float x = st.p[s][(size_t)t * E + i];
#pragma unroll
      for (int j = 0; j < 4; j++) acc[j] += p[s][j] * x;
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (j < nq) out.p[j][(size_t)t * E + i] = acc[j];
  }
}

__global__ void k_concat_rms(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int E) {
  const int t = blockIdx.x;
  const float* ar = a + (size_t)t * E;
  const float* br = b + (size_t)t * E;
  float v[2] = {0.f, 0.f};
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    v[0] += ar[i] * ar[i];
    v[1] += br[i] * br[i];
  }
  block_sum<2>(v);
  const float ra = rsqrtf(v[0] / E + 1e-6f), rb = rsqrtf(v[1] / E + 1e-6f);
  float* yr = y + (size_t)t * 2 * E;
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    yr[i] = ar[i] * ra;
    yr[E + i] = br[i] * rb;
  }
}

__global__ void k_ctrl_combine(float* out, const float* cur, const float* __restrict__ msg, const float* __restrict__ g,
                               const float* __restrict__ f, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float c = cur[i];
  out[i] = c - tanhf(f[i]) * c + tanhf(g[i]) * msg[i];
}

// ---------------------------------------------------------------------------------------------
__global__ void k_conv(const float* __restrict__ x, int T, int C, const float* __restrict__ w,
                       const float* __restrict__ dyn, int dyn_ts, int cph, const float* __restrict__ hist,
                       float* __restrict__ y, float inv) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)T * C) return;
  const int t = (int)(idx / C), c = (int)(idx % C);
  float xin[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int tt = t - 3 + j;
    xin[j] = tt >= 0 ? x[(size_t)tt * C + c] : hist[(size_t)(tt + 3) * C + c];
  }
  float acc = xin[3];
#pragma unroll
  for (int j = 0; j < 4; j++) acc += w[(size_t)c * 4 + j] * xin[j];
  if (dyn) {
    const float* d = dyn + (size_t)t * dyn_ts + (c / cph) * 4;
#pragma unroll
    for (int lag = 0; lag < 4; lag++) acc += xin[3 - lag] * d[lag];
  }
  y[idx] = acc * inv;
}
__global__ void k_conv_hist(const float* __restrict__ x, int T, int C, float* hist) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  // new[i] = input at time T-3+i: x if >= 0, else old hist[T+i] (read before it can be overwritten: T+i > i)
  for (int i = 0; i < 3; i++) {
    const int tt = T - 3 + i;
    hist[(size_t)i * C + c] = tt >= 0 ? x[(size_t)tt * C + c] : hist[(size_t)(T + i) * C + c];
  }
}

// ---------------------------------------------------------------------------------------------
__global__ void k_kv_store(const float* __restrict__ k, int ks, const float* __restrict__ v, int vs, __half* kc,
                           __half* vc, int rowlen, const int* __restrict__ posd, int T, int ring) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)T * rowlen) return;
  const int t = (int)(idx / rowlen), i = (int)(idx % rowlen);
  const int p = *posd + t;
  const size_t row = ring ? (size_t)(p % ring) : (size_t)p;
  kc[row * rowlen + i] = __float2half(k[(size_t)t * ks + i]);
  vc[row * rowlen + i] = __float2half(v[(size_t)t * vs + i]);
}

// relative-position attention with sinks (SIMT flash attention).  Block = (kv head, tile of QT query tokens,
// key split); the G = nh/nkv heads sharing the kv head and the QT tokens form NQ = G*QT query rows.
constexpr int AT_KC = 32;    // keys per chunk
constexpr int AT_HD = 128;   // head dim
constexpr int AT_KS = 65;    // K smem row stride in 32-bit words (conflict-free column reads)
constexpr int AT_NT = 256;
struct AttnArgs {
  const __half* kc;
  const __half* vc;
  int ring;
  const float* q;
  int qs;
  const float* r;
  const float* sink;
  const float* table;
  int extent, window;
  float* out;
  float* part;  // splits > 1: [split][T][nh] (m, l) then [split][T][nh][hd] acc
  const int* posd;
  int T, nh, nkv, G, splits;
};

template <int NQ>
__global__ void __launch_bounds__(AT_NT) k_attn(AttnArgs a) {
  constexpr int QT_MAX = NQ;  // upper bound on tokens per tile (G >= 1)
  (void)QT_MAX;
  const int kvh = blockIdx.x, tile = blockIdx.y, split = blockIdx.z;
  const int pos0 = *a.posd;
  const int G = a.G, QT = NQ / G;
  const int t0 = tile * QT;
  const int nt = min(QT, a.T - t0);
  const int nrow = nt * G;
  const int rowlen = a.nkv * AT_HD;
  __shared__ float Qs[NQ][AT_HD];
  __shared__ float Rs[NQ][16];
  __shared__ uint32_t Ks[AT_KC * AT_KS];
  __shared__ __align__(16) __half Vs[AT_KC][AT_HD];
  __shared__ float Ps[NQ][AT_KC];
  __shared__ float Ms[NQ], Ls[NQ], Cs[NQ];
  __shared__ int Qp[NQ];
  const float qscale = rsqrtf((float)AT_HD);
  for (int i = threadIdx.x; i < NQ * AT_HD; i += AT_NT) {
    const int row = i / AT_HD, d = i % AT_HD;
    const int tt = row / G, g = row % G;
    float v = 0.f;
    if (row < nrow) v = a.q[(size_t)(t0 + tt) * a.qs + (size_t)(kvh * G + g) * AT_HD + d] * qscale;
    Qs[row][d] = v;
  }
  for (int i = threadIdx.x; i < NQ * 16; i += AT_NT) {
    const int row = i / 16, j = i % 16;
    const int tt = row / G, g = row % G;
    Rs[row][j] = row < nrow ? a.r[(size_t)(t0 + tt) * a.nh * 16 + (size_t)(kvh * G + g) * 16 + j] : 0.f;
  }
  for (int i = threadIdx.x; i < NQ; i += AT_NT) {
    Ms[i] = -INFINITY;
    Ls[i] = 0.f;
    Qp[i] = pos0 + t0 + min(i / G, nt - 1);
  }
  // key range of this tile / split
  const int qlo = pos0 + t0, qhi = pos0 + t0 + nt - 1;
  int klo = a.window ? max(0, qlo - a.window + 1) : 0;
  int khi = qhi;
  if (a.splits > 1) {  // equal key ranges (multiples of the chunk) of this tile's keys
    const int sl = (khi - klo + 1 + a.splits * AT_KC - 1) / (a.splits * AT_KC) * AT_KC;
    const int s0 = klo + split * sl;
    khi = min(khi, s0 + sl - 1);
    klo = s0;
  }
  constexpr int NACC = (NQ * AT_HD + AT_NT - 1) / AT_NT;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; i++) acc[i] = 0.f;
  __syncthreads();
  for (int kb = klo; kb <= khi; kb += AT_KC) {
    const int nk = min(AT_KC, khi - kb + 1);
    // stage K (word-wise, padded rows) and V
    for (int i = threadIdx.x; i < AT_KC * (AT_HD / 8); i += AT_NT) {
      const int key = i / (AT_HD / 8), c = i % (AT_HD / 8);
      uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
      if (key < nk) {
        const int p = kb + key;
        const size_t row = a.ring ? (size_t)(p % a.ring) : (size_t)p;
        kv = *(const uint4*)(a.kc + row * rowlen + (size_t)kvh * AT_HD + c * 8);
        vv = *(const uint4*)(a.vc + row * rowlen + (size_t)kvh * AT_HD + c * 8);
      }
      uint32_t* kd = Ks + key * AT_KS + c * 4;
      kd[0] = kv.x; kd[1] = kv.y; kd[2] = kv.z; kd[3] = kv.w;
      *(uint4*)&Vs[key][c * 8] = vv;
    }
    __syncthreads();
    // scores
    for (int i = threadIdx.x; i < NQ * AT_KC; i += AT_NT) {
      const int row = i / AT_KC, key = i % AT_KC;
      float sc = -INFINITY;
      if (row < nrow && key < nk) {
        const int kp = kb + key, qp = Qp[row];
        const int dist = qp - kp;
        if (dist >= 0 && (!a.window || dist < a.window)) {
          float s = 0.f;
          const uint32_t* kr = Ks + key * AT_KS;
#pragma unroll 16
          for (int d = 0; d < AT_HD / 2; d++) {
            const float2 kf = __half22float2(*(const __half2*)&kr[d]);
            s += Qs[row][2 * d] * kf.x + Qs[row][2 * d + 1] * kf.y;
          }
          if (dist < a.extent) {
            const float* tb = a.table + (size_t)dist * 16;
#pragma unroll
            for (int j = 0; j < 16; j++) s += Rs[row][j] * __ldg(tb + j);
          }
          sc = s;
        }
      }
      Ps[row][key] = sc;
    }
    __syncthreads();
    // online softmax: one warp per row, lane = key
    for (int row = threadIdx.x >> 5; row < NQ; row += AT_NT / 32) {
      const int lane = threadIdx.x & 31;
      const float s = Ps[row][lane];
      const float mo = Ms[row];
      const float mn = fmaxf(mo, warp_max(s));
      const float p = mn == -INFINITY ? 0.f : __expf(s - mn);
      const float ls = warp_sum(p);
      Ps[row][lane] = p;
      if (lane == 0) {
        const float c = mn == -INFINITY ? 1.f : __expf(mo - mn);
        Cs[row] = c;
        Ls[row] = Ls[row] * c + ls;
        Ms[row] = mn;
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NACC; i++) {
      const int idx = threadIdx.x + i * AT_NT;
      if (idx < NQ * AT_HD) {
        const int row = idx / AT_HD, d = idx % AT_HD;
        float v = acc[i] * Cs[row];
#pragma unroll 8
        for (int key = 0; key < AT_KC; key++) v += Ps[row][key] * __half2float(Vs[key][d]);
        acc[i] = v;
      }
    }
    __syncthreads();
  }
  // epilogue
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    const int idx = threadIdx.x + i * AT_NT;
    if (idx >= NQ * AT_HD) continue;
    const int row = idx / AT_HD, d = idx % AT_HD;
    if (row >= nrow) continue;
    const int t = t0 + row / G, h = kvh * G + row % G;
    if (a.splits == 1) {
      const float m = Ms[row], sk = a.sink[(size_t)t * a.nh + h];
      const float M = fmaxf(m, sk);
      const float c = __expf(m - M);
      a.out[(size_t)t * a.nh * AT_HD + (size_t)h * AT_HD + d] = acc[i] * c / (Ls[row] * c + __expf(sk - M));
    } else {
      const size_t th = ((size_t)split * a.T + t) * a.nh + h;
      a.part[(size_t)a.splits * a.T * a.nh * 2 + th * AT_HD + d] = acc[i];
      if (d == 0) {
        a.part[th * 2] = Ms[row];
        a.part[th * 2 + 1] = Ls[row];
      }
    }
  }
}

__global__ void k_attn_combine(const float* __restrict__ part, const float* __restrict__ sink, float* out, int T,
                               int nh, int splits) {
  const int th = blockIdx.x;  // t*nh + h
  const int d = threadIdx.x;  // < 128
  float M = sink[th];
  for (int s = 0; s < splits; s++) M = fmaxf(M, part[((size_t)s * T * nh + th) * 2]);
  float L = __expf(sink[th] - M), acc = 0.f;
  const float* accs = part + (size_t)splits * T * nh * 2;
  for (int s = 0; s < splits; s++) {
    const size_t i = (size_t)s * T * nh + th;
    const float m = part[i * 2];
    if (m == -INFINITY) continue;
    const float c = __expf(m - M);
    L += part[i * 2 + 1] * c;
    acc += accs[i * AT_HD + d] * c;
  }
  out[(size_t)th * AT_HD + d] = acc / L;
}

// splits of the key range for an upper bound of the keys per query tile
void attn_plan(int T, int nh, int nkv, int max_keys, int& NQ, int& QT, int& splits) {
  const int G = nh / nkv;
  QT = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
  if (G * QT > 24) QT = std::max(1, 24 / G);
  NQ = G * QT;
  const int tiles = cdiv(T, QT);
  const int base = nkv * tiles;
  splits = 1;
  if (base < 136) splits = std::clamp(std::min(cdiv(max_keys, 256), cdiv(272, base)), 1, 64);
}

// ---------------------------------------------------------------------------------------------
__global__ void k_mul_sigmoid(float* o, const float* __restrict__ g, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] *= sigmoidf(g[i]);
}
__global__ void k_add(float* y, const float* a, const float* b, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}
__global__ void k_softcap(float* x, int64_t n, float cap) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// router: one block of n_exp (<= 256) threads per token
__global__ void k_moe_route(const float* __restrict__ logits, const float* __restrict__ bias, int n_exp, int k,
                            float scale, int* ids, float* w) {
  const int t = blockIdx.x, e = threadIdx.x;
  __shared__ float sel[256], prob[256];
  __shared__ int chosen[16];
  if (e < n_exp) {
    const float p = sigmoidf(logits[(size_t)t * n_exp + e]);
    prob[e] = p;
    sel[e] = p + (bias ? bias[e] : 0.f);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    for (int j = 0; j < k; j++) {
      float bv = -INFINITY;
      int bi = n_exp;
      for (int i = lane; i < n_exp; i += 32)
        if (sel[i] > bv || (sel[i] == bv && i < bi)) { bv = sel[i]; bi = i; }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0) {
        chosen[j] = bi;
        sel[bi] = -INFINITY;
      }
      __syncwarp();
    }
    if (lane == 0) {
      float sum = 0.f;
      for (int j = 0; j < k; j++) sum += prob[chosen[j]];
      const float inv = scale / (sum + 1e-20f);
      for (int j = 0; j < k; j++) {
        ids[(size_t)t * k + j] = chosen[j];
        w[(size_t)t * k + j] = prob[chosen[j]] * inv;
      }
    }
  }
}

__global__ void k_situ(const float* __restrict__ gu, int A, int N, float beta, float lbeta, const int* __restrict__ eid,
                       const float* __restrict__ rscale, float* __restrict__ h) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)A * N) return;
  const int a = (int)(idx / N), i = (int)(idx % N);
  const float sc = rscale ? rscale[eid[a]] : 1.f;
  const float g = gu[(size_t)a * 2 * N + i] * sc, u = gu[(size_t)a * 2 * N + N + i] * sc;
  h[idx] = beta * tanhf(g / beta) * sigmoidf(g) * (lbeta * tanhf(u / lbeta));
}

__global__ void k_moe_combine(const float* __restrict__ y, const int* __restrict__ map, const float* __restrict__ coef,
                              int T, int k, int N, float* __restrict__ out) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)T * N) return;
  const int t = (int)(idx / N), n = (int)(idx % N);
  float acc = 0.f;
  for (int j = 0; j < k; j++) {
    const int a = t * k + j;
    const int r = map ? map[a] : a;
    acc += coef[a] * y[(size_t)r * N + n];
  }
  out[idx] = acc;
}
__global__ void k_moe_coef(const float* w, const int* ids, const float* escale, int A, float* coef) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a < A) coef[a] = w[a] * (escale ? escale[ids[a]] : 1.f);
}
__global__ void k_gather_rows(const float* __restrict__ x, const int* __restrict__ rows, int A, int K, float* out) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)A * K) return;
  const int a = (int)(idx / K), i = (int)(idx % K);
  out[idx] = x[(size_t)rows[a] * K + i];
}

}  // namespace

// =============================================================================================
void rms_rows(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s) {
  if (rows > 0) k_rms_rows<<<rows, 256, 0, s>>>(x, w, y, n, eps);
}
void head_rms(float* x, int hd, int nh, int ts, int T, float eps, cudaStream_t s) {
  if (T > 0) k_head_rms<<<cdiv((int64_t)T * nh * 32, 256), 256, 0, s>>>(x, hd, nh, ts, T, eps);
}
void clip_heads(float* x, int ts, const float* scale, int nh, int hd, int T, float thr, bool tau, const int* posd,
                float alpha, float floor_, cudaStream_t s) {
  if (T > 0)
    k_clip_heads<<<cdiv((int64_t)T * nh * 32, 256), 256, 0, s>>>(x, ts, scale, nh, hd, T, thr, tau ? 1 : 0, posd,
                                                                  alpha, floor_);
}
void mix(const MixSrc& st, const float* query, int nq, int E, int T, const MixDst& out, cudaStream_t s) {
  QE_ASSERT(st.n >= 1 && st.n <= 8 && nq >= 1 && nq <= 4);
  if (T > 0) k_mix<<<T, 256, 0, s>>>(st, query, nq, E, out);
}
void concat_rms(const float* a, const float* b, float* y, int E, int T, cudaStream_t s) {
  if (T > 0) k_concat_rms<<<T, 256, 0, s>>>(a, b, y, E);
}
void ctrl_combine(float* out, const float* cur, const float* msg, const float* g, const float* f, int64_t n,
                  cudaStream_t s) {
  if (n > 0) k_ctrl_combine<<<cdiv(n, 256), 256, 0, s>>>(out, cur, msg, g, f, n);
}
void conv(const float* x, int T, int C, const float* w, const float* dyn, int dyn_ts, int ch_per_head, float* hist,
          float* y, float inv, cudaStream_t s) {
  if (T <= 0) return;
  k_conv<<<cdiv((int64_t)T * C, 256), 256, 0, s>>>(x, T, C, w, dyn, dyn_ts, ch_per_head, hist, y, inv);
  k_conv_hist<<<cdiv(C, 256), 256, 0, s>>>(x, T, C, hist);
}
void kv_store(const float* k, int ks, const float* v, int vs, __half* kc, __half* vc, int rowlen, const int* posd,
              int T, int ring, cudaStream_t s) {
  if (T > 0) k_kv_store<<<cdiv((int64_t)T * rowlen, 256), 256, 0, s>>>(k, ks, v, vs, kc, vc, rowlen, posd, T, ring);
}
size_t attn_work_bytes(int T, int nh, int nkv, int max_keys) {
  int NQ, QT, splits;
  attn_plan(T, nh, nkv, max_keys, NQ, QT, splits);
  return splits > 1 ? (size_t)splits * T * nh * (2 + AT_HD) * sizeof(float) : 0;
}
void attn(const __half* kc, const __half* vc, int ring, const float* q, int qs, const float* r, const float* sink,
          const float* table, int extent, int window, float* out, const int* posd, int max_keys, int T, int nh, int nkv,
          int hd, void* work, size_t work_bytes, cudaStream_t s) {
  QE_ASSERT(hd == AT_HD && nh % nkv == 0);
  if (T <= 0) return;
  AttnArgs a{kc, vc, ring, q, qs, r, sink, table, extent, window, out, (float*)work, posd, T, nh, nkv, nh / nkv, 1};
  int NQ, QT;
  attn_plan(T, nh, nkv, max_keys, NQ, QT, a.splits);
  const size_t per = (size_t)T * nh * (2 + AT_HD) * sizeof(float);
  if (a.splits > 1 && (size_t)a.splits * per > work_bytes) a.splits = std::max(1, (int)(work_bytes / per));
  const dim3 grid(nkv, cdiv(T, QT), a.splits);
  switch (NQ) {
#define CASE(n) case n: k_attn<n><<<grid, AT_NT, 0, s>>>(a); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(6) CASE(8) CASE(12) CASE(16) CASE(24)
#undef CASE
    default: QE_FATAL("attn: unsupported query rows %d (G=%d)", NQ, nh / nkv);
  }
  if (a.splits > 1) k_attn_combine<<<T * nh, AT_HD, 0, s>>>((const float*)work, sink, out, T, nh, a.splits);
}
void mul_sigmoid(float* o, const float* g, int64_t n, cudaStream_t s) {
  if (n > 0) k_mul_sigmoid<<<cdiv(n, 256), 256, 0, s>>>(o, g, n);
}
void add(float* y, const float* a, const float* b, int64_t n, cudaStream_t s) {
  if (n > 0) k_add<<<cdiv(n, 256), 256, 0, s>>>(y, a, b, n);
}
void softcap(float* x, int64_t n, float cap, cudaStream_t s) {
  if (n > 0) k_softcap<<<cdiv(n, 256), 256, 0, s>>>(x, n, cap);
}
void moe_route(const float* logits, const float* bias, int n_exp, int k, float scale, int T, int* ids, float* w,
               cudaStream_t s) {
  QE_ASSERT(n_exp <= 256 && k <= 16);
  if (T > 0) k_moe_route<<<T, std::max(32, cdiv(n_exp, 32) * 32), 0, s>>>(logits, bias, n_exp, k, scale, ids, w);
}
void situ(const float* gu, int A, int N, float beta, float lbeta, const int* eid, const float* rscale, float* h,
          cudaStream_t s) {
  if (A > 0) k_situ<<<cdiv((int64_t)A * N, 256), 256, 0, s>>>(gu, A, N, beta, lbeta, eid, rscale, h);
}
void moe_combine(const float* y, const int* map, const float* coef, int T, int k, int N, float* out, cudaStream_t s) {
  if (T > 0) k_moe_combine<<<cdiv((int64_t)T * N, 256), 256, 0, s>>>(y, map, coef, T, k, N, out);
}
void moe_coef(const float* w, const int* ids, const float* escale, int A, float* coef, cudaStream_t s) {
  if (A > 0) k_moe_coef<<<cdiv(A, 256), 256, 0, s>>>(w, ids, escale, A, coef);
}
void gather_rows(const float* x, const int* rows, int A, int K, float* out, cudaStream_t s) {
  if (A > 0) k_gather_rows<<<cdiv((int64_t)A * K, 256), 256, 0, s>>>(x, rows, A, K, out);
}

}  // namespace ik
