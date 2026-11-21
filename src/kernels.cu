#include <type_traits>
// CUDA kernels for qwen35-engine.
#ifndef QE_HIP
#include <cuda_fp16.h>
#endif

#include <cfloat>

#include "common.h"
#include "kernels.h"

#include <map>
#include <tuple>
#include "quant.h"

namespace qk {

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
static __device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
static inline int cdiv(int64_t a, int64_t b) { return (int)((a + b - 1) / b); }

#ifdef __HIP_PLATFORM_AMD__
// No cp.async on AMD: synchronous copies (same data movement, no async overlap).
static __device__ __forceinline__ void cp_async16(void* smem, const void* gmem) { *(uint4*)smem = *(const uint4*)gmem; }
static __device__ __forceinline__ void cp_async8(void* smem, const void* gmem) { *(uint2*)smem = *(const uint2*)gmem; }
static __device__ __forceinline__ void cp_async_commit() {}
template <int N>
static __device__ __forceinline__ void cp_async_wait() { __syncthreads(); }
#else
static __device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
static __device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(a), "l"(gmem));
}
static __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
static __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
#endif

// =============================================================================================
// Activation quantization
// =============================================================================================
__global__ void k_quantize_q8(const float* __restrict__ x, int8_t* __restrict__ xq,
                              float* __restrict__ xd, int K, int ngroups_total) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= ngroups_total) return;
  const int64_t i = (int64_t)warp * 32 + lane;  // groups are contiguous across rows (K % 32 == 0)
  const float v = x[i];
  const float amax = warp_max(fabsf(v));
  const float d = amax / 127.f;
  const float id = d > 0.f ? 1.f / d : 0.f;
  xq[i] = (int8_t)__float2int_rn(v * id);
  if (lane == 0) xd[warp] = d;
  (void)K;
}

void quantize_q8(const float* x, int8_t* xq, float* xd, int K, int B, cudaStream_t s) {
  const int ng = K / 32 * B;
  if (ng == 0) return;
  k_quantize_q8<<<cdiv(ng, 8), 256, 0, s>>>(x, xq, xd, K, ng);
}

// =============================================================================================
// Quantized weight decoding into a common "W32" form:
//   32 int8 weights packed in 8 ints, per-8-group scale sc[g] and offset off[g] such that
//   dot(w_real, x_real) = xd * sum_g( sc[g]*dot_i8(w_g, xq_g) + off[g]*sum(xq_g) ).
// =============================================================================================
struct W32 {
  int w[8];
  float sc[4];
  float off[4];
};

static __device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*)p; }
static __device__ __forceinline__ uint32_t ld32a2(const uint8_t* p) {  // 2-byte aligned
  const uint16_t* q = (const uint16_t*)p;
  return (uint32_t)q[0] | ((uint32_t)q[1] << 16);
}
static __device__ __forceinline__ uint32_t ld32a4(const uint8_t* p) { return *(const uint32_t*)p; }
static __device__ __forceinline__ float h2f(const uint8_t* p) {
  return __half2float(__ushort_as_half(*(const uint16_t*)p));
}
// expand 4 sign bits into byte masks and apply to 4 unsigned grid bytes
static __device__ __forceinline__ int apply_signs4(uint32_t g, uint32_t s4) {
  const uint32_t m = (((s4 & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu;
  // per-byte negate where m = 0xFF: (g ^ 0xFF) + 1 == -g; grid bytes are never 0, so no carry
  // crosses a byte (__vsub4 is emulated on sm_86)
  return (int)((g ^ m) + (m & 0x01010101u));
}

template <int T>
struct QDec;

template <>
struct QDec<QT_IQ2_XXS> {
  static constexpr int SCALES = 1;  // distinct scale groups per 32 weights
  static constexpr int BB = 66;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const uint32_t a0 = ld32a2(b + 2 + 8 * ib), a1 = ld32a2(b + 2 + 8 * ib + 4);
    const float db = h2f(b) * (0.5f + (a1 >> 28)) * 0.25f;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint64_t g = qtab_dev::iq2xxs_grid[(a0 >> (8 * l)) & 255];
      const uint32_t s = iq2_signs8((a1 >> (7 * l)) & 127);
      o.w[2 * l] = apply_signs4((uint32_t)g, s);
      o.w[2 * l + 1] = apply_signs4((uint32_t)(g >> 32), s >> 4);
      o.sc[l] = db;
    }
  }
};
template <>
struct QDec<QT_IQ2_XS> {
  static constexpr int SCALES = 2;  // distinct scale groups per 32 weights
  static constexpr int BB = 74;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const float d = h2f(b);
    const uint32_t scb = b[66 + ib];
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint32_t q = ld16(b + 2 + 2 * (4 * ib + l));
      const uint64_t g = qtab_dev::iq2xs_grid[q & 511];
      const uint32_t s = iq2_signs8(q >> 9);
      o.w[2 * l] = apply_signs4((uint32_t)g, s);
      o.w[2 * l + 1] = apply_signs4((uint32_t)(g >> 32), s >> 4);
      o.sc[l] = d * (0.5f + ((scb >> (4 * (l / 2))) & 0xf)) * 0.25f;
    }
  }
};
template <>
struct QDec<QT_IQ2_S> {
  static constexpr int SCALES = 2;  // distinct scale groups per 32 weights
  static constexpr int BB = 82;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const float d = h2f(b);
    const uint32_t qs4 = ld32a2(b + 2 + 4 * ib);
    const uint32_t sg4 = ld32a2(b + 2 + 32 + 4 * ib);
    const uint32_t qh = b[66 + ib];
    const uint32_t scb = b[74 + ib];
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint64_t g = qtab_dev::iq2s_grid[((qs4 >> (8 * l)) & 0xff) | ((qh << (8 - 2 * l)) & 0x300)];
      const uint32_t s = sg4 >> (8 * l);
      o.w[2 * l] = apply_signs4((uint32_t)g, s);
      o.w[2 * l + 1] = apply_signs4((uint32_t)(g >> 32), s >> 4);
      o.sc[l] = d * (0.5f + ((scb >> (4 * (l / 2))) & 0xf)) * 0.25f;
    }
  }
};
template <>
struct QDec<QT_IQ3_XXS> {
  static constexpr int SCALES = 1;  // distinct scale groups per 32 weights
  static constexpr int BB = 98;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t a = ld32a2(b + 2 + 64 + 4 * ib);
    const float db = h2f(b) * (0.5f + (a >> 28)) * 0.5f;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint32_t s = iq2_signs8((a >> (7 * l)) & 127);
      o.w[2 * l] = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l]], s);
      o.w[2 * l + 1] = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l + 1]], s >> 4);
      o.sc[l] = db;
    }
  }
};
template <>
struct QDec<QT_IQ3_S> {
  static constexpr int SCALES = 1;  // distinct scale groups per 32 weights
  static constexpr int BB = 110;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const float db = h2f(b) * (1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf));
    const uint32_t qa = ld32a2(b + 2 + 8 * ib), qb = ld32a2(b + 2 + 8 * ib + 4);
    const uint32_t qh = b[66 + ib];
    const uint32_t sg4 = ld32a2(b + 74 + 4 * ib);
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint32_t q2 = l < 2 ? qa : qb;
      const int sh = 16 * (l & 1);
      const uint32_t g1 = qtab_dev::iq3s_grid[((q2 >> sh) & 0xff) | ((qh << (8 - 2 * l)) & 256)];
      const uint32_t g2 = qtab_dev::iq3s_grid[((q2 >> (sh + 8)) & 0xff) | ((qh << (7 - 2 * l)) & 256)];
      const uint32_t s = sg4 >> (8 * l);
      o.w[2 * l] = apply_signs4(g1, s);
      o.w[2 * l + 1] = apply_signs4(g2, s >> 4);
      o.sc[l] = db;
    }
  }
};
template <>
struct QDec<QT_IQ1_S> {
  static constexpr int SCALES = 1;  // distinct scale groups per 32 weights
  static constexpr int BB = 50;
  static constexpr bool OFF = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const uint32_t qh = ld16(b + 34 + 2 * ib);
    const float dl = h2f(b) * (2 * ((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint32_t qs4 = ld32a2(b + 2 + 4 * ib);
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint64_t g = qtab_dev::iq1s_grid[((qs4 >> (8 * l)) & 0xff) | (((qh >> (3 * l)) & 7) << 8)];
      o.w[2 * l] = (int)(uint32_t)g;
      o.w[2 * l + 1] = (int)(uint32_t)(g >> 32);
      o.sc[l] = dl;
      o.off[l] = dl * delta;
    }
  }
};
template <>
struct QDec<QT_IQ1_M> {
  static constexpr int SCALES = 4;  // distinct scale groups per 32 weights
  static constexpr int BB = 56;
  static constexpr bool OFF = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const uint16_t* sc = (const uint16_t*)(b + 48);
    const uint16_t u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = __half2float(__ushort_as_half(u));
    const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
    const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
    const uint8_t* qs = b + 4 * ib;
    const uint8_t* qh = b + 32 + 2 * ib;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint32_t h = qh[l / 2];
      const uint32_t idx = qs[l] | ((l & 1) ? ((h << 4) & 0x700) : ((h << 8) & 0x700));
      const float delta = (h & ((l & 1) ? 0x80 : 0x08)) ? -IQ1S_DELTA : IQ1S_DELTA;
      const uint64_t g = qtab_dev::iq1s_grid[idx];
      o.w[2 * l] = (int)(uint32_t)g;
      o.w[2 * l + 1] = (int)(uint32_t)(g >> 32);
      const float dl = l < 2 ? dl1 : dl2;
      o.sc[l] = dl;
      o.off[l] = dl * delta;
    }
  }
};
// 16-entry signed byte table lookup for 4 nibbles stored in the low nibble of each byte of x
static __device__ __forceinline__ int lut16(uint32_t x, const uint32_t* t) {
  const uint32_t sel = (x & 0xF) | ((x >> 4) & 0xF0) | ((x >> 8) & 0xF00) | ((x >> 12) & 0xF000);
  const uint32_t lo = __byte_perm(t[0], t[1], sel & 0x7777);
  const uint32_t hi = __byte_perm(t[2], t[3], sel & 0x7777);
  const uint32_t m = ((x >> 3) & 0x01010101u) * 0xFFu;
  return (int)((lo & ~m) | (hi & m));
}
template <>
struct QDec<QT_IQ4_XS> {
  static constexpr int SCALES = 1;  // distinct scale groups per 32 weights
  static constexpr int BB = 136;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, W32& o) {
    const uint32_t sh = ld16(b + 2);
    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = h2f(b) * (ls - 32);
    const uint32_t* tab = (const uint32_t*)qtab_dev::kvalues_iq4nl;
    uint32_t t[4] = {tab[0], tab[1], tab[2], tab[3]};
    const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t q = ld32a4(qs + 4 * i);
      o.w[i] = lut16(q & 0x0F0F0F0F, t);
      o.w[4 + i] = lut16((q >> 4) & 0x0F0F0F0F, t);
    }
#pragma unroll
    for (int l = 0; l < 4; l++) o.sc[l] = dl;
  }
};
template <>
struct QDec<QT_Q2_K> {
  static constexpr int SCALES = 2;  // distinct scale groups per 32 weights
  static constexpr int BB = 84;
  static constexpr bool OFF = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, W32& o) {
    const float d = h2f(b + 80), dmin = h2f(b + 82);
    const uint8_t* q = b + 16 + (g / 4) * 32;
    const int shift = 2 * (g % 4);
#pragma unroll
    for (int i = 0; i < 8; i++) o.w[i] = (int)((ld32a4(q + 4 * i) >> shift) & 0x03030303u);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const uint32_t sc = b[2 * g + h];
      o.sc[2 * h] = o.sc[2 * h + 1] = d * (sc & 0xF);
      o.off[2 * h] = o.off[2 * h + 1] = -dmin * (sc >> 4);
    }
  }
};
template <>
struct QDec<QT_Q4_K> {
  static constexpr int SCALES = 1;  // distinct scale groups per 32 weights
  static constexpr int BB = 144;
  static constexpr bool OFF = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, W32& o) {
    uint8_t sc, m;
    q4k_scale_min(g, b + 4, sc, m);
    const float d = h2f(b) * sc, mn = -h2f(b + 2) * m;
    const uint8_t* q = b + 16 + (g / 2) * 32;
    const int sh = 4 * (g & 1);
#pragma unroll
    for (int i = 0; i < 8; i++) o.w[i] = (int)((ld32a4(q + 4 * i) >> sh) & 0x0F0F0F0Fu);
#pragma unroll
    for (int l = 0; l < 4; l++) { o.sc[l] = d; o.off[l] = mn; }
  }
};
template <>
struct QDec<QT_Q5_K> {
  static constexpr int SCALES = 1;
  static constexpr int BB = 176;
  static constexpr bool OFF = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, W32& o) {
    uint8_t sc, m;
    q4k_scale_min(g, b + 4, sc, m);
    const float d = h2f(b) * sc, mn = -h2f(b + 2) * m;
    const uint8_t* q = b + 48 + (g / 2) * 32;
    const uint8_t* qh = b + 16;
    const int sh = 4 * (g & 1);
#pragma unroll
    for (int i = 0; i < 8; i++)
      o.w[i] = (int)(((ld32a4(q + 4 * i) >> sh) & 0x0F0F0F0Fu) | (((ld32a4(qh + 4 * i) >> g) & 0x01010101u) << 4));
#pragma unroll
    for (int l = 0; l < 4; l++) { o.sc[l] = d; o.off[l] = mn; }
  }
};
// Q3_K helpers: the two signed sub-scales (-32..31) of group g, and 2-bit + high-bit -> int8 (-4..3)
static __device__ __forceinline__ void q3k_scale_pair(const uint8_t* s12, int g, int& s0, int& s1) {
  const uint32_t a0 = ld32a2(s12), a1 = ld32a2(s12 + 4), tmp = ld32a2(s12 + 8);
  const uint32_t k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
  const int w = g / 2;  // aux word holding scales 2g, 2g+1
  uint32_t aux;
  if (w == 0) aux = (a0 & k2) | (((tmp >> 0) & k1) << 4);
  else if (w == 1) aux = (a1 & k2) | (((tmp >> 2) & k1) << 4);
  else if (w == 2) aux = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
  else aux = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
  const int sh = 16 * (g & 1);
  s0 = (int)((aux >> sh) & 0xff) - 32;
  s1 = (int)((aux >> (sh + 8)) & 0xff) - 32;
}
static __device__ __forceinline__ uint32_t q3k_vals(uint32_t q, uint32_t hm, int shift, int g) {
  const uint32_t v = ((q >> shift) & 0x03030303u) | (((hm >> g) & 0x01010101u) << 2);  // 0..7
  return (v & 0x03030303u) | ((((~v) >> 2) & 0x01010101u) * 0xFCu);              // v - 4
}
template <>
struct QDec<QT_Q3_K> {
  static constexpr int SCALES = 2;
  static constexpr int BB = 110;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, W32& o) {
    int s0, s1;
    q3k_scale_pair(b + 96, g, s0, s1);
    const float d = h2f(b + 108);
    const uint8_t* q = b + 32 + 32 * (g / 4);
    const int shift = 2 * (g % 4);
#pragma unroll
    for (int i = 0; i < 8; i++) o.w[i] = (int)q3k_vals(ld32a2(q + 4 * i), ld32a2(b + 4 * i), shift, g);
    o.sc[0] = o.sc[1] = d * s0;
    o.sc[2] = o.sc[3] = d * s1;
  }
};
template <>
struct QDec<QT_Q8_0> {
  static constexpr int SCALES = 1;
  static constexpr int BB = 272;  // 8 blocks of 32
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, W32& o) {
    const uint8_t* blk = b + 34 * g;
    const float d = h2f(blk);
#pragma unroll
    for (int i = 0; i < 8; i++) o.w[i] = (int)ld32a2(blk + 2 + 4 * i);
#pragma unroll
    for (int l = 0; l < 4; l++) o.sc[l] = d;
  }
};
template <>
struct QDec<QT_Q6_K> {
  static constexpr int SCALES = 2;  // distinct scale groups per 32 weights
  static constexpr int BB = 210;
  static constexpr bool OFF = false;
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, W32& o) {
    const float d = h2f(b + 208);
    const int h = g / 4, r = g % 4;
    const uint8_t* ql = b + 64 * h + (r & 1) * 32;
    const uint8_t* qh = b + 128 + 32 * h;
    const int8_t* sc = (const int8_t*)(b + 192 + 8 * h + 2 * r);
    const int lsh = (r >= 2) ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t lo = (ld32a2(ql + 4 * i) >> lsh) & 0x0F0F0F0Fu;
      const uint32_t hi = ((ld32a2(qh + 4 * i) >> (2 * r)) & 0x03030303u) << 4;
      o.w[i] = (int)__vsub4(lo | hi, 0x20202020u);
    }
    o.sc[0] = o.sc[1] = d * sc[0];
    o.sc[2] = o.sc[3] = d * sc[1];
  }
};

template <bool OFF, int SC>
static __device__ __forceinline__ float dot_w32(const W32& w, const int4 a, const int4 b) {
  const int x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  if constexpr (SC == 1) {
    int dt = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) dt = __dp4a(w.w[i], x[i], dt);
    float r = w.sc[0] * (float)dt;
    if (OFF) {
      int sx = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) sx = __dp4a(0x01010101, x[i], sx);
      r += w.off[0] * (float)sx;
    }
    return r;
  } else if constexpr (SC == 2) {
    float r = 0.f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int dt = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) dt = __dp4a(w.w[4 * h + i], x[4 * h + i], dt);
      r += w.sc[2 * h] * (float)dt;
      if (OFF) {
        int sx = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) sx = __dp4a(0x01010101, x[4 * h + i], sx);
        r += w.off[2 * h] * (float)sx;
      }
    }
    return r;
  } else {
    float r = 0.f;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const int dt = __dp4a(w.w[2 * l], x[2 * l], __dp4a(w.w[2 * l + 1], x[2 * l + 1], 0));
      r += w.sc[l] * (float)dt;
      if (OFF) {
        const int sx = __dp4a(0x01010101, x[2 * l], __dp4a(0x01010101, x[2 * l + 1], 0));
        r += w.off[l] * (float)sx;
      }
    }
    return r;
  }
}

// =============================================================================================
// Tensor-core small-batch matvec (B = 2..8 tokens) on quantized weights.
//   One mma.m16n8k32.s8 = 8 weight rows (B operand, n) x 32 k x up to 8 tokens (A operand, m; rows
//   8..15 of A are zero).  The 32 k-slots of an mma are permuted so that lane (g, tq) supplies exactly
//   the 8 weights of chunk tq (weights 8tq..8tq+7 of the 32-group) of row g: every weight is decoded
//   once, by one lane, with no shuffles; the activation fragment is one 8-byte load per lane.
//   Scale groups narrower than 32 are handled by masked mmas; the IQ1 offsets (+-1/8) are folded into
//   the int8 weights (8w +- 1, scale / 8).
// =============================================================================================
static __device__ __forceinline__ void mma_tc_k32(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP v1: tensor-core paths are gated off (launch_mmv_tc returns false); never called.
  (void)c; (void)a; (void)b0; (void)b1;
#else
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
// IQ1 grid bytes {-1, 0, 1} -> 8g + s (s = +-1) via a byte-permute table
static __device__ __forceinline__ uint32_t iq1_fold(uint32_t g, int s) {
  const uint32_t t = (uint32_t)(uint8_t)s | ((uint32_t)(uint8_t)(8 + s) << 8) | ((uint32_t)(uint8_t)(-8 + s) << 24);
  const uint32_t sel = (g & 3) | ((g >> 4) & 0x30) | ((g >> 8) & 0x300) | ((g >> 12) & 0x3000);
  return __byte_perm(t, 0, sel);
}
// signed (v - 32) per byte for v in [0, 63]
static __device__ __forceinline__ uint32_t sub32_bytes(uint32_t v) {
  const uint32_t u = v ^ 0x20202020u;
  const uint32_t m = u & 0x20202020u;
  return u | (m << 1) | (m << 2);
}

// Decode chunk l (weights 8l..8l+7) of 32-group ib of super-block b: two int8x4 words and its scale.
template <int T> struct QChunk { static constexpr bool OK = false; };
template <> struct QChunk<QT_IQ2_XXS> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint32_t a1 = ld32a2(b + 2 + 8 * ib + 4);
    const uint64_t g = qtab_dev::iq2xxs_grid[b[2 + 8 * ib + l]];
    const uint32_t s = iq2_signs8((a1 >> (7 * l)) & 127);
    w0 = apply_signs4((uint32_t)g, s);
    w1 = apply_signs4((uint32_t)(g >> 32), s >> 4);
    sc = h2f(b) * (0.5f + (a1 >> 28)) * 0.25f;
  }
};
template <> struct QChunk<QT_IQ2_XS> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint32_t q = ld16(b + 2 + 2 * (4 * ib + l));
    const uint64_t g = qtab_dev::iq2xs_grid[q & 511];
    const uint32_t s = iq2_signs8(q >> 9);
    w0 = apply_signs4((uint32_t)g, s);
    w1 = apply_signs4((uint32_t)(g >> 32), s >> 4);
    sc = h2f(b) * (0.5f + ((b[66 + ib] >> (4 * (l / 2))) & 0xf)) * 0.25f;
  }
};
template <> struct QChunk<QT_IQ2_S> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint32_t qh = b[66 + ib];
    const uint64_t g = qtab_dev::iq2s_grid[b[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300)];
    const uint32_t s = b[34 + 4 * ib + l];
    w0 = apply_signs4((uint32_t)g, s);
    w1 = apply_signs4((uint32_t)(g >> 32), s >> 4);
    sc = h2f(b) * (0.5f + ((b[74 + ib] >> (4 * (l / 2))) & 0xf)) * 0.25f;
  }
};
template <> struct QChunk<QT_IQ3_XXS> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t a = ld32a2(b + 2 + 64 + 4 * ib);
    const uint32_t s = iq2_signs8((a >> (7 * l)) & 127);
    w0 = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l]], s);
    w1 = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l + 1]], s >> 4);
    sc = h2f(b) * (0.5f + (a >> 28)) * 0.5f;
  }
};
template <> struct QChunk<QT_IQ3_S> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t qh = b[66 + ib];
    const uint32_t s = b[74 + 4 * ib + l];
    w0 = apply_signs4(qtab_dev::iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)], s);
    w1 = apply_signs4(qtab_dev::iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)], s >> 4);
    sc = h2f(b) * (1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf));
  }
};
template <> struct QChunk<QT_IQ1_S> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint32_t qh = ld16(b + 34 + 2 * ib);
    const int s = (qh & 0x8000) ? -1 : 1;
    const uint64_t g = qtab_dev::iq1s_grid[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
    w0 = iq1_fold((uint32_t)g, s);
    w1 = iq1_fold((uint32_t)(g >> 32), s);
    sc = h2f(b) * (2 * ((qh >> 12) & 7) + 1) * 0.125f;
  }
};
template <> struct QChunk<QT_IQ1_M> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint16_t* scp = (const uint16_t*)(b + 48);
    const uint16_t u = (scp[0] >> 12) | ((scp[1] >> 8) & 0x00f0) | ((scp[2] >> 4) & 0x0f00) | (scp[3] & 0xf000);
    const float d = __half2float(__ushort_as_half(u));
    const uint32_t h = b[32 + 2 * ib + l / 2];
    const uint32_t idx = b[4 * ib + l] | ((l & 1) ? ((h << 4) & 0x700) : ((h << 8) & 0x700));
    const int s = (h & ((l & 1) ? 0x80 : 0x08)) ? -1 : 1;
    const uint64_t g = qtab_dev::iq1s_grid[idx];
    w0 = iq1_fold((uint32_t)g, s);
    w1 = iq1_fold((uint32_t)(g >> 32), s);
    sc = d * (2 * ((scp[ib / 2] >> (6 * (ib % 2) + (l < 2 ? 0 : 3))) & 0x7) + 1) * 0.125f;
  }
};
template <> struct QChunk<QT_IQ4_XS> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const uint32_t sh = ld16(b + 2);
    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    const uint32_t* tab = (const uint32_t*)qtab_dev::kvalues_iq4nl;
    const uint32_t t[4] = {tab[0], tab[1], tab[2], tab[3]};
    const uint8_t* qs = b + 8 + 16 * ib + 8 * (l & 1);
    const int shift = 4 * (l >> 1);
    w0 = lut16((ld32a4(qs) >> shift) & 0x0F0F0F0F, t);
    w1 = lut16((ld32a4(qs + 4) >> shift) & 0x0F0F0F0F, t);
    sc = h2f(b) * (ls - 32);
  }
};
template <> struct QChunk<QT_Q6_K> {
  static constexpr bool OK = true;
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, int l, uint32_t& w0, uint32_t& w1, float& sc) {
    const int h = g / 4, r = g % 4;
    const uint8_t* ql = b + 64 * h + (r & 1) * 32 + 8 * l;
    const uint8_t* qh = b + 128 + 32 * h + 8 * l;
    const int lsh = (r >= 2) ? 4 : 0;
    w0 = sub32_bytes(((ld32a2(ql) >> lsh) & 0x0F0F0F0Fu) | (((ld32a2(qh) >> (2 * r)) & 0x03030303u) << 4));
    w1 = sub32_bytes(((ld32a2(ql + 4) >> lsh) & 0x0F0F0F0Fu) | (((ld32a2(qh + 4) >> (2 * r)) & 0x03030303u) << 4));
    sc = h2f(b + 208) * ((const int8_t*)(b + 192 + 8 * h + 2 * r))[l / 2];
  }
};

constexpr int TC_WARPS = 8;  // split-K: the warps of a block share a row tile and split the K range
template <int T, int B, int RT>
__global__ void __launch_bounds__(TC_WARPS * 32)
    k_mmv_tc(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
             const float* __restrict__ xd, float* __restrict__ y, int ldy, int acc) {
  constexpr int SC = QDec<T>::SCALES, BB = QDec<T>::BB, CPS = 4 / SC;  // chunks per scale group
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const int rbase = blockIdx.x * 8 * RT;
  const int nsb = K / 32;
  const int j0 = warp * nsb / TC_WARPS, j1 = (warp + 1) * nsb / TC_WARPS;
  const uint8_t* wr[RT];
#pragma unroll
  for (int rt = 0; rt < RT; rt++) wr[rt] = W + (size_t)min(rbase + rt * 8 + g, N - 1) * rb;
  float accv[RT][2];
#pragma unroll
  for (int rt = 0; rt < RT; rt++) accv[rt][0] = accv[rt][1] = 0.f;
  const bool tv = g < B;
  const int8_t* xp = xq + (size_t)(tv ? g : 0) * K + 8 * tq;
  const float* xdp = xd + (size_t)(tv ? g : 0) * nsb;
#pragma unroll 4
  for (int j = j0; j < j1; j++) {
    uint32_t a[4] = {0u, 0u, 0u, 0u};
    float xs = 0.f;
    if (tv) {
      const uint2 v = __ldg((const uint2*)(xp + (size_t)j * 32));
      a[0] = v.x;
      a[2] = v.y;
      xs = __ldg(xdp + j);
    }
#pragma unroll
    for (int rt = 0; rt < RT; rt++) {
      uint32_t w0, w1;
      float sc;
      QChunk<T>::dec(wr[rt] + (size_t)(j >> 3) * BB, j & 7, tq, w0, w1, sc);
#pragma unroll
      for (int h = 0; h < SC; h++) {
        const bool mine = SC == 1 || tq / CPS == h;
        int c[4] = {0, 0, 0, 0};
        mma_tc_k32(c, a, mine ? w0 : 0u, mine ? w1 : 0u);
        // C: (token g, weight rows 2tq / 2tq+1); their scales live in lanes 4*row + chunk
        const float s0 = __shfl_sync(0xffffffffu, sc, 8 * tq + h * CPS);
        const float s1 = __shfl_sync(0xffffffffu, sc, 8 * tq + 4 + h * CPS);
        accv[rt][0] += (float)c[0] * s0 * xs;
        accv[rt][1] += (float)c[1] * s1 * xs;
      }
    }
  }
  __shared__ float red[TC_WARPS][RT][2][32];
#pragma unroll
  for (int rt = 0; rt < RT; rt++) {
    red[warp][rt][0][lane] = accv[rt][0];
    red[warp][rt][1][lane] = accv[rt][1];
  }
  __syncthreads();
  if (warp != 0 || !tv) return;
#pragma unroll
  for (int rt = 0; rt < RT; rt++) {
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < TC_WARPS; w2++) { v0 += red[w2][rt][0][lane]; v1 += red[w2][rt][1][lane]; }
    const int r = rbase + rt * 8 + 2 * tq;
    float* o = y + (size_t)g * ldy + r;
    if (r < N) o[0] = acc ? o[0] + v0 : v0;
    if (r + 1 < N) o[1] = acc ? o[1] + v1 : v1;
  }
}

// Variant 2: lanes decode whole 32-groups (coalesced row reads, the dp4a kernels' decoders), 4 lanes x
// 4 consecutive groups per row; a shared-memory 4x4 transpose turns them into per-group fragments.
template <int T, int B>
__global__ void __launch_bounds__(TC_WARPS * 32)
    k_mmv_tc2(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
              const float* __restrict__ xd, float* __restrict__ y, int ldy, int acc) {
  using D = QDec<T>;
  constexpr int SC = D::SCALES, CPS = 4 / SC;
  constexpr bool FOLD = T == QT_IQ1_S || T == QT_IQ1_M;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const int rbase = blockIdx.x * 8;
  const int nsb = K / 32;
  // warps take interleaved 4-group strips so a block streams each row front to back
  __shared__ uint2 smw[TC_WARPS][8][4][4];   // [warp][row][group][chunk]
  __shared__ float sms[TC_WARPS][8][4][4];
  __shared__ float red[TC_WARPS][2][32];
  const uint8_t* wr = W + (size_t)min(rbase + g, N - 1) * rb;
  float a0 = 0.f, a1 = 0.f;
  const bool tv = g < B;
  const int8_t* xp = xq + (size_t)(tv ? g : 0) * K + 8 * tq;
  const float* xdp = xd + (size_t)(tv ? g : 0) * nsb;
  for (int j = 4 * warp; j < nsb; j += 4 * TC_WARPS) {
    const int jg = j + tq;
    W32 w;
    if (jg < nsb) {
      D::dec(wr + (size_t)(jg >> 3) * D::BB, jg & 7, w);
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) w.w[i] = 0;
#pragma unroll
      for (int l = 0; l < 4; l++) w.sc[l] = w.off[l] = 0.f;
    }
#pragma unroll
    for (int l = 0; l < 4; l++) {
      uint32_t c0 = (uint32_t)w.w[2 * l], c1 = (uint32_t)w.w[2 * l + 1];
      float scl = w.sc[l];
      if constexpr (FOLD) {  // w + off/sc = w +- 1/8: fold into 8w +- 1 with scale / 8
        const int sg = w.off[l] < 0.f ? -1 : 1;
        c0 = iq1_fold(c0, sg);
        c1 = iq1_fold(c1, sg);
        scl *= 0.125f;
      }
      smw[warp][g][tq][l] = make_uint2(c0, c1);
      sms[warp][g][tq][l] = scl;
    }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < 4; s++) {
      if (j + s >= nsb) break;
      uint32_t a[4] = {0u, 0u, 0u, 0u};
      float xs = 0.f;
      if (tv) {
        const uint2 v = __ldg((const uint2*)(xp + (size_t)(j + s) * 32));
        a[0] = v.x;
        a[2] = v.y;
        xs = __ldg(xdp + j + s);
      }
      const uint2 bw = smw[warp][g][s][tq];
#pragma unroll
      for (int h = 0; h < SC; h++) {
        const bool mine = SC == 1 || tq / CPS == h;
        int c[4] = {0, 0, 0, 0};
        mma_tc_k32(c, a, mine ? bw.x : 0u, mine ? bw.y : 0u);
        a0 += (float)c[0] * sms[warp][2 * tq][s][h * CPS] * xs;
        a1 += (float)c[1] * sms[warp][2 * tq + 1][s][h * CPS] * xs;
      }
    }
    __syncwarp();
  }
  red[warp][0][lane] = a0;
  red[warp][1][lane] = a1;
  __syncthreads();
  if (warp != 0 || !tv) return;
  float v0 = 0.f, v1 = 0.f;
#pragma unroll
  for (int w2 = 0; w2 < TC_WARPS; w2++) { v0 += red[w2][0][lane]; v1 += red[w2][1][lane]; }
  const int r = rbase + 2 * tq;
  float* o = y + (size_t)g * ldy + r;
  if (r < N) o[0] = acc ? o[0] + v0 : v0;
  if (r + 1 < N) o[1] = acc ? o[1] + v1 : v1;
}

#ifdef __HIP_PLATFORM_AMD__
// WMMA (RDNA3) tensor-core matvec, IQ3_S only for now. Fragment layout pinned
// empirically by hip-spike/wmma.hip.cpp on a real 7900 XTX (2026-11-20):
//   A fragment: lane l holds token (l % 16), 16 consecutive int8 along K.
//   B fragment: lane l holds weight row (l % 16), 16 consecutive int8 along K.
//   C fragment: element i on lane l -> (token 2*i + (l / 16), row l % 16).
//   The builtin's bool args are per-operand signedness (true = signed); the
//   trailing bool is a saturating clamp (unused). One block covers 16 rows;
//   warps split the 32-group range with a strided loop, partial sums are
//   reduced through shared memory like k_mmv_tc2.
using wmma_i32x4 = int __attribute__((ext_vector_type(4)));
using wmma_i32x8 = int __attribute__((ext_vector_type(8)));

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    k_mmv_wmma_iq3s(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
                    const float* __restrict__ xd, int B, float* __restrict__ y, int ldy, int acc) {
  using D = QDec<QT_IQ3_S>;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rbase = blockIdx.x * 16;
  const int nsb = K / 32;
  const int n = lane % 16;        // this lane's weight row / output column
  const int hi = lane / 16;       // token parity held by this lane's C fragment
  const bool tok_ok = n < B;      // A rows are tokens: lane % 16 doubles as token index
  __shared__ float red[WARPS][32][8];
  float accv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int r = min(rbase + n, N - 1);
  const uint8_t* wr = W + (size_t)r * rb;
  for (int j = warp; j < nsb; j += WARPS) {
    // v2 (best of the WMMA variants): decode this lane's own row in registers
    // (lanes n and n+16 duplicate the work) -- the W32 words are exactly the
    // two B-fragment halves, so no shared-memory staging and no syncwarp in
    // the group loop. v3 (decode on lanes 0..15 + shfl distribution) measured
    // slower: the bpermute chain costs more than the duplicate decode saves.
    W32 w;
    D::dec(wr + (size_t)(j >> 3) * D::BB, j & 7, w);
    const float ws = w.sc[0];
    const float xs_own = tok_ok ? xd[(size_t)n * nsb + j] : 0.f;
    wmma_i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int h = 0; h < 2; h++) {
      wmma_i32x4 a4 = {0, 0, 0, 0};
      if (tok_ok) {
        const uint4 v = *(const uint4*)(xq + (size_t)n * K + 32 * j + 16 * h);
        a4[0] = (int)v.x;
        a4[1] = (int)v.y;
        a4[2] = (int)v.z;
        a4[3] = (int)v.w;
      }
      wmma_i32x4 b4;
      b4[0] = w.w[4 * h + 0];
      b4[1] = w.w[4 * h + 1];
      b4[2] = w.w[4 * h + 2];
      b4[3] = w.w[4 * h + 3];
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a4, true, b4, c, false);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      // token (2i + hi)'s activation scale lives on lane (2i + hi)
      const float xs = __shfl_sync(0xffffffffu, xs_own, 2 * i + hi);
      accv[i] += (float)c[i] * ws * xs;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) red[warp][lane][i] = accv[i];
  __syncthreads();
  if (warp != 0) return;
  const int row = rbase + n;
  if (row >= N) return;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int tok = 2 * i + hi;
    if (tok >= B) continue;
    float v = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < WARPS; w2++) v += red[w2][lane][i];
    float* o = y + (size_t)tok * ldy + row;
    *o = acc ? *o + v : v;
  }
}
#endif  // __HIP_PLATFORM_AMD__

template <int T>
static bool launch_mmv_tc(const uint8_t* w, size_t rb, int K, int N, const int8_t* xq, const float* xd, int B,
                          float* y, int ldy, bool acc, cudaStream_t s) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP: WMMA tensor-core matvec (k_mmv_wmma_iq3s), QE_MMV_TC=2, IQ3_S only.
  static const int tc_mode = getenv("QE_MMV_TC") ? atoi(getenv("QE_MMV_TC")) : 0;
  if constexpr (T == QT_IQ3_S) {
    if (tc_mode == 2 && B >= 2 && B <= 8 && K % 32 == 0) {
      k_mmv_wmma_iq3s<TC_WARPS><<<cdiv(N, 16), TC_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, B, y, ldy, acc);
      return true;
    }
  }
  return false;
#endif
  static const int mode = getenv("QE_MMV_TC") ? atoi(getenv("QE_MMV_TC")) : 0;
  constexpr bool tc2_ok = T != QT_Q2_K && T != QT_Q4_K && T != QT_Q5_K;  // min terms not foldable
  if constexpr (tc2_ok) {
    if (mode == 2 && B >= 2 && B <= 8 && K % 32 == 0) {
      const int grid = cdiv(N, 8);
      switch (B) {
#define CASE(n) case n: k_mmv_tc2<T, n><<<grid, TC_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break;
        CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
      }
      return true;
    }
  }
  if constexpr (!QChunk<T>::OK) {
    return false;
  } else {
    if (mode != 1 || B < 2 || B > 8 || K % 32) return false;
    constexpr int RT = 1;
    const int grid = cdiv(N, 8 * RT);
    switch (B) {
#define CASE(n) case n: k_mmv_tc<T, n, RT><<<grid, TC_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break;
      CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    }
    return true;
  }
}

// =============================================================================================
// Matrix-vector (small batch) product: one warp per output row.
// =============================================================================================
constexpr int MMV_WARPS = 8;

template <int T, int B, int R>
__global__ void __launch_bounds__(MMV_WARPS * 32)
    k_mmv(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
          const float* __restrict__ xd, float* __restrict__ y, int ldy, int acc) {
  using D = QDec<T>;
  const int row0 = (blockIdx.x * MMV_WARPS + (threadIdx.x >> 5)) * R;
  const int lane = threadIdx.x & 31;
  if (row0 >= N) return;
  const int nsb = K / 32;
  float sum[R][B];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int b = 0; b < B; b++) sum[r][b] = 0.f;
  const uint8_t* wr[R];
#pragma unroll
  for (int r = 0; r < R; r++) wr[r] = W + (size_t)min(row0 + r, N - 1) * rb;

  for (int g = lane; g < nsb; g += 32) {
    W32 w[R];
#pragma unroll
    for (int r = 0; r < R; r++) D::dec(wr[r] + (size_t)(g >> 3) * D::BB, g & 7, w[r]);
#pragma unroll
    for (int b = 0; b < B; b++) {
      const int4* xp = (const int4*)(xq + (size_t)b * K + (size_t)g * 32);
      const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
      const float d = __ldg(xd + (size_t)b * nsb + g);
#pragma unroll
      for (int r = 0; r < R; r++) sum[r][b] += dot_w32<D::OFF, D::SCALES>(w[r], x0, x1) * d;
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    if (row0 + r >= N) break;
#pragma unroll
    for (int b = 0; b < B; b++) {
      const float v = warp_sum(sum[r][b]);
      if (lane == 0) {
        float* o = y + (size_t)b * ldy + row0 + r;
        *o = acc ? *o + v : v;
      }
    }
  }
}

// Staged variant: each warp copies the 4 super-blocks its 32 lanes consume (per row) from global
// memory with coalesced 16-byte loads into shared memory, then lanes decode from shared memory.
// The per-lane scattered 1/2/4-byte global loads of the direct kernel become ~1-2 wide loads.
template <int T, int B, int R>
__global__ void __launch_bounds__(MMV_WARPS * 32)
    k_mmv_s(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
            const float* __restrict__ xd, float* __restrict__ y, int ldy, int acc) {
  using D = QDec<T>;
  constexpr int SPAN = 4 * D::BB;                    // bytes of 4 super-blocks
  constexpr int CH = (SPAN + 15 + 15) / 16 + 1;      // 16B chunks covering an unaligned span
  __shared__ __align__(16) uint4 sm[MMV_WARPS][R][CH];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * MMV_WARPS + warp) * R;
  if (row0 >= N) return;
  const int nsb = K / 32;
  const int nsuper = K / 256;
  float sum[R][B];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int b = 0; b < B; b++) sum[r][b] = 0.f;
  for (int j0 = 0; j0 < nsuper; j0 += 4) {
    const int nblk = min(4, nsuper - j0);
    int off[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const uint8_t* a = W + (size_t)min(row0 + r, N - 1) * rb + (size_t)j0 * D::BB;
      const uintptr_t A = (uintptr_t)a & ~(uintptr_t)15;
      off[r] = (int)((uintptr_t)a - A);
      const int nch = (off[r] + nblk * D::BB + 15) / 16;
      for (int ch = lane; ch < nch; ch += 32) sm[warp][r][ch] = __ldg((const uint4*)A + ch);
    }
    __syncwarp();
    const int g = j0 * 8 + lane;
    if (lane < nblk * 8) {
      W32 w[R];
#pragma unroll
      for (int r = 0; r < R; r++)
        D::dec((const uint8_t*)&sm[warp][r][0] + off[r] + (lane >> 3) * D::BB, lane & 7, w[r]);
#pragma unroll
      for (int b = 0; b < B; b++) {
        const int4* xp = (const int4*)(xq + (size_t)b * K + (size_t)g * 32);
        const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
        const float d = __ldg(xd + (size_t)b * nsb + g);
#pragma unroll
        for (int r = 0; r < R; r++) sum[r][b] += dot_w32<D::OFF, D::SCALES>(w[r], x0, x1) * d;
      }
    }
    __syncwarp();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    if (row0 + r >= N) break;
#pragma unroll
    for (int b = 0; b < B; b++) {
      const float v = warp_sum(sum[r][b]);
      if (lane == 0) {
        float* o = y + (size_t)b * ldy + row0 + r;
        *o = acc ? *o + v : v;
      }
    }
  }
}

// B >= 2 variant: the int8 activations (B x K) and their scales are staged once per CTA in shared
// memory and reused by all warps (instead of every warp re-reading them through L1).
template <int T, int B, int R>
__global__ void __launch_bounds__(MMV_WARPS * 32)
    k_mmv_x(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
            const float* __restrict__ xd, float* __restrict__ y, int ldy, int acc) {
  using D = QDec<T>;
  extern __shared__ __align__(16) unsigned char mx_smem[];
  int8_t* xs = (int8_t*)mx_smem;                       // [B][K]
  float* xds = (float*)(mx_smem + (size_t)B * K);      // [B][K/32]
  const int nsb = K / 32;
  for (int i = threadIdx.x; i < B * K / 16; i += blockDim.x) ((int4*)xs)[i] = __ldg((const int4*)xq + i);
  for (int i = threadIdx.x; i < B * nsb; i += blockDim.x) xds[i] = __ldg(xd + i);
  __syncthreads();
  const int row0 = (blockIdx.x * MMV_WARPS + (threadIdx.x >> 5)) * R;
  const int lane = threadIdx.x & 31;
  if (row0 >= N) return;
  float sum[R][B];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int b = 0; b < B; b++) sum[r][b] = 0.f;
  const uint8_t* wr[R];
#pragma unroll
  for (int r = 0; r < R; r++) wr[r] = W + (size_t)min(row0 + r, N - 1) * rb;

  for (int g = lane; g < nsb; g += 32) {
    W32 w[R];
#pragma unroll
    for (int r = 0; r < R; r++) D::dec(wr[r] + (size_t)(g >> 3) * D::BB, g & 7, w[r]);
#pragma unroll
    for (int b = 0; b < B; b++) {
      const int4* xp = (const int4*)(xs + (size_t)b * K + (size_t)g * 32);
      const int4 x0 = xp[0], x1 = xp[1];
      const float d = xds[b * nsb + g];
#pragma unroll
      for (int r = 0; r < R; r++) sum[r][b] += dot_w32<D::OFF, D::SCALES>(w[r], x0, x1) * d;
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    if (row0 + r >= N) break;
#pragma unroll
    for (int b = 0; b < B; b++) {
      const float v = warp_sum(sum[r][b]);
      if (lane == 0) {
        float* o = y + (size_t)b * ldy + row0 + r;
        *o = acc ? *o + v : v;
      }
    }
  }
}

// Pipelined small-batch matvec: each warp owns R rows; the raw weight bytes of the next two 32-sub-
// block chunks of its rows are in flight (cp.async, 3 stages in shared memory) while the current
// chunk is decoded and dotted, so the weight stream overlaps the per-token dot work (which grows with
// B and otherwise serializes behind the loads).
template <int T, int B, int R>
__global__ void __launch_bounds__(MMV_WARPS * 32)
    k_mmv_p(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
            const float* __restrict__ xd, float* __restrict__ y, int ldy, int acc) {
  using D = QDec<T>;
  constexpr int CH = 4 * D::BB, CHP = (CH + 15) & ~15, S = 3;
  extern __shared__ __align__(16) unsigned char pm_smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint8_t* buf = pm_smem + (size_t)warp * S * R * CHP;
  const int row0 = (blockIdx.x * MMV_WARPS + warp) * R;
  if (row0 >= N) return;
  const int nsb = K / 32, nit = nsb / 32;
  const uint8_t* wr[R];
#pragma unroll
  for (int r = 0; r < R; r++) wr[r] = W + (size_t)min(row0 + r, N - 1) * rb;
  auto issue = [&](int it, int st) {
#pragma unroll
    for (int r = 0; r < R; r++) {
      const uint8_t* src = wr[r] + (size_t)it * CH;
      uint8_t* dst = buf + (st * R + r) * CHP;
      for (int i = lane; i < CH / 8; i += 32) cp_async8(dst + 8 * i, src + 8 * i);
    }
    cp_async_commit();
  };
  float sum[R][B];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int b = 0; b < B; b++) sum[r][b] = 0.f;
  issue(0, 0);
  if (nit > 1) issue(1, 1); else cp_async_commit();
  for (int it = 0; it < nit; it++) {
    const int st = it % S;
    if (it + 2 < nit) issue(it + 2, (it + 2) % S); else cp_async_commit();
    cp_async_wait<2>();
    __syncwarp();
    W32 w[R];
#pragma unroll
    for (int r = 0; r < R; r++) D::dec(buf + (st * R + r) * CHP + (lane >> 3) * D::BB, lane & 7, w[r]);
    const int g = it * 32 + lane;
#pragma unroll
    for (int b = 0; b < B; b++) {
      const int4* xp = (const int4*)(xq + (size_t)b * K + (size_t)g * 32);
      const int4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
      const float d = __ldg(xd + (size_t)b * nsb + g);
#pragma unroll
      for (int r = 0; r < R; r++) sum[r][b] += dot_w32<D::OFF, D::SCALES>(w[r], x0, x1) * d;
    }
    __syncwarp();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    if (row0 + r >= N) break;
#pragma unroll
    for (int b = 0; b < B; b++) {
      const float v = warp_sum(sum[r][b]);
      if (lane == 0) {
        float* o = y + (size_t)b * ldy + row0 + r;
        *o = acc ? *o + v : v;
      }
    }
  }
}

// f32 / f16 / bf16 weights with f32 activations
template <int WT, int B>
__global__ void k_mmv_float(const uint8_t* __restrict__ W, size_t rb, int K, int N,
                            const float* __restrict__ x, float* __restrict__ y, int ldy, int acc) {
  const int row = blockIdx.x * MMV_WARPS + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const uint8_t* wr = W + (size_t)row * rb;
  float sum[B];
#pragma unroll
  for (int b = 0; b < B; b++) sum[b] = 0.f;
  for (int k = lane; k < K; k += 32) {
    float w;
    if (WT == QT_F32) w = ((const float*)wr)[k];
    else if (WT == QT_F16) w = __half2float(((const __half*)wr)[k]);
    else w = __uint_as_float((uint32_t)((const uint16_t*)wr)[k] << 16);
#pragma unroll
    for (int b = 0; b < B; b++) sum[b] += w * x[(size_t)b * K + k];
  }
#pragma unroll
  for (int b = 0; b < B; b++) {
    const float v = warp_sum(sum[b]);
    if (lane == 0) {
      float* o = y + (size_t)b * ldy + row;
      *o = acc ? *o + v : v;
    }
  }
}

// Few-row float matvec (GDN alpha/beta: 48 x 5120 BF16): one 256-thread block per row, K split over
// all threads with 16-byte weight loads, block reduction.  The warp-per-row kernel above launches only
// N/8 blocks for such matrices and leaves most SMs idle.
template <int WT, int B>
__global__ void __launch_bounds__(256) k_mmv_float_row(const uint8_t* __restrict__ W, size_t rb, int K,
                                                       const float* __restrict__ x, float* __restrict__ y, int ldy, int acc,
                                                       const uint8_t* __restrict__ W2 = nullptr, float* y2 = nullptr, int N1 = 0) {
  int row = blockIdx.x;
  const uint8_t* Wm = W;
  if (W2 && row >= N1) { row -= N1; Wm = W2; y = y2; }  // second matrix of a fused pair
  const uint8_t* wr = Wm + (size_t)row * rb;
  constexpr int E = WT == QT_F32 ? 4 : 8;  // elements per 16-byte load
  float sum[B];
#pragma unroll
  for (int b = 0; b < B; b++) sum[b] = 0.f;
  for (int k0 = threadIdx.x * E; k0 < K; k0 += 256 * E) {
    const uint4 raw = __ldg((const uint4*)(wr + (size_t)k0 * (WT == QT_F32 ? 4 : 2)));
    float w[E];
    if constexpr (WT == QT_F32) {
      w[0] = __uint_as_float(raw.x); w[1] = __uint_as_float(raw.y); w[2] = __uint_as_float(raw.z); w[3] = __uint_as_float(raw.w);
    } else {
      const uint32_t u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if constexpr (WT == QT_F16) {
          const float2 f = __half22float2(*(const __half2*)&u[i]);
          w[2 * i] = f.x; w[2 * i + 1] = f.y;
        } else {
          w[2 * i] = __uint_as_float(u[i] << 16); w[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u);
        }
      }
    }
#pragma unroll
    for (int b = 0; b < B; b++) {
      const float4* xp = (const float4*)(x + (size_t)b * K + k0);
#pragma unroll
      for (int i = 0; i < E / 4; i++) {
        const float4 xv = __ldg(xp + i);
        sum[b] += w[4 * i] * xv.x + w[4 * i + 1] * xv.y + w[4 * i + 2] * xv.z + w[4 * i + 3] * xv.w;
      }
    }
  }
  __shared__ float red[8][B];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int b = 0; b < B; b++) {
    const float v = warp_sum(sum[b]);
    if (lane == 0) red[warp][b] = v;
  }
  __syncthreads();
  if (threadIdx.x < B) {
    float v = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < 8; w2++) v += red[w2][threadIdx.x];
    float* o = y + (size_t)threadIdx.x * ldy + row;
    *o = acc ? *o + v : v;
  }
}

// 4 rows per warp at B >= 3: the activation loads (B x 32 bytes per 32 weights, from L1) dominate
// the per-weight work, so sharing them across more rows wins; IQ4_XS decodes cheaply and prefers R=2
static bool mmv_r4() {
  static bool r = !getenv("QE_MMV_R4") || atoi(getenv("QE_MMV_R4")) != 0;
  return r;
}
static int mmv_mode() {
  static int m = getenv("QE_MMV") ? atoi(getenv("QE_MMV")) : 0;  // 0 = direct (faster on sm_86)
  return m;
}

// Default small-batch matvec choice (hand-tuned heuristics; variant 0 of the tuner below)
template <int T>
static void mmv_default(const uint8_t* w, size_t rb, int K, int N, const int8_t* xq, const float* xd,
                        int B, float* y, int ldy, bool acc, cudaStream_t s) {
  const dim3 block(MMV_WARPS * 32);
  constexpr bool smem_ok = !(T == QT_IQ1_S || T == QT_IQ1_M || T == QT_IQ2_XXS);
  constexpr bool grid_type = T == QT_IQ2_XXS || T == QT_IQ2_XS || T == QT_IQ2_S || T == QT_IQ3_XXS ||
                             T == QT_IQ3_S || T == QT_IQ1_S || T == QT_IQ1_M;
  if (mmv_mode() == 0 && smem_ok && B >= 2 && B <= 4 && (size_t)B * K <= 24576) {
    const size_t sh = (size_t)B * K + (size_t)B * (K / 32) * 4;
    if (B == 2 && !grid_type) {
      k_mmv_x<T, 2, 1><<<cdiv(N, MMV_WARPS), block, sh, s>>>(w, rb, K, N, xq, xd, y, ldy, acc);
      return;
    }
    if (mmv_r4() && B >= 3 && T != QT_IQ4_XS) {
      if (B == 3) k_mmv_x<T, 3, 4><<<cdiv(N, MMV_WARPS * 4), block, sh, s>>>(w, rb, K, N, xq, xd, y, ldy, acc);
      else k_mmv_x<T, 4, 4><<<cdiv(N, MMV_WARPS * 4), block, sh, s>>>(w, rb, K, N, xq, xd, y, ldy, acc);
      return;
    }
    switch (B) {
#define CASE(n) case n: k_mmv_x<T, n, 2><<<cdiv(N, MMV_WARPS * 2), block, sh, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); return;
      CASE(2) CASE(3) CASE(4)
#undef CASE
    }
  }
  if (mmv_mode() == 1) {
    switch (B) {
#define CASE(n, R) case n: k_mmv_s<T, n, R><<<cdiv(N, MMV_WARPS * R), block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); return;
      CASE(1, 1) CASE(2, 2) CASE(3, 2) CASE(4, 2) CASE(5, 2) CASE(6, 2) CASE(7, 2) CASE(8, 2)
#undef CASE
    }
  }
  // B >= 2: two rows per warp share the activation loads (B == 1 keeps more warps in flight)
  if (B == 1 || (B == 2 && !grid_type)) {
    if (B == 1) k_mmv<T, 1, 1><<<cdiv(N, MMV_WARPS), block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc);
    else k_mmv<T, 2, 1><<<cdiv(N, MMV_WARPS), block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc);
    return;
  }
  if (mmv_r4() && (B == 3 || B == 4) && T != QT_IQ4_XS) {
    if (B == 3) k_mmv<T, 3, 4><<<cdiv(N, MMV_WARPS * 4), block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc);
    else k_mmv<T, 4, 4><<<cdiv(N, MMV_WARPS * 4), block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc);
    return;
  }
  const dim3 grid(cdiv(N, MMV_WARPS * 2));
  switch (B) {
#define CASE(n) case n: k_mmv<T, n, 2><<<grid, block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break;
    CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    default: QE_FATAL("matvec: unsupported batch %d", B);
  }
}

// Variants of the B = 2..8 matvec (same product, different work split); false if not applicable.
//   0 default, 1 shared-activation R=2, 2 shared-activation R=4, 3 direct R=2, 4 direct R=4,
//   5 cp.async-pipelined R=2
constexpr int kMmvVariants = 6;
template <int T>
static bool mmv_variant(int v, const uint8_t* w, size_t rb, int K, int N, const int8_t* xq, const float* xd,
                        int B, float* y, int ldy, bool acc, cudaStream_t s) {
  const dim3 block(MMV_WARPS * 32);
  constexpr bool smem_ok = !(T == QT_IQ1_S || T == QT_IQ1_M || T == QT_IQ2_XXS);
  if (B < 2 || B > 8) return false;
  switch (v) {
    case 0: mmv_default<T>(w, rb, K, N, xq, xd, B, y, ldy, acc, s); return true;
    case 1: case 2: {
      const size_t sh = (size_t)B * K + (size_t)B * (K / 32) * 4;
      if (!smem_ok || sh > 99 * 1024 || (v == 2 && B < 3)) return false;
#ifdef __HIP_PLATFORM_AMD__
      if (sh > 64 * 1024) return false;  // RDNA3 LDS cap: fall back to mmv_default
#endif
      if (sh > 48 * 1024) {
        static bool init = false;
        if (!init) {
#ifdef __HIP_PLATFORM_AMD__
#define ATTR(n, r) cudaFuncSetAttribute(k_mmv_x<T, n, r>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
#else
#define ATTR(n, r) cudaFuncSetAttribute(k_mmv_x<T, n, r>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
#endif
          ATTR(2, 2) ATTR(3, 2) ATTR(4, 2) ATTR(5, 2) ATTR(6, 2) ATTR(7, 2) ATTR(8, 2)
          ATTR(3, 4) ATTR(4, 4) ATTR(5, 4) ATTR(6, 4) ATTR(7, 4) ATTR(8, 4)
#undef ATTR
          init = true;
        }
      }
      if (v == 1) {
        switch (B) {
#define CASE(n) case n: k_mmv_x<T, n, 2><<<cdiv(N, MMV_WARPS * 2), block, sh, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); return true;
          CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
        }
      } else {
        switch (B) {
#define CASE(n) case n: k_mmv_x<T, n, 4><<<cdiv(N, MMV_WARPS * 4), block, sh, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); return true;
          CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
        }
      }
      return false;
    }
    case 3:
      switch (B) {
#define CASE(n) case n: k_mmv<T, n, 2><<<cdiv(N, MMV_WARPS * 2), block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); return true;
        CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
      }
      return false;
    case 4:
      switch (B) {
#define CASE(n) case n: k_mmv<T, n, 4><<<cdiv(N, MMV_WARPS * 4), block, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); return true;
        CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
      }
      return false;
    case 5: {
      constexpr int CHP = (4 * QDec<T>::BB + 15) & ~15;
      if ((K / 32) % 32 != 0 || rb % 8 != 0 || ((uintptr_t)w & 7) != 0 || T == QT_Q8_0) return false;
      const size_t sh = (size_t)MMV_WARPS * 3 * 2 * CHP;
      if (sh > 48 * 1024) return false;
      switch (B) {
#define CASE(n) case n: k_mmv_p<T, n, 2><<<cdiv(N, MMV_WARPS * 2), block, sh, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); return true;
        CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
      }
      return false;
    }
  }
  return false;
}

// B >= 3: the shared-activation kernel with 4 rows per warp (variant 2) wherever it applies (its
// staged activations fit and the type decodes from shared memory well); measured 3-8% faster per
// decode step than the older per-type choice on the 27B (a per-shape timing autotuner was tried and
// did not beat it: isolated timings do not predict the in-pass cost).  QE_MMV_VARIANT=v forces one.
template <int T>
static void launch_mmv_t(const void* W, size_t rb, int K, int N, const int8_t* xq, const float* xd,
                         int B, float* y, int ldy, bool acc, cudaStream_t s) {
  const uint8_t* w = (const uint8_t*)W;
  if (launch_mmv_tc<T>(w, rb, K, N, xq, xd, B, y, ldy, acc, s)) return;
  static const int force = getenv("QE_MMV_VARIANT") ? atoi(getenv("QE_MMV_VARIANT")) : -1;
  if (B >= 2 && force >= 0 && mmv_variant<T>(force, w, rb, K, N, xq, xd, B, y, ldy, acc, s)) return;
  if (B >= 3 && force < 0 && mmv_variant<T>(2, w, rb, K, N, xq, xd, B, y, ldy, acc, s)) return;
  mmv_default<T>(w, rb, K, N, xq, xd, B, y, ldy, acc, s);
}
template <int WT>
static void launch_mmv_float(const void* W, size_t rb, int K, int N, const float* x, int B,
                             float* y, int ldy, bool acc, cudaStream_t s) {
  if (N <= 256 && K % 8 == 0 && rb % 16 == 0 && !getenv("QE_FLOAT_OLD")) {
    const uint8_t* w8 = (const uint8_t*)W;
    switch (B) {
#define CASE(n) case n: k_mmv_float_row<WT, n><<<N, 256, 0, s>>>(w8, rb, K, x, y, ldy, acc); return;
      CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    }
  }
  const dim3 grid(cdiv(N, MMV_WARPS)), block(MMV_WARPS * 32);
  const uint8_t* w = (const uint8_t*)W;
  switch (B) {
#define CASE(n) case n: k_mmv_float<WT, n><<<grid, block, 0, s>>>(w, rb, K, N, x, y, ldy, acc); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    default: QE_FATAL("matvec: unsupported batch %d", B);
  }
}

// Gathered single-row matvec: y[i] = W[rows[i]] . x  (draft head over a vocabulary subset)
template <int T>
__global__ void __launch_bounds__(MMV_WARPS * 32)
    k_mmv_rows(const uint8_t* __restrict__ W, size_t rb, int K, const int* __restrict__ rows, int N,
               const int8_t* __restrict__ xq, const float* __restrict__ xd, float* __restrict__ y) {
  using D = QDec<T>;
  const int i = blockIdx.x * MMV_WARPS + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (i >= N) return;
  const uint8_t* wr = W + (size_t)__ldg(rows + i) * rb;
  const int nsb = K / 32;
  float sum = 0.f;
  for (int g = lane; g < nsb; g += 32) {
    W32 w;
    D::dec(wr + (size_t)(g >> 3) * D::BB, g & 7, w);
    const int4* xp = (const int4*)(xq + (size_t)g * 32);
    sum += dot_w32<D::OFF, D::SCALES>(w, __ldg(xp), __ldg(xp + 1)) * __ldg(xd + g);
  }
  sum = warp_sum(sum);
  if (lane == 0) y[i] = sum;
}
void matvec_rows(uint32_t type, const void* W, size_t rb, int K, const int* rows, int N, const int8_t* xq,
                 const float* xd, float* y, cudaStream_t s) {
  const uint8_t* w = (const uint8_t*)W;
  switch (type) {
#define T(t) case t: k_mmv_rows<t><<<cdiv(N, MMV_WARPS), MMV_WARPS * 32, 0, s>>>(w, rb, K, rows, N, xq, xd, y); break;
    T(QT_Q8_0) T(QT_Q2_K) T(QT_Q3_K) T(QT_Q4_K) T(QT_Q5_K) T(QT_Q6_K) T(QT_IQ2_XXS) T(QT_IQ2_XS) T(QT_IQ2_S) T(QT_IQ3_XXS)
    T(QT_IQ3_S) T(QT_IQ1_S) T(QT_IQ1_M) T(QT_IQ4_XS)
#undef T
    default: QE_FATAL("matvec_rows: unsupported type %u", type);
  }
}

void matvec(uint32_t type, const void* W, size_t rb, int K, int N, const int8_t* xq,
            const float* xd, const float* xf, int B, float* y, int ldy, bool acc, cudaStream_t s) {
  switch (type) {
    case QT_F32: launch_mmv_float<QT_F32>(W, rb, K, N, xf, B, y, ldy, acc, s); break;
    case QT_F16: launch_mmv_float<QT_F16>(W, rb, K, N, xf, B, y, ldy, acc, s); break;
    case QT_BF16: launch_mmv_float<QT_BF16>(W, rb, K, N, xf, B, y, ldy, acc, s); break;
#define T(t) case t: launch_mmv_t<t>(W, rb, K, N, xq, xd, B, y, ldy, acc, s); break;
    T(QT_Q8_0) T(QT_Q2_K) T(QT_Q3_K) T(QT_Q4_K) T(QT_Q5_K) T(QT_Q6_K) T(QT_IQ2_XXS) T(QT_IQ2_XS) T(QT_IQ2_S) T(QT_IQ3_XXS)
    T(QT_IQ3_S) T(QT_IQ1_S) T(QT_IQ1_M) T(QT_IQ4_XS)
#undef T
    default: QE_FATAL("matvec: unsupported type %u", type);
  }
}

// =============================================================================================
// MMQ: int8 tensor-core GEMM directly on GGUF-quantized weights (prefill, T > kMaxB tokens).
//   C[t][n] = sum_k W[n][k] x[t][k],  x int8 with one scale per 256 (quantize_q8_256).
// Weights are decoded per 256-wide super-block into int8 + *integer* sub-scales (and min/offset
// coefficients), so the per-group epilogue is a single integer multiply-add per output element and
// the float conversion happens once per super-block (same factorization as the CPU kernels).
// CTA tile = 64 weight rows x 128 tokens, 8 warps (2 x 4), warp tile 32 x 32.
// =============================================================================================
constexpr int MQ_BM = 64, MQ_BN = 64, MQ_LD = 256 + 16;
constexpr int MQ_NT = MQ_BN / 32;  // n8 tiles per warp (warps are 2 x 4)

static __device__ __forceinline__ void mma_s8_k32(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP v1: MMQ is gated off (mmq_supported returns false); never called.
  (void)c; (void)a; (void)b0; (void)b1;
#else
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
static __device__ __forceinline__ void mma_s8_k16(int* c, uint32_t a0, uint32_t a1, uint32_t b0) {
#ifdef __HIP_PLATFORM_AMD__
  (void)c; (void)a0; (void)a1; (void)b0;
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// Integer-scale decode of one 32-weight sub-block: int8 weights (8 ints), integer scale per scale
// group (SC groups of 32/SC), integer min coefficient (MIN types), plus per-super-block factors:
//   value = fac * s * w  + mfac * m        (summed against x)
struct WI {
  int w[8];
  int s[2];
  int m[2];
};
template <int T> struct QInt;
template <> struct QInt<QT_IQ2_XXS> {
  static constexpr int SC = 1; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b) * 0.125f; }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, WI& o) {
    const uint32_t a0 = ld32a2(b + 2 + 8 * ib), a1 = ld32a2(b + 2 + 8 * ib + 4);
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint64_t g = qtab_dev::iq2xxs_grid[(a0 >> (8 * l)) & 255];
      const uint32_t sg = iq2_signs8((a1 >> (7 * l)) & 127);
      o.w[2 * l] = apply_signs4((uint32_t)g, sg);
      o.w[2 * l + 1] = apply_signs4((uint32_t)(g >> 32), sg >> 4);
    }
    o.s[0] = 2 * (a1 >> 28) + 1;
  }
};
template <> struct QInt<QT_IQ2_XS> {
  static constexpr int SC = 2; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b) * 0.125f; }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, WI& o) {
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint32_t q = ld16(b + 2 + 2 * (4 * ib + l));
      const uint64_t g = qtab_dev::iq2xs_grid[q & 511];
      const uint32_t sg = iq2_signs8(q >> 9);
      o.w[2 * l] = apply_signs4((uint32_t)g, sg);
      o.w[2 * l + 1] = apply_signs4((uint32_t)(g >> 32), sg >> 4);
    }
    const uint32_t scb = b[66 + ib];
    o.s[0] = 2 * (scb & 0xf) + 1;
    o.s[1] = 2 * (scb >> 4) + 1;
  }
};
template <> struct QInt<QT_IQ2_S> {
  static constexpr int SC = 2; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b) * 0.125f; }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, WI& o) {
    const uint8_t* qs = b + 2 + 4 * ib;
    const uint8_t* signs = b + 2 + 32 + 4 * ib;
    const uint32_t qh = b[66 + ib];
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint64_t g = qtab_dev::iq2s_grid[qs[l] | ((qh << (8 - 2 * l)) & 0x300)];
      o.w[2 * l] = apply_signs4((uint32_t)g, signs[l]);
      o.w[2 * l + 1] = apply_signs4((uint32_t)(g >> 32), (uint32_t)signs[l] >> 4);
    }
    const uint32_t scb = b[74 + ib];
    o.s[0] = 2 * (scb & 0xf) + 1;
    o.s[1] = 2 * (scb >> 4) + 1;
  }
};
template <> struct QInt<QT_IQ3_XXS> {
  static constexpr int SC = 1; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b) * 0.25f; }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, WI& o) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t a = ld32a2(b + 2 + 64 + 4 * ib);
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint32_t sg = iq2_signs8((a >> (7 * l)) & 127);
      o.w[2 * l] = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l]], sg);
      o.w[2 * l + 1] = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l + 1]], sg >> 4);
    }
    o.s[0] = 2 * (a >> 28) + 1;
  }
};
template <> struct QInt<QT_IQ3_S> {
  static constexpr int SC = 1; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b); }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, WI& o) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t qh = b[66 + ib];
    const uint8_t* signs = b + 74 + 4 * ib;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint32_t g1 = qtab_dev::iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)];
      const uint32_t g2 = qtab_dev::iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
      o.w[2 * l] = apply_signs4(g1, signs[l]);
      o.w[2 * l + 1] = apply_signs4(g2, (uint32_t)signs[l] >> 4);
    }
    o.s[0] = 1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf);
  }
};
template <> struct QInt<QT_IQ4_XS> {
  static constexpr int SC = 1; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b); }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, WI& o) {
    const uint32_t sh = ld16(b + 2);
    o.s[0] = (((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4)) - 32;
    const uint32_t* tab = (const uint32_t*)qtab_dev::kvalues_iq4nl;
    uint32_t t[4] = {tab[0], tab[1], tab[2], tab[3]};
    const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t q = ld32a4(qs + 4 * i);
      o.w[i] = lut16(q & 0x0F0F0F0F, t);
      o.w[4 + i] = lut16((q >> 4) & 0x0F0F0F0F, t);
    }
  }
};
template <> struct QInt<QT_Q2_K> {
  static constexpr int SC = 2; static constexpr bool MIN = true;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b + 80); }
  static __device__ __forceinline__ float mfac(const uint8_t* b) { return -h2f(b + 82); }
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, WI& o) {
    const uint8_t* q = b + 16 + (g / 4) * 32;
    const int shift = 2 * (g % 4);
#pragma unroll
    for (int i = 0; i < 8; i++) o.w[i] = (int)((ld32a4(q + 4 * i) >> shift) & 0x03030303u);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const uint32_t sc = b[2 * g + h];
      o.s[h] = sc & 0xF;
      o.m[h] = sc >> 4;
    }
  }
};
template <> struct QInt<QT_Q4_K> {
  static constexpr int SC = 1; static constexpr bool MIN = true;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b); }
  static __device__ __forceinline__ float mfac(const uint8_t* b) { return -h2f(b + 2); }
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, WI& o) {
    uint8_t sc, m;
    q4k_scale_min(g, b + 4, sc, m);
    const uint8_t* q = b + 16 + (g / 2) * 32;
    const int sh = 4 * (g & 1);
#pragma unroll
    for (int i = 0; i < 8; i++) o.w[i] = (int)((ld32a4(q + 4 * i) >> sh) & 0x0F0F0F0Fu);
    o.s[0] = sc;
    o.m[0] = m;
  }
};
template <> struct QInt<QT_Q5_K> {
  static constexpr int SC = 1; static constexpr bool MIN = true;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b); }
  static __device__ __forceinline__ float mfac(const uint8_t* b) { return -h2f(b + 2); }
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, WI& o) {
    uint8_t sc, m;
    q4k_scale_min(g, b + 4, sc, m);
    const uint8_t* q = b + 48 + (g / 2) * 32;
    const int sh = 4 * (g & 1);
#pragma unroll
    for (int i = 0; i < 8; i++)
      o.w[i] = (int)(((ld32a4(q + 4 * i) >> sh) & 0x0F0F0F0Fu) | (((ld32a4(b + 16 + 4 * i) >> g) & 0x01010101u) << 4));
    o.s[0] = sc;
    o.m[0] = m;
  }
};
template <> struct QInt<QT_Q3_K> {
  static constexpr int SC = 2; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b + 108); }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, WI& o) {
    q3k_scale_pair(b + 96, g, o.s[0], o.s[1]);
    const uint8_t* q = b + 32 + 32 * (g / 4);
    const int shift = 2 * (g % 4);
#pragma unroll
    for (int i = 0; i < 8; i++) o.w[i] = (int)q3k_vals(ld32a2(q + 4 * i), ld32a2(b + 4 * i), shift, g);
  }
};
template <> struct QInt<QT_Q6_K> {
  static constexpr int SC = 2; static constexpr bool MIN = false;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b + 208); }
  static __device__ __forceinline__ float mfac(const uint8_t*) { return 0.f; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int g, WI& o) {
    const int h = g / 4, r = g % 4;
    const uint8_t* ql = b + 64 * h + (r & 1) * 32;
    const uint8_t* qh = b + 128 + 32 * h;
    const int8_t* sc = (const int8_t*)(b + 192 + 8 * h + 2 * r);
    const int lsh = (r >= 2) ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t lo = (ld32a2(ql + 4 * i) >> lsh) & 0x0F0F0F0Fu;
      const uint32_t hi = ((ld32a2(qh + 4 * i) >> (2 * r)) & 0x03030303u) << 4;
      o.w[i] = (int)__vsub4(lo | hi, 0x20202020u);
    }
    o.s[0] = sc[0];
    o.s[1] = sc[1];
  }
};
template <> struct QInt<QT_IQ1_S> {
  static constexpr int SC = 1; static constexpr bool MIN = true;
  static __device__ __forceinline__ float fac(const uint8_t* b) { return h2f(b); }
  static __device__ __forceinline__ float mfac(const uint8_t* b) { return h2f(b) * IQ1S_DELTA; }
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, WI& o) {
    const uint32_t qh = ld16(b + 34 + 2 * ib);
    const uint32_t qs4 = ld32a2(b + 2 + 4 * ib);
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const uint64_t g = qtab_dev::iq1s_grid[((qs4 >> (8 * l)) & 0xff) | (((qh >> (3 * l)) & 7) << 8)];
      o.w[2 * l] = (int)(uint32_t)g;
      o.w[2 * l + 1] = (int)(uint32_t)(g >> 32);
    }
    o.s[0] = 2 * ((qh >> 12) & 7) + 1;
    o.m[0] = (qh & 0x8000) ? -o.s[0] : o.s[0];
  }
};

template <int SC, bool MIN, int GS>
struct MqSmem {
  int8_t W[MQ_BM][MQ_LD];
  int8_t X[MQ_BN][MQ_LD];
  int ws[MQ_BM][8 * SC];
  int wm[MIN ? MQ_BM : 1][8 * SC];
  float fac[MQ_BM], mfac[MQ_BM];
  float xd[MQ_BN][256 / GS];
  int xs[MIN ? MQ_BN : 1][8 * SC];  // integer sums of x per scale group
};

template <int T, int GS>
__global__ void __launch_bounds__(256, 2)
    k_mmq(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
          const float* __restrict__ xd256, int Tt, float* __restrict__ y, int ldy, int accum, int dbg) {
  using D = QInt<T>;
  constexpr int SC = D::SC;
  constexpr bool MIN = D::MIN;
  constexpr int NG = 8 * SC;
  constexpr int BB = QDec<T>::BB;
  extern __shared__ __align__(16) unsigned char mq_raw[];
  constexpr int XG = 256 / GS;  // activation scale groups per super-block
  MqSmem<SC, MIN, GS>& sm = *reinterpret_cast<MqSmem<SC, MIN, GS>*>(mq_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int n0 = blockIdx.x * MQ_BM, t0 = blockIdx.y * MQ_BN;
  const int wm = warp & 1, wn = warp >> 1;
  const int nb = K / 256;
  float acc[2][MQ_NT][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < MQ_NT; b++) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  for (int kb = 0; kb < nb; kb++) {
    __syncthreads();
    for (int i = tid; i < MQ_BN * 16; i += 256) {
      const int tok = i >> 4, ch = i & 15;
      const int t = t0 + tok;
      int4 v = make_int4(0, 0, 0, 0);
      if (t < Tt) v = __ldg((const int4*)(xq + (size_t)t * K + kb * 256 + ch * 16));
      *(int4*)&sm.X[tok][ch * 16] = v;
    }
    for (int i = tid; i < MQ_BN * XG; i += 256) {
      const int tok = i / XG, j = i % XG;
      const int t = t0 + tok;
      sm.xd[tok][j] = t < Tt ? __ldg(xd256 + ((size_t)t * nb + kb) * XG + j) : 0.f;
    }
    for (int i = tid; i < MQ_BM * 8 && !(dbg & 1); i += 256) {
      const int row = i >> 3, gg = i & 7;
      const int r = min(n0 + row, N - 1);
      const uint8_t* blk = W + (size_t)r * rb + (size_t)kb * BB;
      WI w;
      D::dec(blk, gg, w);
      *(int4*)&sm.W[row][gg * 32] = make_int4(w.w[0], w.w[1], w.w[2], w.w[3]);
      *(int4*)&sm.W[row][gg * 32 + 16] = make_int4(w.w[4], w.w[5], w.w[6], w.w[7]);
#pragma unroll
      for (int h = 0; h < SC; h++) {
        sm.ws[row][gg * SC + h] = w.s[h];
        if (MIN) sm.wm[row][gg * SC + h] = w.m[h];
      }
      if (gg == 0) {
        sm.fac[row] = D::fac(blk);
        sm.mfac[row] = D::mfac(blk);
      }
    }
    __syncthreads();
    if (MIN) {
      for (int i = tid; i < MQ_BN * NG; i += 256) {
        const int tok = i / NG, grp = i % NG;
        const int* xp = (const int*)&sm.X[tok][grp * (32 / SC)];
        int sx = 0;
#pragma unroll
        for (int j = 0; j < 8 / SC; j++) sx = __dp4a(0x01010101, xp[j], sx);
        sm.xs[tok][grp] = sx;
      }
      __syncthreads();
    }
    int tot[2][MQ_NT][4], mtot[2][MQ_NT][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
      for (int b = 0; b < MQ_NT; b++)
#pragma unroll
        for (int e = 0; e < 4; e++) { tot[a][b][e] = 0; mtot[a][b][e] = 0; }
    // float epilogue: acc += xd * (fac * sum_int + mfac * minsum_int); resets the integer sums
    auto epilogue = [&](int xg) {
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        const int ra = wm * 32 + mt * 16 + g;
        const float fa = sm.fac[ra], fb = sm.fac[ra + 8];
        const float ma = MIN ? sm.mfac[ra] : 0.f, mb = MIN ? sm.mfac[ra + 8] : 0.f;
#pragma unroll
        for (int nt = 0; nt < MQ_NT; nt++) {
          const int ta = wn * (8 * MQ_NT) + nt * 8 + 2 * tq;
          const float x0 = sm.xd[ta][xg], x1 = sm.xd[ta + 1][xg];
          float v0 = fa * (float)tot[mt][nt][0], v1 = fa * (float)tot[mt][nt][1];
          float v2 = fb * (float)tot[mt][nt][2], v3 = fb * (float)tot[mt][nt][3];
          if (MIN) {
            v0 += ma * (float)mtot[mt][nt][0]; v1 += ma * (float)mtot[mt][nt][1];
            v2 += mb * (float)mtot[mt][nt][2]; v3 += mb * (float)mtot[mt][nt][3];
          }
          acc[mt][nt][0] += v0 * x0;
          acc[mt][nt][1] += v1 * x1;
          acc[mt][nt][2] += v2 * x0;
          acc[mt][nt][3] += v3 * x1;
#pragma unroll
          for (int e = 0; e < 4; e++) { tot[mt][nt][e] = 0; mtot[mt][nt][e] = 0; }
        }
      }
    };

    for (int gg = 0; gg < 8 && !(dbg & 2); gg++) {
#pragma unroll
      for (int h = 0; h < SC; h++) {
        const int grp = gg * SC + h;
        const int kofs = gg * 32 + h * 16;
#pragma unroll
        for (int mt = 0; mt < 2; mt++) {
          const int rbase = wm * 32 + mt * 16;
          int ci[MQ_NT][4];
#pragma unroll
          for (int b = 0; b < MQ_NT; b++) ci[b][0] = ci[b][1] = ci[b][2] = ci[b][3] = 0;
          if (SC == 1) {
            uint32_t a[4];
            a[0] = *(const uint32_t*)&sm.W[rbase + g][kofs + tq * 4];
            a[1] = *(const uint32_t*)&sm.W[rbase + g + 8][kofs + tq * 4];
            a[2] = *(const uint32_t*)&sm.W[rbase + g][kofs + 16 + tq * 4];
            a[3] = *(const uint32_t*)&sm.W[rbase + g + 8][kofs + 16 + tq * 4];
#pragma unroll
            for (int nt = 0; nt < MQ_NT; nt++) {
              const int tb = wn * (8 * MQ_NT) + nt * 8 + g;
              mma_s8_k32(ci[nt], a, *(const uint32_t*)&sm.X[tb][kofs + tq * 4],
                         *(const uint32_t*)&sm.X[tb][kofs + 16 + tq * 4]);
            }
          } else {
            const uint32_t a0 = *(const uint32_t*)&sm.W[rbase + g][kofs + tq * 4];
            const uint32_t a1 = *(const uint32_t*)&sm.W[rbase + g + 8][kofs + tq * 4];
#pragma unroll
            for (int nt = 0; nt < MQ_NT; nt++) {
              const int tb = wn * (8 * MQ_NT) + nt * 8 + g;
              mma_s8_k16(ci[nt], a0, a1, *(const uint32_t*)&sm.X[tb][kofs + tq * 4]);
            }
          }
          const int s_lo = sm.ws[rbase + g][grp], s_hi = sm.ws[rbase + g + 8][grp];
#pragma unroll
          for (int nt = 0; nt < MQ_NT; nt++) {
            tot[mt][nt][0] += ci[nt][0] * s_lo;
            tot[mt][nt][1] += ci[nt][1] * s_lo;
            tot[mt][nt][2] += ci[nt][2] * s_hi;
            tot[mt][nt][3] += ci[nt][3] * s_hi;
          }
          if (MIN) {
            const int m_lo = sm.wm[rbase + g][grp], m_hi = sm.wm[rbase + g + 8][grp];
#pragma unroll
            for (int nt = 0; nt < MQ_NT; nt++) {
              const int ta = wn * (8 * MQ_NT) + nt * 8 + 2 * tq;
              const int xa = sm.xs[ta][grp], xb = sm.xs[ta + 1][grp];
              mtot[mt][nt][0] += m_lo * xa;
              mtot[mt][nt][1] += m_lo * xb;
              mtot[mt][nt][2] += m_hi * xa;
              mtot[mt][nt][3] += m_hi * xb;
            }
          }
        }
      }
      if (GS == 32) epilogue(gg);
    }
    if (GS == 256) epilogue(0);
  }
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int nt = 0; nt < MQ_NT; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int row = n0 + wm * 32 + mt * 16 + g + 8 * (e >> 1);
        const int tok = t0 + wn * (8 * MQ_NT) + nt * 8 + 2 * tq + (e & 1);
        if (row < N && tok < Tt) {
          float* o = y + (size_t)tok * ldy + row;
          *o = accum ? *o + acc[mt][nt][e] : acc[mt][nt][e];
        }
      }
}

template <int T, int GS>
static void launch_mmq_t(const void* W, size_t rb, int K, int N, const int8_t* xq, const float* xd, int Tt, float* y,
                         int ldy, bool acc, cudaStream_t s) {
  using D = QInt<T>;
  const size_t sh = sizeof(MqSmem<D::SC, D::MIN, GS>);
  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaFuncSetAttribute(k_mmq<T, GS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    init = true;
  }
  const dim3 grid(cdiv(N, MQ_BM), cdiv(Tt, MQ_BN));
  static const int dbg = getenv("QE_MMQ_DBG") ? atoi(getenv("QE_MMQ_DBG")) : 0;
  k_mmq<T, GS><<<grid, 256, sh, s>>>((const uint8_t*)W, rb, K, N, xq, xd, Tt, y, ldy, acc, dbg);
}

// =============================================================================================
// GQ: "decode once, then int8 tensor-core GEMM" for long prefill chunks.
//   1) gq_decode: GGUF blocks -> int8 weights W8[N][K], int8 sub-scales S8[N][K/16] (and min
//      coefficients M8 for MIN types), float factors FAC/MFAC[N][K/256]   (QInt factorization)
//   2) k_gq_gemm: 64x128 CTA tile, 4-stage cp.async pipeline, ldmatrix fragments, integer sub-scale
//      accumulation per 16/32 group and one float epilogue per 256 (activations int8 per 256).
// Each weight is decoded once per chunk instead of once per token tile (MMQ).
// =============================================================================================
template <int T>
__global__ void k_gq_decode(const uint8_t* __restrict__ W, size_t rb, int K, int N, int8_t* __restrict__ w8,
                            int8_t* __restrict__ s8, int8_t* __restrict__ m8, float* __restrict__ fac,
                            float* __restrict__ mfac) {
  using D = QInt<T>;
  constexpr int BB = QDec<T>::BB;
  const int nsub = K / 32;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)N * nsub) return;
  const int row = (int)(i / nsub), j = (int)(i % nsub);
  const int kb = j >> 3, gg = j & 7;
  const uint8_t* blk = W + (size_t)row * rb + (size_t)kb * BB;
  WI w;
  D::dec(blk, gg, w);
  constexpr bool FOLD = T == QT_IQ1_S;  // d*s*(w +- 1/8) = (d/8)*s*(8w +- 1): exact int8, no min term
  if constexpr (FOLD) {
    const int sg = w.m[0] > 0 ? 1 : -1;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint32_t v = 0;
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const int q = (int)(int8_t)((uint32_t)w.w[i] >> (8 * b));
        v |= (uint32_t)(uint8_t)(int8_t)(8 * q + sg) << (8 * b);
      }
      w.w[i] = (int)v;
    }
  }
  int4* dst = (int4*)(w8 + (size_t)row * K + (size_t)j * 32);
  dst[0] = make_int4(w.w[0], w.w[1], w.w[2], w.w[3]);
  dst[1] = make_int4(w.w[4], w.w[5], w.w[6], w.w[7]);
  const size_t so = (size_t)row * (K / 16) + (size_t)j * 2;
  s8[so] = (int8_t)w.s[0];
  s8[so + 1] = (int8_t)(D::SC == 2 ? w.s[1] : w.s[0]);
  if constexpr (D::MIN && !FOLD) {
    m8[so] = (int8_t)w.m[0];
    m8[so + 1] = (int8_t)(D::SC == 2 ? w.m[1] : w.m[0]);
  }
  if (gg == 0) {
    fac[(size_t)row * (K / 256) + kb] = FOLD ? D::fac(blk) * 0.125f : D::fac(blk);
    if constexpr (D::MIN && !FOLD) mfac[(size_t)row * (K / 256) + kb] = D::mfac(blk);
  }
}

__global__ void k_gq_xsum(const int8_t* __restrict__ xq, int K, int T, int16_t* __restrict__ xs) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int ng = K / 16;
  if (i >= (long long)T * ng) return;
  const int4 v = __ldg((const int4*)(xq + (size_t)i * 16));
  int s = __dp4a(v.x, 0x01010101, 0);
  s = __dp4a(v.y, 0x01010101, s);
  s = __dp4a(v.z, 0x01010101, s);
  s = __dp4a(v.w, 0x01010101, s);
  xs[i] = (int16_t)s;
}

static __device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
#ifdef __HIP_PLATFORM_AMD__
  *(uint32_t*)smem = *(const uint32_t*)gmem;
#else
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a), "l"(gmem));
#endif
}
static __device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP v1: GQ GEMM is gated off (gq_supported returns false); never called.
  (void)r; (void)p;
#else
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}

constexpr int GQ_BN = 128, GQ_BK = 64, GQ_ST = 3, GQ_LDS = GQ_BK + 16;
template <int MT, bool MIN>
struct GQL {  // smem layout for a CTA of 32*MT weight rows x 128 tokens
  static constexpr int BM = 32 * MT;
  static constexpr int A = BM * GQ_LDS, B = GQ_BN * GQ_LDS, S = BM * (GQ_BK / 16), XS = MIN ? GQ_BN * (GQ_BK / 16) * 2 : 0;
  static constexpr int F = (2 * BM + GQ_BN) * 4;  // fac, mfac per row and xd per token (last stage of a 256)
  static constexpr int STAGE = A + B + 2 * S + XS + F;
  static constexpr size_t SMEM = (size_t)STAGE * GQ_ST;
};

template <bool SC2, bool MIN, int MT>
__global__ void __launch_bounds__(256, 1)
    k_gq_gemm(const int8_t* __restrict__ w8, const int8_t* __restrict__ s8, const int8_t* __restrict__ m8,
              const float* __restrict__ fac, const float* __restrict__ mfac, int K, int N,
              const int8_t* __restrict__ xq, const float* __restrict__ xd, const int16_t* __restrict__ xs,
              int Tt, float* __restrict__ y, int ldy, int accum) {
  extern __shared__ __align__(16) uint8_t gq_sm[];
  using LY = GQL<MT, MIN>;
  constexpr int GQ_BM = LY::BM, GQ_A = LY::A, GQ_B = LY::B, GQ_S = LY::S, GQ_STAGE = LY::STAGE;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int wm = warp & 1, wn = warp >> 1;  // 2 x 4 warps, warp tile 16*MT rows x 32 tokens
  const int t0 = blockIdx.x * GQ_BN, n0 = blockIdx.y * GQ_BM;
  const int nk = K / GQ_BK, K16 = K / 16, K256 = K / 256;

  auto load = [&](int ks, int st) {
    uint8_t* base = gq_sm + (size_t)st * GQ_STAGE;
    const int k0 = ks * GQ_BK;
#pragma unroll
    for (int i = 0; i < MT / 2; i++) {
      const int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
      const int r = min(n0 + row, N - 1);
      cp_async16(base + row * GQ_LDS + ch * 16, w8 + (size_t)r * K + k0 + ch * 16);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int idx = tid + i * 256, tok = idx >> 2, ch = idx & 3;
      const int t = min(t0 + tok, Tt - 1);
      cp_async16(base + GQ_A + tok * GQ_LDS + ch * 16, xq + (size_t)t * K + k0 + ch * 16);
    }
    if (tid < GQ_BM) {
      const int r = min(n0 + tid, N - 1);
      cp_async4(base + GQ_A + GQ_B + tid * 4, s8 + (size_t)r * K16 + k0 / 16);
      if (MIN) cp_async4(base + GQ_A + GQ_B + GQ_S + tid * 4, m8 + (size_t)r * K16 + k0 / 16);
    }
    if (MIN && tid >= 128) {
      const int tok = tid - 128, t = min(t0 + tok, Tt - 1);
      cp_async8(base + GQ_A + GQ_B + 2 * GQ_S + tok * 8, xs + (size_t)t * K16 + k0 / 16);
    }
    if ((ks & 3) == 3) {  // per-256 factors, consumed by the epilogue of this stage
      float* F = (float*)(base + GQ_A + GQ_B + 2 * GQ_S + LY::XS);
      const int kb = ks >> 2;
      if (tid < GQ_BM) {
        const int r = min(n0 + tid, N - 1);
        cp_async4(F + tid, fac + (size_t)r * K256 + kb);
        if (MIN) cp_async4(F + GQ_BM + tid, mfac + (size_t)r * K256 + kb);
      }
      if (tid >= GQ_BM && tid < GQ_BM + GQ_BN) {
        const int tok = tid - GQ_BM, t = min(t0 + tok, Tt - 1);
        cp_async4(F + 2 * GQ_BM + tok, xd + (size_t)t * K256 + kb);
      }
    }
  };

  float acc[MT][4][4];
  int tot[MT][4][4], mtot[MT][4][4];
#pragma unroll
  for (int a = 0; a < MT; a++)
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
      for (int e = 0; e < 4; e++) { acc[a][b][e] = 0.f; tot[a][b][e] = 0; mtot[a][b][e] = 0; }

#pragma unroll
  for (int s = 0; s < GQ_ST - 1; s++) {
    if (s < nk) load(s, s);
    cp_async_commit();
  }
  for (int ks = 0; ks < nk; ks++) {
    cp_async_wait<GQ_ST - 2>();
    __syncthreads();
    if (ks + GQ_ST - 1 < nk) load(ks + GQ_ST - 1, (ks + GQ_ST - 1) % GQ_ST);
    cp_async_commit();
    const uint8_t* base = gq_sm + (size_t)(ks % GQ_ST) * GQ_STAGE;
    const uint8_t* As = base;
    const uint8_t* Bs = base + GQ_A;
    const int8_t* Ss = (const int8_t*)(base + GQ_A + GQ_B);
    const int8_t* Ms = Ss + GQ_S;
    const int16_t* Xs = (const int16_t*)(base + GQ_A + GQ_B + 2 * GQ_S);
#pragma unroll
    for (int kk = 0; kk < GQ_BK; kk += 32) {
      uint32_t a[MT][4], b[4][2];
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
        ldsm_x4(a[mt], As + (wm * (16 * MT) + mt * 16 + (lane & 15)) * GQ_LDS + kk + (lane >> 4) * 16);
#pragma unroll
      for (int np = 0; np < 2; np++) {
        uint32_t r[4];
        ldsm_x4(r, Bs + (wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * GQ_LDS + kk + ((lane >> 3) & 1) * 16);
        b[2 * np][0] = r[0]; b[2 * np][1] = r[1]; b[2 * np + 1][0] = r[2]; b[2 * np + 1][1] = r[3];
      }
      const int gi = kk / 16;  // 16-group index within the stage
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        const int rl = wm * (16 * MT) + mt * 16 + g, rh = rl + 8;
        if constexpr (!SC2) {
          const int sl = Ss[rl * 4 + gi], sh = Ss[rh * 4 + gi];
#pragma unroll
          for (int nt = 0; nt < 4; nt++) {
            int c[4] = {0, 0, 0, 0};
            mma_s8_k32(c, a[mt], b[nt][0], b[nt][1]);
            tot[mt][nt][0] += c[0] * sl; tot[mt][nt][1] += c[1] * sl;
            tot[mt][nt][2] += c[2] * sh; tot[mt][nt][3] += c[3] * sh;
          }
          if constexpr (MIN) {
            const int ml = Ms[rl * 4 + gi], mh = Ms[rh * 4 + gi];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
              const int ta = wn * 32 + nt * 8 + 2 * tq;
              const int x0 = Xs[ta * 4 + gi] + Xs[ta * 4 + gi + 1];
              const int x1 = Xs[(ta + 1) * 4 + gi] + Xs[(ta + 1) * 4 + gi + 1];
              mtot[mt][nt][0] += ml * x0; mtot[mt][nt][1] += ml * x1;
              mtot[mt][nt][2] += mh * x0; mtot[mt][nt][3] += mh * x1;
            }
          }
        } else {
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int sl = Ss[rl * 4 + gi + h], sh = Ss[rh * 4 + gi + h];
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
              int c[4] = {0, 0, 0, 0};
              mma_s8_k16(c, a[mt][2 * h], a[mt][2 * h + 1], b[nt][h]);
              tot[mt][nt][0] += c[0] * sl; tot[mt][nt][1] += c[1] * sl;
              tot[mt][nt][2] += c[2] * sh; tot[mt][nt][3] += c[3] * sh;
            }
            if constexpr (MIN) {
              const int ml = Ms[rl * 4 + gi + h], mh = Ms[rh * 4 + gi + h];
#pragma unroll
              for (int nt = 0; nt < 4; nt++) {
                const int ta = wn * 32 + nt * 8 + 2 * tq;
                const int x0 = Xs[ta * 4 + gi + h], x1 = Xs[(ta + 1) * 4 + gi + h];
                mtot[mt][nt][0] += ml * x0; mtot[mt][nt][1] += ml * x1;
                mtot[mt][nt][2] += mh * x0; mtot[mt][nt][3] += mh * x1;
              }
            }
          }
        }
      }
    }
    if ((ks & 3) == 3) {  // end of a 256-wide super-block: float epilogue
      const float* F = (const float*)(base + GQ_A + GQ_B + 2 * GQ_S + LY::XS);
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        const int rl = wm * (16 * MT) + mt * 16 + g;
        const float fa = F[rl], fb = F[rl + 8];
        const float ma = MIN ? F[GQ_BM + rl] : 0.f, mb = MIN ? F[GQ_BM + rl + 8] : 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          const int tl = wn * 32 + nt * 8 + 2 * tq;
          const float x0 = F[2 * GQ_BM + tl], x1 = F[2 * GQ_BM + tl + 1];
          float v0 = fa * (float)tot[mt][nt][0], v1 = fa * (float)tot[mt][nt][1];
          float v2 = fb * (float)tot[mt][nt][2], v3 = fb * (float)tot[mt][nt][3];
          if (MIN) {
            v0 += ma * (float)mtot[mt][nt][0]; v1 += ma * (float)mtot[mt][nt][1];
            v2 += mb * (float)mtot[mt][nt][2]; v3 += mb * (float)mtot[mt][nt][3];
          }
          acc[mt][nt][0] += v0 * x0; acc[mt][nt][1] += v1 * x1;
          acc[mt][nt][2] += v2 * x0; acc[mt][nt][3] += v3 * x1;
#pragma unroll
          for (int e = 0; e < 4; e++) { tot[mt][nt][e] = 0; mtot[mt][nt][e] = 0; }
        }
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    const int ra = n0 + wm * (16 * MT) + mt * 16 + g, rb2 = ra + 8;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const int ta = t0 + wn * 32 + nt * 8 + 2 * tq;
      const int rr[4] = {ra, ra, rb2, rb2}, tt[4] = {ta, ta + 1, ta, ta + 1};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        if (rr[e] >= N || tt[e] >= Tt) continue;
        float* o = y + (size_t)tt[e] * ldy + rr[e];
        *o = accum ? *o + acc[mt][nt][e] : acc[mt][nt][e];
      }
    }
  }
}

size_t gq_buffer_bytes(int K, int rows) {
  // W8 + S8 + M8 + FAC + MFAC, each region 256-aligned
  auto al = [](size_t b) { return (b + 255) / 256 * 256; };
  return al((size_t)rows * K) + 2 * al((size_t)rows * (K / 16)) + 2 * al((size_t)rows * (K / 256) * 4);
}

void gq_decode(uint32_t type, const void* W, size_t rb, int K, int N, void* buf, cudaStream_t s) {
  auto al = [](size_t b) { return (b + 255) / 256 * 256; };
  int8_t* w8 = (int8_t*)buf;
  int8_t* s8 = w8 + al((size_t)N * K);
  int8_t* m8 = s8 + al((size_t)N * (K / 16));
  float* fac = (float*)(m8 + al((size_t)N * (K / 16)));
  float* mfac = (float*)((uint8_t*)fac + al((size_t)N * (K / 256) * 4));
  const long long n = (long long)N * (K / 32);
  const int blocks = (int)((n + 255) / 256);
  const uint8_t* w = (const uint8_t*)W;
  switch (type) {
#define T(t) case t: k_gq_decode<t><<<blocks, 256, 0, s>>>(w, rb, K, N, w8, s8, m8, fac, mfac); break;
    T(QT_Q2_K) T(QT_Q3_K) T(QT_Q4_K) T(QT_Q5_K) T(QT_Q6_K) T(QT_IQ2_XXS) T(QT_IQ2_XS) T(QT_IQ2_S) T(QT_IQ3_XXS)
    T(QT_IQ3_S) T(QT_IQ1_S) T(QT_IQ4_XS)
#undef T
    default: QE_FATAL("gq_decode: unsupported type %u", type);
  }
}

void gq_xsum(const int8_t* xq, int K, int T, int16_t* xs, cudaStream_t s) {
  const long long n = (long long)T * (K / 16);
  k_gq_xsum<<<(int)((n + 255) / 256), 256, 0, s>>>(xq, K, T, xs);
}

bool gq_needs_xsum(uint32_t type) { return type == QT_Q2_K || type == QT_Q4_K || type == QT_Q5_K; }  // IQ1_S: min folded
bool gq_supported(uint32_t type) {
#ifdef __HIP_PLATFORM_AMD__
  (void)type;
  return false;  // follows from mmq_supported() == false on HIP
#else
  static const bool all = getenv("QE_GQ_MIN") && atoi(getenv("QE_GQ_MIN")) != 0;  // MIN types: +0.4% PPL so far
  return mmq_supported(type) && (all || !gq_needs_xsum(type));
#endif
}

void gq_gemm(uint32_t type, const void* buf, int K, int N, const int8_t* xq, const float* xd256,
             const int16_t* xs, int T, float* y, int ldy, bool accumulate, cudaStream_t s) {
  auto al = [](size_t b) { return (b + 255) / 256 * 256; };
  const int8_t* w8 = (const int8_t*)buf;
  const int8_t* s8 = w8 + al((size_t)N * K);
  const int8_t* m8 = s8 + al((size_t)N * (K / 16));
  const float* fac = (const float*)(m8 + al((size_t)N * (K / 16)));
  const float* mfac = (const float*)((const uint8_t*)fac + al((size_t)N * (K / 256) * 4));
  const bool sc2 = type == QT_Q2_K || type == QT_Q3_K || type == QT_Q6_K || type == QT_IQ2_XS || type == QT_IQ2_S;
  const bool mn = gq_needs_xsum(type);
#define L(a, b, mt)                                                                                    \
  {                                                                                                   \
    static bool init = false;                                                                         \
    if (!init) {                                                                                      \
      CUDA_CHECK(cudaFuncSetAttribute(k_gq_gemm<a, b, mt>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GQL<mt, b>::SMEM)); \
      init = true;                                                                                    \
    }                                                                                                 \
    const dim3 grid(cdiv(T, GQ_BN), cdiv(N, GQL<mt, b>::BM));                                         \
    k_gq_gemm<a, b, mt><<<grid, 256, GQL<mt, b>::SMEM, s>>>(w8, s8, m8, fac, mfac, K, N, xq, xd256, xs, T, y, ldy, accumulate); \
  }
  if (sc2) { if (mn) L(true, true, 2) else L(true, false, 4) }
  else { if (mn) L(false, true, 2) else L(false, false, 4) }
#undef L
}

// Row-wise top-K (K <= 128) of logits for sampling: 64 chunks per row are bitonic-sorted in shared
// memory keeping their top 128, then one block merges the 64x128 candidates.  Only K (id, logit)
// pairs per row go back to the host instead of the whole 248K-entry row.
constexpr int TK_CH = 64, TK_KEEP = 128, TK_S1 = 4096, TK_S2 = TK_CH * TK_KEEP;
template <int N>
static __device__ __forceinline__ void bitonic_desc(float* v, int* ix) {
  for (int k = 2; k <= N; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const int p = i ^ j;
        if (p > i) {
          const bool desc = (i & k) == 0;
          if (desc ? v[i] < v[p] : v[i] > v[p]) {
            const float tv = v[i]; v[i] = v[p]; v[p] = tv;
            const int ti = ix[i]; ix[i] = ix[p]; ix[p] = ti;
          }
        }
      }
      __syncthreads();
    }
}
__global__ void __launch_bounds__(1024) k_topk1(const float* __restrict__ l, int V, float* wv, int* wi) {
  __shared__ float v[TK_S1];
  __shared__ int ix[TK_S1];
  const int row = blockIdx.y, ch = blockIdx.x;
  const int len = (V + TK_CH - 1) / TK_CH, b = ch * len;
  for (int i = threadIdx.x; i < TK_S1; i += blockDim.x) {
    const int g = b + i;
    const bool ok = i < len && g < V;
    v[i] = ok ? l[(size_t)row * V + g] : -INFINITY;
    ix[i] = ok ? g : 0;
  }
  __syncthreads();
  bitonic_desc<TK_S1>(v, ix);
  for (int i = threadIdx.x; i < TK_KEEP; i += blockDim.x) {
    wv[((size_t)row * TK_CH + ch) * TK_KEEP + i] = v[i];
    wi[((size_t)row * TK_CH + ch) * TK_KEEP + i] = ix[i];
  }
}
__global__ void __launch_bounds__(1024) k_topk2(const float* wv, const int* wi, int K, float* ov, int* oi) {
  extern __shared__ float sm2[];
  float* v = sm2;
  int* ix = (int*)(sm2 + TK_S2);
  const int row = blockIdx.x;
  for (int i = threadIdx.x; i < TK_S2; i += blockDim.x) {
    v[i] = wv[(size_t)row * TK_S2 + i];
    ix[i] = wi[(size_t)row * TK_S2 + i];
  }
  __syncthreads();
  bitonic_desc<TK_S2>(v, ix);
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    ov[(size_t)row * K + i] = v[i];
    oi[(size_t)row * K + i] = ix[i];
  }
}
size_t topk_work_bytes(int rows) { return (size_t)rows * TK_S2 * 8; }
void topk_rows(const float* logits, int V, int rows, int K, float* out_val, int* out_idx, void* work, cudaStream_t s) {
  QE_ASSERT(K >= 1 && K <= TK_KEEP && (V + TK_CH - 1) / TK_CH <= TK_S1);
  float* wv = (float*)work;
  int* wi = (int*)(wv + (size_t)rows * TK_S2);
  k_topk1<<<dim3(TK_CH, rows), 1024, 0, s>>>(logits, V, wv, wi);
  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaFuncSetAttribute(k_topk2, cudaFuncAttributeMaxDynamicSharedMemorySize, TK_S2 * 8));
    init = true;
  }
  k_topk2<<<rows, 1024, TK_S2 * 8, s>>>(wv, wi, K, out_val, out_idx);
}

// two float-weight matrices with the same input and K in one launch (GDN alpha / beta)
void matvec_float_pair(uint32_t type, const void* W1, const void* W2, size_t rb, int K, int N, const float* x, int B,
                       float* y1, float* y2, int ldy, cudaStream_t s) {
  QE_ASSERT(type == QT_BF16 && N <= 256 && K % 8 == 0 && rb % 16 == 0);
  const uint8_t* a = (const uint8_t*)W1;
  const uint8_t* b = (const uint8_t*)W2;
  switch (B) {
#define CASE(n) case n: k_mmv_float_row<QT_BF16, n><<<2 * N, 256, 0, s>>>(a, rb, K, x, y1, ldy, 0, b, y2, N); return;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
  }
  QE_FATAL("matvec_float_pair: batch %d", B);
}

// =============================================================================================
// Tensor-core small-batch matvec (verify passes, 2 <= B <= 8 tokens): mma.m16n8k32.s8 with the
// weights as A (16 rows x 32 k) and the B tokens as the n = 8 columns.  Lane (g, tq) decodes only
// the 8 weights k = 8tq..8tq+7 of rows g and g+8 of each 32-sub-block (words 2tq, 2tq+1 of the WI
// layout, where word w holds k = 4w..4w+3), and the activation fragment uses the same k order (one
// 8-byte load).  Sub-block scales are integers applied to the int32 MMA result; one float multiply
// per 256-super-block with the activation scale (activations quantized per 256).  The dp4a kernels'
// per-token dot work, which grows with B and dominates at B >= 4, becomes one MMA per tile.
// =============================================================================================
template <int T> struct QPiece;
template <> struct QPiece<QT_IQ2_XXS> {
  static constexpr int SC = 1;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, int& s0, int& s1) {
    const uint32_t a0 = ld32a2(b + 2 + 8 * ib), a1 = ld32a2(b + 2 + 8 * ib + 4);
    const uint64_t g = qtab_dev::iq2xxs_grid[(a0 >> (8 * l)) & 255];
    const uint32_t sg = iq2_signs8((a1 >> (7 * l)) & 127);
    w0 = apply_signs4((uint32_t)g, sg);
    w1 = apply_signs4((uint32_t)(g >> 32), sg >> 4);
    s0 = s1 = 2 * (a1 >> 28) + 1;
  }
};
template <> struct QPiece<QT_IQ2_XS> {
  static constexpr int SC = 2;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, int& s0, int& s1) {
    const uint32_t q = ld16(b + 2 + 2 * (4 * ib + l));
    const uint64_t g = qtab_dev::iq2xs_grid[q & 511];
    const uint32_t sg = iq2_signs8(q >> 9);
    w0 = apply_signs4((uint32_t)g, sg);
    w1 = apply_signs4((uint32_t)(g >> 32), sg >> 4);
    const uint32_t scb = b[66 + ib];
    s0 = 2 * (scb & 0xf) + 1;
    s1 = 2 * (scb >> 4) + 1;
  }
};
template <> struct QPiece<QT_IQ2_S> {
  static constexpr int SC = 2;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, int& s0, int& s1) {
    const uint32_t qh = b[66 + ib];
    const uint64_t g = qtab_dev::iq2s_grid[b[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300)];
    const uint32_t sg = b[34 + 4 * ib + l];
    w0 = apply_signs4((uint32_t)g, sg);
    w1 = apply_signs4((uint32_t)(g >> 32), sg >> 4);
    const uint32_t scb = b[74 + ib];
    s0 = 2 * (scb & 0xf) + 1;
    s1 = 2 * (scb >> 4) + 1;
  }
};
template <> struct QPiece<QT_IQ3_XXS> {
  static constexpr int SC = 1;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, int& s0, int& s1) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t a = ld32a2(b + 2 + 64 + 4 * ib);
    const uint32_t sg = iq2_signs8((a >> (7 * l)) & 127);
    w0 = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l]], sg);
    w1 = apply_signs4(qtab_dev::iq3xxs_grid[qs[2 * l + 1]], sg >> 4);
    s0 = s1 = 2 * (a >> 28) + 1;
  }
};
template <> struct QPiece<QT_IQ3_S> {
  static constexpr int SC = 1;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, int& s0, int& s1) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t qh = b[66 + ib];
    const uint32_t sg = b[74 + 4 * ib + l];
    w0 = apply_signs4(qtab_dev::iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)], sg);
    w1 = apply_signs4(qtab_dev::iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)], sg >> 4);
    s0 = s1 = 1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf);
  }
};
template <> struct QPiece<QT_IQ4_XS> {
  static constexpr int SC = 1;
  static __device__ __forceinline__ void dec(const uint8_t* b, int ib, int l, uint32_t& w0, uint32_t& w1, int& s0, int& s1) {
    const uint32_t sh = ld16(b + 2);
    s0 = s1 = (((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4)) - 32;
    const uint32_t* tab = (const uint32_t*)qtab_dev::kvalues_iq4nl;
    const uint32_t t[4] = {tab[0], tab[1], tab[2], tab[3]};
    const uint8_t* qs = b + 8 + 16 * ib + 8 * (l & 1);
    const int shift = 4 * (l >> 1);  // words 0..3: low nibbles of bytes 0..15, words 4..7: high nibbles
    w0 = lut16((ld32a4(qs) >> shift) & 0x0F0F0F0F, t);
    w1 = lut16((ld32a4(qs + 4) >> shift) & 0x0F0F0F0F, t);
  }
};

constexpr int TQ_WARPS = 4;  // the warps of a CTA split the K range of one 16-row tile
// Each warp stages the raw bytes of its 16 rows x one super-block in shared memory (coalesced 4-byte
// cp.async, double-buffered: the next super-block streams in while this one is decoded), since the
// fragment layout would otherwise read a few bytes from each of 16 rows per load.
// RP: register-prefetched single stage (the next super-block's raw words wait in registers while the
// current one is decoded from shared memory; half the shared memory of the cp.async double buffer,
// so about twice the warps per SM)
// MB: min CTAs per SM for the register allocation (a cap that raises occupancy where it pays)
// X32: activations with one scale per 32 (quantize_q8, as the dp4a kernels): the scale is applied per
// sub-block in float; else per 256 (quantize_q8_256), one float multiply per super-block
template <int T, int B, bool RP, int MB, bool X32 = false>
__global__ void __launch_bounds__(TQ_WARPS * 32, MB)
    k_mmv_tq(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
             const float* __restrict__ xd, float* __restrict__ y, int ldy, int acc) {
  using Q = QInt<T>;
  using P = QPiece<T>;
  constexpr int BB = QDec<T>::BB;
  constexpr int BW = (BB + 3 + 3) / 4 + 1;  // 4-byte words per staged row (aligned-down start)
  constexpr int NS = RP ? 1 : 2, PW = (16 * BW + 31) / 32;
  // double-buffer rows are copied in 16-byte chunks from the 16-byte boundary below the super-block
  // (cp.async.cg, L2 only): BC chunks per row; the register-prefetch stage keeps 4-byte words
  // row stride in words: an odd number of 16-byte chunks, so the 16 rows fall on different banks
  constexpr int BC = (BB + 15 + 15) / 16, RW = RP ? BW : 4 * (BC | 1);
  __shared__ __align__(16) uint32_t stage[TQ_WARPS][NS][16][RW];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const int row0 = blockIdx.x * 16;
  const int nb = K / 256;
  const int j0 = warp * nb / TQ_WARPS, j1 = (warp + 1) * nb / TQ_WARPS;
  const bool tv = g < B;
  const int8_t* xp = xq + (size_t)(tv ? g : 0) * K + 8 * tq;
  const int t0 = 2 * tq, t1 = 2 * tq + 1;  // the C columns (tokens) of this lane
  // the staged copy of row r starts at the 4-byte boundary below its super-block: offset (r*rb + j*BB) & 3
  auto issue = [&](int j, int st) {
    for (int i = lane; i < 16 * BC; i += 32) {
      const int r = i / BC, ch = i % BC;
      const uint8_t* src = W + (size_t)min(row0 + r, N - 1) * rb + (size_t)j * BB;
      const uint8_t* al = (const uint8_t*)((uintptr_t)src & ~(uintptr_t)15);
      cp_async16(&stage[warp][st][r][4 * ch], al + 16 * ch);
    }
    cp_async_commit();
  };
  uint32_t pre[RP ? PW : 1];
  auto gload = [&](int j) {
#pragma unroll
    for (int k = 0; k < PW; k++) {
      const int i = lane + 32 * k;
      if (i < 16 * BW) {
        const int r = i / BW, wd = i % BW;
        const uint8_t* src = W + (size_t)min(row0 + r, N - 1) * rb + (size_t)j * BB;
        pre[RP ? k : 0] = __ldg((const uint32_t*)((uintptr_t)src & ~(uintptr_t)3) + wd);
      }
    }
  };
  auto sstore = [&]() {
#pragma unroll
    for (int k = 0; k < PW; k++) {
      const int i = lane + 32 * k;
      if (i < 16 * BW) stage[warp][0][i / BW][i % BW] = pre[RP ? k : 0];
    }
  };
  float acc4[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (RP) {
    if (j0 < j1) { gload(j0); sstore(); }
  } else {
    if (j0 < j1) issue(j0, 0);
  }
  for (int j = j0; j < j1; j++) {
    int st = 0;
    if constexpr (RP) {
      __syncwarp();
      if (j + 1 < j1) gload(j + 1);  // in flight during this super-block's decode
    } else {
      st = (j - j0) & 1;
      if (j + 1 < j1) issue(j + 1, st ^ 1); else cp_async_commit();
      cp_async_wait<1>();
      __syncwarp();
    }
    constexpr uintptr_t OM = RP ? 3 : 15;
    const uintptr_t oa = (uintptr_t)(W + (size_t)min(row0 + g, N - 1) * rb + (size_t)j * BB) & OM;
    const uintptr_t ob = (uintptr_t)(W + (size_t)min(row0 + g + 8, N - 1) * rb + (size_t)j * BB) & OM;
    const uint8_t* ba = (const uint8_t*)stage[warp][st][g] + oa;
    const uint8_t* bbk = (const uint8_t*)stage[warp][st][g + 8] + ob;
    int ci[4] = {0, 0, 0, 0};
    float cf[4] = {0.f, 0.f, 0.f, 0.f};  // X32: per-sub-block scaled partials
#pragma unroll
    for (int ib = 0; ib < 8; ib++) {
      if constexpr (X32) ci[0] = ci[1] = ci[2] = ci[3] = 0;
      uint32_t a[4];
      int sa0, sa1, sb0, sb1;
      P::dec(ba, ib, tq, a[0], a[2], sa0, sa1);
      P::dec(bbk, ib, tq, a[1], a[3], sb0, sb1);
      uint32_t b0 = 0u, b1 = 0u;
      if (tv) {
        const uint2 v = __ldg((const uint2*)(xp + (size_t)j * 256 + ib * 32));
        b0 = v.x; b1 = v.y;
      }
      if constexpr (P::SC == 1) {
        int c[4] = {0, 0, 0, 0};
        mma_s8_k32(c, a, b0, b1);
        ci[0] += c[0] * sa0; ci[1] += c[1] * sa0; ci[2] += c[2] * sb0; ci[3] += c[3] * sb0;
      } else {  // two 16-weight halves with their own scales: lanes tq < 2 hold half 0
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const bool mine = (tq >> 1) == h;
          const uint32_t am[4] = {mine ? a[0] : 0u, mine ? a[1] : 0u, mine ? a[2] : 0u, mine ? a[3] : 0u};
          int c[4] = {0, 0, 0, 0};
          mma_s8_k32(c, am, b0, b1);
          const int s_a = h ? sa1 : sa0, s_b = h ? sb1 : sb0;
          ci[0] += c[0] * s_a; ci[1] += c[1] * s_a; ci[2] += c[2] * s_b; ci[3] += c[3] * s_b;
        }
      }
      if constexpr (X32) {
        const int nsb = K / 32;
        const float xs0 = t0 < B ? __ldg(xd + (size_t)t0 * nsb + j * 8 + ib) : 0.f;
        const float xs1 = t1 < B ? __ldg(xd + (size_t)t1 * nsb + j * 8 + ib) : 0.f;
        cf[0] += (float)ci[0] * xs0; cf[1] += (float)ci[1] * xs1;
        cf[2] += (float)ci[2] * xs0; cf[3] += (float)ci[3] * xs1;
      }
    }
    const float fa = Q::fac(ba), fb = Q::fac(bbk);
    if constexpr (X32) {
      acc4[0] += cf[0] * fa; acc4[1] += cf[1] * fa; acc4[2] += cf[2] * fb; acc4[3] += cf[3] * fb;
    } else {
    const float x0 = t0 < B ? __ldg(xd + (size_t)t0 * nb + j) : 0.f;
    const float x1 = t1 < B ? __ldg(xd + (size_t)t1 * nb + j) : 0.f;
    acc4[0] += (float)ci[0] * fa * x0;
    acc4[1] += (float)ci[1] * fa * x1;
    acc4[2] += (float)ci[2] * fb * x0;
    acc4[3] += (float)ci[3] * fb * x1;
    }
    __syncwarp();  // the stage is refilled by the next iteration's issue
    if constexpr (RP) {
      if (j + 1 < j1) sstore();
    }
  }
  __shared__ float red[TQ_WARPS][4][32];
#pragma unroll
  for (int i = 0; i < 4; i++) red[warp][i][lane] = acc4[i];
  __syncthreads();
  if (warp != 0) return;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    v[i] = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < TQ_WARPS; w2++) v[i] += red[w2][i][lane];
  }
  const int ra = row0 + g, rbw = row0 + g + 8;
  auto put = [&](int row, int tok, float val) {
    if (row < N && tok < B) {
      float* o = y + (size_t)tok * ldy + row;
      *o = acc ? *o + val : val;
    }
  };
  put(ra, t0, v[0]); put(ra, t1, v[1]); put(rbw, t0, v[2]); put(rbw, t1, v[3]);
}

bool matvec_tq_supported(uint32_t type) {
#ifdef __HIP_PLATFORM_AMD__
  (void)type;
  return false;  // HIP v1: tensor-core matvec disabled; qk::matvec takes over
#else
  return type == QT_IQ2_XXS || type == QT_IQ2_XS || type == QT_IQ2_S || type == QT_IQ3_XXS || type == QT_IQ3_S ||
         type == QT_IQ4_XS;
#endif
}
template <int T>
static void launch_tq(const void* W, size_t rb, int K, int N, const int8_t* xq, const float* xd, int B, float* y,
                      int ldy, bool acc, cudaStream_t s, bool x32) {
  const uint8_t* w = (const uint8_t*)W;
  const int grid = cdiv(N, 16);
  // per type and shape (qe_bench_mv ... types, RTX 3080): the uncapped 16-byte double buffer, except
  // the long rows (K >= 8192, down projections) of IQ2_XS / IQ2_S / IQ4_XS, which prefer registers
  // capped for 8 CTAs / SM.  QE_TQ_MODE=0|1|2 forces double buffer / register prefetch / capped.
  static const int force = getenv("QE_TQ_MODE") ? atoi(getenv("QE_TQ_MODE")) : -1;
  const bool capped = K >= 8192 && (T == QT_IQ2_XS || T == QT_IQ2_S || T == QT_IQ4_XS);
  const int mode = force >= 0 ? force : capped ? 2 : 0;
#define LAUNCH1(RP, MB, X)                                                                                     \
  switch (B) {                                                                                                 \
    case 1: k_mmv_tq<T, 1, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    case 2: k_mmv_tq<T, 2, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    case 3: k_mmv_tq<T, 3, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    case 4: k_mmv_tq<T, 4, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    case 5: k_mmv_tq<T, 5, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    case 6: k_mmv_tq<T, 6, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    case 7: k_mmv_tq<T, 7, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    case 8: k_mmv_tq<T, 8, RP, MB, X><<<grid, TQ_WARPS * 32, 0, s>>>(w, rb, K, N, xq, xd, y, ldy, acc); break; \
    default: QE_FATAL("matvec_tq: batch %d", B);                                                               \
  }
#define LAUNCH(RP, MB) if (x32) { LAUNCH1(RP, MB, true) } else { LAUNCH1(RP, MB, false) }
  if (mode == 1) { LAUNCH(true, 1) }
  else if (mode == 2) { LAUNCH(false, 8) }
  else { LAUNCH(false, 1) }
#undef LAUNCH
#undef LAUNCH1
}
int matvec_tq_min_batch(uint32_t type, int K) {
  // smallest batch where the tensor-core kernel beats the dp4a kernels (same benchmark)
  const bool down = K >= 8192, wide = K >= 6144;
  switch (type) {
    case QT_IQ3_S: return 2;
    case QT_IQ3_XXS: return 4;
    case QT_IQ2_XS: return down ? 2 : 4;
    case QT_IQ2_XXS: return 4;
    case QT_IQ2_S: return wide ? 4 : 6;
    case QT_IQ4_XS: return wide ? 4 : 7;
  }
  return 99;
}
void matvec_tq(uint32_t type, const void* W, size_t rb, int K, int N, const int8_t* xq256, const float* xd256, int B,
               float* y, int ldy, bool acc, cudaStream_t s, int gs) {
  const bool x32 = gs == 32;
  QE_ASSERT(K % 256 == 0);
  switch (type) {
    case QT_IQ2_XXS: launch_tq<QT_IQ2_XXS>(W, rb, K, N, xq256, xd256, B, y, ldy, acc, s, x32); break;
    case QT_IQ2_XS: launch_tq<QT_IQ2_XS>(W, rb, K, N, xq256, xd256, B, y, ldy, acc, s, x32); break;
    case QT_IQ2_S: launch_tq<QT_IQ2_S>(W, rb, K, N, xq256, xd256, B, y, ldy, acc, s, x32); break;
    case QT_IQ3_XXS: launch_tq<QT_IQ3_XXS>(W, rb, K, N, xq256, xd256, B, y, ldy, acc, s, x32); break;
    case QT_IQ3_S: launch_tq<QT_IQ3_S>(W, rb, K, N, xq256, xd256, B, y, ldy, acc, s, x32); break;
    case QT_IQ4_XS: launch_tq<QT_IQ4_XS>(W, rb, K, N, xq256, xd256, B, y, ldy, acc, s, x32); break;
    default: QE_FATAL("matvec_tq: unsupported type %u", type);
  }
}

#ifdef __HIP_PLATFORM_AMD__
// WMMA int8 GEMM for HIP prefill (MMQ), IQ3_S only. Block: 64 output rows x
// 64 tokens, 8 warps owning two 16x16 WMMA tiles each. K is staged in
// 128-wide chunks: weights are decoded once per block into shared memory
// (QDec, exact float scales per 32-group -- the same numbers the dp4a decode
// path produces), activations are staged alongside, so no f16 weight
// expansion ever happens. Fragment layout per hip-spike/wmma.hip.cpp.
constexpr int MQH_BM = 64, MQH_BN = 64, MQH_LD = 144, MQH_GC = 4;  // groups per K chunk

__global__ void __launch_bounds__(256)
    k_mmq_wmma_iq3s(const uint8_t* __restrict__ W, size_t rb, int K, int N, const int8_t* __restrict__ xq,
                    const float* __restrict__ xd, int Tt, float* __restrict__ y, int ldy, int accum, int gs) {
  using D = QDec<QT_IQ3_S>;
  __shared__ __align__(16) int8_t sW[MQH_BM][MQH_LD];
  __shared__ __align__(16) int8_t sX[MQH_BN][MQH_LD];
  __shared__ float sws[MQH_BM][MQH_GC];
  __shared__ float sxs[MQH_BN][MQH_GC];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n0 = blockIdx.x * MQH_BM, t0 = blockIdx.y * MQH_BN;
  const int nsb = K / 32;                          // 32-groups in K
  const int xstride = gs == 32 ? nsb : nsb / 8;    // xd floats per token
  const int tn = (warp & 3) * 16;                  // this warp's token tile
  const int nn0 = (warp >> 2) * 16;                // first of its two row tiles (+32)
  const int nloc = lane % 16, hi = lane / 16;
  float acc[2][8] = {};
  for (int kb = 0; kb < nsb; kb += MQH_GC) {
    {  // stage weights: thread tid decodes row (tid >> 2), group (kb + (tid & 3))
      const int row = tid >> 2, gl = tid & 3;
      const int r = min(n0 + row, N - 1);
      W32 w;
      D::dec(W + (size_t)r * rb + (size_t)((kb + gl) >> 3) * D::BB, (kb + gl) & 7, w);
      int* dst = (int*)&sW[row][gl * 32];
#pragma unroll
      for (int i = 0; i < 8; i++) dst[i] = w.w[i];
      sws[row][gl] = w.sc[0];
    }
#pragma unroll
    for (int u = 0; u < 2; u++) {  // stage activations: 512 uint4 over 256 threads
      const int idx = tid + u * 256;
      const int row = idx >> 3, ch = idx & 7;
      const int t = t0 + row;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (t < Tt) v = *(const uint4*)(xq + (size_t)t * K + kb * 32 + ch * 16);
      *(uint4*)&sX[row][ch * 16] = v;
    }
    {  // stage activation scales: 64 tokens x 4 groups, one float per thread
      const int row = tid >> 2, gl = tid & 3;
      const int t = t0 + row;
      const int g = kb + gl;
      sxs[row][gl] = t < Tt ? xd[(size_t)t * xstride + (gs == 32 ? g : (g >> 3))] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int gl = 0; gl < MQH_GC; gl++) {
#pragma unroll
      for (int b = 0; b < 2; b++) {
        const int nn = nn0 + b * 32;
        wmma_i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const uint4 av = *(const uint4*)&sX[tn + nloc][gl * 32 + h * 16];
          wmma_i32x4 a4;
          a4[0] = (int)av.x; a4[1] = (int)av.y; a4[2] = (int)av.z; a4[3] = (int)av.w;
          const uint4 bv = *(const uint4*)&sW[nn + nloc][gl * 32 + h * 16];
          wmma_i32x4 b4;
          b4[0] = (int)bv.x; b4[1] = (int)bv.y; b4[2] = (int)bv.z; b4[3] = (int)bv.w;
          c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a4, true, b4, c, false);
        }
        const float wsc = sws[nn + nloc][gl];
#pragma unroll
        for (int i = 0; i < 8; i++) acc[b][i] += (float)c[i] * wsc * sxs[tn + 2 * i + hi][gl];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int b = 0; b < 2; b++) {
    const int n = n0 + nn0 + b * 32 + nloc;
    if (n >= N) continue;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int t = t0 + tn + 2 * i + hi;
      if (t >= Tt) continue;
      float* o = y + (size_t)t * ldy + n;
      *o = accum ? *o + acc[b][i] : acc[b][i];
    }
  }
}
#endif  // __HIP_PLATFORM_AMD__

bool mmq_supported(uint32_t type) {
#ifdef __HIP_PLATFORM_AMD__
  return type == QT_IQ3_S;  // WMMA int8 GEMM (k_mmq_wmma_iq3s); other types keep dequant + hipBLAS
#else
  switch (type) {
    case QT_Q2_K: case QT_Q3_K: case QT_Q4_K: case QT_Q5_K: case QT_Q6_K: case QT_IQ2_XXS: case QT_IQ2_XS: case QT_IQ2_S:
    case QT_IQ3_XXS: case QT_IQ3_S: case QT_IQ1_S: case QT_IQ4_XS:
      return true;
    default:
      return false;
  }
#endif
}

void mmq(uint32_t type, const void* W, size_t rb, int K, int N, const int8_t* xq, const float* xd, int Tt, float* y,
         int ldy, bool acc, int gs, cudaStream_t s) {
#ifdef __HIP_PLATFORM_AMD__
  if (type == QT_IQ3_S) {
    QE_ASSERT(K % 128 == 0);  // K staging runs in 4-group (128-wide) chunks
    const dim3 grid(cdiv(N, MQH_BM), cdiv(Tt, MQH_BN));
    k_mmq_wmma_iq3s<<<grid, 256, 0, s>>>((const uint8_t*)W, rb, K, N, xq, xd, Tt, y, ldy, acc, gs);
    return;
  }
  QE_FATAL("mmq: unsupported type %u on HIP", type);
#endif
  switch (type) {
#define T(t) case t: if (gs == 32) launch_mmq_t<t, 32>(W, rb, K, N, xq, xd, Tt, y, ldy, acc, s); \
                     else launch_mmq_t<t, 256>(W, rb, K, N, xq, xd, Tt, y, ldy, acc, s); break;
    T(QT_Q2_K) T(QT_Q3_K) T(QT_Q4_K) T(QT_Q5_K) T(QT_Q6_K) T(QT_IQ2_XXS) T(QT_IQ2_XS) T(QT_IQ2_S) T(QT_IQ3_XXS) T(QT_IQ3_S)
    T(QT_IQ1_S) T(QT_IQ4_XS)
#undef T
    default: QE_FATAL("mmq: unsupported type %u", type);
  }
}

// x[T][K] -> int8 with one scale per 256 (prefill GEMM activations)
__global__ void k_quantize_q8_256(const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xd,
                                  int nblk) {
  // one warp per 256-block: 8 values per lane
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= nblk) return;
  const float* p = x + (size_t)wid * 256 + 8 * lane;
  const float4 a = *(const float4*)p, b = *(const float4*)(p + 4);
  float m = fmaxf(fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))),
                  fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
  m = warp_max(m);
  const float d = m / 127.f, id = d > 0 ? 1.f / d : 0.f;
  const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) lo |= (uint32_t)(uint8_t)(int8_t)__float2int_rn(v[i] * id) << (8 * i);
#pragma unroll
  for (int i = 0; i < 4; i++) hi |= (uint32_t)(uint8_t)(int8_t)__float2int_rn(v[4 + i] * id) << (8 * i);
  *(uint2*)(xq + (size_t)wid * 256 + 8 * lane) = make_uint2(lo, hi);
  if (lane == 0) xd[wid] = d;
}
void quantize_q8_256(const float* x, int8_t* xq, float* xd, int K, int T, cudaStream_t s) {
  const int nblk = K / 256 * T;
  if (nblk == 0) return;
  k_quantize_q8_256<<<cdiv(nblk, 8), 256, 0, s>>>(x, xq, xd, nblk);
}

// =============================================================================================
// Dequantization to f16 (for tensor-core GEMM in prefill)
// =============================================================================================
template <int T>
__global__ void k_dequant_f16(const uint8_t* __restrict__ W, size_t rb, int K, int N,
                              __half* __restrict__ dst) {
  const int nsb = K / 32;
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)N * nsb) return;
  const int row = (int)(idx / nsb), g = (int)(idx % nsb);
  float v[32];
  dequant32<T>(W + (size_t)row * rb + (size_t)(g >> 3) * QDec<T>::BB, g & 7, v);
  __half2 h[16];
#pragma unroll
  for (int i = 0; i < 16; i++) h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
  uint4* o = (uint4*)(dst + (size_t)row * K + (size_t)g * 32);
  const uint4* hs = (const uint4*)h;
#pragma unroll
  for (int i = 0; i < 4; i++) o[i] = hs[i];
}
template <int WT>
__global__ void k_dequant_f16_float(const uint8_t* __restrict__ W, size_t rb, int K, int N,
                                    __half* __restrict__ dst) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)N * K) return;
  const int row = (int)(idx / K), k = (int)(idx % K);
  const uint8_t* wr = W + (size_t)row * rb;
  float w;
  if (WT == QT_F32) w = ((const float*)wr)[k];
  else if (WT == QT_F16) w = __half2float(((const __half*)wr)[k]);
  else w = __uint_as_float((uint32_t)((const uint16_t*)wr)[k] << 16);
  dst[idx] = __float2half(w);
}

void dequant_f16(uint32_t type, const void* W, size_t rb, int K, int N, __half* dst,
                 cudaStream_t s) {
  const uint8_t* w = (const uint8_t*)W;
  const int64_t n32 = (int64_t)N * (K / 32);
  switch (type) {
    case QT_F32: k_dequant_f16_float<QT_F32><<<cdiv((int64_t)N * K, 256), 256, 0, s>>>(w, rb, K, N, dst); break;
    case QT_F16: k_dequant_f16_float<QT_F16><<<cdiv((int64_t)N * K, 256), 256, 0, s>>>(w, rb, K, N, dst); break;
    case QT_BF16: k_dequant_f16_float<QT_BF16><<<cdiv((int64_t)N * K, 256), 256, 0, s>>>(w, rb, K, N, dst); break;
#define T(t) case t: k_dequant_f16<t><<<cdiv(n32, 128), 128, 0, s>>>(w, rb, K, N, dst); break;
    T(QT_Q8_0) T(QT_Q2_K) T(QT_Q3_K) T(QT_Q4_K) T(QT_Q5_K) T(QT_Q6_K) T(QT_IQ2_XXS) T(QT_IQ2_XS) T(QT_IQ2_S) T(QT_IQ3_XXS)
    T(QT_IQ3_S) T(QT_IQ1_S) T(QT_IQ1_M) T(QT_IQ4_XS)
#undef T
    default: QE_FATAL("dequant_f16: unsupported type %u", type);
  }
}

__global__ void k_h2f_accum(const __half* __restrict__ src, int lds, int rows, int cols, float* dst, int ldd, int acc,
                            int* ovf) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)rows * cols) return;
  const int t = (int)(i / cols), c = (int)(i % cols);
  const float v = __half2float(src[(size_t)t * lds + c]);
  if (!isfinite(v)) *ovf = 1;
  float* d = dst + (size_t)t * ldd + c;
  *d = acc ? *d + v : v;
}
void h2f_accum(const __half* src, int lds, int rows, int cols, float* dst, int ldd, bool acc, int* ovf, cudaStream_t s) {
  k_h2f_accum<<<cdiv((int64_t)rows * cols, 256), 256, 0, s>>>(src, lds, rows, cols, dst, ldd, acc, ovf);
}

__global__ void k_f32_to_f16(const float* __restrict__ x, __half* __restrict__ y, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half(x[i]);
}
void f32_to_f16(const float* x, __half* y, int64_t n, cudaStream_t s) {
  k_f32_to_f16<<<cdiv(n, 256), 256, 0, s>>>(x, y, n);
}

// =============================================================================================
// Norms, RoPE, elementwise
// =============================================================================================
static __device__ float block_sum(float v, float* sh) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  v = (l < nw) ? sh[l] : 0.f;
  v = warp_sum(v);
  return v;
}

// Fused decode-path layer input: x += r (deferred residual, optional), y = rmsnorm(x) * w, and the
// int8-per-32 activation quantization of y for the matvec kernels (xq/xd optional).
__global__ void __launch_bounds__(256) k_add_rmsnorm_q8(float* __restrict__ x, const float* __restrict__ r,
                                                       const float* __restrict__ w, float* __restrict__ y, int n,
                                                       float eps, int8_t* __restrict__ xq, float* __restrict__ xd) {
  __shared__ float sh[32];
  float* xr = x + (size_t)blockIdx.x * n;
  const float* rr = r ? r + (size_t)blockIdx.x * n : nullptr;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += 256) {
    float v = xr[i];
    if (rr) { v += rr[i]; xr[i] = v; }
    ss += v * v;
  }
  ss = block_sum(ss, sh);
  const float inv = rsqrtf(ss / n + eps);
  float* yr = y + (size_t)blockIdx.x * n;
  const int lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < n; i += 256) {  // a warp covers 32 contiguous elements per iteration
    const float v = xr[i] * inv * w[i];
    yr[i] = v;
    if (xq) {
      const float amax = warp_max(fabsf(v));
      const float d = amax / 127.f;
      const float id = d > 0.f ? 1.f / d : 0.f;
      const size_t g = (size_t)blockIdx.x * n + i;
      xq[g] = (int8_t)__float2int_rn(v * id);
      if (lane == 0) xd[g / 32] = d;
    }
  }
}
void add_rmsnorm_q8(float* x, const float* r, const float* w, float* y, int n, int rows, float eps, int8_t* xq,
                    float* xd, cudaStream_t s) {
  k_add_rmsnorm_q8<<<rows, 256, 0, s>>>(x, r, w, y, n, eps, xq, xd);
}
// h = silu(g) * u, plus its int8-per-32 quantization
__global__ void k_silu_mul_q8(const float* __restrict__ g, const float* __restrict__ u, float* h, int64_t n,
                              int8_t* __restrict__ xq, float* __restrict__ xd) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const float a = i < n ? g[i] : 0.f;
  const float v = i < n ? a / (1.f + __expf(-a)) * u[i] : 0.f;
  const float amax = warp_max(fabsf(v));
  if (i >= n) return;
  h[i] = v;
  const float d = amax / 127.f;
  const float id = d > 0.f ? 1.f / d : 0.f;
  xq[i] = (int8_t)__float2int_rn(v * id);
  if ((threadIdx.x & 31) == 0) xd[i / 32] = d;
}
void silu_mul_q8(const float* g, const float* u, float* h, int64_t n, int8_t* xq, float* xd, cudaStream_t s) {
  k_silu_mul_q8<<<cdiv(n, 256), 256, 0, s>>>(g, u, h, n, xq, xd);
}

__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ y, int n, float eps, __half* __restrict__ y16) {
  __shared__ float sh[32];
  const float* xr = x + (size_t)blockIdx.x * n;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
  ss = block_sum(ss, sh);
  const float r = rsqrtf(ss / n + eps);
  float* yr = y + (size_t)blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = xr[i] * r * w[i];
    yr[i] = v;
    if (y16) y16[(size_t)blockIdx.x * n + i] = __float2half(v);
  }
}
void rmsnorm(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t s,
             __half* y16) {
  k_rmsnorm<<<rows, 512, 0, s>>>(x, w, y, n, eps, y16);
}

// one warp per (token, head)
__global__ void k_rmsnorm_heads(float* x, const float* __restrict__ w, int hd, int nh, int hstride,
                                int tokens, int tstride, float eps) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= nh * tokens) return;
  float* p = x + (size_t)(wid / nh) * tstride + (size_t)(wid % nh) * hstride;
  float ss = 0.f;
  for (int i = lane; i < hd; i += 32) ss += p[i] * p[i];
  ss = warp_sum(ss);
  const float r = rsqrtf(ss / hd + eps);
  for (int i = lane; i < hd; i += 32) p[i] = p[i] * r * w[i];
}
void rmsnorm_heads(float* x, const float* w, int hd, int nh, int hstride, int tokens, int tstride,
                   float eps, cudaStream_t s) {
  k_rmsnorm_heads<<<cdiv(nh * tokens, 8), 256, 0, s>>>(x, w, hd, nh, hstride, tokens, tstride, eps);
}

// RoPE frequency table (per rotated pair) and cos / sin scale; set by set_rope_table (YaRN models).
// A launch uses it when the caller passes base < 0 (log2_base < 0 here), else the plain NeoX
// frequencies base^(-2i/rot): several models in one process (target + draft) each pick their own.
__constant__ double c_rope_freq[64];
__constant__ float c_rope_mscale = 1.f;
__constant__ int c_rope_tab = 0;
void set_rope_table(const double* freq, int half, float mscale) {
  QE_ASSERT(half <= 64);
  const int on = 1;
  CUDA_CHECK(cudaMemcpyToSymbol(c_rope_freq, freq, half * sizeof(double)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_rope_mscale, &mscale, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_rope_tab, &on, sizeof(int)));
}
static __device__ __forceinline__ double rope_freq(int i, int rot, float log2_base) {
  return (c_rope_tab && log2_base < 0) ? c_rope_freq[i] : exp2(-(double)log2_base * (2.0 * i / rot));
}

__global__ void k_rope_neox(float* x, const int* __restrict__ pos, int hd, int nh, int hstride,
                            int tokens, int tstride, int rot, float log2_base) {
  const int half = rot / 2;
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)tokens * nh * half) return;
  const int i = idx % half;
  const int h = (idx / half) % nh;
  const int t = idx / ((int64_t)half * nh);
  float* p = x + (size_t)t * tstride + (size_t)h * hstride;
  // theta_i = pos * base^(-2i/rot)
  // double precision: at pos ~1e5 the angle is ~1e5 rad and fast-math sincosf would be noise
  const double freq = rope_freq(i, rot, log2_base);
  double sd, cd;
  sincos((double)pos[t] * freq, &sd, &cd);
  const float ms = log2_base < 0 ? c_rope_mscale : 1.f;
  const float sn = (float)sd * ms, cs = (float)cd * ms;
  const float a = p[i], b = p[i + half];
  p[i] = a * cs - b * sn;
  p[i + half] = a * sn + b * cs;
  (void)hd;
}
void rope_neox(float* x, const int* pos, int hd, int nh, int hstride, int tokens, int tstride,
               int rot, float base, cudaStream_t s) {
  const int64_t n = (int64_t)tokens * nh * (rot / 2);
  k_rope_neox<<<cdiv(n, 256), 256, 0, s>>>(x, pos, hd, nh, hstride, tokens, tstride, rot, base < 0 ? -1.f : log2f(base));
}

// small host->device transfers done by the SMs (zero-copy reads of pinned host memory) so they do
// not queue behind the big weight-streaming copies on the copy engine
__global__ void k_copy_f32(float* __restrict__ y, const float* __restrict__ x, int64_t n) {
  const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) *(float4*)(y + i) = *(const float4*)(x + i);
  else for (int64_t j = i; j < n; j++) y[j] = x[j];
}
void copy_f32_zc(float* dst, const float* host_src, int64_t n, cudaStream_t s) {
  if (n > 0) k_copy_f32<<<(int)((n / 4 + 255) / 256 + 1), 256, 0, s>>>(dst, host_src, n);
}
__global__ void k_iota(int* p, int p0, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = p0 + i;
}
void iota_i32(int* dst, int p0, int n, cudaStream_t s) {
  if (n > 0) k_iota<<<(n + 255) / 256, 256, 0, s>>>(dst, p0, n);
}

__global__ void k_add(float* y, const float* x, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
}
void add_inplace(float* y, const float* x, int64_t n, cudaStream_t s) {
  k_add<<<cdiv(n, 256), 256, 0, s>>>(y, x, n);
}
__global__ void k_silu_mul(const float* g, const float* u, float* h, int64_t n, __half* h16) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = g[i];
  const float v = a / (1.f + __expf(-a)) * u[i];
  h[i] = v;
  if (h16) h16[i] = __float2half(v);
}
void silu_mul(const float* g, const float* u, float* h, int64_t n, cudaStream_t s, __half* h16) {
  k_silu_mul<<<cdiv(n, 256), 256, 0, s>>>(g, u, h, n, h16);
}
__global__ void k_gate_sigmoid(float* o, const float* qg, int hd, int nh, int64_t n, __half* o16, int8_t* xq,
                               float* xd) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int d = i % hd;
  const int h = (i / hd) % nh;
  const int64_t t = i / ((int64_t)hd * nh);
  const float g = qg[(size_t)t * nh * hd * 2 + (size_t)h * hd * 2 + hd + d];
  const float v = o[i] / (1.f + __expf(-g));
  o[i] = v;
  if (o16) o16[i] = __float2half(v);
  if (xq) {  // n % 32 == 0 and blocks of 256: every warp is one full 32-group
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    xq[i] = (int8_t)__float2int_rn(d > 0.f ? v / d : 0.f);
    if ((threadIdx.x & 31) == 0) xd[i / 32] = d;
  }
}
void gate_sigmoid(float* o, const float* qg, int hd, int nh, int tokens, cudaStream_t s, __half* o16, int8_t* xq,
                  float* xd) {
  const int64_t n = (int64_t)tokens * nh * hd;
  k_gate_sigmoid<<<cdiv(n, 256), 256, 0, s>>>(o, qg, hd, nh, n, o16, xq, xd);
}

// Per-head RMS norm of q and k followed by NeoX RoPE on the first `rot` dims, one block per token:
// the rotation angles (double precision) are computed once per token instead of once per head.
__global__ void k_qk_norm_rope(float* q, const float* __restrict__ qw, int qhs, int qts, int nhq, float* k,
                               const float* __restrict__ kw, int khs, int kts, int nhk, const int* __restrict__ pos,
                               int hd, int rot, float log2_base, float eps) {
  __shared__ float cs[64], sn[64];
  const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, half = rot / 2;
  if (threadIdx.x < half) {
    const double freq = rope_freq(threadIdx.x, rot, log2_base);
    double sd, cd;
    sincos((double)pos[t] * freq, &sd, &cd);
    const float ms = log2_base < 0 ? c_rope_mscale : 1.f;
    cs[threadIdx.x] = (float)cd * ms;
    sn[threadIdx.x] = (float)sd * ms;
  }
  __syncthreads();
  if (warp >= nhq + nhk) return;
  float* p = warp < nhq ? q + (size_t)t * qts + (size_t)warp * qhs : k + (size_t)t * kts + (size_t)(warp - nhq) * khs;
  const float* w = warp < nhq ? qw : kw;
  float ss = 0.f;
  for (int i = lane; i < hd; i += 32) ss += p[i] * p[i];
  ss = warp_sum(ss);
  const float r = rsqrtf(ss / hd + eps);
  for (int i = lane; i < hd; i += 32) p[i] = p[i] * r * w[i];
  __syncwarp();
  for (int i = lane; i < half; i += 32) {
    const float a = p[i], b = p[i + half];
    p[i] = a * cs[i] - b * sn[i];
    p[i + half] = a * sn[i] + b * cs[i];
  }
}
bool qk_norm_rope(float* q, const float* qw, int qhs, int qts, int nhq, float* k, const float* kw, int khs, int kts,
                  int nhk, const int* pos, int hd, int tokens, int rot, float base, float eps, cudaStream_t s) {
  if ((nhq + nhk) * 32 > 1024 || rot > 128) return false;
  k_qk_norm_rope<<<tokens, (nhq + nhk) * 32, 0, s>>>(q, qw, qhs, qts, nhq, k, kw, khs, kts, nhk, pos, hd, rot, base < 0 ? -1.f : log2f(base), eps);
  return true;
}

// =============================================================================================
// KV cache.  Per (position, kv-head) record of hd=256 values:
//   F16: 256 x f16 (512 B)
//   Q8 : 8 x f16 scale | 256 x int8        (272 B)
//   Q4 : 8 x f16 scale | 128 B nibbles      (144 B)   value = scale * (nibble - 8)
//   Q3 : 8 x f16 rms   | 96 B (3 bit/value)  (112 B)   value = rms * L[q], L = Lloyd-Max levels of N(0,1)
// For Q8/Q4 the vectors are rotated by a normalized Walsh-Hadamard transform before quantization
// (Q is rotated identically at attention time, and the output rotated back), which spreads
// outliers and makes low-bit KV much more accurate.  Each lane of a warp owns 8 consecutive dims.
// =============================================================================================
size_t kv_bytes_per_token_head(int kvtype, int hd) {
  QE_ASSERT(hd == 256);
  switch (kvtype) {
    case KV_F16: return 512;
    case KV_Q8: return 272;
    case KV_Q4: return 144;
    case KV_Q3: return 112;
  }
  QE_FATAL("bad kv type");
}

// in-warp normalized WHT of 256 values, 8 per lane (v[0..7] = dims 8*lane .. 8*lane+7)
static __device__ __forceinline__ void wht256(float v[8]) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if ((i & h) == 0) {
        const float a = v[i], b = v[i + h];
        v[i] = a + b;
        v[i + h] = a - b;
      }
    }
  }
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
      v[i] = (lane & m) ? (o - v[i]) : (v[i] + o);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] *= 0.0625f;  // 1/sqrt(256)
}

template <int KT>
static __device__ __forceinline__ void kv_write8(uint8_t* rec, const float v[8]) {
  const int lane = threadIdx.x & 31;
  if (KT == KV_F16) {
    __half2 h[4];
#pragma unroll
    for (int i = 0; i < 4; i++) h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
    *(uint4*)(rec + 16 * lane) = *(uint4*)h;
  } else if (KT == KV_Q3) {
    // 8-level Lloyd-Max quantizer for a unit Gaussian, scaled by the group RMS (values are ~Gaussian
    // after the Hadamard rotation)
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += v[i] * v[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rms = sqrtf(ss * (1.f / 32.f));
    const float ir = rms > 0 ? 1.f / rms : 0.f;
    uint32_t pk = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float t = v[i] * ir, a = fabsf(t);
      const int k = (a > 0.5006f) + (a > 1.0500f) + (a > 1.7480f);
      const int q = t < 0.f ? 3 - k : 4 + k;
      pk |= (uint32_t)q << (3 * i);
    }
    uint8_t* d = rec + 16 + 3 * lane;
    d[0] = (uint8_t)pk; d[1] = (uint8_t)(pk >> 8); d[2] = (uint8_t)(pk >> 16);
    if ((lane & 3) == 0) ((__half*)rec)[lane >> 2] = __float2half(rms);
  } else {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    // 4 lanes share a group of 32
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (KT == KV_Q8) {
      const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) lo |= (uint32_t)(uint8_t)(int8_t)__float2int_rn(v[i] * id) << (8 * i);
#pragma unroll
      for (int i = 0; i < 4; i++) hi |= (uint32_t)(uint8_t)(int8_t)__float2int_rn(v[4 + i] * id) << (8 * i);
      *(uint2*)(rec + 16 + 8 * lane) = make_uint2(lo, hi);
      if ((lane & 3) == 0) ((__half*)rec)[lane >> 2] = __float2half(d);
    } else {
      // symmetric 4-bit: q in [-8, 7], scale = amax / 7.5 with nearest rounding of v/d - 0.5? use
      // q = clamp(round(v/d), -8, 7) with d = amax/7.
      const float d = amax / 7.f, id = d > 0 ? 1.f / d : 0.f;
      uint32_t pk = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int q = __float2int_rn(v[i] * id);
        q = max(-8, min(7, q)) + 8;
        pk |= (uint32_t)q << (4 * i);
      }
      *(uint32_t*)(rec + 16 + 4 * lane) = pk;
      if ((lane & 3) == 0) ((__half*)rec)[lane >> 2] = __float2half(d);
    }
  }
}
template <int KT>
static __device__ __forceinline__ void kv_read8(const uint8_t* rec, int lane, float v[8]) {
  if (KT == KV_F16) {
    const uint4 u = *(const uint4*)(rec + 16 * lane);
    const __half2* h = (const __half2*)&u;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float2 f = __half22float2(h[i]);
      v[2 * i] = f.x;
      v[2 * i + 1] = f.y;
    }
  } else if (KT == KV_Q8) {
    const float d = __half2float(((const __half*)rec)[lane >> 2]);
    const uint2 u = *(const uint2*)(rec + 16 + 8 * lane);
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = d * (float)(int8_t)(u.x >> (8 * i));
#pragma unroll
    for (int i = 0; i < 4; i++) v[4 + i] = d * (float)(int8_t)(u.y >> (8 * i));
  } else if (KT == KV_Q3) {
    const float r = __half2float(((const __half*)rec)[lane >> 2]);
    const uint8_t* p = rec + 16 + 3 * lane;
    const uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    constexpr float L[8] = {-2.1520f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1520f};
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = r * L[(u >> (3 * i)) & 7];
  } else {
    const float d = __half2float(((const __half*)rec)[lane >> 2]);
    const uint32_t u = *(const uint32_t*)(rec + 16 + 4 * lane);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = d * (float)((int)((u >> (4 * i)) & 15) - 8);
  }
}

static __device__ __forceinline__ uint32_t pack_h2_(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *(uint32_t*)&h;
}

// KV pool addressing: positions are grouped in blocks of KV_BLOCK; record (h, p) lives at
//   base + (p / KV_BLOCK) * bstride + (h * KV_BLOCK + p % KV_BLOCK) * recb
static __device__ __forceinline__ size_t kv_off(int h, int p, KVMap m, int recb) {
  const int lb = p >> KV_BLOCK_SHIFT;
  const int pb = m.btab ? m.btab[lb] : lb;  // logical -> physical block (paged KV)
  return (size_t)pb * (size_t)m.bs + ((size_t)h * KV_BLOCK + (p & (KV_BLOCK - 1))) * recb;
}

template <int KT>
static __device__ __forceinline__ float kv_read1(const uint8_t* rec, int d) {
  if (KT == KV_F16) return __half2float(((const __half*)rec)[d]);
  const float sc = __half2float(((const __half*)rec)[d >> 5]);
  if (KT == KV_Q8) return sc * (float)((const int8_t*)(rec + 16))[d];
  const uint32_t b = rec[16 + (d >> 1)];
  return sc * (float)((int)((b >> (4 * (d & 1))) & 15) - 8);
}

// 8 consecutive dims of a record as packed f16 (FA tile fill).  Q4 uses the magic-number int4->f16
// conversion: half(0x6400 | n) == 1024 + n, so v = (h - 1032) * d (exact subtract, one rounding).
template <int KT>
static __device__ __forceinline__ uint4 kv_read8h(const uint8_t* rec, int c) {
  if constexpr (KT == KV_Q4) {
    const uint32_t u = *(const uint32_t*)(rec + 16 + 4 * c);
    const __half d = ((const __half*)rec)[c >> 2];
    const __half2 d2 = __halves2half2(d, d);
    const __half2 c2 = __halves2half2(__float2half(1032.f), __float2half(1032.f));
    uint32_t q[4];
#pragma unroll
    for (int k = 0; k < 4; k++) q[k] = ((u >> (4 * k)) & 0x000F000Fu) | 0x64006400u;  // (n_k, n_{k+4})
    // reorder to (n0,n1),(n2,n3),(n4,n5),(n6,n7)
    uint32_t r[4];
    r[0] = __byte_perm(q[0], q[1], 0x5410);
    r[1] = __byte_perm(q[2], q[3], 0x5410);
    r[2] = __byte_perm(q[0], q[1], 0x7632);
    r[3] = __byte_perm(q[2], q[3], 0x7632);
    uint4 o;
    uint32_t* op = (uint32_t*)&o;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      __half2 h = *(__half2*)&r[k];
      h = __hmul2(__hsub2(h, c2), d2);  // (1024 + n) - 1032 = n - 8 exactly, then one rounding
      op[k] = *(uint32_t*)&h;
    }
    return o;
  } else if constexpr (KT == KV_Q3) {
    // Lloyd-Max levels +-{0.2451, 0.7560, 1.3439, 2.1520} as f16, picked with byte permutes:
    // field q (3 bits): q >= 4 -> +mag[q & 3], q < 4 -> -mag[(q & 3) ^ 3]
    const uint32_t* w = (const uint32_t*)rec;
    const int off = 16 + 3 * c, wi = off >> 2;
    const uint32_t u = __funnelshift_r(w[wi], w[min(wi + 1, 27)], (off & 3) * 8);
    const uint32_t nb2 = (~u >> 2) & 0x249249u;     // 1 where the field is negative (bit 2 clear)
    const uint32_t x = u ^ (nb2 * 3);               // low two bits of each field = magnitude index
    constexpr uint32_t T0 = 0x3a0c33d8u;            // f16(0.7560) << 16 | f16(0.2451)
    constexpr uint32_t T1 = 0x404e3d60u;            // f16(2.1520) << 16 | f16(1.3439)
    const __half r = ((const __half*)rec)[c >> 2];
    const __half2 r2 = __halves2half2(r, r);
    uint4 o;
    uint32_t* op = (uint32_t*)&o;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint32_t m0 = (x >> (6 * k)) & 3, m1 = (x >> (6 * k + 3)) & 3;
      const uint32_t sel = (m0 | (m1 << 8)) * 0x22u + 0x1010u;
      uint32_t h = __byte_perm(T0, T1, sel);
      h ^= ((nb2 >> (6 * k)) & 1) << 15 | ((nb2 >> (6 * k + 3)) & 1) << 31;
      __half2 hv = __hmul2(*(__half2*)&h, r2);
      op[k] = *(uint32_t*)&hv;
    }
    return o;
  } else {
    float v[8];
    kv_read8<KT>(rec, c, v);
    uint4 pk;
    pk.x = pack_h2_(v[0], v[1]); pk.y = pack_h2_(v[2], v[3]);
    pk.z = pack_h2_(v[4], v[5]); pk.w = pack_h2_(v[6], v[7]);
    return pk;
  }
}

template <int KT>
__global__ void k_kv_store(uint8_t* kc, uint8_t* vc, const float* __restrict__ k,
                           const float* __restrict__ v, const int* __restrict__ pos, int nkv,
                           int tokens, int kts, int vts, int recb, KVMap bs) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= tokens * nkv * 2) return;
  const int which = wid & 1;  // 0 = K, 1 = V
  const int h = (wid >> 1) % nkv, t = (wid >> 1) / nkv;
  const float* src = (which ? v + (size_t)t * vts : k + (size_t)t * kts) + h * 256 + 8 * lane;
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = src[i];
  if (KT != KV_F16) wht256(x);
  uint8_t* rec = (which ? vc : kc) + kv_off(h, pos[t], bs, recb);
  kv_write8<KT>(rec, x);
}

void kv_store(int kvtype, void* kcache, void* vcache, const float* k, const float* v,
              const int* pos, int nkv, int hd, int tokens, int kts, int vts, KVMap bs, cudaStream_t s) {
  const int recb = (int)kv_bytes_per_token_head(kvtype, hd);
  const int nw = tokens * nkv * 2;
  uint8_t* kc = (uint8_t*)kcache;
  uint8_t* vc = (uint8_t*)vcache;
  switch (kvtype) {
    case KV_F16: k_kv_store<KV_F16><<<cdiv(nw, 8), 256, 0, s>>>(kc, vc, k, v, pos, nkv, tokens, kts, vts, recb, bs); break;
    case KV_Q8: k_kv_store<KV_Q8><<<cdiv(nw, 8), 256, 0, s>>>(kc, vc, k, v, pos, nkv, tokens, kts, vts, recb, bs); break;
    case KV_Q4: k_kv_store<KV_Q4><<<cdiv(nw, 8), 256, 0, s>>>(kc, vc, k, v, pos, nkv, tokens, kts, vts, recb, bs); break;
    case KV_Q3: k_kv_store<KV_Q3><<<cdiv(nw, 8), 256, 0, s>>>(kc, vc, k, v, pos, nkv, tokens, kts, vts, recb, bs); break;
  }
}

// ---------------------------------------------------------------------------------------------
// Split-K decode attention (T <= 2 queries per head typically; NQ = G*T <= 12 per kv head).
// Cache layout: see kv_off (block-major, head-major inside a block).
// grid = (nkv, nsplit); CTA = 8 warps handles ATT_SPLIT positions for all NQ queries of a kv head.
//   phase 1: lane-per-position scores (q broadcast from shared memory)
//   phase 2: per-query max/sum over the split
//   phase 3: warp w takes positions w, w+8, ...; lane owns 8 dims; shared-memory reduction
// Partials (m, l, o[256]) per (split, kvh, query) are merged by k_attn_combine.
// ---------------------------------------------------------------------------------------------
constexpr int ATT_SPLIT = 512;

template <int KT, int NQ, int SPLIT>
__global__ void __launch_bounds__(256)
    k_attn_split(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                 const float* __restrict__ q, int qhs, int qts, int nh, int nkv, int pos0, int T,
                 float scale, float* __restrict__ work, KVMap bs, int recb) {
  __shared__ __align__(16) float qs[NQ][256];
  __shared__ float sc[NQ][SPLIT];
  float (*osh)[256] = qs;  // reused after phase 1
  const int G = nh / nkv;
  const int kvh = blockIdx.x, sp = blockIdx.y;
  const int p_begin = sp * SPLIT;
  const int p_end = min(p_begin + SPLIT, pos0 + T);
  const int np = p_end - p_begin;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // a split never crosses a KV block (ATT_SPLIT divides KV_BLOCK): positions are contiguous from p_begin
  const uint8_t* kbase = kc + kv_off(kvh, p_begin, bs, recb) - (size_t)p_begin * recb;
  const uint8_t* vbase = vc + kv_off(kvh, p_begin, bs, recb) - (size_t)p_begin * recb;

  for (int qi = warp; qi < NQ; qi += 8) {
    const int t = qi / G, g = qi % G;
    float x[8];
    if (t < T) {
      const float* src = q + (size_t)t * qts + (size_t)(kvh * G + g) * qhs + 8 * lane;
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] = src[i] * scale;
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] = 0.f;
    }
    if (KT != KV_F16) wht256(x);
#pragma unroll
    for (int i = 0; i < 8; i++) qs[qi][8 * lane + i] = x[i];
  }
  __syncthreads();

  // phase 1: lane-per-position
  for (int pi = threadIdx.x; pi < np; pi += 256) {
    const int p = p_begin + pi;
    const uint8_t* rec = kbase + (size_t)p * recb;
    float acc[NQ];
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) acc[qi] = 0.f;
#pragma unroll 4
    for (int c = 0; c < 32; c++) {
      float kv[8];
      kv_read8<KT>(rec, c, kv);
#pragma unroll
      for (int qi = 0; qi < NQ; qi++) {
        const float4 a = *(const float4*)&qs[qi][8 * c];
        const float4 b = *(const float4*)&qs[qi][8 * c + 4];
        acc[qi] += a.x * kv[0] + a.y * kv[1] + a.z * kv[2] + a.w * kv[3] + b.x * kv[4] + b.y * kv[5] + b.z * kv[6] + b.w * kv[7];
      }
    }
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) {
      const int t = qi / G;
      sc[qi][pi] = (t < T && p <= pos0 + t) ? acc[qi] : -INFINITY;
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < NQ * 256; i += 256) (&osh[0][0])[i] = 0.f;
  // phase 2: softmax stats per query (warp per query)
  float* stats = work + ((size_t)(sp * nkv + kvh) * NQ) * 258;
  for (int qi = warp; qi < NQ; qi += 8) {
    float m = -INFINITY;
    for (int pi = lane; pi < np; pi += 32) m = fmaxf(m, sc[qi][pi]);
    m = warp_max(m);
    float l = 0.f;
    for (int pi = lane; pi < np; pi += 32) {
      const float s = sc[qi][pi];
      const float e = (s == -INFINITY) ? 0.f : __expf(s - m);
      sc[qi][pi] = e;
      l += e;
    }
    l = warp_sum(l);
    if (lane == 0) {
      stats[qi * 258 + 0] = m;
      stats[qi * 258 + 1] = l;
    }
  }
  __syncthreads();

  // phase 3: o[q][d] = sum_p e[q][p] V[p][d], in groups of 6 queries (bounded registers)
#pragma unroll 1
  for (int q0 = 0; q0 < NQ; q0 += 6) {
    float acc[6][8];
#pragma unroll
    for (int qi = 0; qi < 6; qi++)
#pragma unroll
      for (int i = 0; i < 8; i++) acc[qi][i] = 0.f;
    for (int pi = warp; pi < np; pi += 8) {
      float vv[8];
      kv_read8<KT>(vbase + (size_t)(p_begin + pi) * recb, lane, vv);
#pragma unroll
      for (int qi = 0; qi < 6; qi++) {
        const float e = sc[q0 + qi][pi];
#pragma unroll
        for (int i = 0; i < 8; i++) acc[qi][i] += e * vv[i];
      }
    }
#pragma unroll
    for (int qi = 0; qi < 6; qi++)
#pragma unroll
      for (int i = 0; i < 8; i++) atomicAdd(&osh[q0 + qi][8 * lane + i], acc[qi][i]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NQ * 256; i += 256) stats[(i / 256) * 258 + 2 + (i % 256)] = (&osh[0][0])[i];
}

// one CTA (256 threads = dims) per (kv head, query)
template <int KT>
__global__ void __launch_bounds__(256)
    k_attn_combine(const float* __restrict__ work, float* __restrict__ out, int nh, int nkv, int T,
                   int NQ, int pos0, int split) {
  __shared__ float wsh[512];
  __shared__ float red[32];
  const int G = nh / nkv;
  const int kvh = blockIdx.x / NQ, qi = blockIdx.x % NQ;
  const int t = qi / G, g = qi % G;
  if (t >= T) return;
  const int nsplit = (pos0 + t) / split + 1;
  const int d = threadIdx.x;
  auto st = [&](int sp) { return work + ((size_t)(sp * nkv + kvh) * NQ + qi) * 258; };
  float M = -INFINITY;
  for (int sp = d; sp < nsplit; sp += 256) M = fmaxf(M, st(sp)[0]);
  M = warp_max(M);
  if ((d & 31) == 0) red[d >> 5] = M;
  __syncthreads();
  M = -INFINITY;
  for (int i = 0; i < 8; i++) M = fmaxf(M, red[i]);
  __syncthreads();
  float L = 0.f;
  for (int sp = d; sp < nsplit; sp += 256) {
    const float m = st(sp)[0];
    const float w = (m == -INFINITY) ? 0.f : __expf(m - M);
    if (sp < 512) wsh[sp] = w;
    L += w * st(sp)[1];
  }
  L = warp_sum(L);
  if ((d & 31) == 0) red[d >> 5] = L;
  __syncthreads();
  L = 0.f;
  for (int i = 0; i < 8; i++) L += red[i];
  float o = 0.f;
  for (int sp = 0; sp < nsplit; sp++) {
    const float w = sp < 512 ? wsh[sp] : __expf(st(sp)[0] - M);
    if (w != 0.f) o += w * st(sp)[2 + d];
  }
  o /= L;
  // un-rotate: warp-level WHT needs 8 consecutive dims per lane -> go through shared memory
  __shared__ float ob[256];
  ob[d] = o;
  __syncthreads();
  if (d < 32) {
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = ob[8 * d + i];
    if (KT != KV_F16) wht256(v);
    float* dst = out + (size_t)t * nh * 256 + (size_t)(kvh * G + g) * 256 + 8 * d;
#pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = v[i];
  }
}

size_t attn_decode_work_bytes(int max_ctx, int nh, int hd, int tokens) {
  const int nsplit = (max_ctx + 255) / 256;  // smallest split used (T >= 3)
  return (size_t)nsplit * nh * tokens * (hd + 2) * sizeof(float);
}

void attn_decode_mma(int kvtype, const void* kcache, const void* vcache, const float* q, int qhs, int qts,
                     float* out, int nh, int nkv, int hd, int pos0, int T, float scale, float* work,
                     size_t work_bytes, KVMap bs, cudaStream_t s);

static void attn_decode_simt(int kvtype, const void* kcache, const void* vcache, const float* q, int qhs,
                 int qts, float* out, int nh, int nkv, int hd, int pos0, int T, float scale,
                 float* work, size_t work_bytes, KVMap bs, cudaStream_t s);

void attn_decode(int kvtype, const void* kcache, const void* vcache, const float* q, int qhs,
                 int qts, float* out, int nh, int nkv, int hd, int pos0, int T, float scale,
                 float* work, size_t work_bytes, KVMap bs, cudaStream_t s) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP v1: SIMT path only (the MMA kernel needs tensor cores), chunked to its
  // T <= 4 limit, with a private scratch buffer sized by the SIMT formula (the
  // caller's buffer follows the MMA sizing).
  (void)work; (void)work_bytes;
  static float* hwork = nullptr;
  static size_t hwork_bytes = 0;
  const size_t need = attn_decode_work_bytes(pos0 + T, nh, hd, 4);
  if (need > hwork_bytes) {
    if (hwork) (void)cudaFree(hwork);
    CUDA_CHECK(cudaMalloc(&hwork, need));
    hwork_bytes = need;
  }
  for (int t0 = 0; t0 < T; t0 += 4) {
    const int nt = T - t0 < 4 ? T - t0 : 4;
    attn_decode_simt(kvtype, kcache, vcache, q + (size_t)t0 * qts, qhs, qts, out + (size_t)t0 * nh * hd,
                     nh, nkv, hd, pos0 + t0, nt, scale, hwork, hwork_bytes, bs, s);
  }
  return;
#endif
  static const int mode = getenv("QE_ATT") ? atoi(getenv("QE_ATT")) : -1;  // 0 simt, 1 mma, -1 auto
  const bool mma = mode != 0;  // the MMA kernel is as fast or faster at T = 1 too
  if (mma) attn_decode_mma(kvtype, kcache, vcache, q, qhs, qts, out, nh, nkv, hd, pos0, T, scale, work, work_bytes, bs, s);
  else attn_decode_simt(kvtype, kcache, vcache, q, qhs, qts, out, nh, nkv, hd, pos0, T, scale, work, work_bytes, bs, s);
}

static void attn_decode_simt(int kvtype, const void* kcache, const void* vcache, const float* q, int qhs,
                 int qts, float* out, int nh, int nkv, int hd, int pos0, int T, float scale,
                 float* work, size_t work_bytes, KVMap bs, cudaStream_t s) {
  QE_ASSERT(hd == 256 && T >= 1 && T <= 4 && nh / nkv == 6);
  static_assert(KV_BLOCK % ATT_SPLIT == 0, "split must not cross KV blocks");
  const int split = T <= 2 ? ATT_SPLIT : 256;
  const int nsplit = (pos0 + T + split - 1) / split;
  QE_ASSERT(attn_decode_work_bytes(pos0 + T, nh, hd, T) <= work_bytes);
  const int recb = (int)kv_bytes_per_token_head(kvtype, hd);
  const dim3 grid(nkv, nsplit);
  const uint8_t* kc = (const uint8_t*)kcache;
  const uint8_t* vc = (const uint8_t*)vcache;
  const int NQ = 6 * T;
#define LAUNCH(KT)                                                                                                         \
  {                                                                                                                        \
    switch (T) {                                                                                                           \
      case 1: k_attn_split<KT, 6, ATT_SPLIT><<<grid, 256, 0, s>>>(kc, vc, q, qhs, qts, nh, nkv, pos0, T, scale, work, bs, recb); break; \
      case 2: k_attn_split<KT, 12, ATT_SPLIT><<<grid, 256, 0, s>>>(kc, vc, q, qhs, qts, nh, nkv, pos0, T, scale, work, bs, recb); break; \
      case 3: k_attn_split<KT, 18, 256><<<grid, 256, 0, s>>>(kc, vc, q, qhs, qts, nh, nkv, pos0, T, scale, work, bs, recb); break; \
      default: k_attn_split<KT, 24, 256><<<grid, 256, 0, s>>>(kc, vc, q, qhs, qts, nh, nkv, pos0, T, scale, work, bs, recb); break; \
    }                                                                                                                      \
    k_attn_combine<KT><<<nkv * NQ, 256, 0, s>>>(work, out, nh, nkv, T, NQ, pos0, split);                                  \
  }
  switch (kvtype) {
    case KV_F16: LAUNCH(KV_F16) break;
    case KV_Q8: LAUNCH(KV_Q8) break;
    case KV_Q4: LAUNCH(KV_Q4) break;
    case KV_Q3: LAUNCH(KV_Q3) break;
  }
#undef LAUNCH
}

// ---------------------------------------------------------------------------------------------
// Prefill flash attention on tensor cores (mma.sync m16n8k16, f16 in / f32 accumulate).
// CTA = 4 warps = 64 query rows of one head; K/V tiles of 32 positions dequantized from the
// cache into shared memory (V transposed).  Works in the WHT-rotated domain for Q8/Q4 caches.
// ---------------------------------------------------------------------------------------------
constexpr int FA_BM = 64, FA_BN = 32, FA_D = 256;
constexpr int FA_QLD = FA_D + 8;

static __device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP v1: tensor-core attention paths are not launched (SIMT decode + chunked
  // prefill take over); never called.
  (void)c; (void)a; (void)b0; (void)b1;
#else
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

static __device__ __forceinline__ void mma16816_h(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef __HIP_PLATFORM_AMD__
  (void)c; (void)a; (void)b0; (void)b1;
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// 4 transposed 8x8 b16 matrices from shared memory (B fragments of m16n8k16 from row-major [k][n])
static __device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
#ifdef __HIP_PLATFORM_AMD__
  (void)r; (void)p;
#else
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
#endif
}
static __device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *(uint32_t*)&h;
}

// q (f32, strided) -> q16 [T][nh][256], scaled by scale*log2(e), rotated for quantized caches
template <int KT>
__global__ void k_fa_prep_q(const float* __restrict__ q, int qhs, int qts, int nh, int T,
                            float scale_log2, __half* __restrict__ q16) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= T * nh) return;
  const int t = wid / nh, h = wid % nh;
  const float* src = q + (size_t)t * qts + (size_t)h * qhs + 8 * lane;
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = src[i];
  if (KT != KV_F16) wht256(x);
  __half2 hh[4];
#pragma unroll
  for (int i = 0; i < 4; i++) hh[i] = __floats2half2_rn(x[2 * i] * scale_log2, x[2 * i + 1] * scale_log2);
  *(uint4*)(q16 + (size_t)wid * FA_D + 8 * lane) = *(uint4*)hh;
}

template <int KT, int NW>  // NW warps x 16 query rows; Q in registers, so 2 CTAs / SM fit (dequant overlaps MMA)
__global__ void __launch_bounds__(NW * 32, 8 / NW)
    k_fa(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
         const __half* __restrict__ q16, float* __restrict__ out, int nh, int nkv, int pos0, int T,
         int recb, KVMap bs) {
  extern __shared__ __align__(16) unsigned char fa_smem[];
  __half* Ks = (__half*)fa_smem;             // [32][FA_QLD]
  __half* Vs = Ks + FA_BN * FA_QLD;          // [32][FA_QLD] (row-major, read with ldmatrix.trans)
  const int h = blockIdx.y, kvh = h / (nh / nkv);
  constexpr int BM = NW * 16, NTH = NW * 32;
  const int row0 = blockIdx.x * BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3;

  // Q fragments of this warp's 16 rows (16 k-steps x 4 regs), straight from global
  uint32_t qa[16][4];
  {
    const int ra = row0 + warp * 16 + g, rb = ra + 8;
    const bool va = ra < T, vb = rb < T;
    const __half* qa_p = q16 + ((size_t)min(ra, T - 1) * nh + h) * FA_D + tq * 2;
    const __half* qb_p = q16 + ((size_t)min(rb, T - 1) * nh + h) * FA_D + tq * 2;
#pragma unroll
    for (int kk = 0; kk < 16; kk++) {
      qa[kk][0] = va ? *(const uint32_t*)(qa_p + kk * 16) : 0u;
      qa[kk][1] = vb ? *(const uint32_t*)(qb_p + kk * 16) : 0u;
      qa[kk][2] = va ? *(const uint32_t*)(qa_p + kk * 16 + 8) : 0u;
      qa[kk][3] = vb ? *(const uint32_t*)(qb_p + kk * 16 + 8) : 0u;
    }
  }

  float o[32][4];
#pragma unroll
  for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  const int r_lo = warp * 16 + g, r_hi = r_lo + 8;   // rows of this thread inside the tile
  const int lim_lo = pos0 + row0 + r_lo, lim_hi = pos0 + row0 + r_hi;  // max key pos (causal)
  const int pend = pos0 + min(row0 + BM, T) - 1;

  // Q4 cache: raw records of the next tile are prefetched with cp.async (double-buffered) so the
  // global-memory latency overlaps the tensor-core work of the current tile.
  constexpr bool PIPE = KT == KV_Q4 || KT == KV_Q3;
  constexpr int RECB = KT == KV_Q3 ? 112 : 144;
  constexpr int RAWB = FA_BN * RECB;
  uint8_t* raw = (uint8_t*)(Vs + FA_BN * FA_QLD);  // [stage][K|V][RAWB]
  auto issue = [&](int kt, int st) {
    const uint8_t* kb = kc + kv_off(kvh, kt, bs, recb);  // a tile never crosses a KV block
    const uint8_t* vb = vc + kv_off(kvh, kt, bs, recb);
    const int nch = min(FA_BN, pend - kt + 1) * (RECB / 16);
    for (int i = tid; i < nch; i += NTH) {
      cp_async16(raw + (st * 2 + 0) * RAWB + i * 16, kb + i * 16);
      cp_async16(raw + (st * 2 + 1) * RAWB + i * 16, vb + i * 16);
    }
    cp_async_commit();
  };
  if constexpr (PIPE) issue(0, 0);
  int st = 0;
  for (int kt = 0; kt <= pend; kt += FA_BN, st ^= 1) {
    if constexpr (PIPE) {
      if (kt + FA_BN <= pend) issue(kt + FA_BN, st ^ 1);
      else cp_async_commit();
      cp_async_wait<1>();
    }
    __syncthreads();
    for (int i = tid; i < FA_BN * (FA_D / 8); i += NTH) {
      const int pi = i % FA_BN, c = i / FA_BN;  // lanes = positions: conflict-free stores
      const int p = kt + pi;
      uint4 pkk = make_uint4(0, 0, 0, 0), pkv = make_uint4(0, 0, 0, 0);
      if (p <= pend) {
        if constexpr (PIPE) {
          pkk = kv_read8h<KT>(raw + (st * 2 + 0) * RAWB + pi * RECB, c);
          pkv = kv_read8h<KT>(raw + (st * 2 + 1) * RAWB + pi * RECB, c);
        } else {
          pkk = kv_read8h<KT>(kc + kv_off(kvh, p, bs, recb), c);
          pkv = kv_read8h<KT>(vc + kv_off(kvh, p, bs, recb), c);
        }
      }
      *(uint4*)(Ks + pi * FA_QLD + 8 * c) = pkk;
      *(uint4*)(Vs + pi * FA_QLD + 8 * c) = pkv;
    }
    __syncthreads();
    if (kt > pos0 + row0 + warp * 16 + 15) continue;  // tile fully masked for every row of this warp (warp-uniform)

    // S = Q K^T  (16 rows x 32 positions per warp)
    float s[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
    // f16 accumulation over 64-dim chunks (2x tensor rate on GeForce Ampere), chunks summed in f32
#pragma unroll
    for (int kc = 0; kc < FA_D / 64; kc++) {
      uint32_t hc[4][2] = {};
#pragma unroll
      for (int k4 = 0; k4 < 4; k4++) {
        const int kk = kc * 4 + k4;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          const __half* kb = Ks + (nt * 8 + g) * FA_QLD + kk * 16 + tq * 2;
          mma16816_h(hc[nt], qa[kk], *(const uint32_t*)kb, *(const uint32_t*)(kb + 8));
        }
      }
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const float2 lo = __half22float2(*(__half2*)&hc[nt][0]), hi = __half22float2(*(__half2*)&hc[nt][1]);
        s[nt][0] += lo.x; s[nt][1] += lo.y; s[nt][2] += hi.x; s[nt][3] += hi.y;
      }
    }
    // causal mask + online softmax (log2 domain; q was pre-scaled by log2e)
    float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const int p = kt + nt * 8 + tq * 2;
      if (p > lim_lo) s[nt][0] = -1e30f;
      if (p + 1 > lim_lo) s[nt][1] = -1e30f;
      if (p > lim_hi) s[nt][2] = -1e30f;
      if (p + 1 > lim_hi) s[nt][3] = -1e30f;
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
    m0 = mn0; m1 = mn1;
    l0 *= al0; l1 *= al1;
#pragma unroll
    for (int dn = 0; dn < 32; dn++) { o[dn][0] *= al0; o[dn][1] *= al0; o[dn][2] *= al1; o[dn][3] *= al1; }
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      s[nt][0] = s[nt][0] <= -1e29f ? 0.f : exp2f(s[nt][0] - m0);
      s[nt][1] = s[nt][1] <= -1e29f ? 0.f : exp2f(s[nt][1] - m0);
      s[nt][2] = s[nt][2] <= -1e29f ? 0.f : exp2f(s[nt][2] - m1);
      s[nt][3] = s[nt][3] <= -1e29f ? 0.f : exp2f(s[nt][3] - m1);
      l0 += s[nt][0] + s[nt][1];
      l1 += s[nt][2] + s[nt][3];
    }
    // O += P V.  Within the tile (32 positions, P <= 1) the products are accumulated in f16 (2x the
    // f32-accumulate tensor rate on GeForce Ampere), then added to the f32 accumulators.
    {
      uint32_t a[FA_BN / 16][4];
#pragma unroll
      for (int j = 0; j < FA_BN / 16; j++) {
        a[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
        a[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
        a[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
        a[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
      }
      // lane -> (matrix, row): positions j*16 + (mi&1)*8 + row, dims base + (mi>>1)*8
      const int mi = lane >> 3, rr = lane & 7;
      const __half* vrow = Vs + ((mi & 1) * 8 + rr) * FA_QLD + (mi >> 1) * 8;
#pragma unroll
      for (int dn = 0; dn < 32; dn += 2) {
        uint32_t c0[2] = {0u, 0u}, c1[2] = {0u, 0u};
#pragma unroll
        for (int j = 0; j < FA_BN / 16; j++) {
          uint32_t b[4];
          ldsm_x4_t(b, vrow + j * 16 * FA_QLD + dn * 8);
          mma16816_h(c0, a[j], b[0], b[1]);
          mma16816_h(c1, a[j], b[2], b[3]);
        }
        const float2 p0 = __half22float2(*(__half2*)&c0[0]), p1 = __half22float2(*(__half2*)&c0[1]);
        const float2 q0 = __half22float2(*(__half2*)&c1[0]), q1 = __half22float2(*(__half2*)&c1[1]);
        o[dn][0] += p0.x; o[dn][1] += p0.y; o[dn][2] += p1.x; o[dn][3] += p1.y;
        o[dn + 1][0] += q0.x; o[dn + 1][1] += q0.y; o[dn + 1][2] += q1.x; o[dn + 1][3] += q1.y;
      }
    }
  }
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
  const int t_lo = row0 + r_lo, t_hi = row0 + r_hi;
#pragma unroll
  for (int dn = 0; dn < 32; dn++) {
    const int d = dn * 8 + tq * 2;
    if (t_lo < T) *(float2*)(out + ((size_t)t_lo * nh + h) * FA_D + d) = make_float2(o[dn][0] * il0, o[dn][1] * il0);
    if (t_hi < T) *(float2*)(out + ((size_t)t_hi * nh + h) * FA_D + d) = make_float2(o[dn][2] * il1, o[dn][3] * il1);
  }
}

// ---------------------------------------------------------------------------------------------
// Prefill flash attention, GQA-packed: CTA = one kv head x 16 tokens; warp w = query head
// kvh*G + w (G = 6 warps), 16 rows each.  Every dequantized K/V tile feeds all G heads, Q lives in
// registers (A fragments), and Q4 raw records are prefetched with cp.async (double buffer).
// ---------------------------------------------------------------------------------------------
constexpr int F2_BN = 32, F2_TOK = 16;

template <int KT, int G>
__global__ void __launch_bounds__(G * 32, 1)
    k_fa2(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
          const __half* __restrict__ q16, float* __restrict__ out, int nh, int nkv, int pos0, int T,
          int recb, KVMap bs) {
  extern __shared__ __align__(16) unsigned char f2_smem[];
  __half* Ks = (__half*)f2_smem;                 // [32][FA_QLD]
  __half* Vs = Ks + F2_BN * FA_QLD;              // [32][FA_QLD]
  uint8_t* raw = (uint8_t*)(Vs + F2_BN * FA_QLD);  // Q4: [stage][K|V][32*144]
  constexpr int NT = G * 32;
  const int kvh = blockIdx.y, t0 = blockIdx.x * F2_TOK;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int h = kvh * G + warp;

  // Q fragments (16 k-steps x 4 regs) straight from global
  uint32_t qa[16][4];
  {
    const int ra = min(t0 + g, T - 1), rb = min(t0 + g + 8, T - 1);
    const __half* qa_p = q16 + ((size_t)ra * nh + h) * FA_D + tq * 2;
    const __half* qb_p = q16 + ((size_t)rb * nh + h) * FA_D + tq * 2;
#pragma unroll
    for (int kk = 0; kk < 16; kk++) {
      qa[kk][0] = *(const uint32_t*)(qa_p + kk * 16);
      qa[kk][1] = *(const uint32_t*)(qb_p + kk * 16);
      qa[kk][2] = *(const uint32_t*)(qa_p + kk * 16 + 8);
      qa[kk][3] = *(const uint32_t*)(qb_p + kk * 16 + 8);
    }
  }
  float o[32][4];
#pragma unroll
  for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  const int lim_lo = pos0 + t0 + g, lim_hi = lim_lo + 8;
  const int pend = pos0 + min(t0 + F2_TOK, T) - 1;

  constexpr bool PIPE = KT == KV_Q4 || KT == KV_Q3;
  constexpr int RECB = KT == KV_Q3 ? 112 : 144;
  constexpr int RAWB = F2_BN * RECB;
  auto issue = [&](int kt, int st) {
    const uint8_t* kb = kc + kv_off(kvh, kt, bs, recb);
    const uint8_t* vb = vc + kv_off(kvh, kt, bs, recb);
    const int nch = min(F2_BN, pend - kt + 1) * (RECB / 16);
    for (int i = tid; i < nch; i += NT) {
      cp_async16(raw + (st * 2 + 0) * RAWB + i * 16, kb + i * 16);
      cp_async16(raw + (st * 2 + 1) * RAWB + i * 16, vb + i * 16);
    }
    cp_async_commit();
  };
  if constexpr (PIPE) issue(0, 0);
  int st = 0;
  for (int kt = 0; kt <= pend; kt += F2_BN, st ^= 1) {
    if constexpr (PIPE) {
      if (kt + F2_BN <= pend) issue(kt + F2_BN, st ^ 1);
      else cp_async_commit();
      cp_async_wait<1>();
    }
    __syncthreads();
    for (int i = tid; i < F2_BN * (FA_D / 8); i += NT) {
      const int pi = i % F2_BN, c = i / F2_BN;
      const int p = kt + pi;
      uint4 pkk = make_uint4(0, 0, 0, 0), pkv = make_uint4(0, 0, 0, 0);
      if (p <= pend) {
        if constexpr (PIPE) {
          pkk = kv_read8h<KT>(raw + (st * 2 + 0) * RAWB + pi * RECB, c);
          pkv = kv_read8h<KT>(raw + (st * 2 + 1) * RAWB + pi * RECB, c);
        } else {
          pkk = kv_read8h<KT>(kc + kv_off(kvh, p, bs, recb), c);
          pkv = kv_read8h<KT>(vc + kv_off(kvh, p, bs, recb), c);
        }
      }
      *(uint4*)(Ks + pi * FA_QLD + 8 * c) = pkk;
      *(uint4*)(Vs + pi * FA_QLD + 8 * c) = pkv;
    }
    __syncthreads();
    float s[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 16; kk++) {
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const __half* kp = Ks + (nt * 8 + g) * FA_QLD + kk * 16 + tq * 2;
        mma16816(s[nt], qa[kk], *(const uint32_t*)kp, *(const uint32_t*)(kp + 8));
      }
    }
    float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const int p = kt + nt * 8 + tq * 2;
      if (p > lim_lo) s[nt][0] = -1e30f;
      if (p + 1 > lim_lo) s[nt][1] = -1e30f;
      if (p > lim_hi) s[nt][2] = -1e30f;
      if (p + 1 > lim_hi) s[nt][3] = -1e30f;
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
    m0 = mn0; m1 = mn1;
    l0 *= al0; l1 *= al1;
#pragma unroll
    for (int dn = 0; dn < 32; dn++) { o[dn][0] *= al0; o[dn][1] *= al0; o[dn][2] *= al1; o[dn][3] *= al1; }
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      s[nt][0] = s[nt][0] <= -1e29f ? 0.f : exp2f(s[nt][0] - m0);
      s[nt][1] = s[nt][1] <= -1e29f ? 0.f : exp2f(s[nt][1] - m0);
      s[nt][2] = s[nt][2] <= -1e29f ? 0.f : exp2f(s[nt][2] - m1);
      s[nt][3] = s[nt][3] <= -1e29f ? 0.f : exp2f(s[nt][3] - m1);
      l0 += s[nt][0] + s[nt][1];
      l1 += s[nt][2] + s[nt][3];
    }
#pragma unroll
    for (int j = 0; j < F2_BN / 16; j++) {
      uint32_t a[4];
      a[0] = pack_h2(s[2 * j][0], s[2 * j][1]);
      a[1] = pack_h2(s[2 * j][2], s[2 * j][3]);
      a[2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
      a[3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
      const int mi = lane >> 3, rr = lane & 7;
      const __half* vrow = Vs + (j * 16 + (mi & 1) * 8 + rr) * FA_QLD + (mi >> 1) * 8;
#pragma unroll
      for (int dn = 0; dn < 32; dn += 2) {
        uint32_t b[4];
        ldsm_x4_t(b, vrow + dn * 8);
        mma16816(o[dn], a, b[0], b[1]);
        mma16816(o[dn + 1], a, b[2], b[3]);
      }
    }
  }
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
  const int ta = t0 + g, tb = ta + 8;
#pragma unroll
  for (int dn = 0; dn < 32; dn++) {
    const int d = dn * 8 + tq * 2;
    if (ta < T) *(float2*)(out + ((size_t)ta * nh + h) * FA_D + d) = make_float2(o[dn][0] * il0, o[dn][1] * il0);
    if (tb < T) *(float2*)(out + ((size_t)tb * nh + h) * FA_D + d) = make_float2(o[dn][2] * il1, o[dn][3] * il1);
  }
}

// in-place inverse rotation of attention output rows [T*nh][256]
__global__ void k_wht_rows(float* x, int rows) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= rows) return;
  float* p = x + (size_t)wid * 256 + 8 * lane;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = p[i];
  wht256(v);
#pragma unroll
  for (int i = 0; i < 8; i++) p[i] = v[i];
}

size_t attn_prefill_work_bytes(int T, int nh, int hd) { return (size_t)T * nh * hd * 2; }

#ifdef __HIP_PLATFORM_AMD__
// SIMT flash attention for HIP prefill (QE_FA_SIMT=1). One block = one kv head
// x BQ queries (BQ*G rows); KV is streamed in 32-position tiles, dequantized
// once into shared memory and reused by all rows of the block -- the v1 HIP
// fallback (attn_decode in chunks of 4) re-dequantizes and re-streams the KV
// prefix for every chunk and pays a split+combine launch pair per chunk.
// Numerics match the CUDA k_fa: Q comes from k_fa_prep_q (WHT-rotated for
// quantized caches, pre-scaled by scale*log2(e)), scores/softmax run in log2
// space with exp2f, and k_wht_rows undoes the output rotation afterwards.
constexpr int FA_SIMT_BQ = 8;   // queries per block
constexpr int FA_SIMT_BN = 32;  // KV tile positions

template <int KT, int G>  // G = nh / nkv (6 for the target model)
__global__ void __launch_bounds__(256)
    k_fa_simt(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
              const __half* __restrict__ q16, float* __restrict__ out, int nh, int pos0, int T,
              int recb, KVMap bs) {
  __shared__ __align__(16) __half Ks[FA_SIMT_BN][FA_QLD];
  __shared__ __align__(16) __half Vs[FA_SIMT_BN][FA_QLD];
  __shared__ float qs[8][FA_D];   // current row's q (per warp)
  __shared__ float wsm[8][FA_SIMT_BN];
  constexpr int ROWS = FA_SIMT_BQ * G, RPW = ROWS / 8;  // rows per warp
  const int kvh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int q0 = blockIdx.x * FA_SIMT_BQ;
  const int ptot = pos0 + T;
  const int kt_max = pos0 + min(q0 + FA_SIMT_BQ, T) - 1;  // last key any row may see

  int rt_[RPW], rh_[RPW];
  float o_[RPW][8], m_[RPW], l_[RPW];
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int rr = warp * RPW + i;
    rt_[i] = q0 + rr / G;
    rh_[i] = kvh * G + rr % G;
    m_[i] = -1e30f;
    l_[i] = 0.f;
#pragma unroll
    for (int d = 0; d < 8; d++) o_[i][d] = 0.f;
  }

  for (int kt = 0; kt <= kt_max; kt += FA_SIMT_BN) {
    // cooperative tile fill: 32 positions x 32 chunks of 8 dims
    for (int idx = tid; idx < FA_SIMT_BN * 32; idx += 256) {
      const int pi = idx >> 5, c = idx & 31;
      const int p = min(kt + pi, ptot - 1);
      const uint4 kk = kv_read8h<KT>(kc + kv_off(kvh, p, bs, recb), c);
      const uint4 vv = kv_read8h<KT>(vc + kv_off(kvh, p, bs, recb), c);
      *(uint4*)&Ks[pi][8 * c] = kk;
      *(uint4*)&Vs[pi][8 * c] = vv;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < RPW; i++) {
      if (rt_[i] >= T) continue;
      const int t = rt_[i];
      // stage this row's q (lane owns dims 8*lane..+7)
      const __half* qp = q16 + ((size_t)t * nh + rh_[i]) * FA_D + 8 * lane;
#pragma unroll
      for (int d = 0; d < 8; d++) qs[warp][8 * lane + d] = __half2float(qp[d]);
      __syncwarp();
      const int p = kt + lane;
      float score = -1e30f;
      if (p <= pos0 + t) {
        score = 0.f;
#pragma unroll
        for (int c = 0; c < 32; c++) {
          const uint4 kv4 = *(const uint4*)&Ks[lane][8 * c];
          const __half* kh = (const __half*)&kv4;
#pragma unroll
          for (int d = 0; d < 8; d++) score += qs[warp][8 * c + d] * __half2float(kh[d]);
        }
      }
      float mt = score;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
      const float mn = fmaxf(m_[i], mt);
      const float al = exp2f(m_[i] - mn);
      const float wp = score <= -1e29f ? 0.f : exp2f(score - mn);
      wsm[warp][lane] = wp;
      float ls = wp;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, off);
      l_[i] = l_[i] * al + ls;
      m_[i] = mn;
#pragma unroll
      for (int d = 0; d < 8; d++) o_[i][d] *= al;
      __syncwarp();
#pragma unroll
      for (int pi = 0; pi < FA_SIMT_BN; pi++) {
        const float wgt = wsm[warp][pi];
        if (wgt == 0.f) continue;
        const uint4 vv = *(const uint4*)&Vs[pi][8 * lane];
        const __half* vh = (const __half*)&vv;
#pragma unroll
        for (int d = 0; d < 8; d++) o_[i][d] += wgt * __half2float(vh[d]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    if (rt_[i] >= T) continue;
    const float il = l_[i] > 0.f ? 1.f / l_[i] : 0.f;
    float* op = out + ((size_t)rt_[i] * nh + rh_[i]) * FA_D + 8 * lane;
#pragma unroll
    for (int d = 0; d < 8; d++) op[d] = o_[i][d] * il;
  }
}
#endif  // __HIP_PLATFORM_AMD__

void attn_prefill(int kvtype, const void* kcache, const void* vcache, const float* q, int qhs,
                  int qts, float* out, int nh, int nkv, int hd, int pos0, int T, float scale,
                  void* work, KVMap bs, cudaStream_t s) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP prefill attention: SIMT flash attention when QE_FA_SIMT=1 (and the
  // shape matches), else the v1 fallback -- the decode path in chunks of 4
  // (exactly the engine's own QE_SLOW_ATTN reference route).
  static const bool fa_simt = getenv("QE_FA_SIMT") && atoi(getenv("QE_FA_SIMT")) == 1;
  if (fa_simt && hd == FA_D && nh % nkv == 0 && nh / nkv == 6 && work) {
    const int recb = (int)kv_bytes_per_token_head(kvtype, hd);
    const float sl2 = scale * 1.4426950408889634f;
    __half* q16 = (__half*)work;
    const uint8_t* kc = (const uint8_t*)kcache;
    const uint8_t* vc = (const uint8_t*)vcache;
    const dim3 grid(cdiv(T, FA_SIMT_BQ), nkv);
    switch (kvtype) {
#define FA_SIMT_CASE(KT)                                                                 \
  case KT:                                                                               \
    k_fa_prep_q<KT><<<cdiv(T * nh, 8), 256, 0, s>>>(q, qhs, qts, nh, T, sl2, q16);       \
    k_fa_simt<KT, 6><<<grid, 256, 0, s>>>(kc, vc, q16, out, nh, pos0, T, recb, bs);      \
    if (KT != KV_F16) k_wht_rows<<<cdiv(T * nh, 8), 256, 0, s>>>(out, T * nh);           \
    break;
      FA_SIMT_CASE(KV_Q4) FA_SIMT_CASE(KV_Q8) FA_SIMT_CASE(KV_Q3) FA_SIMT_CASE(KV_F16)
#undef FA_SIMT_CASE
      default:
        break;
    }
    if (kvtype == KV_Q4 || kvtype == KV_Q8 || kvtype == KV_Q3 || kvtype == KV_F16) return;
  }
  for (int t0 = 0; t0 < T; t0 += 4) {
    const int nt = T - t0 < 4 ? T - t0 : 4;
    attn_decode(kvtype, kcache, vcache, q + (size_t)t0 * qts, qhs, qts, out + (size_t)t0 * nh * hd,
                nh, nkv, hd, pos0 + t0, nt, scale, nullptr, 0, bs, s);
  }
  return;
#endif
  QE_ASSERT(hd == FA_D);
  __half* q16 = (__half*)work;
  const int recb = (int)kv_bytes_per_token_head(kvtype, hd);
  const size_t sh = ((size_t)(2 * FA_BN) * FA_QLD) * sizeof(__half) +
                    (kvtype == KV_Q4 ? (size_t)2 * 2 * FA_BN * 144 : kvtype == KV_Q3 ? (size_t)2 * 2 * FA_BN * 112 : 0);
  const float sl2 = scale * 1.4426950408889634f;
  static const int fa_nw = getenv("QE_FA_NW") ? atoi(getenv("QE_FA_NW")) : 4;  // warps per CTA (4; 8 measured no faster)
  const dim3 grid(cdiv(T, fa_nw * 16), nh);
  static const bool use_fa2 = getenv("QE_FA2") && atoi(getenv("QE_FA2")) == 1 && nh / nkv == 6;  // off: slower on sm_86
  const size_t sh2 = (size_t)2 * F2_BN * FA_QLD * sizeof(__half) + (kvtype == KV_Q4 ? (size_t)2 * 2 * F2_BN * 144 : kvtype == KV_Q3 ? (size_t)2 * 2 * F2_BN * 112 : 0);
  const uint8_t* kc = (const uint8_t*)kcache;
  const uint8_t* vc = (const uint8_t*)vcache;
  switch (kvtype) {
#define CASE(KT)                                                                                         \
  case KT: {                                                                                             \
    static bool init = false;                                                                            \
    if (!init) {                                                                                         \
      CUDA_CHECK(cudaFuncSetAttribute(k_fa<KT, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); \
      CUDA_CHECK(cudaFuncSetAttribute(k_fa<KT, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); \
      init = true;                                                                                       \
    }                                                                                                    \
    k_fa_prep_q<KT><<<cdiv(T * nh, 8), 256, 0, s>>>(q, qhs, qts, nh, T, sl2, q16);                       \
    if (use_fa2) {                                                                                       \
      static bool init2 = false;                                                                         \
      if (!init2) {                                                                                      \
        CUDA_CHECK(cudaFuncSetAttribute(k_fa2<KT, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2)); \
        init2 = true;                                                                                    \
      }                                                                                                  \
      k_fa2<KT, 6><<<dim3(cdiv(T, F2_TOK), nkv), 6 * 32, sh2, s>>>(kc, vc, q16, out, nh, nkv, pos0, T, recb, bs); \
    } else                                                                                               \
    if (fa_nw == 4) k_fa<KT, 4><<<grid, 128, sh, s>>>(kc, vc, q16, out, nh, nkv, pos0, T, recb, bs);      \
    else k_fa<KT, 8><<<grid, 256, sh, s>>>(kc, vc, q16, out, nh, nkv, pos0, T, recb, bs);                  \
    if (KT != KV_F16) k_wht_rows<<<cdiv(T * nh, 8), 256, 0, s>>>(out, T * nh);                         \
    break;                                                                                               \
  }
    CASE(KV_F16) CASE(KV_Q8) CASE(KV_Q4) CASE(KV_Q3)
#undef CASE
  }
}

// ---------------------------------------------------------------------------------------------
// Tensor-core decode / verify attention.  All G*T (<= 32) queries of one kv head share each
// dequantized K/V tile.  CTA = 4 warps: warp = (m-tile of 16 query rows) x (half of the 64-position
// tile).  Emits per-split partials (natural-log max, sum, unnormalized o) for k_attn_combine.
// ---------------------------------------------------------------------------------------------
constexpr int AD_BN = 32;
constexpr int AD_QLD = FA_D + 8;

template <int KT, int MT>  // MT m-tiles of 16 query rows (G*T <= 16*MT), 2*MT warps
__global__ void __launch_bounds__(64 * MT, 1)
    k_attn_dec_mma(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                   const float* __restrict__ q, int qhs, int qts, int nh, int nkv, int pos0, int T,
                   float scale_log2, float* __restrict__ work, KVMap bs, int recb, int split) {
  extern __shared__ __align__(16) unsigned char ad_smem[];
  constexpr int NTH = 64 * MT, NR = 16 * MT;
  __half* Qs = (__half*)ad_smem;          // [NR][AD_QLD]
  __half* Ks = Qs + NR * AD_QLD;          // [32][AD_QLD]
  __half* Vs = Ks + AD_BN * AD_QLD;       // [32][AD_QLD] row-major (ldmatrix.trans)
  uint8_t* raw = (uint8_t*)(Vs + AD_BN * AD_QLD);  // Q4: [stage][K|V][32 records] (cp.async prefetch)
  const int G = nh / nkv, NQ = G * T;
  const int kvh = blockIdx.x, sp = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int p_begin = sp * split;
  const int pend = min(p_begin + split, pos0 + T) - 1;

  // queries: rotate (quantized caches), scale by softmax scale * log2(e), f16
  for (int qi = warp; qi < NR; qi += 2 * MT) {
    float x[8];
    if (qi < NQ) {
      const int t = qi / G, gg = qi % G;
      const float* src = q + (size_t)t * qts + (size_t)(kvh * G + gg) * qhs + 8 * lane;
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] = src[i];
      if (KT != KV_F16) wht256(x);
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] = 0.f;
    }
    uint4 pk;
    pk.x = pack_h2(x[0] * scale_log2, x[1] * scale_log2); pk.y = pack_h2(x[2] * scale_log2, x[3] * scale_log2);
    pk.z = pack_h2(x[4] * scale_log2, x[5] * scale_log2); pk.w = pack_h2(x[6] * scale_log2, x[7] * scale_log2);
    *(uint4*)(Qs + qi * AD_QLD + 8 * lane) = pk;
  }

  const int mt = warp % MT, ph = warp / MT;
  const bool active = mt * 16 < NQ;
  const int r_lo = mt * 16 + g, r_hi = r_lo + 8;
  const int lim_lo = r_lo < NQ ? pos0 + r_lo / G : -1;
  const int lim_hi = r_hi < NQ ? pos0 + r_hi / G : -1;
  float o[32][4];
#pragma unroll
  for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  const uint8_t* kb = kc + kv_off(kvh, p_begin, bs, recb) - (size_t)p_begin * recb;
  const uint8_t* vb = vc + kv_off(kvh, p_begin, bs, recb) - (size_t)p_begin * recb;

  constexpr bool PIPE = KT == KV_Q4 || KT == KV_Q3;
  constexpr int RECB = KT == KV_Q3 ? 112 : 144;
  constexpr int RAWB = AD_BN * RECB;
  auto issue = [&](int kt, int st) {
    const int nch = min(AD_BN, pend - kt + 1) * (RECB / 16);
    for (int i = tid; i < nch; i += NTH) {
      cp_async16(raw + (st * 2 + 0) * RAWB + i * 16, kb + (size_t)kt * recb + i * 16);
      cp_async16(raw + (st * 2 + 1) * RAWB + i * 16, vb + (size_t)kt * recb + i * 16);
    }
    cp_async_commit();
  };
  if constexpr (PIPE) issue(p_begin, 0);
  int st = 0;
  for (int kt = p_begin; kt <= pend; kt += AD_BN, st ^= 1) {
    if constexpr (PIPE) {
      if (kt + AD_BN <= pend) issue(kt + AD_BN, st ^ 1);
      else cp_async_commit();
      cp_async_wait<1>();
    }
    __syncthreads();
    for (int i = tid; i < AD_BN * (FA_D / 8); i += NTH) {
      const int pi = i % AD_BN, c = i / AD_BN;
      const int p = kt + pi;
      uint4 pkk = make_uint4(0, 0, 0, 0), pkv = make_uint4(0, 0, 0, 0);
      if (p <= pend) {
        if constexpr (PIPE) {
          pkk = kv_read8h<KT>(raw + (st * 2 + 0) * RAWB + pi * RECB, c);
          pkv = kv_read8h<KT>(raw + (st * 2 + 1) * RAWB + pi * RECB, c);
        } else {
          pkk = kv_read8h<KT>(kb + (size_t)p * recb, c);
          pkv = kv_read8h<KT>(vb + (size_t)p * recb, c);
        }
      }
      *(uint4*)(Ks + pi * AD_QLD + 8 * c) = pkk;
      *(uint4*)(Vs + pi * AD_QLD + 8 * c) = pkv;
    }
    __syncthreads();
    if (!active) continue;
    const int pbase = ph * 16;  // this warp's 16 positions of the tile
    float s[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < FA_D / 16; kk++) {
      uint32_t a[4];
      const __half* qa = Qs + (mt * 16 + g) * AD_QLD + kk * 16 + tq * 2;
      a[0] = *(const uint32_t*)(qa);
      a[1] = *(const uint32_t*)(qa + 8 * AD_QLD);
      a[2] = *(const uint32_t*)(qa + 8);
      a[3] = *(const uint32_t*)(qa + 8 * AD_QLD + 8);
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        const __half* kp = Ks + (pbase + nt * 8 + g) * AD_QLD + kk * 16 + tq * 2;
        mma16816(s[nt], a, *(const uint32_t*)kp, *(const uint32_t*)(kp + 8));
      }
    }
    float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      const int p = kt + pbase + nt * 8 + tq * 2;
      if (p > lim_lo) s[nt][0] = -1e30f;
      if (p + 1 > lim_lo) s[nt][1] = -1e30f;
      if (p > lim_hi) s[nt][2] = -1e30f;
      if (p + 1 > lim_hi) s[nt][3] = -1e30f;
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
    m0 = mn0; m1 = mn1;
    l0 *= al0; l1 *= al1;
#pragma unroll
    for (int dn = 0; dn < 32; dn++) { o[dn][0] *= al0; o[dn][1] *= al0; o[dn][2] *= al1; o[dn][3] *= al1; }
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      s[nt][0] = s[nt][0] <= -1e29f ? 0.f : exp2f(s[nt][0] - m0);
      s[nt][1] = s[nt][1] <= -1e29f ? 0.f : exp2f(s[nt][1] - m0);
      s[nt][2] = s[nt][2] <= -1e29f ? 0.f : exp2f(s[nt][2] - m1);
      s[nt][3] = s[nt][3] <= -1e29f ? 0.f : exp2f(s[nt][3] - m1);
      l0 += s[nt][0] + s[nt][1];
      l1 += s[nt][2] + s[nt][3];
    }
#pragma unroll
    for (int j = 0; j < 1; j++) {
      uint32_t a[4];
      a[0] = pack_h2(s[2 * j][0], s[2 * j][1]);
      a[1] = pack_h2(s[2 * j][2], s[2 * j][3]);
      a[2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
      a[3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
      const int mi = lane >> 3, rr = lane & 7;
      const __half* vrow = Vs + (pbase + j * 16 + (mi & 1) * 8 + rr) * AD_QLD + (mi >> 1) * 8;
#pragma unroll
      for (int dn = 0; dn < 32; dn += 2) {
        uint32_t b[4];
        ldsm_x4_t(b, vrow + dn * 8);
        mma16816(o[dn], a, b[0], b[1]);
        mma16816(o[dn + 1], a, b[2], b[3]);
      }
    }
  }
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);

  // merge the two position halves of each m-tile (through shared memory), write the partials
  __syncthreads();
  float* xm = (float*)Ks;  // [mt][16 rows][2]  m, l
  float* xo = xm + MT * 16 * 2;  // [mt][16 rows][256]
  if (active && ph == 1) {
    if (tq == 0) {
      xm[(mt * 16 + g) * 2 + 0] = m0; xm[(mt * 16 + g) * 2 + 1] = l0;
      xm[(mt * 16 + g + 8) * 2 + 0] = m1; xm[(mt * 16 + g + 8) * 2 + 1] = l1;
    }
#pragma unroll
    for (int dn = 0; dn < 32; dn++) {
      const int d = dn * 8 + tq * 2;
      *(float2*)&xo[(mt * 16 + g) * 256 + d] = make_float2(o[dn][0], o[dn][1]);
      *(float2*)&xo[(mt * 16 + g + 8) * 256 + d] = make_float2(o[dn][2], o[dn][3]);
    }
  }
  __syncthreads();
  if (active && ph == 0) {
    const float LN2 = 0.6931471805599453f;
    float* stats = work + ((size_t)(sp * nkv + kvh) * NQ) * 258;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = mt * 16 + g + 8 * h;
      const float ma = h ? m1 : m0, la = h ? l1 : l0;
      const float mb = xm[row * 2 + 0], lb = xm[row * 2 + 1];
      const float M = fmaxf(ma, mb);
      const float wa = exp2f(ma - M), wb = exp2f(mb - M);
      if (row < NQ) {
        if (tq == 0) {
          stats[row * 258 + 0] = (M <= -1e29f) ? -INFINITY : M * LN2;
          stats[row * 258 + 1] = la * wa + lb * wb;
        }
#pragma unroll
        for (int dn = 0; dn < 32; dn++) {
          const int d = dn * 8 + tq * 2;
          const float2 ob = *(const float2*)&xo[row * 256 + d];
          const float oa0 = h ? o[dn][2] : o[dn][0], oa1 = h ? o[dn][3] : o[dn][1];
          *(float2*)&stats[row * 258 + 2 + d] = make_float2(oa0 * wa + ob.x * wb, oa1 * wa + ob.y * wb);
        }
      }
    }
  }
}

static int g_num_sms = 68;
void set_num_sms(int n) { g_num_sms = n > 0 ? n : 68; }
static int mma_split(int L, int nkv) {
  // largest split (multiple of the tile, divides KV_BLOCK) that still gives >= ~2 CTAs per SM
  int split = 2048;
  while (split > 256 && (long long)nkv * ((L + split - 1) / split) < 2 * g_num_sms) split /= 2;
  return split;
}
size_t attn_decode_work_bytes_auto(int max_ctx, int nh, int nkv, int hd, int tokens) {
  static const int mode = getenv("QE_ATT") ? atoi(getenv("QE_ATT")) : -1;
  if (mode == 0) return attn_decode_work_bytes(max_ctx, nh, hd, tokens);
  size_t mx = 0;
  for (int L = 1; L <= max_ctx + tokens; L = L < 256 ? 256 : L + 256) {  // nsplit*T is not monotone in L
    const int split = mma_split(L, nkv);
    mx = std::max(mx, (size_t)((L + split - 1) / split) * nh * tokens * 258 * sizeof(float));
  }
  return mx;
}

void attn_decode_mma(int kvtype, const void* kcache, const void* vcache, const float* q, int qhs, int qts,
                     float* out, int nh, int nkv, int hd, int pos0, int T, float scale, float* work,
                     size_t work_bytes, KVMap bs, cudaStream_t s) {
  QE_ASSERT(hd == FA_D && nh / nkv * T <= 48);
  const int L = pos0 + T;
  const int split = mma_split(L, nkv);
  const int nsplit = (L + split - 1) / split;
  QE_ASSERT((size_t)nsplit * nh * T * 258 * sizeof(float) <= work_bytes);
  const int recb = (int)kv_bytes_per_token_head(kvtype, hd);
  const int NQ = nh / nkv * T;
  const int MT = NQ > 32 ? 3 : 2;
  // Q rows + K/V tiles + raw staging; the end-of-kernel merge reuses everything after Q
  const size_t sh_tile = ((size_t)(16 * MT + 2 * AD_BN) * AD_QLD) * sizeof(__half) +
                    (kvtype == KV_Q4 ? (size_t)2 * 2 * AD_BN * 144 : kvtype == KV_Q3 ? (size_t)2 * 2 * AD_BN * 112 : 0);
  const size_t sh_merge = (size_t)16 * MT * AD_QLD * sizeof(__half) + (size_t)MT * 16 * (2 + 256) * sizeof(float);
  const size_t sh = std::max(sh_tile, sh_merge);
  const float sl2 = scale * 1.4426950408889634f;
  const dim3 grid(nkv, nsplit);
  const uint8_t* kc = (const uint8_t*)kcache;
  const uint8_t* vc = (const uint8_t*)vcache;
  switch (kvtype) {
#define CASE(KT)                                                                                                       \
  case KT: {                                                                                                           \
    static bool init = false;                                                                                          \
    if (!init) {                                                                                                       \
      const int mx = (int)std::max((size_t)(48 + 2 * AD_BN) * AD_QLD * 2 + 2 * 2 * AD_BN * 144,                       \
                                   (size_t)48 * AD_QLD * 2 + 48 * (2 + 256) * 4);                                       \
      CUDA_CHECK(cudaFuncSetAttribute(k_attn_dec_mma<KT, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx));        \
      CUDA_CHECK(cudaFuncSetAttribute(k_attn_dec_mma<KT, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx));        \
      init = true;                                                                                                     \
    }                                                                                                                  \
    if (MT == 2)                                                                                                       \
      k_attn_dec_mma<KT, 2><<<grid, 128, sh, s>>>(kc, vc, q, qhs, qts, nh, nkv, pos0, T, sl2, work, bs, recb, split);  \
    else                                                                                                               \
      k_attn_dec_mma<KT, 3><<<grid, 192, sh, s>>>(kc, vc, q, qhs, qts, nh, nkv, pos0, T, sl2, work, bs, recb, split);  \
    k_attn_combine<KT><<<nkv * NQ, 256, 0, s>>>(work, out, nh, nkv, T, NQ, pos0, split);                              \
    break;                                                                                                             \
  }
    CASE(KV_F16) CASE(KV_Q8) CASE(KV_Q4) CASE(KV_Q3)
#undef CASE
  }
}

// =============================================================================================
// Gated DeltaNet
// =============================================================================================
__global__ void k_ssm_conv(const float* __restrict__ x, int xts, float* conv_state,
                           const float* __restrict__ w, float* __restrict__ y, int C, int T,
                           float* hist, int snap_t, float* snap) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float win[3];
#pragma unroll
  for (int i = 0; i < 3; i++) win[i] = conv_state[(size_t)i * C + c];
  if (hist) {
#pragma unroll
    for (int i = 0; i < 3; i++) hist[(size_t)i * C + c] = win[i];
  }
  const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  for (int t = 0; t < T; t++) {
    const float xi = x[(size_t)t * xts + c];
    if (hist) hist[(size_t)(3 + t) * C + c] = xi;
    const float o = w0 * win[0] + w1 * win[1] + w2 * win[2] + w3 * xi;
    y[(size_t)t * C + c] = o / (1.f + __expf(-o));
    win[0] = win[1];
    win[1] = win[2];
    win[2] = xi;
    if (t == snap_t) {  // checkpoint: conv state after token snap_t (host-mapped memory)
#pragma unroll
      for (int i = 0; i < 3; i++) snap[(size_t)i * C + c] = win[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 3; i++) conv_state[(size_t)i * C + c] = win[i];
}
void ssm_conv(const float* x, int xts, float* conv_state, const float* w, float* y, int C, int T,
              float* hist, cudaStream_t s, int snap_t, float* snap) {
  k_ssm_conv<<<cdiv(C, 256), 256, 0, s>>>(x, xts, conv_state, w, y, C, T, hist, snap ? snap_t : -1, snap);
}
__global__ void k_conv_rollback_b(const GdnBatch* __restrict__ tab, int C, int n_acc) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const GdnBatch& b = tab[blockIdx.y];
#pragma unroll
  for (int i = 0; i < 3; i++) b.conv_state[(size_t)i * C + c] = b.conv_hist[(size_t)(n_acc + i) * C + c];
}
__global__ void k_conv_rollback(float* conv_state, const float* hist, int C, int n_acc) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
#pragma unroll
  for (int i = 0; i < 3; i++) conv_state[(size_t)i * C + c] = hist[(size_t)(n_acc + i) * C + c];
}
void ssm_conv_rollback(float* conv_state, const float* hist, int C, int n_acc, cudaStream_t s) {
  k_conv_rollback<<<cdiv(C, 256), 256, 0, s>>>(conv_state, hist, C, n_acc);
}

// one warp per (token, head-slot): slots [0,nk) = q heads, [nk,2nk) = k heads, [2nk, 2nk+nv) = g/beta
__global__ void k_gdn_prep(float* qkv, int qts, const float* __restrict__ alpha,
                           const float* __restrict__ beta_raw, int abts,
                           const float* __restrict__ dt_bias, const float* __restrict__ a,
                           float* __restrict__ g, float* __restrict__ beta, int nk, int nv, int dk,
                           int T, float eps) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const int per = 2 * nk + 1;
  if (wid >= T * per) return;
  const int t = wid / per, slot = wid % per;
  if (slot < 2 * nk) {
    float* p = qkv + (size_t)t * qts + (size_t)slot * dk;
    float ss = 0.f;
    for (int i = lane; i < dk; i += 32) ss += p[i] * p[i];
    ss = warp_sum(ss);
    const float r = rsqrtf(ss + eps);
    for (int i = lane; i < dk; i += 32) p[i] *= r;
  } else {
    for (int h = lane; h < nv; h += 32) {
      const float al = alpha[(size_t)t * abts + h] + dt_bias[h];
      const float sp = al > 20.f ? al : log1pf(__expf(al));
      g[(size_t)t * nv + h] = sp * a[h];
      beta[(size_t)t * nv + h] = 1.f / (1.f + __expf(-beta_raw[(size_t)t * abts + h]));
    }
  }
}
void gdn_prep(float* qkv, int qts, const float* alpha, const float* beta_raw, int abts,
              const float* dt_bias, const float* a, float* g, float* beta, int nk, int nv, int dk,
              int T, float eps, cudaStream_t s) {
  const int nw = T * (2 * nk + 1);
  k_gdn_prep<<<cdiv(nw, 8), 256, 0, s>>>(qkv, qts, alpha, beta_raw, abts, dt_bias, a, g, beta, nk, nv,
                                         dk, T, eps);
}

// grid (nv, dv/32): CTA = 8 warps x 4 columns; lane holds rows 4*lane..4*lane+3 (dk = 128)
// GDN state element access: fp32, or fp16 (QE_STATE16: half the state traffic and VRAM)
template <bool H>
static __device__ __forceinline__ float4 state_ld(const float* p, size_t i) {
  if constexpr (H) {
    const uint2 u = *(const uint2*)((const __half*)p + i);
    const float2 a = __half22float2(*(const __half2*)&u.x), b = __half22float2(*(const __half2*)&u.y);
    return make_float4(a.x, a.y, b.x, b.y);
  } else {
    return *(const float4*)(p + i);
  }
}
template <bool H>
static __device__ __forceinline__ void state_st(float* p, size_t i, float4 v) {
  if constexpr (H) {
    const __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
    uint2 u;
    u.x = *(const uint32_t*)&a;
    u.y = *(const uint32_t*)&b;
    *(uint2*)((__half*)p + i) = u;
  } else {
    *(float4*)(p + i) = v;
  }
}
template <bool H>
__global__ void __launch_bounds__(256)
    k_gdn_rec(const float* __restrict__ qkv, int qts, const float* __restrict__ g,
              const float* __restrict__ beta, const float* state, float* state_out,
              float* __restrict__ out, int nk, int nv, int T, int snap_t, float* snap,
              const GdnBatch* __restrict__ tab = nullptr) {
  constexpr int DK = 128, DV = 128;
  if (tab) {  // batched rollback: one launch for all GDN layers (blockIdx.z = layer)
    const GdnBatch& b = tab[blockIdx.z];
    qkv = b.qkv; g = b.g; beta = b.beta; state = b.state; state_out = b.state;
  }
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j0 = blockIdx.y * 32 + warp * 4;
  const int hk = h % nk;
  float S[4][4];  // [column jj][row r]
  const size_t hoff = (size_t)h * DV * DK;
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    const float4 v = state_ld<H>(state, hoff + (size_t)(j0 + jj) * DK + 4 * lane);
    S[jj][0] = v.x; S[jj][1] = v.y; S[jj][2] = v.z; S[jj][3] = v.w;
  }
  const float qscale = rsqrtf((float)DK);
  // the per-token inputs do not depend on the state: prefetch PF tokens ahead so the sequential
  // chain only sees the FMAs and warp reductions, not the memory latency
  constexpr int PF = 4;
  float4 pq[PF], pk[PF], pv[PF];
  float pg[PF], pb[PF];
  auto fetch = [&](int t, int slot) {
    if (t < T) {
      const float* base = qkv + (size_t)t * qts;
      pq[slot] = __ldg((const float4*)(base + hk * DK + 4 * lane));
      pk[slot] = __ldg((const float4*)(base + nk * DK + hk * DK + 4 * lane));
      pv[slot] = __ldg((const float4*)(base + 2 * nk * DK + h * DV + j0));
      pg[slot] = __ldg(g + (size_t)t * nv + h);
      pb[slot] = __ldg(beta + (size_t)t * nv + h);
    }
  };
#pragma unroll
  for (int i = 0; i < PF; i++) fetch(i, i);
  for (int t0 = 0; t0 < T; t0 += PF) {
#pragma unroll
  for (int u = 0; u < PF; u++) {
    const int t = t0 + u;
    if (t >= T) break;
    const float4 q4 = pq[u], k4 = pk[u], v4 = pv[u];
    const float decay = __expf(pg[u]);
    const float b = pb[u];
    fetch(t + PF, u);
    const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
    const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
    const float vj[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
      float sk = 0.f;
#pragma unroll
      for (int r = 0; r < 4; r++) {
        S[jj][r] *= decay;
        sk += S[jj][r] * kk[r];
      }
      sk = warp_sum(sk);
      const float delta = (vj[jj] - sk) * b;
      float o = 0.f;
#pragma unroll
      for (int r = 0; r < 4; r++) {
        S[jj][r] += kk[r] * delta;
        o += S[jj][r] * qq[r];
      }
      if (out) {
        o = warp_sum(o);
        if (lane == 0) out[(size_t)t * nv * DV + h * DV + j0 + jj] = o * qscale;
      }
    }
    if (t == snap_t) {  // checkpoint: state after token snap_t (host-mapped memory)
      float* sn = snap + (size_t)h * DV * DK;
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
        *(float4*)(sn + (size_t)(j0 + jj) * DK + 4 * lane) = make_float4(S[jj][0], S[jj][1], S[jj][2], S[jj][3]);
    }
  }
  }
  if (!state_out) return;  // verify pass: state is read-only
#pragma unroll
  for (int jj = 0; jj < 4; jj++)
    state_st<H>(state_out, hoff + (size_t)(j0 + jj) * DK + 4 * lane, make_float4(S[jj][0], S[jj][1], S[jj][2], S[jj][3]));
}
void gdn_rollback_batched(const GdnBatch* tab, int nl, int qts, int nk, int nv, int dk, int n_keep, int C, bool conv,
                          cudaStream_t s, bool half) {
  QE_ASSERT(dk == 128);
  if (half)
    k_gdn_rec<true><<<dim3(nv, 128 / 32, nl), 256, 0, s>>>(nullptr, qts, nullptr, nullptr, nullptr, nullptr, nullptr, nk, nv,
                                                         n_keep, -1, nullptr, tab);
  else
    k_gdn_rec<false><<<dim3(nv, 128 / 32, nl), 256, 0, s>>>(nullptr, qts, nullptr, nullptr, nullptr, nullptr, nullptr, nk, nv,
                                                          n_keep, -1, nullptr, tab);
  if (conv) k_conv_rollback_b<<<dim3(cdiv(C, 256), nl), 256, 0, s>>>(tab, C, n_keep);
}
void gdn_recurrence(const float* qkv, int qts, const float* g, const float* beta, const float* state_in,
                    float* state_out, float* out, int nk, int nv, int dk, int T, cudaStream_t s, int snap_t,
                    float* snap, bool half) {
  QE_ASSERT(dk == 128);
  if (half)
    k_gdn_rec<true><<<dim3(nv, 128 / 32), 256, 0, s>>>(qkv, qts, g, beta, state_in, state_out, out, nk, nv, T,
                                                     snap ? snap_t : -1, snap);
  else
    k_gdn_rec<false><<<dim3(nv, 128 / 32), 256, 0, s>>>(qkv, qts, g, beta, state_in, state_out, out, nk, nv, T,
                                                      snap ? snap_t : -1, snap);
}

__global__ void k_gdn_gated_norm(float* o, const float* __restrict__ z, int zts,
                                 const float* __restrict__ w, int nv, int dv, int T, float eps,
                                 __half* o16, int8_t* xq, float* xd) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= T * nv) return;
  const int t = wid / nv, h = wid % nv;
  float* p = o + (size_t)t * nv * dv + (size_t)h * dv;
  const float* zz = z + (size_t)t * zts + (size_t)h * dv;
  float ss = 0.f;
  for (int i = lane; i < dv; i += 32) ss += p[i] * p[i];
  ss = warp_sum(ss);
  const float r = rsqrtf(ss / dv + eps);
  for (int i = lane; i < dv; i += 32) {
    const float zv = zz[i];
    const float v = p[i] * r * w[i] * (zv / (1.f + __expf(-zv)));
    p[i] = v;
    const size_t gi = (size_t)t * nv * dv + (size_t)h * dv + i;
    if (o16) o16[gi] = __float2half(v);
    if (xq) {  // int8-per-32 quantization for the following matvec (dv % 32 == 0: a warp step = a group)
      const float amax = warp_max(fabsf(v));
      const float d = amax / 127.f;
      xq[gi] = (int8_t)__float2int_rn(d > 0.f ? v / d : 0.f);
      if (lane == 0) xd[gi / 32] = d;
    }
  }
}
void gdn_gated_norm(float* o, const float* z, int zts, const float* w, int nv, int dv, int T,
                    float eps, cudaStream_t s, __half* o16, int8_t* xq, float* xd) {
  k_gdn_gated_norm<<<cdiv(T * nv, 8), 256, 0, s>>>(o, z, zts, w, nv, dv, T, eps, o16, xq, xd);
}

// =============================================================================================
// argmax
// =============================================================================================
__global__ void k_argmax(const float* __restrict__ x, int n, int* idx) {
  const float* r = x + (size_t)blockIdx.x * n;
  float best = -INFINITY;
  int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = r[i];
    if (v > best) { best = v; bi = i; }
  }
  __shared__ float sv[32];
  __shared__ int si[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
  }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { sv[w] = best; si[w] = bi; }
  __syncthreads();
  if (w == 0) {
    const int nw = blockDim.x >> 5;
    best = l < nw ? sv[l] : -INFINITY;
    bi = l < nw ? si[l] : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (l == 0) idx[blockIdx.x] = bi;
  }
}
void argmax_rows(const float* logits, int n, int rows, int* idx, cudaStream_t s) {
  k_argmax<<<rows, 1024, 0, s>>>(logits, n, idx);
}

// softmax probability of the argmax (idx computed by k_argmax just before)
__global__ void k_max_prob(const float* __restrict__ x, int n, const int* __restrict__ idx, float* prob) {
  __shared__ float sh[32];
  const float m = x[idx[0]];
  float se = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) se += __expf(x[i] - m);
  se = block_sum(se, sh);
  if (threadIdx.x == 0) prob[0] = 1.f / se;
}
void argmax_prob(const float* logits, int n, int* idx, float* prob, cudaStream_t s) {
  k_argmax<<<1, 1024, 0, s>>>(logits, n, idx);
  k_max_prob<<<1, 1024, 0, s>>>(logits, n, idx, prob);
}


// =============================================================================================
// sensenova_loop residual streams (mHC, two streams).  route: RMS over the two streams together,
// 8 router logits (fp32 weights [8][2E] + bias): sublayer input pre0*a + pre1*b (sigmoids), output
// gates post = 2*sigmoid, and a 2x2 mixing matrix made doubly stochastic by 20 log-space Sinkhorn
// passes.  merge: a' = m00 a + m01 b + post0 x, b' = m10 a + m11 b + post1 x.  (GuideboardLabs'
// ggml_mhc_route / ggml_mhc_merge, one block per token.)
// =============================================================================================
__global__ void k_mhc_route(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ w,
                            const float* __restrict__ bias, int d, float* __restrict__ branch, float* __restrict__ pk) {
  const int t = blockIdx.x;
  const float* va = a + (size_t)t * d;
  const float* vb = b + (size_t)t * d;
  float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // sum of squares, 8 dots
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const float xa = va[i], xb = vb[i];
    acc[0] += xa * xa + xb * xb;
#pragma unroll
    for (int k = 0; k < 8; k++) acc[1 + k] += w[(size_t)k * 2 * d + i] * xa + w[(size_t)k * 2 * d + d + i] * xb;
  }
  __shared__ float red[9][32];
  __shared__ float g[4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
  for (int k = 0; k < 9; k++) {
    const float v = warp_sum(acc[k]);
    if (lane == 0) red[k][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot[9];
    for (int k = 0; k < 9; k++) {
      tot[k] = 0.f;
      for (int j = 0; j < nw; j++) tot[k] += red[k][j];
    }
    const float rms = 1.f / sqrtf(tot[0] / (2.f * d) + 1e-6f);
    float lg[8];
    for (int k = 0; k < 8; k++) lg[k] = tot[1 + k] * rms + bias[k];
    float m[2][2] = {{lg[4], lg[5]}, {lg[6], lg[7]}};
    for (int it = 0; it < 20; it++) {
      for (int r = 0; r < 2; r++) {
        const float mx = fmaxf(m[r][0], m[r][1]), lse = mx + logf(expf(m[r][0] - mx) + expf(m[r][1] - mx));
        m[r][0] -= lse; m[r][1] -= lse;
      }
      for (int cc = 0; cc < 2; cc++) {
        const float mx = fmaxf(m[0][cc], m[1][cc]), lse = mx + logf(expf(m[0][cc] - mx) + expf(m[1][cc] - mx));
        m[0][cc] -= lse; m[1][cc] -= lse;
      }
    }
    g[0] = 1.f / (1.f + expf(-lg[0]));
    g[1] = 1.f / (1.f + expf(-lg[1]));
    float* p = pk + (size_t)t * 8;
    p[0] = 2.f / (1.f + expf(-lg[2]));
    p[1] = 2.f / (1.f + expf(-lg[3]));
    p[2] = expf(m[0][0]); p[3] = expf(m[0][1]); p[4] = expf(m[1][0]); p[5] = expf(m[1][1]);
  }
  __syncthreads();
  const float p0 = g[0], p1 = g[1];
  for (int i = threadIdx.x; i < d; i += blockDim.x) branch[(size_t)t * d + i] = p0 * va[i] + p1 * vb[i];
}
void mhc_route(const float* a, const float* b, const float* w, const float* bias, int d, int T, float* branch, float* pk,
               cudaStream_t s) {
  k_mhc_route<<<T, 512, 0, s>>>(a, b, w, bias, d, branch, pk);
}
__global__ void k_mhc_merge(float* a, float* b, const float* __restrict__ x, const float* __restrict__ pk, int d, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* p = pk + (i / d) * 8;
  const float va = a[i], vb = b[i], vx = x ? x[i] : 0.f;
  a[i] = p[2] * va + p[3] * vb + p[0] * vx;
  b[i] = p[4] * va + p[5] * vb + p[1] * vx;
}
void mhc_merge(float* a, float* b, const float* x, const float* pk, int d, int T, cudaStream_t s) {
  const int64_t n = (int64_t)T * d;
  k_mhc_merge<<<cdiv(n, 256), 256, 0, s>>>(a, b, x, pk, d, n);
}
__global__ void k_mhc_leave(float* y, const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 0.5f * (a[i] + b[i]);
}
void mhc_leave(float* y, const float* a, const float* b, int64_t n, cudaStream_t s) {
  k_mhc_leave<<<cdiv(n, 256), 256, 0, s>>>(y, a, b, n);
}
}  // namespace qk
