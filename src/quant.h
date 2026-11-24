// Block layouts and reference (de)quantization of GGUF quant formats.
// Shared between host C++ and CUDA device code.  All formats used by the
// target model are handled at 32-element sub-block granularity ("ib32"),
// which is the unit both the GPU dp4a kernels and the CPU AVX2 kernels use.
#pragma once
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#include <intrin.h>  // __popcnt for the MSVC host build
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QE_HD __host__ __device__ __forceinline__
#else
#define QE_HD inline
#endif

namespace qtab {
#define QT_DECL(type, name, size) static const type name[size]
#include "quant_tables.inc"
#undef QT_DECL
}  // namespace qtab

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace qtab_dev {
#define QT_DECL(type, name, size) static __device__ const type name[size]
#include "quant_tables.inc"
#undef QT_DECL
}  // namespace qtab_dev
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define QTAB(name) (qtab_dev::name)
#else
#define QTAB(name) (qtab::name)
#endif
#else
#define QTAB(name) (qtab::name)
#endif

#define QK_K 256

struct blk_q2_K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
struct blk_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct blk_q5_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct blk_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct blk_q8_0 { uint16_t d; int8_t qs[32]; };
struct blk_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct blk_iq2_xxs { uint16_t d; uint16_t qs[32]; };
struct blk_iq2_xs { uint16_t d; uint16_t qs[32]; uint8_t scales[8]; };
struct blk_iq2_s { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; };
struct blk_iq3_xxs { uint16_t d; uint8_t qs[96]; };
struct blk_iq3_s { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; };
struct blk_iq1_s { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
struct blk_iq1_m { uint8_t qs[32]; uint8_t qh[16]; uint8_t scales[8]; };
struct blk_iq4_xs { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };

static_assert(sizeof(blk_q2_K) == 84, "");
static_assert(sizeof(blk_q4_K) == 144, "");
static_assert(sizeof(blk_q6_K) == 210, "");
static_assert(sizeof(blk_iq2_xxs) == 66, "");
static_assert(sizeof(blk_iq2_xs) == 74, "");
static_assert(sizeof(blk_iq2_s) == 82, "");
static_assert(sizeof(blk_iq3_xxs) == 98, "");
static_assert(sizeof(blk_iq3_s) == 110, "");
static_assert(sizeof(blk_iq1_s) == 50, "");
static_assert(sizeof(blk_iq1_m) == 56, "");
static_assert(sizeof(blk_iq4_xs) == 136, "");

#define IQ1S_DELTA 0.125f

QE_HD float qe_h2f(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000) << 16;
  uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
  if (e == 0) {
    if (m == 0) {
      bits = s;
    } else {  // subnormal
      e = 1;
      while (!(m & 0x400)) { m <<= 1; e--; }
      m &= 0x3ff;
      bits = s | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 31) {
    bits = s | 0x7f800000u | (m << 13);
  } else {
    bits = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
QE_HD float qe_bf2f(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

QE_HD void q4k_scale_min(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63; m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

QE_HD uint8_t iq2_signs8(uint32_t s7) {  // == ksigns_iq2xs[s7]
#ifdef __CUDA_ARCH__
  return (uint8_t)(s7 | ((__popc(s7) & 1) << 7));
#elif defined(_MSC_VER)
  return (uint8_t)(s7 | ((__popcnt(s7) & 1) << 7));  // nvcc's MSVC host pass has no __builtin_popcount
#else
  return (uint8_t)(s7 | ((__builtin_popcount(s7) & 1) << 7));
#endif
}

// Reference dequantization of the 32 values of sub-block ib32 (0..7) of a
// 256-element super block.  `blk` points at the super block.
template <int T>
QE_HD void dequant32(const uint8_t* blk, int ib, float* y);

template <> QE_HD void dequant32<10>(const uint8_t* p, int g, float* y) {  // Q2_K
  const blk_q2_K* x = (const blk_q2_K*)p;
  const float d = qe_h2f(x->d), dmin = qe_h2f(x->dmin);
  const uint8_t* q = x->qs + (g / 4) * 32;
  const int shift = 2 * (g % 4);
  for (int h = 0; h < 2; h++) {
    const uint8_t sc = x->scales[2 * g + h];
    const float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
    for (int l = 0; l < 16; l++) y[16 * h + l] = dl * ((q[16 * h + l] >> shift) & 3) - ml;
  }
}
template <> QE_HD void dequant32<12>(const uint8_t* p, int g, float* y) {  // Q4_K
  const blk_q4_K* x = (const blk_q4_K*)p;
  uint8_t sc, m;
  q4k_scale_min(g, x->scales, sc, m);
  const float d = qe_h2f(x->d) * sc, mn = qe_h2f(x->dmin) * m;
  const uint8_t* q = x->qs + (g / 2) * 32;
  for (int l = 0; l < 32; l++) y[l] = d * ((g & 1) ? (q[l] >> 4) : (q[l] & 0xF)) - mn;
}
template <> QE_HD void dequant32<13>(const uint8_t* p, int g, float* y) {  // Q5_K
  const blk_q5_K* x = (const blk_q5_K*)p;
  uint8_t sc, m;
  q4k_scale_min(g, x->scales, sc, m);
  const float d = qe_h2f(x->d) * sc, mn = qe_h2f(x->dmin) * m;
  const uint8_t* q = x->qs + (g / 2) * 32;
  for (int l = 0; l < 32; l++) {
    const int lo = (g & 1) ? (q[l] >> 4) : (q[l] & 0xF);
    y[l] = d * (lo + 16 * ((x->qh[l] >> g) & 1)) - mn;
  }
}
// Q3_K: 16 signed 6-bit sub-scales packed in 12 bytes (ggml layout)
QE_HD void q3k_scales(const uint8_t* s12, int8_t* out) {
  uint32_t aux[4];
  memcpy(aux, s12, 12);
  const uint32_t tmp = aux[2], k1 = 0x03030303, k2 = 0x0f0f0f0f;
  aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
  aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
  aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
  aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
  memcpy(out, aux, 16);
}
template <> QE_HD void dequant32<11>(const uint8_t* p, int g, float* y) {  // Q3_K
  const blk_q3_K* x = (const blk_q3_K*)p;
  int8_t sc[16];
  q3k_scales(x->scales, sc);
  const float d = qe_h2f(x->d);
  const uint8_t* q = x->qs + 32 * (g / 4);
  const int shift = 2 * (g % 4);
  for (int h = 0; h < 2; h++) {
    const float dl = d * (sc[2 * g + h] - 32);
    for (int l = 0; l < 16; l++) {
      const int i = 16 * h + l;
      y[i] = dl * (((q[i] >> shift) & 3) - ((x->hmask[i] >> g) & 1 ? 0 : 4));
    }
  }
}
// Q8_0: 32-element blocks; a "super block" here is 8 consecutive blocks (272 bytes)
template <> QE_HD void dequant32<8>(const uint8_t* p, int g, float* y) {  // Q8_0
  const blk_q8_0* x = (const blk_q8_0*)(p + 34 * g);
  const float d = qe_h2f(x->d);
  for (int l = 0; l < 32; l++) y[l] = d * x->qs[l];
}
template <> QE_HD void dequant32<14>(const uint8_t* p, int g, float* y) {  // Q6_K
  const blk_q6_K* x = (const blk_q6_K*)p;
  const float d = qe_h2f(x->d);
  const int h = g / 4, r = g % 4;
  const uint8_t* ql = x->ql + 64 * h + (r & 1) * 32;
  const uint8_t* qh = x->qh + 32 * h;
  const int8_t* sc = x->scales + 8 * h + 2 * r;
  for (int l = 0; l < 32; l++) {
    const int lo = (r >= 2) ? (ql[l] >> 4) : (ql[l] & 0xF);
    const int q = (lo | (((qh[l] >> (2 * r)) & 3) << 4)) - 32;
    y[l] = d * sc[l / 16] * q;
  }
}
template <> QE_HD void dequant32<16>(const uint8_t* p, int ib, float* y) {  // IQ2_XXS
  const blk_iq2_xxs* x = (const blk_iq2_xxs*)p;
  uint32_t a0, a1;
  memcpy(&a0, (const uint8_t*)x->qs + 8 * ib, 4);
  memcpy(&a1, (const uint8_t*)x->qs + 8 * ib + 4, 4);
  const float db = qe_h2f(x->d) * (0.5f + (a1 >> 28)) * 0.25f;
  for (int l = 0; l < 4; l++) {
    const uint8_t* grid = (const uint8_t*)(QTAB(iq2xxs_grid) + ((a0 >> (8 * l)) & 255));
    const uint8_t s = iq2_signs8((a1 >> (7 * l)) & 127);
    for (int j = 0; j < 8; j++) y[8 * l + j] = db * grid[j] * ((s >> j) & 1 ? -1.f : 1.f);
  }
}
template <> QE_HD void dequant32<17>(const uint8_t* p, int ib, float* y) {  // IQ2_XS
  const blk_iq2_xs* x = (const blk_iq2_xs*)p;
  const float d = qe_h2f(x->d);
  for (int l = 0; l < 4; l++) {
    const float db = d * (0.5f + ((x->scales[ib] >> (4 * (l / 2))) & 0xf)) * 0.25f;
    const uint16_t q = x->qs[4 * ib + l];
    const uint8_t* grid = (const uint8_t*)(QTAB(iq2xs_grid) + (q & 511));
    const uint8_t s = iq2_signs8(q >> 9);
    for (int j = 0; j < 8; j++) y[8 * l + j] = db * grid[j] * ((s >> j) & 1 ? -1.f : 1.f);
  }
}
template <> QE_HD void dequant32<22>(const uint8_t* p, int ib, float* y) {  // IQ2_S
  const blk_iq2_s* x = (const blk_iq2_s*)p;
  const float d = qe_h2f(x->d);
  const uint8_t* qs = x->qs + 4 * ib;
  const uint8_t* signs = x->qs + 32 + 4 * ib;
  for (int l = 0; l < 4; l++) {
    const float db = d * (0.5f + ((x->scales[ib] >> (4 * (l / 2))) & 0xf)) * 0.25f;
    const uint8_t* grid = (const uint8_t*)(QTAB(iq2s_grid) + (qs[l] | ((x->qh[ib] << (8 - 2 * l)) & 0x300)));
    for (int j = 0; j < 8; j++) y[8 * l + j] = db * grid[j] * ((signs[l] >> j) & 1 ? -1.f : 1.f);
  }
}
template <> QE_HD void dequant32<18>(const uint8_t* p, int ib, float* y) {  // IQ3_XXS
  const blk_iq3_xxs* x = (const blk_iq3_xxs*)p;
  const uint8_t* qs = x->qs + 8 * ib;
  uint32_t a;
  memcpy(&a, x->qs + 64 + 4 * ib, 4);
  const float db = qe_h2f(x->d) * (0.5f + (a >> 28)) * 0.5f;
  for (int l = 0; l < 4; l++) {
    const uint8_t s = iq2_signs8((a >> (7 * l)) & 127);
    const uint8_t* g1 = (const uint8_t*)(QTAB(iq3xxs_grid) + qs[2 * l + 0]);
    const uint8_t* g2 = (const uint8_t*)(QTAB(iq3xxs_grid) + qs[2 * l + 1]);
    for (int j = 0; j < 4; j++) {
      y[8 * l + j] = db * g1[j] * ((s >> j) & 1 ? -1.f : 1.f);
      y[8 * l + j + 4] = db * g2[j] * ((s >> (j + 4)) & 1 ? -1.f : 1.f);
    }
  }
}
template <> QE_HD void dequant32<21>(const uint8_t* p, int ib, float* y) {  // IQ3_S
  const blk_iq3_s* x = (const blk_iq3_s*)p;
  const float db = qe_h2f(x->d) * (1 + 2 * ((x->scales[ib / 2] >> (4 * (ib & 1))) & 0xf));
  const uint8_t* qs = x->qs + 8 * ib;
  const uint8_t qh = x->qh[ib];
  const uint8_t* signs = x->signs + 4 * ib;
  for (int l = 0; l < 4; l++) {
    const uint8_t* g1 = (const uint8_t*)(QTAB(iq3s_grid) + (qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)));
    const uint8_t* g2 = (const uint8_t*)(QTAB(iq3s_grid) + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
    for (int j = 0; j < 4; j++) {
      y[8 * l + j] = db * g1[j] * ((signs[l] >> j) & 1 ? -1.f : 1.f);
      y[8 * l + j + 4] = db * g2[j] * ((signs[l] >> (j + 4)) & 1 ? -1.f : 1.f);
    }
  }
}
template <> QE_HD void dequant32<19>(const uint8_t* p, int ib, float* y) {  // IQ1_S
  const blk_iq1_s* x = (const blk_iq1_s*)p;
  const uint16_t qh = x->qh[ib];
  const float dl = qe_h2f(x->d) * (2 * ((qh >> 12) & 7) + 1);
  const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
  for (int l = 0; l < 4; l++) {
    const int8_t* grid = (const int8_t*)(QTAB(iq1s_grid) + (x->qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)));
    for (int j = 0; j < 8; j++) y[8 * l + j] = dl * (grid[j] + delta);
  }
}
QE_HD float iq1m_d(const blk_iq1_m* x) {
  uint16_t sc[4];
  memcpy(sc, x->scales, 8);
  const uint16_t u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
  return qe_h2f(u);
}
template <> QE_HD void dequant32<29>(const uint8_t* p, int ib, float* y) {  // IQ1_M
  const blk_iq1_m* x = (const blk_iq1_m*)p;
  uint16_t sc[4];
  memcpy(sc, x->scales, 8);
  const float d = iq1m_d(x);
  const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
  const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
  const uint8_t* qs = x->qs + 4 * ib;
  const uint8_t* qh = x->qh + 2 * ib;
  const uint16_t idx[4] = {(uint16_t)(qs[0] | ((qh[0] << 8) & 0x700)), (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700)),
                           (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700)), (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700))};
  const float delta[4] = {qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA,
                          qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA};
  for (int l = 0; l < 4; l++) {
    const int8_t* grid = (const int8_t*)(QTAB(iq1s_grid) + idx[l]);
    const float dl = l < 2 ? dl1 : dl2;
    for (int j = 0; j < 8; j++) y[8 * l + j] = dl * (grid[j] + delta[l]);
  }
}
template <> QE_HD void dequant32<23>(const uint8_t* p, int ib, float* y) {  // IQ4_XS
  const blk_iq4_xs* x = (const blk_iq4_xs*)p;
  const int ls = ((x->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((x->scales_h >> (2 * ib)) & 3) << 4);
  const float dl = qe_h2f(x->d) * (ls - 32);
  const uint8_t* qs = x->qs + 16 * ib;
  for (int j = 0; j < 16; j++) {
    y[j] = dl * QTAB(kvalues_iq4nl)[qs[j] & 0xf];
    y[j + 16] = dl * QTAB(kvalues_iq4nl)[qs[j] >> 4];
  }
}

template <> QE_HD void dequant32<20>(const uint8_t* p, int ib, float* y) {  // IQ4_NL: 8 blocks of 32 per "super-block"
  const uint8_t* b = p + 18 * ib;
  uint16_t d;
  memcpy(&d, b, 2);
  const float dl = qe_h2f(d);
  for (int j = 0; j < 16; j++) {
    y[j] = dl * QTAB(kvalues_iq4nl)[b[2 + j] & 0xf];
    y[j + 16] = dl * QTAB(kvalues_iq4nl)[b[2 + j] >> 4];
  }
}
// NVFP4 (ggml): 64-element blocks of 4 UE4M3 scales (one per 16) + 32 bytes of E2M1 nibbles; sub-block s holds
// elements 16s+j (low nibble of qs[8s+j]) and 16s+8+j (high nibble), values = kFp4x2[q] * ue4m3(d) (the table is
// doubled and the scale halved, as in ggml)
QE_HD float qe_ue4m3(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.f;
  const int e = (x >> 3) & 0xF, m = x & 7;
  float v;
  if (e == 0) v = (float)m * (1.f / 1024.f);  // m * 2^-9 * 0.5
  else {
    uint32_t bits = (uint32_t)(e + 119) << 23 | (uint32_t)m << 20;  // (1 + m/8) * 2^(e-7) * 0.5
    memcpy(&v, &bits, 4);
  }
  return v;
}
template <> QE_HD void dequant32<40>(const uint8_t* p, int ib, float* y) {  // NVFP4: 4 blocks of 64 per "super-block"
  const int8_t kFp4x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  const uint8_t* b = p + 36 * (ib >> 1);
  for (int h = 0; h < 2; h++) {
    const int s = 2 * (ib & 1) + h;
    const float d = qe_ue4m3(b[s]);
    const uint8_t* q = b + 4 + 8 * s;
    for (int j = 0; j < 8; j++) {
      y[16 * h + j] = d * kFp4x2[q[j] & 0xf];
      y[16 * h + 8 + j] = d * kFp4x2[q[j] >> 4];
    }
  }
}

// Runtime-dispatched reference dequant of a whole row (host only, used by tests / CPU fallback).
inline void dequant_row_ref(uint32_t type, const uint8_t* row, int64_t n, float* y) {
  if (type == 0) { memcpy(y, row, n * 4); return; }
  if (type == 1) { for (int64_t i = 0; i < n; i++) y[i] = qe_h2f(((const uint16_t*)row)[i]); return; }
  if (type == 30) { for (int64_t i = 0; i < n; i++) y[i] = qe_bf2f(((const uint16_t*)row)[i]); return; }
  int bb;
  void (*f)(const uint8_t*, int, float*);
  switch (type) {
    case 8: f = dequant32<8>; bb = 272; break;
    case 10: f = dequant32<10>; bb = 84; break;
    case 11: f = dequant32<11>; bb = 110; break;
    case 13: f = dequant32<13>; bb = 176; break;
    case 12: f = dequant32<12>; bb = 144; break;
    case 14: f = dequant32<14>; bb = 210; break;
    case 16: f = dequant32<16>; bb = 66; break;
    case 17: f = dequant32<17>; bb = 74; break;
    case 18: f = dequant32<18>; bb = 98; break;
    case 19: f = dequant32<19>; bb = 50; break;
    case 21: f = dequant32<21>; bb = 110; break;
    case 22: f = dequant32<22>; bb = 82; break;
    case 23: f = dequant32<23>; bb = 136; break;
    case 29: f = dequant32<29>; bb = 56; break;
    case 20: f = dequant32<20>; bb = 144; break;
    case 40: f = dequant32<40>; bb = 144; break;
    default: return;
  }
  for (int64_t g = 0; g < n / 32; g++) f(row + (g >> 3) * bb, (int)(g & 7), y + g * 32);
}
