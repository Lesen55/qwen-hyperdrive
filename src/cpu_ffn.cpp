#include "cpu_ffn.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <cmath>
#include <vector>

#include "quant.h"

namespace cpu {

bool type_supported(uint32_t t) {
  switch (t) {
    case QT_IQ2_XXS: case QT_IQ2_XS: case QT_IQ2_S: case QT_IQ3_XXS: case QT_IQ3_S:
    case QT_IQ4_XS: case QT_Q2_K: case QT_Q4_K:
      return true;
    default:
      return false;
  }
}

void Q8Act::resize(int K_, int B_) {
  K = K_;
  B = B_;
  q.resize((size_t)K * B);
  d.resize((size_t)K / 256 * B);
  bs16.resize((size_t)K / 16 * B);
}

void quantize(const float* x, int K, int b, Q8Act& a, int blk0, int blk1) {
  const float* xr = x + (size_t)b * K;
  int8_t* q = a.q.data() + (size_t)b * K;
  const __m256 sign = _mm256_set1_ps(-0.f);
  for (int blk = blk0; blk < blk1; blk++) {
    const float* p = xr + 256 * blk;
    __m256 m = _mm256_setzero_ps();
    for (int i = 0; i < 256; i += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(p + i)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    const float amax = _mm_cvtss_f32(m4);
    const float dd = amax / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
    a.d[(size_t)b * (K / 256) + blk] = dd;
    const __m256 vid = _mm256_set1_ps(id);
    int8_t* qb = q + 256 * blk;
    for (int i = 0; i < 256; i += 32) {
      __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i), vid));
      __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i + 8), vid));
      __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i + 16), vid));
      __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i + 24), vid));
      __m256i s01 = _mm256_packs_epi32(i0, i1), s23 = _mm256_packs_epi32(i2, i3);
      __m256i v = _mm256_packs_epi16(s01, s23);
      v = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
      _mm256_storeu_si256((__m256i*)(qb + i), v);
    }
    int16_t* bs = a.bs16.data() + (size_t)b * (K / 16) + 16 * blk;
    for (int j = 0; j < 16; j++) {
      int s = 0;
      for (int k = 0; k < 16; k++) s += qb[16 * j + k];
      bs[j] = (int16_t)s;
    }
  }
}

// ------------------------------------------------------------------------------------------------
namespace {

uint64_t g_smask[256];  // byte j = 0xFF if bit j of the index is set
struct InitMask {
  InitMask() {
    for (int i = 0; i < 256; i++) {
      uint64_t m = 0;
      for (int j = 0; j < 8; j++)
        if (i & (1 << j)) m |= 0xFFull << (8 * j);
      g_smask[i] = m;
    }
  }
} g_init_mask;

inline float h2f(const uint8_t* p) { return _cvtsh_ss(*(const uint16_t*)p); }
inline uint32_t ld32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline uint16_t ld16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
inline uint32_t s8_of(uint32_t s7) { return s7 | ((__builtin_popcount(s7) & 1) << 7); }

inline __m256i apply_signs(__m256i g, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
  const __m256i m = _mm256_set_epi64x((int64_t)g_smask[s3], (int64_t)g_smask[s2], (int64_t)g_smask[s1], (int64_t)g_smask[s0]);
  return _mm256_sub_epi8(_mm256_xor_si256(g, m), m);
}

// Decoded 32-weight sub-block: signed int8 weights and integer scale per 16-bit madd lane.
// dot(w_real, x_real) = factor * d_w * d_x * sum_i (s16 . maddubs(|w|, sign(x,w)))  (+ min term)
struct D32 {
  __m256i w;
  __m256i s16;
};
inline __m256i s16_one(int s) { return _mm256_set1_epi16((int16_t)s); }
inline __m256i s16_two(int s0, int s1) { return _mm256_set_m128i(_mm_set1_epi16((int16_t)s1), _mm_set1_epi16((int16_t)s0)); }

template <int T> struct Dec;

template <> struct Dec<QT_IQ2_XXS> {
  static constexpr int BB = 66; static constexpr bool MIN = false;
  static inline float factor(const uint8_t* b) { return h2f(b) * 0.125f; }
  static inline void dec(const uint8_t* b, int ib, D32& o) {
    const uint32_t a0 = ld32(b + 2 + 8 * ib), a1 = ld32(b + 2 + 8 * ib + 4);
    const __m256i g = _mm256_set_epi64x((int64_t)qtab::iq2xxs_grid[(a0 >> 24) & 255], (int64_t)qtab::iq2xxs_grid[(a0 >> 16) & 255],
                                        (int64_t)qtab::iq2xxs_grid[(a0 >> 8) & 255], (int64_t)qtab::iq2xxs_grid[a0 & 255]);
    o.w = apply_signs(g, s8_of(a1 & 127), s8_of((a1 >> 7) & 127), s8_of((a1 >> 14) & 127), s8_of((a1 >> 21) & 127));
    o.s16 = s16_one(2 * (a1 >> 28) + 1);
  }
};
template <> struct Dec<QT_IQ2_XS> {
  static constexpr int BB = 74; static constexpr bool MIN = false;
  static inline float factor(const uint8_t* b) { return h2f(b) * 0.125f; }
  static inline void dec(const uint8_t* b, int ib, D32& o) {
    const uint8_t* q = b + 2 + 8 * ib;
    const uint16_t q0 = ld16(q), q1 = ld16(q + 2), q2 = ld16(q + 4), q3 = ld16(q + 6);
    const __m256i g = _mm256_set_epi64x((int64_t)qtab::iq2xs_grid[q3 & 511], (int64_t)qtab::iq2xs_grid[q2 & 511],
                                        (int64_t)qtab::iq2xs_grid[q1 & 511], (int64_t)qtab::iq2xs_grid[q0 & 511]);
    o.w = apply_signs(g, s8_of(q0 >> 9), s8_of(q1 >> 9), s8_of(q2 >> 9), s8_of(q3 >> 9));
    const uint32_t sc = b[66 + ib];
    o.s16 = s16_two(2 * (sc & 0xf) + 1, 2 * (sc >> 4) + 1);
  }
};
template <> struct Dec<QT_IQ2_S> {
  static constexpr int BB = 82; static constexpr bool MIN = false;
  static inline float factor(const uint8_t* b) { return h2f(b) * 0.125f; }
  static inline void dec(const uint8_t* b, int ib, D32& o) {
    const uint8_t* qs = b + 2 + 4 * ib;
    const uint8_t* sg = b + 34 + 4 * ib;
    const uint32_t qh = b[66 + ib];
    const __m256i g = _mm256_set_epi64x((int64_t)qtab::iq2s_grid[qs[3] | ((qh << 2) & 0x300)],
                                        (int64_t)qtab::iq2s_grid[qs[2] | ((qh << 4) & 0x300)],
                                        (int64_t)qtab::iq2s_grid[qs[1] | ((qh << 6) & 0x300)],
                                        (int64_t)qtab::iq2s_grid[qs[0] | ((qh << 8) & 0x300)]);
    o.w = apply_signs(g, sg[0], sg[1], sg[2], sg[3]);
    const uint32_t sc = b[74 + ib];
    o.s16 = s16_two(2 * (sc & 0xf) + 1, 2 * (sc >> 4) + 1);
  }
};
template <> struct Dec<QT_IQ3_XXS> {
  static constexpr int BB = 98; static constexpr bool MIN = false;
  static inline float factor(const uint8_t* b) { return h2f(b) * 0.25f; }
  static inline void dec(const uint8_t* b, int ib, D32& o) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t a = ld32(b + 66 + 4 * ib);
    const __m256i g = _mm256_setr_epi32((int)qtab::iq3xxs_grid[qs[0]], (int)qtab::iq3xxs_grid[qs[1]], (int)qtab::iq3xxs_grid[qs[2]],
                                        (int)qtab::iq3xxs_grid[qs[3]], (int)qtab::iq3xxs_grid[qs[4]], (int)qtab::iq3xxs_grid[qs[5]],
                                        (int)qtab::iq3xxs_grid[qs[6]], (int)qtab::iq3xxs_grid[qs[7]]);
    o.w = apply_signs(g, s8_of(a & 127), s8_of((a >> 7) & 127), s8_of((a >> 14) & 127), s8_of((a >> 21) & 127));
    o.s16 = s16_one(2 * (a >> 28) + 1);
  }
};
template <> struct Dec<QT_IQ3_S> {
  static constexpr int BB = 110; static constexpr bool MIN = false;
  static inline float factor(const uint8_t* b) { return h2f(b); }
  static inline void dec(const uint8_t* b, int ib, D32& o) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t qh = b[66 + ib];
    const uint8_t* sg = b + 74 + 4 * ib;
    const __m256i g = _mm256_setr_epi32(
        (int)qtab::iq3s_grid[qs[0] | ((qh << 8) & 256)], (int)qtab::iq3s_grid[qs[1] | ((qh << 7) & 256)],
        (int)qtab::iq3s_grid[qs[2] | ((qh << 6) & 256)], (int)qtab::iq3s_grid[qs[3] | ((qh << 5) & 256)],
        (int)qtab::iq3s_grid[qs[4] | ((qh << 4) & 256)], (int)qtab::iq3s_grid[qs[5] | ((qh << 3) & 256)],
        (int)qtab::iq3s_grid[qs[6] | ((qh << 2) & 256)], (int)qtab::iq3s_grid[qs[7] | ((qh << 1) & 256)]);
    o.w = apply_signs(g, sg[0], sg[1], sg[2], sg[3]);
    o.s16 = s16_one(1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf));
  }
};
template <> struct Dec<QT_IQ4_XS> {
  static constexpr int BB = 136; static constexpr bool MIN = false;
  static inline float factor(const uint8_t* b) { return h2f(b); }
  static inline void dec(const uint8_t* b, int ib, D32& o) {
    const uint32_t sh = ld16(b + 2);
    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    const __m128i q = _mm_loadu_si128((const __m128i*)(b + 8 + 16 * ib));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(q, m4), hi = _mm_and_si128(_mm_srli_epi16(q, 4), m4);
    const __m128i tab = _mm_loadu_si128((const __m128i*)qtab::kvalues_iq4nl);
    o.w = _mm256_set_m128i(_mm_shuffle_epi8(tab, hi), _mm_shuffle_epi8(tab, lo));
    o.s16 = s16_one(ls - 32);
  }
};
template <> struct Dec<QT_Q2_K> {
  static constexpr int BB = 84; static constexpr bool MIN = true;
  static inline float factor(const uint8_t* b) { return h2f(b + 80); }
  static inline float min_factor(const uint8_t* b) { return h2f(b + 82); }
  static inline int min_dot(const uint8_t* b, const int16_t* bs) {  // sum_j m_j * bsum16_j
    int s = 0;
    for (int j = 0; j < 16; j++) s += (b[j] >> 4) * bs[j];
    return s;
  }
  static inline void dec(const uint8_t* b, int g, D32& o) {
    const __m256i q = _mm256_loadu_si256((const __m256i*)(b + 16 + (g / 4) * 32));
    const __m128i sh = _mm_cvtsi32_si128(2 * (g % 4));
    o.w = _mm256_and_si256(_mm256_srl_epi16(q, sh), _mm256_set1_epi8(3));
    o.s16 = s16_two(b[2 * g] & 0xF, b[2 * g + 1] & 0xF);
  }
};
template <> struct Dec<QT_Q4_K> {
  static constexpr int BB = 144; static constexpr bool MIN = true;
  static inline float factor(const uint8_t* b) { return h2f(b); }
  static inline float min_factor(const uint8_t* b) { return h2f(b + 2); }
  static inline int min_dot(const uint8_t* b, const int16_t* bs) {
    int s = 0;
    for (int g = 0; g < 8; g++) {
      uint8_t sc, m;
      q4k_scale_min(g, b + 4, sc, m);
      s += m * (bs[2 * g] + bs[2 * g + 1]);
    }
    return s;
  }
  static inline void dec(const uint8_t* b, int g, D32& o) {
    uint8_t sc, m;
    q4k_scale_min(g, b + 4, sc, m);
    const __m256i q = _mm256_loadu_si256((const __m256i*)(b + 16 + (g / 2) * 32));
    const __m128i sh = _mm_cvtsi32_si128(4 * (g & 1));
    o.w = _mm256_and_si256(_mm256_srl_epi16(q, sh), _mm256_set1_epi8(0x0F));
    o.s16 = s16_one(sc);
  }
};

inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

template <int T, int B>
void mv(const uint8_t* W, size_t rb, int K, int r0, int r1, const Q8Act& x, float* y, int ldy) {
  using D = Dec<T>;
  const int nb = K / 256;
  for (int r = r0; r < r1; r++) {
    const uint8_t* wr = W + (size_t)r * rb;
    if (r + 1 < r1) _mm_prefetch((const char*)(wr + rb), _MM_HINT_T0);
    __m256 acc[B];
    float macc[B];
    for (int b = 0; b < B; b++) { acc[b] = _mm256_setzero_ps(); macc[b] = 0.f; }
    for (int j = 0; j < nb; j++) {
      const uint8_t* blk = wr + (size_t)j * D::BB;
      __m256i sumi[B];
      for (int b = 0; b < B; b++) sumi[b] = _mm256_setzero_si256();
#pragma GCC unroll 8
      for (int ib = 0; ib < 8; ib++) {
        D32 w;
        D::dec(blk, ib, w);
        const __m256i aw = _mm256_sign_epi8(w.w, w.w);
        for (int b = 0; b < B; b++) {
          const __m256i xv = _mm256_loadu_si256((const __m256i*)(x.q.data() + (size_t)b * K + 256 * j + 32 * ib));
          const __m256i dot = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, w.w));
          sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(dot, w.s16));
        }
      }
      const float f = D::factor(blk);
      for (int b = 0; b < B; b++) {
        const float xd = x.d[(size_t)b * nb + j];
        acc[b] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[b]), _mm256_set1_ps(f * xd), acc[b]);
        if constexpr (D::MIN) macc[b] += D::min_factor(blk) * xd * (float)D::min_dot(blk, x.bs16.data() + (size_t)b * (K / 16) + 16 * j);
      }
    }
    for (int b = 0; b < B; b++) y[(size_t)b * ldy + r] = hsum(acc[b]) - macc[b];
  }
}

template <int T>
void mv_b(const uint8_t* W, size_t rb, int K, int r0, int r1, const Q8Act& x, int B, float* y, int ldy) {
  switch (B) {
    case 1: mv<T, 1>(W, rb, K, r0, r1, x, y, ldy); break;
    case 2: mv<T, 2>(W, rb, K, r0, r1, x, y, ldy); break;
    case 3: mv<T, 3>(W, rb, K, r0, r1, x, y, ldy); break;
    case 4: mv<T, 4>(W, rb, K, r0, r1, x, y, ldy); break;
    default: QE_FATAL("cpu matvec: batch %d > 4 not supported", B);
  }
}

template <int B>
void mv_r4(const int8_t* tabp, const uint8_t* W, int K, int r0, int r1, const Q8Act& x, float* y, int ldy) {
  const int nb = K / 256;
  const size_t rb = (size_t)nb * R4_BLOCK_BYTES;
  const __m128i tab = _mm_loadu_si128((const __m128i*)tabp);
  const __m128i m4 = _mm_set1_epi8(0x0F);
  for (int r = r0; r < r1; r++) {
    const uint8_t* wr = W + (size_t)r * rb;
    if (r + 1 < r1) _mm_prefetch((const char*)(wr + rb), _MM_HINT_T0);
    __m256 acc[B];
    for (int b = 0; b < B; b++) acc[b] = _mm256_setzero_ps();
    for (int j = 0; j < nb; j++) {
      const uint8_t* blk = wr + (size_t)j * R4_BLOCK_BYTES;
      const uint8_t* sc = blk + 4;
      const uint8_t* q = blk + 20;
      __m256i sumi[B];
      for (int b = 0; b < B; b++) sumi[b] = _mm256_setzero_si256();
#pragma GCC unroll 8
      for (int ib = 0; ib < 8; ib++) {
        const __m128i qq = _mm_loadu_si128((const __m128i*)(q + 16 * ib));
        const __m128i lo = _mm_and_si128(qq, m4), hi = _mm_and_si128(_mm_srli_epi16(qq, 4), m4);
        const __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(tab, hi), _mm_shuffle_epi8(tab, lo));
        const __m256i aw = _mm256_sign_epi8(w, w);
        const __m256i s16 = _mm256_set_m128i(_mm_set1_epi16(sc[2 * ib + 1]), _mm_set1_epi16(sc[2 * ib]));
        for (int b = 0; b < B; b++) {
          const __m256i xv = _mm256_loadu_si256((const __m256i*)(x.q.data() + (size_t)b * K + 256 * j + 32 * ib));
          const __m256i dot = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, w));
          sumi[b] = _mm256_add_epi32(sumi[b], _mm256_madd_epi16(dot, s16));
        }
      }
      float f;
      memcpy(&f, blk, 4);
      for (int b = 0; b < B; b++)
        acc[b] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[b]), _mm256_set1_ps(f * x.d[(size_t)b * nb + j]), acc[b]);
    }
    for (int b = 0; b < B; b++) y[(size_t)b * ldy + r] = hsum(acc[b]);
  }
}

struct R4Tables {
  int8_t tab[32][16];  // indexed by ggml type id
  int8_t idx[32][256]; // signed value (as uint8) -> nibble, -1 if not representable
  R4Tables() {
    memset(tab, 0, sizeof tab);
    memset(idx, -1, sizeof idx);
    auto build = [&](int t, std::vector<int> mags) {
      for (size_t i = 0; i < mags.size(); i++) {
        tab[t][i] = (int8_t)mags[i];
        tab[t][8 + i] = (int8_t)-mags[i];
        idx[t][(uint8_t)(int8_t)mags[i]] = (int8_t)i;
        idx[t][(uint8_t)(int8_t)-mags[i]] = (int8_t)(8 + i);
      }
    };
    for (int t : {QT_IQ2_XXS, QT_IQ2_XS, QT_IQ2_S}) build(t, {8, 25, 43});
    build(QT_IQ3_XXS, {4, 12, 20, 28, 36, 44, 52, 62});
    build(QT_IQ3_S, {1, 3, 5, 7, 9, 11, 13, 15});
  }
} g_r4;

template <int T>
void r4_convert_t(const uint8_t* src, size_t src_rb, int K, int N, uint8_t* dst) {
  using D = Dec<T>;
  const int nb = K / 256;
  for (int r = 0; r < N; r++) {
    for (int j = 0; j < nb; j++) {
      const uint8_t* blk = src + (size_t)r * src_rb + (size_t)j * D::BB;
      uint8_t* o = dst + ((size_t)r * nb + j) * R4_BLOCK_BYTES;
      const float f = D::factor(blk);
      memcpy(o, &f, 4);
      uint8_t* sc = o + 4;
      uint8_t* q = o + 20;
      for (int ib = 0; ib < 8; ib++) {
        D32 w;
        D::dec(blk, ib, w);
        alignas(32) int8_t v[32];
        alignas(32) int16_t s[16];
        _mm256_store_si256((__m256i*)v, w.w);
        _mm256_store_si256((__m256i*)s, w.s16);
        sc[2 * ib] = (uint8_t)s[0];
        sc[2 * ib + 1] = (uint8_t)s[8];
        for (int e = 0; e < 16; e++) {
          const int8_t a = g_r4.idx[T][(uint8_t)v[e]], b = g_r4.idx[T][(uint8_t)v[e + 16]];
          if (a < 0 || b < 0) QE_FATAL("r4: value not in codebook");
          q[16 * ib + e] = (uint8_t)(a | (b << 4));
        }
      }
    }
  }
}

}  // namespace

bool r4_supported(uint32_t t) {
  return t == QT_IQ2_XXS || t == QT_IQ2_XS || t == QT_IQ2_S || t == QT_IQ3_XXS || t == QT_IQ3_S;
}
const int8_t* r4_table(uint32_t t) { return g_r4.tab[t]; }
void r4_convert(uint32_t type, const uint8_t* src, size_t src_rb, int K, int N, uint8_t* dst) {
  switch (type) {
    case QT_IQ2_XXS: r4_convert_t<QT_IQ2_XXS>(src, src_rb, K, N, dst); break;
    case QT_IQ2_XS: r4_convert_t<QT_IQ2_XS>(src, src_rb, K, N, dst); break;
    case QT_IQ2_S: r4_convert_t<QT_IQ2_S>(src, src_rb, K, N, dst); break;
    case QT_IQ3_XXS: r4_convert_t<QT_IQ3_XXS>(src, src_rb, K, N, dst); break;
    case QT_IQ3_S: r4_convert_t<QT_IQ3_S>(src, src_rb, K, N, dst); break;
    default: QE_FATAL("r4_convert: unsupported type %u", type);
  }
}
void matvec_r4(const int8_t* tab, const uint8_t* W, int K, int r0, int r1, const Q8Act& x, int B, float* y,
               int ldy) {
  switch (B) {
    case 1: mv_r4<1>(tab, W, K, r0, r1, x, y, ldy); break;
    case 2: mv_r4<2>(tab, W, K, r0, r1, x, y, ldy); break;
    case 3: mv_r4<3>(tab, W, K, r0, r1, x, y, ldy); break;
    case 4: mv_r4<4>(tab, W, K, r0, r1, x, y, ldy); break;
    default: QE_FATAL("cpu matvec_r4: batch %d > 4", B);
  }
}

void matvec(uint32_t type, const uint8_t* W, size_t rb, int K, int r0, int r1, const Q8Act& x, int B,
            float* y, int ldy) {
  switch (type) {
#define T(t) case t: mv_b<t>(W, rb, K, r0, r1, x, B, y, ldy); break;
    T(QT_IQ2_XXS) T(QT_IQ2_XS) T(QT_IQ2_S) T(QT_IQ3_XXS) T(QT_IQ3_S) T(QT_IQ4_XS) T(QT_Q2_K) T(QT_Q4_K)
#undef T
    default: QE_FATAL("cpu matvec: unsupported type %u", type);
  }
}

// ------------------------------------------------------------------------------------------------
ThreadPool::ThreadPool(int n) : n_(n) {
  for (int i = 0; i < n; i++) threads_.emplace_back([this, i] { worker(i); });
}
ThreadPool::~ThreadPool() {
  stop_.store(true);
  gen_.fetch_add(1);
  gen_.notify_all();
  for (auto& t : threads_) t.join();
}
void ThreadPool::start(std::function<void(int)> fn) {
  job_ = std::move(fn);
  done_.store(0, std::memory_order_relaxed);
  gen_.fetch_add(1, std::memory_order_release);
  gen_.notify_all();
}
void ThreadPool::wait() {
  while (done_.load(std::memory_order_acquire) < n_) _mm_pause();
}
void ThreadPool::barrier() {
  const uint64_t g = bar_gen_.load(std::memory_order_acquire);
  if (bar_count_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
    bar_count_.store(0, std::memory_order_relaxed);
    bar_gen_.fetch_add(1, std::memory_order_release);
  } else {
    while (bar_gen_.load(std::memory_order_acquire) == g) _mm_pause();
  }
}
void ThreadPool::worker(int idx) {
  {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(idx, &cs);  // logical CPUs 0..5 are distinct physical cores on this Ryzen
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
  }
  uint64_t seen = 0;
  for (;;) {
    uint64_t g;
    int spins = 0;
    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
      if (++spins < 200000) _mm_pause();
      else gen_.wait(seen, std::memory_order_acquire);
    }
    seen = g;
    if (stop_.load()) return;
    job_(idx);
    done_.fetch_add(1, std::memory_order_acq_rel);
  }
}

}  // namespace cpu
