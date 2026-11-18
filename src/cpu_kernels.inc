// CPU matvec kernels, compiled once per instruction set (included by cpu_ffn.cpp inside namespaces
// isa_avx2 / isa_vnni / isa_avx512 under the matching `#pragma GCC target`) and picked at run time.
// The int8 dot is maddubs(|w|, sign(x, w)) -> int16 pairs, then a multiply-add with the integer
// sub-block scales: madd + add on AVX2, one vpdpwssd (VNNI) on AVX-VNNI / AVX-512 VNNI.

inline __m256i acc_madd(__m256i acc, __m256i dot, __m256i s16) {
#if defined(QE_ISA_AVX512)
  return _mm256_dpwssd_epi32(acc, dot, s16);
#elif defined(QE_ISA_VNNI)
  return _mm256_dpwssd_avx_epi32(acc, dot, s16);
#else
  return _mm256_add_epi32(acc, _mm256_madd_epi16(dot, s16));
#endif
}

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
          sumi[b] = acc_madd(sumi[b], dot, w.s16);
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
    case 5: mv<T, 5>(W, rb, K, r0, r1, x, y, ldy); break;
    case 6: mv<T, 6>(W, rb, K, r0, r1, x, y, ldy); break;
    case 7: mv<T, 7>(W, rb, K, r0, r1, x, y, ldy); break;
    case 8: mv<T, 8>(W, rb, K, r0, r1, x, y, ldy); break;
    default: QE_FATAL("cpu matvec: batch %d > 8 not supported", B);
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
          sumi[b] = acc_madd(sumi[b], dot, s16);
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

