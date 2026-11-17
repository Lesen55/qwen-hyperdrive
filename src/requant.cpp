// Load-time requantization to IQ4_XS (used for the MTP block, which only proposes drafts: a lighter
// format makes every draft step cheaper and frees VRAM; verification keeps the output exact).
#include "requant.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "quant.h"

namespace {

const int8_t kV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline int nearest_kv(float v) {
  if (v <= kV[0]) return 0;
  if (v >= kV[15]) return 15;
  int lo = 0, hi = 15;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    (v < kV[mid] ? hi : lo) = mid;
  }
  return v - kV[lo] < kV[hi] - v ? lo : hi;
}

// best scale for 32 values (weights = x^2, as llama.cpp's iq4_nl search)
float best_scale(const float* x) {
  float amax = 0, xmax = 0;
  for (int j = 0; j < 32; j++)
    if (std::fabs(x[j]) > amax) { amax = std::fabs(x[j]); xmax = x[j]; }
  if (amax < 1e-30f) return 0.f;
  float best_d = xmax / -kV[0], best = -1.f;
  for (int side = 0; side < 2; side++)
    for (int itry = -7; itry <= 7; itry++) {
      const float id = (itry + (side ? kV[15] : kV[0])) / xmax;
      float sqx = 0, sq2 = 0;
      for (int j = 0; j < 32; j++) {
        const float w = x[j] * x[j];
        const float q = kV[nearest_kv(id * x[j])];
        sqx += w * q * x[j];
        sq2 += w * q * q;
      }
      if (sq2 > 0 && sqx * sqx > best * sq2) {
        best = sqx * sqx / sq2;
        best_d = sqx / sq2;
      }
    }
  return best_d;
}

void quantize_block_iq4_xs(const float* x, uint8_t* out) {
  float sc[8];
  float max_scale = 0, amax_scale = 0;
  for (int ib = 0; ib < 8; ib++) {
    sc[ib] = best_scale(x + 32 * ib);
    if (std::fabs(sc[ib]) > amax_scale) { amax_scale = std::fabs(sc[ib]); max_scale = sc[ib]; }
  }
  std::memset(out, 0, 136);
  const float d = -max_scale / 32;
  const float id = d != 0.f ? 1.f / d : 0.f;
  const uint16_t dh = _cvtss_sh(d, 0);
  std::memcpy(out, &dh, 2);
  uint16_t sh = 0;
  for (int ib = 0; ib < 8; ib++) {
    int l = (int)std::lround(id * sc[ib]);
    l = std::clamp(l, -32, 31);
    const float dl = _cvtsh_ss(dh) * l;
    const float idl = dl != 0.f ? 1.f / dl : 0.f;
    const int ls = l + 32;
    out[4 + ib / 2] |= (uint8_t)((ls & 0xf) << (4 * (ib % 2)));
    sh |= (uint16_t)((ls >> 4) << (2 * ib));
    uint8_t* qs = out + 8 + 16 * ib;
    const float* xb = x + 32 * ib;
    for (int j = 0; j < 16; j++)
      qs[j] = (uint8_t)(nearest_kv(idl * xb[j]) | (nearest_kv(idl * xb[j + 16]) << 4));
  }
  std::memcpy(out + 2, &sh, 2);
}

}  // namespace

size_t iq4_xs_row_bytes(int K) { return (size_t)K / 256 * 136; }

void requant_rows_iq4_xs(uint32_t src_type, const uint8_t* src, size_t src_row_bytes, int K, int N, uint8_t* dst) {
  const size_t drb = iq4_xs_row_bytes(K);
  const int nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      std::vector<float> row(K);
      for (int r = t; r < N; r += nt) {
        dequant_row_ref(src_type, src + (size_t)r * src_row_bytes, K, row.data());
        for (int b = 0; b < K / 256; b++) quantize_block_iq4_xs(row.data() + 256 * b, dst + (size_t)r * drb + 136 * (size_t)b);
      }
    });
  for (auto& x : th) x.join();
}
