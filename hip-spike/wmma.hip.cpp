// WMMA spike for the HIP phase-2 tensor-core matvec port: verify that
// __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 compiles for gfx1100 (wave32),
// determine the A/B fragment loading and C fragment mapping empirically, and
// pin down the meaning of the two bool flags (signedness selectors, as in
// sudot4). One wave computes one 16x16x16 tile; the host checks every
// combination of load orientation x C mapping x flag combo against CPU
// references for signed/unsigned interpretations, and prints what matched.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                     \
  do {                                                               \
    hipError_t e = (x);                                              \
    if (e != hipSuccess) {                                           \
      printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
      return 3;                                                      \
    }                                                                \
  } while (0)

using i32x4 = int __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));

// A is 16x16 row-major (A[m][k] at A[m*16+k]), B is 16x16 row-major
// (B[k][n] at B[k*16+n]). Load modes:
//   mode 0: A fragment = row (lane%16) of A, B fragment = column (lane%16) of B
//   mode 1: A fragment = column (lane%16) of A, B fragment = row (lane%16) of B
//   mode 2: A fragment = row (lane%16) of A, B fragment = row (lane%16) of B
//   mode 3: A fragment = column (lane%16) of A, B fragment = column (lane%16) of B
// Fragment element e (0..15) is byte e of the packed i32x4 (little-endian).
// out[lane*8 + i] receives accumulator fragment element i.
template <int MODE, bool FA, bool FB>
__global__ void wmma_tile(const int8_t* A, const int8_t* B, int* out) {
  const int lane = (int)threadIdx.x;
  const int l = lane % 16;
  i32x4 a4 = {0, 0, 0, 0}, b4 = {0, 0, 0, 0};
  int8_t* pa = (int8_t*)&a4;
  int8_t* pb = (int8_t*)&b4;
  for (int e = 0; e < 16; e++) {
    if (MODE == 0 || MODE == 2) pa[e] = A[l * 16 + e];        // A row l
    else pa[e] = A[e * 16 + l];                               // A column l
    if (MODE == 0 || MODE == 3) pb[e] = B[e * 16 + l];        // B column l
    else pb[e] = B[l * 16 + e];                               // B row l
  }
  i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(FA, a4, FB, b4, c, false);
  for (int i = 0; i < 8; i++) out[lane * 8 + i] = c[i];
}

int main() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    printf("no HIP device\n");
    return 2;
  }
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  printf("device 0: %s (%s, warp %d)\n", p.name, p.gcnArchName, p.warpSize);

  // Deterministic test data with negative values, small magnitudes.
  int8_t A[256], B[256];
  for (int i = 0; i < 256; i++) {
    A[i] = (int8_t)((i * 37 + 11) % 13 - 6);
    B[i] = (int8_t)((i * 53 + 7) % 11 - 5);
  }
  // CPU references for the four signedness interpretations of D = A * B.
  auto aval = [&](int8_t v, bool s) { return s ? (int)v : (int)(uint8_t)v; };
  int ref[4][256];
  const bool signs[4][2] = {{true, true}, {false, false}, {true, false}, {false, true}};
  for (int r = 0; r < 4; r++)
    for (int m = 0; m < 16; m++)
      for (int n = 0; n < 16; n++) {
        int s = 0;
        for (int k = 0; k < 16; k++) s += aval(A[m * 16 + k], signs[r][0]) * aval(B[k * 16 + n], signs[r][1]);
        ref[r][m * 16 + n] = s;
      }

  int8_t *dA, *dB;
  int* dOut;
  CHECK(hipMalloc(&dA, 256));
  CHECK(hipMalloc(&dB, 256));
  CHECK(hipMalloc(&dOut, 32 * 8 * 4));
  CHECK(hipMemcpy(dA, A, 256, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dB, B, 256, hipMemcpyHostToDevice));

  // C fragment mapping candidates: (row, col) of fragment element i on lane.
  auto cmap = [](int cand, int lane, int i, int& m, int& n) {
    switch (cand) {
      case 0: m = lane % 16; n = (lane / 16) + 2 * i; break;
      case 1: m = 2 * i + (lane / 16); n = lane % 16; break;
      case 2: m = lane / 2; n = (lane % 2) * 8 + i; break;
      case 3: m = (lane % 16); n = i + 8 * (lane / 16); break;
      default: m = 0; n = 0;
    }
  };

  int found = 0;
  auto run = [&](auto kernel, const char* name) {
    kernel<<<1, 32>>>(dA, dB, dOut);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      printf("%s: launch failed: %s\n", name, hipGetErrorString(e));
      return;
    }
    int out[256];
    if (hipMemcpy(out, dOut, sizeof(out), hipMemcpyDeviceToHost) != hipSuccess) {
      printf("%s: copy back failed\n", name);
      return;
    }
    for (int cand = 0; cand < 4; cand++) {
      int D[256];
      bool covered[256] = {};
      for (int lane = 0; lane < 32; lane++)
        for (int i = 0; i < 8; i++) {
          int m, n;
          cmap(cand, lane, i, m, n);
          D[m * 16 + n] = out[lane * 8 + i];
          covered[m * 16 + n] = true;
        }
      bool all = true;
      for (int j = 0; j < 256; j++) all &= covered[j];
      if (!all) continue;
      for (int r = 0; r < 4; r++) {
        bool eq = true;
        for (int j = 0; j < 256; j++) eq &= (D[j] == ref[r][j]);
        if (eq) {
          printf("MATCH: %s, C-map %d == ref %d (A %s, B %s)\n", name, cand, r,
                 signs[r][0] ? "signed" : "unsigned", signs[r][1] ? "signed" : "unsigned");
          found++;
        }
      }
    }
  };

  run(wmma_tile<0, true, true>, "mode0 flags(T,T)");
  run(wmma_tile<0, false, false>, "mode0 flags(F,F)");
  run(wmma_tile<0, true, false>, "mode0 flags(T,F)");
  run(wmma_tile<0, false, true>, "mode0 flags(F,T)");
  run(wmma_tile<1, true, true>, "mode1 flags(T,T)");
  run(wmma_tile<2, true, true>, "mode2 flags(T,T)");
  run(wmma_tile<3, true, true>, "mode3 flags(T,T)");

  if (found) {
    printf("WMMA SPIKE PASS (%d match(es))\n", found);
    return 0;
  }
  printf("WMMA SPIKE: no mapping matched; raw fragment dump (mode0, flags T,T):\n");
  wmma_tile<0, true, true><<<1, 32>>>(dA, dB, dOut);
  int out[256];
  CHECK(hipMemcpy(out, dOut, sizeof(out), hipMemcpyDeviceToHost));
  for (int lane = 0; lane < 32; lane++) {
    printf("lane %2d:", lane);
    for (int i = 0; i < 8; i++) printf(" %d", out[lane * 8 + i]);
    printf("\n");
  }
  printf("expected D (signed x signed), row 0: ");
  for (int n = 0; n < 16; n++) printf(" %d", ref[0][n]);
  printf("\nFAIL\n");
  return 4;
}
