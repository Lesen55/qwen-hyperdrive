// Stage-0 spike for the WMMA flash-attention project: determine the fp16
// WMMA (v_wmma_f32_16x16x16_f16, wave32, gfx1100) fragment layouts on real
// hardware. One wave runs one WMMA with uniquely-coded inputs; the raw
// accumulator slots are dumped per lane so the true (row, col) of every
// (lane, slot) can be decoded offline, whatever the layout turns out to be.
//
//   Run 1: A[m][k] = (m*16 + k + 1) / 256 (unique per element), B = identity
//          (B[n][k] = 1 iff k == n). Every C value is then a single A value
//          (times a permutation), so value*256-1 decodes to (m*16 + k).
//   Run 2: roles swapped (A = identity, B coded) for the B-side view.
//
// Fragments are loaded with the hypothesized standard layout:
//   A/B frag: lane l, element e -> matrix[l % 16][(l / 16) * 8 + e]
// The dump does not depend on the hypothesis being right: we print exactly
// which values each lane fed and exactly which values came back per slot.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

typedef __half __attribute__((ext_vector_type(8))) half8;
typedef float __attribute__((ext_vector_type(8))) float8;

__global__ void wmma_probe(const __half* __restrict__ A, const __half* __restrict__ B,
                           float* __restrict__ dumpA, float* __restrict__ dumpB,
                           float* __restrict__ dumpC) {
  const int lane = threadIdx.x & 31;
  half8 a, b;
  const int r = lane & 15, half = (lane >> 4) * 8;
  for (int e = 0; e < 8; e++) {
    a[e] = A[r * 16 + half + e];
    b[e] = B[r * 16 + half + e];
    dumpA[lane * 8 + e] = __half2float(a[e]);
    dumpB[lane * 8 + e] = __half2float(b[e]);
  }
  float8 c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  for (int i = 0; i < 8; i++) dumpC[lane * 8 + i] = c[i];
}

static void run_one(const char* label, const __half* hA, const __half* hB) {
  __half *dA, *dB;
  float *dDA, *dDB, *dDC;
  hipMalloc(&dA, 256 * sizeof(__half));
  hipMalloc(&dB, 256 * sizeof(__half));
  hipMalloc(&dDA, 256 * sizeof(float));
  hipMalloc(&dDB, 256 * sizeof(float));
  hipMalloc(&dDC, 256 * sizeof(float));
  hipMemcpy(dA, hA, 256 * sizeof(__half), hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, 256 * sizeof(__half), hipMemcpyHostToDevice);
  wmma_probe<<<1, 32>>>(dA, dB, dDA, dDB, dDC);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("%s: HIP error %s\n", label, hipGetErrorString(err));
    return;
  }
  float DA[256], DB[256], DC[256];
  hipMemcpy(DA, dDA, 256 * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(DB, dDB, 256 * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(DC, dDC, 256 * sizeof(float), hipMemcpyDeviceToHost);
  printf("== %s ==\n", label);
  for (int lane = 0; lane < 32; lane++) {
    printf("lane %2d A:", lane);
    for (int e = 0; e < 8; e++) printf(" %d", (int)(DA[lane * 8 + e] * 256.f + 0.5f) - 1);
    printf("  B:");
    for (int e = 0; e < 8; e++) printf(" %d", (int)(DB[lane * 8 + e] * 256.f + 0.5f) - 1);
    printf("  C:");
    for (int i = 0; i < 8; i++) printf(" %d", (int)(DC[lane * 8 + i] * 256.f + 0.5f) - 1);
    printf("\n");
  }
  hipFree(dA);
  hipFree(dB);
  hipFree(dDA);
  hipFree(dDB);
  hipFree(dDC);
}

int main() {
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  if (err != hipSuccess) {
    printf("no HIP device: %s\n", hipGetErrorString(err));
    return 1;
  }
  printf("device: %s, gcnArch %s, warpSize %d\n", prop.name, prop.gcnArchName, prop.warpSize);

  __half coded[256], ident[256];
  for (int m = 0; m < 16; m++)
    for (int k = 0; k < 16; k++) {
      coded[m * 16 + k] = __float2half((m * 16 + k + 1) / 256.f);
      ident[m * 16 + k] = __float2half(m == k ? 1.f : 0.f);
    }
  run_one("run1: A=coded, B=identity", coded, ident);
  run_one("run2: A=identity, B=coded", ident, coded);
  printf("done\n");
  return 0;
}
