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
// Fragment loads mirror the int8 WMMA style used by the engine's GEMM
// kernels: lane l feeds row (l % 16), all 16 K elements (v16f16 operand).
// The dump does not depend on that being the hardware's interpretation: we
// print exactly which values each lane fed and what came back per slot.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

typedef __fp16 __attribute__((ext_vector_type(16))) f16x16;
typedef float __attribute__((ext_vector_type(8))) f32x8;

__global__ void wmma_probe(const __half* __restrict__ A, const __half* __restrict__ B,
                           float* __restrict__ dumpA, float* __restrict__ dumpB,
                           float* __restrict__ dumpC) {
  const int lane = threadIdx.x & 31;
  f16x16 a, b;
  const int r = lane & 15;
  for (int e = 0; e < 16; e++) {
    a[e] = (__fp16)__half2float(A[r * 16 + e]);
    b[e] = (__fp16)__half2float(B[r * 16 + e]);
    dumpA[lane * 16 + e] = __half2float(A[r * 16 + e]);
    dumpB[lane * 16 + e] = __half2float(B[r * 16 + e]);
  }
  f32x8 c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  for (int i = 0; i < 8; i++) dumpC[lane * 8 + i] = c[i];
}

static void run_one(const char* label, const __half* hA, const __half* hB) {
  __half *dA, *dB;
  float *dDA, *dDB, *dDC;
  (void)hipMalloc(&dA, 256 * sizeof(__half));
  (void)hipMalloc(&dB, 256 * sizeof(__half));
  (void)hipMalloc(&dDA, 512 * sizeof(float));
  (void)hipMalloc(&dDB, 512 * sizeof(float));
  (void)hipMalloc(&dDC, 256 * sizeof(float));
  (void)hipMemcpy(dA, hA, 256 * sizeof(__half), hipMemcpyHostToDevice);
  (void)hipMemcpy(dB, hB, 256 * sizeof(__half), hipMemcpyHostToDevice);
  wmma_probe<<<1, 32>>>(dA, dB, dDA, dDB, dDC);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("%s: HIP error %s\n", label, hipGetErrorString(err));
    return;
  }
  float DA[512], DB[512], DC[256];
  (void)hipMemcpy(DA, dDA, 512 * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipMemcpy(DB, dDB, 512 * sizeof(float), hipMemcpyDeviceToHost);
  (void)hipMemcpy(DC, dDC, 256 * sizeof(float), hipMemcpyDeviceToHost);
  printf("== %s ==\n", label);
  for (int lane = 0; lane < 32; lane++) {
    printf("lane %2d A:", lane);
    for (int e = 0; e < 16; e++) printf(" %d", (int)(DA[lane * 16 + e] * 256.f + 0.5f) - 1);
    printf("  B:");
    for (int e = 0; e < 16; e++) printf(" %d", (int)(DB[lane * 16 + e] * 256.f + 0.5f) - 1);
    printf("  C:");
    for (int i = 0; i < 8; i++) printf(" %d", (int)(DC[lane * 8 + i] * 256.f + 0.5f) - 1);
    printf("\n");
  }
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dDA);
  (void)hipFree(dDB);
  (void)hipFree(dDC);
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
