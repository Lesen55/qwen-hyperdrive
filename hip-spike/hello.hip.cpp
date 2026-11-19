// Phase-0 HIP spike: prove the Windows HIP SDK toolchain can build a gfx1100
// binary that runs on the RX 7900 XTX. Prints device properties, verifies a
// vector-add kernel, and measures a device-to-device copy bandwidth.
#include <hip/hip_runtime.h>

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

__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void dcopy(const float* a, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i];
}

int main() {
  int count = 0;
  hipError_t e = hipGetDeviceCount(&count);
  if (e != hipSuccess) {
    printf("hipGetDeviceCount failed: %s\n", hipGetErrorString(e));
    return 2;
  }
  printf("devices: %d\n", count);
  if (count == 0) return 2;
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  printf("device 0: %s\n", p.name);
  printf("  arch: %s, warp %d, CUs %d, VRAM %.0f MiB\n", p.gcnArchName, p.warpSize,
         p.multiProcessorCount, p.totalGlobalMem / 1048576.0);

  const int n = 1 << 20;
  std::vector<float> ha(n), hb(n);
  for (int i = 0; i < n; i++) { ha[i] = i * 0.5f; hb[i] = 100.f - i * 0.25f; }
  float *da, *db, *dc;
  CHECK(hipMalloc(&da, n * 4));
  CHECK(hipMalloc(&db, n * 4));
  CHECK(hipMalloc(&dc, n * 4));
  CHECK(hipMemcpy(da, ha.data(), n * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(db, hb.data(), n * 4, hipMemcpyHostToDevice));
  vadd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  CHECK(hipGetLastError());
  std::vector<float> hc(n);
  CHECK(hipMemcpy(hc.data(), dc, n * 4, hipMemcpyDeviceToHost));
  double maxerr = 0;
  for (int i = 0; i < n; i++) {
    double d = hc[i] - (ha[i] + hb[i]);
    if (d < 0) d = -d;
    if (d > maxerr) maxerr = d;
  }
  printf("vadd max err: %g\n", maxerr);
  if (maxerr > 1e-4) { printf("VERIFY FAILED\n"); return 4; }

  // crude copy bandwidth: 256 MiB device-to-device, 20 reps
  const int m = 1 << 26;
  float *src, *dst;
  CHECK(hipMalloc(&src, (size_t)m * 4));
  CHECK(hipMalloc(&dst, (size_t)m * 4));
  dcopy<<<(m + 255) / 256, 256>>>(src, dst, m);
  CHECK(hipDeviceSynchronize());
  hipEvent_t t0, t1;
  CHECK(hipEventCreate(&t0));
  CHECK(hipEventCreate(&t1));
  CHECK(hipEventRecord(t0));
  for (int r = 0; r < 20; r++) dcopy<<<(m + 255) / 256, 256>>>(src, dst, m);
  CHECK(hipEventRecord(t1));
  CHECK(hipEventSynchronize(t1));
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, t0, t1));
  const double gbs = 20.0 * m * 4 * 2 / (ms / 1e3) / 1e9;  // read + write
  printf("copy bandwidth: %.0f GB/s\n", gbs);
  printf("PASS\n");
  return 0;
}
