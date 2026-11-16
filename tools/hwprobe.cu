// Hardware probe: DRAM read bandwidth (CPU, multi-thread), pinned H2D bandwidth, GPU DRAM bandwidth,
// and concurrent CPU-read + H2D (does PCIe streaming add bandwidth on top of CPU compute?).
#include <cuda_runtime.h>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }
static float g_sink;
static void cpu_read(const float* p, size_t n, int nt, double* gbs) {
  std::vector<std::thread> th; std::vector<float> acc(nt*16);
  double t0 = now();
  for (int t = 0; t < nt; t++) th.emplace_back([&, t] {
    size_t a = n * t / nt, b = n * (t + 1) / nt; __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (size_t i = a; i + 32 <= b; i += 32) {
      s0 = _mm256_add_ps(s0, _mm256_load_ps(p + i)); s1 = _mm256_add_ps(s1, _mm256_load_ps(p + i + 8));
      s2 = _mm256_add_ps(s2, _mm256_load_ps(p + i + 16)); s3 = _mm256_add_ps(s3, _mm256_load_ps(p + i + 24)); }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)); _mm256_storeu_ps(&acc[t*16], s0); });
  for (auto& x : th) x.join();
  double t = now() - t0; *gbs = n * 4.0 / t / 1e9; g_sink += acc[0];
}
__global__ void gread(const float4* p, size_t n, float* out) {
  float s = 0; for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = p[i]; s += v.x + v.y + v.z + v.w; }
  if (s == 12345.f) *out = s;
}
int main() {
  size_t bytes = 2ull << 30; size_t n = bytes / 4;
  float* h; cudaMallocHost(&h, bytes); memset(h, 1, bytes);
  float* d; cudaMalloc(&d, bytes); float* o; cudaMalloc(&o, 4);
  for (int nt : {1, 3, 6, 12}) { double g; cpu_read(h, n, nt, &g); cpu_read(h, n, nt, &g); printf("CPU read %2d threads: %6.1f GB/s\n", nt, g); }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
  cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(e0); cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); cudaEventRecord(e1); cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms, e0, e1); printf("H2D pinned: %6.1f GB/s\n", bytes / ms / 1e6);
  gread<<<68*8, 256>>>((float4*)d, n / 4, o); cudaDeviceSynchronize();
  cudaEventRecord(e0); for (int i = 0; i < 5; i++) gread<<<68*8, 256>>>((float4*)d, n / 4, o); cudaEventRecord(e1); cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms, e0, e1); printf("GPU DRAM read: %6.1f GB/s\n", 5.0 * bytes / ms / 1e6);
  // concurrent: H2D of first half while CPU reads second half
  size_t half = bytes / 2; cudaStream_t s; cudaStreamCreate(&s);
  for (int nt : {5, 6}) {
    double t0 = now(); cudaMemcpyAsync(d, h, half, cudaMemcpyHostToDevice, s); double g; cpu_read(h + n / 2, n / 2, nt, &g); double tc = now() - t0;
    cudaStreamSynchronize(s); double tt = now() - t0;
    printf("concurrent (%d thr): CPU %6.1f GB/s (cpu done %.1f ms), total %.1f ms -> combined %.1f GB/s\n", nt, g, tc * 1e3, tt * 1e3, bytes / tt / 1e9);
  }
  size_t fr, tot; cudaMemGetInfo(&fr, &tot); printf("VRAM free after ctx+2GiB: %zu MiB\n", fr >> 20);
  printf("%f\n", g_sink * 0);
}
