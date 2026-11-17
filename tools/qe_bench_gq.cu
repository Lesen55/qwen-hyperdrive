// Micro-benchmark of the GQ int8 GEMM (synthetic data): TOPS per type class.
#include <cstdio>
#include <vector>
#include "../src/common.h"
#include "../src/kernels.h"
int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 2048;
  struct C { const char* name; uint32_t type; int K, N; };
  const C cases[] = {{"IQ4_XS 10240x5120", QT_IQ4_XS, 5120, 10240}, {"IQ2_XS 5120x17408", QT_IQ2_XS, 17408, 5120},
                     {"IQ1_S 17408x5120", QT_IQ1_S, 5120, 17408}, {"IQ2_XXS 17408x5120", QT_IQ2_XXS, 5120, 17408}};
  for (const C& c : cases) {
    const size_t bb = qk::gq_buffer_bytes(c.K, c.N);
    void* buf; CUDA_CHECK(cudaMalloc(&buf, bb)); CUDA_CHECK(cudaMemset(buf, 1, bb));
    int8_t* x; float* xd; int16_t* xs; float* y;
    CUDA_CHECK(cudaMalloc(&x, (size_t)T * c.K)); CUDA_CHECK(cudaMemset(x, 1, (size_t)T * c.K));
    CUDA_CHECK(cudaMalloc(&xd, (size_t)T * c.K / 256 * 4)); CUDA_CHECK(cudaMemset(xd, 0, (size_t)T * c.K / 256 * 4));
    CUDA_CHECK(cudaMalloc(&xs, (size_t)T * c.K / 16 * 2)); CUDA_CHECK(cudaMemset(xs, 0, (size_t)T * c.K / 16 * 2));
    CUDA_CHECK(cudaMalloc(&y, (size_t)T * c.N * 4));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 2; i++) qk::gq_gemm(c.type, buf, c.K, c.N, x, xd, xs, T, y, c.N, false, 0);
    cudaEventRecord(a);
    const int it = 10;
    for (int i = 0; i < it; i++) qk::gq_gemm(c.type, buf, c.K, c.N, x, xd, xs, T, y, c.N, false, 0);
    cudaEventRecord(b); CUDA_CHECK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= it;
    printf("%-20s T=%d: %.3f ms  %.1f TOPS\n", c.name, T, ms, 2.0 * c.K * c.N * T / ms / 1e9);
    cudaFree(buf); cudaFree(x); cudaFree(xd); cudaFree(xs); cudaFree(y);
  }
}
