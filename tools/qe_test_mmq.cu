// MMQ (prefill int8 tensor-core GEMM) vs fp64 reference on real tensors, and throughput.
#include <cmath>
#include <map>
#include <random>
#include "gguf.h"
#include "kernels.h"
#include "quant.h"

int main(int argc, char** argv) {
  GGUFFile f(argv[1]);
  std::map<uint32_t, const GGUFTensor*> pick;
  for (auto& t : f.tensors())
    if (t.ne.size() == 2 && t.ne[1] >= 1024 && qk::mmq_supported(t.type) && !pick.count(t.type)) pick[t.type] = &t;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 1.f);
  bool ok_all = true;
  const int T = 200;
  for (auto [type, t] : pick) {
    const int K = (int)t->ne[0], N = (int)t->rows();
    const size_t rb = qtype_row_bytes(type, K);
    std::vector<float> x((size_t)T * K);
    for (auto& v : x) v = nd(rng);
    void* dW; float *dx, *dy, *dxd; int8_t* dxq;
    CUDA_CHECK(cudaMalloc(&dW, t->nbytes));
    CUDA_CHECK(cudaMemcpy(dW, t->data, t->nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dx, x.size() * 4));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dy, (size_t)T * N * 4));
    CUDA_CHECK(cudaMalloc(&dxq, (size_t)T * K));
    CUDA_CHECK(cudaMalloc(&dxd, (size_t)T * K / 32 * 4));
    const int GS = getenv("GS") ? atoi(getenv("GS")) : 32;
    if (GS == 32) qk::quantize_q8(dx, dxq, dxd, K, T, 0);
    else qk::quantize_q8_256(dx, dxq, dxd, K, T, 0);
    qk::mmq(type, dW, rb, K, N, dxq, dxd, T, dy, N, false, GS, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> y((size_t)T * N);
    CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    // check 32 rows x all tokens
    std::vector<float> w(K);
    double worst = 0, rms = 0;
    std::vector<double> ref(32 * T);
    for (int ri = 0; ri < 32; ri++) {
      const int r = (int)((int64_t)ri * (N - 1) / 31);
      dequant_row_ref(type, t->data + r * rb, K, w.data());
      for (int tt = 0; tt < T; tt++) {
        double s = 0;
        for (int k = 0; k < K; k++) s += (double)w[k] * x[(size_t)tt * K + k];
        ref[ri * T + tt] = s;
        rms += s * s;
      }
    }
    rms = std::sqrt(rms / (32.0 * T));
    for (int ri = 0; ri < 32; ri++) {
      const int r = (int)((int64_t)ri * (N - 1) / 31);
      for (int tt = 0; tt < T; tt++) worst = std::max(worst, std::fabs(y[(size_t)tt * N + r] - ref[ri * T + tt]) / rms);
    }
    // throughput at T=512
    const int TB = 512;
    float* dy2; int8_t* dxq2; float* dxd2;
    CUDA_CHECK(cudaMalloc(&dy2, (size_t)TB * N * 4));
    CUDA_CHECK(cudaMalloc(&dxq2, (size_t)TB * K));
    CUDA_CHECK(cudaMemset(dxq2, 1, (size_t)TB * K));
    CUDA_CHECK(cudaMalloc(&dxd2, (size_t)TB * K / 32 * 4));
    CUDA_CHECK(cudaMemset(dxd2, 0, (size_t)TB * K / 32 * 4));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    qk::mmq(type, dW, rb, K, N, dxq2, dxd2, TB, dy2, N, false, GS, 0);
    cudaEventRecord(e0);
    for (int i = 0; i < 5; i++) qk::mmq(type, dW, rb, K, N, dxq2, dxd2, TB, dy2, N, false, GS, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
    const bool ok = worst < 0.03;
    ok_all &= ok;
    printf("%-8s %-28s %6d x %6d  err/rms=%.4f  T=512: %.3f ms  %.1f TOPS  %s\n", qtype_info(type).name, t->name.c_str(), N, K,
           worst, ms, 2.0 * TB * (double)N * K / (ms * 1e-3) / 1e12, ok ? "OK" : "FAIL");
    cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dxq); cudaFree(dxd); cudaFree(dy2); cudaFree(dxq2); cudaFree(dxd2);
  }
  printf(ok_all ? "ALL OK\n" : "FAILURES\n");
  return ok_all ? 0 : 1;
}
