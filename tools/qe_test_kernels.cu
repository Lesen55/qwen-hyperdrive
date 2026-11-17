// Unit tests: GPU matvec (dp4a) and dequant_f16 vs host reference dequant, for every weight
// type present in the model, batch 1..8.  Uses real tensors from the GGUF.
#include <cmath>
#include <map>
#include <random>

#include "gguf.h"
#include "kernels.h"
#include "quant.h"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: qe_test_kernels model.gguf\n"); return 1; }
  GGUFFile f(argv[1]);
  std::map<uint32_t, const GGUFTensor*> pick;
  for (auto& t : f.tensors())
    if (t.ne.size() == 2 && t.ne[1] >= 64 && !pick.count(t.type)) pick[t.type] = &t;
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  bool all_ok = true;
  for (auto [type, t] : pick) {
    const int K = (int)t->ne[0];
    const int N = (int)std::min<int64_t>(256, t->rows());  // test the first 256 rows
    const size_t rb = qtype_row_bytes(type, K);
    // host reference
    std::vector<float> W((size_t)N * K);
    for (int r = 0; r < N; r++) dequant_row_ref(type, t->data + r * rb, K, &W[(size_t)r * K]);
    const int B = 8;
    std::vector<float> x((size_t)B * K);
    for (auto& v : x) v = nd(rng);
    std::vector<double> ref((size_t)B * N);
    for (int b = 0; b < B; b++)
      for (int r = 0; r < N; r++) {
        double s = 0;
        for (int k = 0; k < K; k++) s += (double)W[(size_t)r * K + k] * x[(size_t)b * K + k];
        ref[(size_t)b * N + r] = s;
      }
    // device
    void* dW; float *dx, *dy, *dxd; int8_t* dxq; __half* d16;
    CUDA_CHECK(cudaMalloc(&dW, rb * N));
    CUDA_CHECK(cudaMemcpy(dW, t->data, rb * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dx, x.size() * 4));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dy, (size_t)B * N * 4));
    CUDA_CHECK(cudaMalloc(&dxq, (size_t)B * K));
    CUDA_CHECK(cudaMalloc(&dxd, (size_t)B * K / 32 * 4));
    CUDA_CHECK(cudaMalloc(&d16, (size_t)N * K * 2));
    qk::quantize_q8(dx, dxq, dxd, K, B, 0);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    double worst = 0;
    for (int bb = 1; bb <= B; bb++) {
      qk::matvec(type, dW, rb, K, N, dxq, dxd, dx, bb, dy, N, false, 0);
      std::vector<float> y((size_t)bb * N);
      CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
      // error relative to the rms of the reference outputs (int8 activation quantization adds ~0.5%)
      double rms = 0;
      for (int i = 0; i < bb * N; i++) rms += ref[i] * ref[i];
      rms = std::sqrt(rms / (bb * N));
      double e = 0;
      for (int i = 0; i < bb * N; i++) e = std::max(e, std::fabs(y[i] - ref[i]) / rms);
      worst = std::max(worst, e);
    }
    qk::dequant_f16(type, dW, rb, K, N, d16, 0);
    std::vector<__half> h((size_t)N * K);
    CUDA_CHECK(cudaMemcpy(h.data(), d16, h.size() * 2, cudaMemcpyDeviceToHost));
    double derr = 0, wmax = 0;
    for (size_t i = 0; i < h.size(); i++) {
      derr = std::max(derr, (double)std::fabs(__half2float(h[i]) - W[i]));
      wmax = std::max(wmax, (double)std::fabs(W[i]));
    }
    const bool is_float = type == QT_F32 || type == QT_F16 || type == QT_BF16;
    const bool ok = worst < (is_float ? 1e-4 : 0.03) && derr / wmax < 2e-3;
    all_ok &= ok;
    printf("%-8s %-30s K=%5d  matvec max_err/rms=%.4f  dequant_f16 rel=%.2e  %s\n", qtype_info(type).name,
           t->name.c_str(), K, worst, derr / wmax, ok ? "OK" : "FAIL");
    cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dxq); cudaFree(dxd); cudaFree(d16);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  printf(all_ok ? "ALL OK\n" : "FAILURES\n");
  return all_ok ? 0 : 1;
}
