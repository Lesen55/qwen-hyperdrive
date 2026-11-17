// GPU top-K vs CPU reference on random rows
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include "../src/common.h"
#include "../src/kernels.h"
int main() {
  const int V = 248320, R = 8, K = 20;
  std::vector<float> h((size_t)R * V);
  std::mt19937 g(1);
  std::normal_distribution<float> nd(0, 3);
  for (auto& x : h) x = nd(g);
  float *d, *ov; int* oi; void* w;
  CUDA_CHECK(cudaMalloc(&d, h.size() * 4)); CUDA_CHECK(cudaMalloc(&ov, R * K * 4)); CUDA_CHECK(cudaMalloc(&oi, R * K * 4));
  CUDA_CHECK(cudaMalloc(&w, qk::topk_work_bytes(R)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  qk::topk_rows(d, V, R, K, ov, oi, w, 0);
  std::vector<float> gv(R * K); std::vector<int> gi(R * K);
  CUDA_CHECK(cudaMemcpy(gv.data(), ov, R * K * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gi.data(), oi, R * K * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int r = 0; r < R; r++) {
    std::vector<int> id(V);
    for (int i = 0; i < V; i++) id[i] = i;
    std::partial_sort(id.begin(), id.begin() + K, id.end(), [&](int a, int b) { return h[(size_t)r * V + a] > h[(size_t)r * V + b]; });
    for (int i = 0; i < K; i++)
      if (id[i] != gi[r * K + i]) { if (bad++ < 5) printf("row %d pos %d: cpu %d (%f) gpu %d (%f)\n", r, i, id[i], h[(size_t)r*V+id[i]], gi[r*K+i], gv[r*K+i]); }
  }
  printf(bad ? "FAIL %d\n" : "OK\n", bad);
}
