// MoE small-batch matvec bandwidth: qe_bench_moe model.gguf layer
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "kernels.h"

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  GGUFFile g(argv[1]);
  const std::string L = std::string("blk.") + argv[2] + ".";
  for (const char* nm : {"ffn_gate_up_exps.weight", "ffn_down_exps.weight"}) {
    const GGUFTensor& t = g.need(L + nm);
    const int K = (int)t.ne[0], N = (int)t.ne[1], E = (int)t.ne[2];
    const size_t rb = qtype_row_bytes(t.type, K), stride = rb * N;
    void* w;
    CUDA_CHECK(cudaMalloc(&w, t.nbytes));
    CUDA_CHECK(cudaMemcpy(w, t.data, t.nbytes, cudaMemcpyHostToDevice));
    for (int A : {8, 32, 64}) {
      std::vector<int> ids(A);
      std::mt19937 r(1);
      for (auto& x : ids) x = r() % E;
      int* idd;
      int8_t* xq;
      float *xd, *y;
      CUDA_CHECK(cudaMalloc(&idd, A * 4));
      CUDA_CHECK(cudaMemcpy(idd, ids.data(), A * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&xq, (size_t)A * K));
      CUDA_CHECK(cudaMemset(xq, 1, (size_t)A * K));
      CUDA_CHECK(cudaMalloc(&xd, (size_t)A * K / 32 * 4));
      CUDA_CHECK(cudaMemset(xd, 0, (size_t)A * K / 32 * 4));
      CUDA_CHECK(cudaMalloc(&y, (size_t)A * N * 4));
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      for (int it = 0; it < 3; it++) qk::moe_matvec(t.type, w, stride, rb, K, N, A, idd, idd, nullptr, xq, xd, 1, y, 0);
      cudaEventRecord(a);
      const int R = 50;
      for (int it = 0; it < R; it++) qk::moe_matvec(t.type, w, stride, rb, K, N, A, idd, idd, nullptr, xq, xd, 1, y, 0);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float ms;
      cudaEventElapsedTime(&ms, a, b);
      printf("%-24s %-7s K=%d N=%d A=%2d: %.1f us, %.0f GB/s\n", nm, qtype_info(t.type).name, K, N, A, ms * 1e3 / R,
             (double)stride * A * R / (ms * 1e-3) / 1e9);
    }
  }
  return 0;
}
