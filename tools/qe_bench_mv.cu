// Microbenchmark of the decode matvec kernels on real tensors (B = 1..4): achieved bandwidth.
#include <algorithm>
#include <map>
#include <vector>
#include "gguf.h"
#include "kernels.h"
#include "quant.h"

int main(int argc, char** argv) {
  GGUFFile f(argv[1]);
  const char* names[] = {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.attn_qkv.weight",
                         "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ssm_alpha.weight",
                         "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_output.weight",
                         "blk.10.ffn_up.weight", "blk.20.ffn_down.weight", "output.weight"};
  float *x, *y; int8_t* xq; float* xd;
  CUDA_CHECK(cudaMalloc(&x, 8 * 32768 * 4));
  {
    std::vector<float> hx(8 * 32768);
    for (auto& v : hx) v = rand() / (float)RAND_MAX - 0.5f;
    CUDA_CHECK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
  }
  const bool check = getenv("QE_BENCH_CHECK") != nullptr;  // print an output checksum per (tensor, B)
  CUDA_CHECK(cudaMalloc(&y, 8 * 248320 * 4));
  CUDA_CHECK(cudaMalloc(&xq, 8 * 32768));
  CUDA_CHECK(cudaMalloc(&xd, 8 * 1024 * 4));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (const char* n : names) {
    const GGUFTensor* t = f.tensor(n);
    if (!t) continue;
    const int K = (int)t->ne[0], N = (int)t->rows();
    const size_t rb = qtype_row_bytes(t->type, K);
    void* W; CUDA_CHECK(cudaMalloc(&W, t->nbytes));
    CUDA_CHECK(cudaMemcpy(W, t->data, t->nbytes, cudaMemcpyHostToDevice));
    printf("%-24s %-8s %5.1f MB GB/s:", n, qtype_info(t->type).name, t->nbytes / 1e6);
    for (int B : {1, 2, 4, 5, 8}) {
      qk::quantize_q8(x, xq, xd, K, B, 0);
      for (int i = 0; i < 10; i++) qk::matvec(t->type, W, rb, K, N, xq, xd, x, B, y, N, false, 0);
      const int reps = 20;
      float ms = 1e30f;
      for (int trial = 0; trial < 7; trial++) {  // best of 7 (clocks and the L2 state vary run to run)
        cudaEventRecord(e0);
        for (int i = 0; i < reps; i++) qk::matvec(t->type, W, rb, K, N, xq, xd, x, B, y, N, false, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float m; cudaEventElapsedTime(&m, e0, e1);
        ms = std::min(ms, m);
      }
      printf(" B%d %4.0f", B, t->nbytes / (ms / reps * 1e-3) / 1e9);
      if (check) {
        std::vector<float> hy((size_t)B * N);
        CUDA_CHECK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
        double cs = 0;
        for (size_t i = 0; i < hy.size(); i++) cs += hy[i] * (double)((i % 97) + 1);
        printf(" [%.6e]", cs);
      }
    }
    printf("\n");
    cudaFree(W);
  }
}
