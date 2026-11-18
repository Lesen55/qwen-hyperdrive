// Microbenchmark of the decode matvec kernels on real tensors (B = 1..4): achieved bandwidth.
#include <map>
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
  CUDA_CHECK(cudaMalloc(&x, 4 * 32768 * 4));
  CUDA_CHECK(cudaMemset(x, 0, 4 * 32768 * 4));
  CUDA_CHECK(cudaMalloc(&y, 4 * 248320 * 4));
  CUDA_CHECK(cudaMalloc(&xq, 4 * 32768));
  CUDA_CHECK(cudaMalloc(&xd, 4 * 1024 * 4));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (const char* n : names) {
    const GGUFTensor* t = f.tensor(n);
    if (!t) continue;
    const int K = (int)t->ne[0], N = (int)t->rows();
    const size_t rb = qtype_row_bytes(t->type, K);
    void* W; CUDA_CHECK(cudaMalloc(&W, t->nbytes));
    CUDA_CHECK(cudaMemcpy(W, t->data, t->nbytes, cudaMemcpyHostToDevice));
    printf("%-28s %-8s %6d x %6d %7.1f MB:", n, qtype_info(t->type).name, N, K, t->nbytes / 1e6);
    for (int B : {1, 2, 4, 5, 8}) {
      qk::quantize_q8(x, xq, xd, K, B, 0);
      qk::matvec(t->type, W, rb, K, N, xq, xd, x, B, y, N, false, 0);
      cudaEventRecord(e0);
      const int reps = 20;
      for (int i = 0; i < reps; i++) qk::matvec(t->type, W, rb, K, N, xq, xd, x, B, y, N, false, 0);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      printf("  B=%d %6.1f us %4.0f GB/s", B, ms / reps * 1e3, t->nbytes / (ms / reps * 1e-3) / 1e9);
    }
    printf("\n");
    cudaFree(W);
  }
}
