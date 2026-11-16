// Microbenchmark: decode/prefill attention against a 128K q4 KV cache (one layer).
#include <vector>
#include "common.h"
#include "kernels.h"

int main(int argc, char** argv) {
  const int ctx = argc > 1 ? atoi(argv[1]) : 131072;
  const int kvt = qk::KV_Q4, nh = 24, nkv = 4, hd = 256;
  const size_t recb = qk::kv_bytes_per_token_head(kvt, hd);
  void *kc, *vc;
  CUDA_CHECK(cudaMalloc(&kc, (size_t)ctx * nkv * recb));
  CUDA_CHECK(cudaMalloc(&vc, (size_t)ctx * nkv * recb));
  const qk::KVMap bs{(long long)nkv * qk::KV_BLOCK * recb, nullptr};  // one layer, blocks back to back
  // fill with valid records via kv_store of random data
  const int T = 512;
  std::vector<float> h((size_t)T * nkv * hd);
  for (auto& x : h) x = (rand() / (float)RAND_MAX - 0.5f);
  float *k, *q, *o; int* pos;
  CUDA_CHECK(cudaMalloc(&k, h.size() * 4));
  CUDA_CHECK(cudaMemcpy(k, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&q, (size_t)T * nh * 2 * hd * 4));
  CUDA_CHECK(cudaMemset(q, 0, (size_t)T * nh * 2 * hd * 4));
  CUDA_CHECK(cudaMalloc(&o, (size_t)T * nh * hd * 4));
  CUDA_CHECK(cudaMalloc(&pos, T * 4));
  for (int p0 = 0; p0 < ctx; p0 += T) {
    std::vector<int> pp(T);
    for (int i = 0; i < T; i++) pp[i] = p0 + i;
    CUDA_CHECK(cudaMemcpy(pos, pp.data(), T * 4, cudaMemcpyHostToDevice));
    qk::kv_store(kvt, kc, vc, k, k, pos, nkv, hd, T, nkv * hd, nkv * hd, bs, 0);
  }
  const size_t wb = qk::attn_decode_work_bytes(ctx, nh, hd, 4);
  float* work; CUDA_CHECK(cudaMalloc(&work, wb));
  void* fw; CUDA_CHECK(cudaMalloc(&fw, qk::attn_prefill_work_bytes(T, nh, hd)));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int Tq : {1, 2, 3, 4}) {
    const int p = ctx - Tq;
    qk::attn_decode(kvt, kc, vc, q, 2 * hd, nh * 2 * hd, o, nh, nkv, hd, p, Tq, 0.0625f, work, wb, bs, 0);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; i++) qk::attn_decode(kvt, kc, vc, q, 2 * hd, nh * 2 * hd, o, nh, nkv, hd, p, Tq, 0.0625f, work, wb, bs, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    const double bytes = 2.0 * ctx * nkv * recb;
    printf("decode T=%d at pos %d: %.3f ms/layer (%.0f GB/s of KV)\n", Tq, p, ms / 20, bytes / (ms / 20 * 1e-3) / 1e9);
  }
  for (int p0 : {0, ctx / 2, ctx - T}) {
    cudaEventRecord(e0);
    qk::attn_prefill(kvt, kc, vc, q, 2 * hd, nh * 2 * hd, o, nh, nkv, hd, p0, T, 0.0625f, fw, bs, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    const double flop = 4.0 * T * (p0 + T / 2.0) * hd * nh;
    printf("prefill T=%d at pos %d: %.2f ms/layer (%.1f TFLOP/s)\n", T, p0, ms, flop / (ms * 1e-3) / 1e12);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}
