// Microbenchmark: decode/prefill attention against a 128K q4 KV cache (one layer).
#include <algorithm>
#include <cmath>
#include <vector>
#include "common.h"
#include "kernels.h"

int main(int argc, char** argv) {
  const int ctx = argc > 1 ? atoi(argv[1]) : 131072;
  const int kvt = argc > 2 ? atoi(argv[2]) : qk::KV_Q4, nh = 24, nkv = 4, hd = 256;  // argv[2]: KV type
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
  const size_t wb = std::max(qk::attn_decode_work_bytes(ctx, nh, hd, 4), qk::attn_decode_work_bytes_auto(ctx, nh, nkv, hd, 8));
  float* work; CUDA_CHECK(cudaMalloc(&work, wb));
  void* fw; CUDA_CHECK(cudaMalloc(&fw, qk::attn_prefill_work_bytes(T, nh, hd)));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int Tq : {1, 2, 3, 4, 5, 6, 8}) {
    const int p = ctx - Tq;
    qk::attn_decode(kvt, kc, vc, q, 2 * hd, nh * 2 * hd, o, nh, nkv, hd, p, Tq, 0.0625f, work, wb, bs, 0);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; i++) qk::attn_decode(kvt, kc, vc, q, 2 * hd, nh * 2 * hd, o, nh, nkv, hd, p, Tq, 0.0625f, work, wb, bs, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    const double bytes = 2.0 * ctx * nkv * recb;
    printf("decode T=%d at pos %d: %.3f ms/layer (%.0f GB/s of KV)\n", Tq, p, ms / 20, bytes / (ms / 20 * 1e-3) / 1e9);
  }
  {  // correctness: one T=8 (or 6) launch == launches of 4 (+ rest) at the same positions
    std::vector<float> hq((size_t)8 * nh * 2 * hd);
    for (auto& x : hq) x = (rand() / (float)RAND_MAX - 0.5f) * 4.f;
    CUDA_CHECK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    for (int Tq : {5, 6, 8}) {
      const int p = ctx / 2 + 7;
      std::vector<float> a((size_t)Tq * nh * hd), b(a.size());
      qk::attn_decode(kvt, kc, vc, q, 2 * hd, nh * 2 * hd, o, nh, nkv, hd, p, Tq, 0.0625f, work, wb, bs, 0);
      CUDA_CHECK(cudaMemcpy(a.data(), o, a.size() * 4, cudaMemcpyDeviceToHost));
      for (int t0 = 0; t0 < Tq; t0 += 4) {
        const int nt = std::min(4, Tq - t0);
        qk::attn_decode(kvt, kc, vc, q + (size_t)t0 * nh * 2 * hd, 2 * hd, nh * 2 * hd, o + (size_t)t0 * nh * hd, nh, nkv, hd, p + t0, nt, 0.0625f, work, wb, bs, 0);
      }
      CUDA_CHECK(cudaMemcpy(b.data(), o, b.size() * 4, cudaMemcpyDeviceToHost));
      double md = 0, mx = 0;
      for (size_t i = 0; i < a.size(); i++) { md = std::max(md, (double)fabsf(a[i] - b[i])); mx = std::max(mx, (double)fabsf(b[i])); }
      printf("check T=%d vs chunks of 4: max |diff| %.3g (max |o| %.3g)\n", Tq, md, mx);
    }
  }
  for (int p0 : {0, ctx / 4, ctx / 2, ctx - T}) {
    cudaEventRecord(e0);
    qk::attn_prefill(kvt, kc, vc, q, 2 * hd, nh * 2 * hd, o, nh, nkv, hd, p0, T, 0.0625f, fw, bs, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    const double flop = 4.0 * T * (p0 + T / 2.0) * hd * nh;
    printf("prefill T=%d at pos %d: %.2f ms/layer (%.1f TFLOP/s)\n", T, p0, ms, flop / (ms * 1e-3) / 1e12);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}
