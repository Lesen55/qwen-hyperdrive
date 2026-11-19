// Microbenchmark of the decode matvec kernels on real tensors (B = 1..4): achieved bandwidth.
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
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
  std::vector<std::string> list(std::begin(names), std::end(names));
  if (argc > 2 && std::string(argv[2]) == "types") {  // one ffn / attention / ssm tensor per (type, shape)
    list.clear();
    std::map<std::pair<uint32_t, std::pair<int64_t, int64_t>>, int> seen;
    for (auto& t : f.tensors()) {
      if (t.name.rfind("blk.", 0) != 0 || t.ne.size() != 2 || t.ne[0] % 256) continue;
      auto k = std::make_pair((uint32_t)t.type, std::make_pair(t.ne[0], t.rows()));
      if (seen[k]++ == 0) list.push_back(t.name);
    }
  }
  for (const std::string& ns : list) {
    const char* n = ns.c_str();
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
      if (B >= 2 && qk::matvec_tq_supported(t->type) && K % 256 == 0) {  // tensor-core variant + max rel. diff
        static int8_t* xq2 = nullptr; static float* xd2 = nullptr; static float* y2 = nullptr;
        if (!xq2) { CUDA_CHECK(cudaMalloc(&xq2, 8 * 32768)); CUDA_CHECK(cudaMalloc(&xd2, 8 * 256 * 4)); CUDA_CHECK(cudaMalloc(&y2, 8 * 248320 * 4)); }
        qk::quantize_q8_256(x, xq2, xd2, K, B, 0);
        for (int i = 0; i < 10; i++) qk::matvec_tq(t->type, W, rb, K, N, xq2, xd2, B, y2, N, false, 0);
        float ms2 = 1e30f;
        for (int trial = 0; trial < 7; trial++) {
          cudaEventRecord(e0);
          for (int i = 0; i < reps; i++) qk::matvec_tq(t->type, W, rb, K, N, xq2, xd2, B, y2, N, false, 0);
          cudaEventRecord(e1); cudaEventSynchronize(e1);
          float m; cudaEventElapsedTime(&m, e0, e1);
          ms2 = std::min(ms2, m);
        }
        std::vector<float> ha((size_t)B * N), hb((size_t)B * N);
        CUDA_CHECK(cudaMemcpy(ha.data(), y, ha.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), y2, hb.size() * 4, cudaMemcpyDeviceToHost));
        double md = 0, mx = 0;
        for (size_t i = 0; i < ha.size(); i++) { md = std::max(md, (double)fabsf(ha[i] - hb[i])); mx = std::max(mx, (double)fabsf(ha[i])); }
        printf("/tq %4.0f (d %.0e)", t->nbytes / (ms2 / reps * 1e-3) / 1e9, md / std::max(mx, 1e-30));
        {  // per-32 activations (the dp4a kernels' xq / xd)
          for (int i = 0; i < 10; i++) qk::matvec_tq(t->type, W, rb, K, N, xq, xd, B, y2, N, false, 0, 32);
          float ms3 = 1e30f;
          for (int trial = 0; trial < 7; trial++) {
            cudaEventRecord(e0);
            for (int i = 0; i < reps; i++) qk::matvec_tq(t->type, W, rb, K, N, xq, xd, B, y2, N, false, 0, 32);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float m; cudaEventElapsedTime(&m, e0, e1);
            ms3 = std::min(ms3, m);
          }
          CUDA_CHECK(cudaMemcpy(hb.data(), y2, hb.size() * 4, cudaMemcpyDeviceToHost));
          double md3 = 0;
          for (size_t i = 0; i < ha.size(); i++) md3 = std::max(md3, (double)fabsf(ha[i] - hb[i]));
          printf("/x32 %4.0f (d %.0e)", t->nbytes / (ms3 / reps * 1e-3) / 1e9, md3 / std::max(mx, 1e-30));
        }
      }
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
