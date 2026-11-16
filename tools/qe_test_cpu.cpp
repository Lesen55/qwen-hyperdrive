// CPU AVX2 matvec: correctness vs reference dequant, and multi-thread throughput on real FFN tensors.
#include <cmath>
#include <map>
#include <random>

#include "cpu_ffn.h"
#include "gguf.h"
#include "quant.h"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: qe_test_cpu model.gguf [threads]\n"); return 1; }
  GGUFFile f(argv[1]);
  const int nth = argc > 2 ? atoi(argv[2]) : 6;
  cpu::ThreadPool pool(nth);
  std::map<uint32_t, const GGUFTensor*> pick;
  for (auto& t : f.tensors())
    if (t.name.find("ffn_") != std::string::npos && cpu::type_supported(t.type) && !pick.count(t.type)) pick[t.type] = &t;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  bool ok_all = true;
  for (auto [type, t] : pick) {
    const int K = (int)t->ne[0], N = (int)t->rows();
    const size_t rb = qtype_row_bytes(type, K);
    for (int B : {1, 2, 4}) {
      std::vector<float> x((size_t)B * K);
      for (auto& v : x) v = nd(rng);
      cpu::Q8Act q;
      q.resize(K, B);
      for (int b = 0; b < B; b++) cpu::quantize(x.data(), K, b, q, 0, K / 256);
      // correctness on 64 rows
      const int NR = 64;
      std::vector<float> y((size_t)B * N);
      cpu::matvec(type, t->data, rb, K, 0, NR, q, B, y.data(), N);
      std::vector<float> w(K);
      double worst = 0, rms = 0;
      std::vector<double> ref((size_t)B * NR);
      for (int r = 0; r < NR; r++) {
        dequant_row_ref(type, t->data + r * rb, K, w.data());
        for (int b = 0; b < B; b++) {
          double s = 0;
          for (int k = 0; k < K; k++) s += (double)w[k] * x[(size_t)b * K + k];
          ref[(size_t)b * NR + r] = s;
          rms += s * s;
        }
      }
      rms = std::sqrt(rms / (B * NR));
      for (int r = 0; r < NR; r++)
        for (int b = 0; b < B; b++) worst = std::max(worst, std::fabs(y[(size_t)b * N + r] - ref[(size_t)b * NR + r]) / rms);
      // throughput, full matrix, nth threads
      const int reps = 5;
      double t0 = qe_now();
      for (int rep = 0; rep < reps; rep++)
        pool.run([&](int i) {
          const int r0 = (int)((int64_t)N * i / nth), r1 = (int)((int64_t)N * (i + 1) / nth);
          cpu::matvec(type, t->data, rb, K, r0, r1, q, B, y.data(), N);
        });
      const double dt = (qe_now() - t0) / reps;
      double dt4 = 0, worst4 = 0;
      if (cpu::r4_supported(type)) {
        std::vector<uint8_t> r4((size_t)N * (K / 256) * cpu::R4_BLOCK_BYTES);
        cpu::r4_convert(type, t->data, rb, K, N, r4.data());
        std::vector<float> y4((size_t)B * N);
        cpu::matvec_r4(cpu::r4_table(type), r4.data(), K, 0, NR, q, B, y4.data(), N);
        for (int r = 0; r < NR; r++)
          for (int b = 0; b < B; b++) worst4 = std::max(worst4, (double)std::fabs(y4[(size_t)b * N + r] - y[(size_t)b * N + r]) / rms);
        const double t1 = qe_now();
        for (int rep = 0; rep < reps; rep++)
          pool.run([&](int i) {
            const int r0 = (int)((int64_t)N * i / nth), r1 = (int)((int64_t)N * (i + 1) / nth);
            cpu::matvec_r4(cpu::r4_table(type), r4.data(), K, r0, r1, q, B, y4.data(), N);
          });
        dt4 = (qe_now() - t1) / reps;
        printf("   R4 vs native diff/rms=%.2e  %6.2f ms (%.2fx)\n", worst4, dt4 * 1e3, dt / dt4);
        ok_all &= worst4 < 1e-4;
      }
      const bool ok = worst < 0.03;
      ok_all &= ok;
      printf("%-8s %-26s B=%d err/rms=%.4f  %6.2f ms  %5.1f GB/s  %s\n", qtype_info(type).name, t->name.c_str(), B, worst,
             dt * 1e3, t->nbytes / dt / 1e9, ok ? "OK" : "FAIL");
    }
  }
  printf(ok_all ? "ALL OK\n" : "FAILURES\n");
  return ok_all ? 0 : 1;
}
