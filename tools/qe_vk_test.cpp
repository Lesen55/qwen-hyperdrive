// Vulkan kernel tests: for one tensor of every weight type in a GGUF, compare the shader dequant,
// matvec (T = 1, 3, 8) and GEMM (T = 40) against the host reference dequant in fp64.
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "quant.h"
#include "vk_ctx.h"

struct MvPC { uint64_t w, x, y, tab; uint32_t K, N, T, rb, ldx, ldy, acc, wide, kg; };

static const char* tag(uint32_t t) {
  switch (t) {
    case QT_F32: return "f32"; case QT_F16: return "f16"; case QT_BF16: return "bf16"; case QT_Q8_0: return "q8_0";
    case QT_Q2_K: return "q2_k"; case QT_Q3_K: return "q3_k"; case QT_Q4_K: return "q4_k"; case QT_Q5_K: return "q5_k";
    case QT_Q6_K: return "q6_k"; case QT_IQ2_XXS: return "iq2_xxs"; case QT_IQ2_XS: return "iq2_xs";
    case QT_IQ2_S: return "iq2_s"; case QT_IQ3_XXS: return "iq3_xxs"; case QT_IQ3_S: return "iq3_s";
    case QT_IQ1_S: return "iq1_s"; case QT_IQ1_M: return "iq1_m"; case QT_IQ4_XS: return "iq4_xs";
    default: return nullptr;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: qe_vk_test model.gguf\n"); return 2; }
  GGUFFile f(argv[1]);
  std::map<uint32_t, const GGUFTensor*> pick;
  for (auto& t : f.tensors())
    if (t.ne.size() == 2 && t.ne[0] % 256 == 0 && tag(t.type) && !pick.count(t.type)) pick[t.type] = &t;
  VkCtx vk;
  printf("device: %s (%s), subgroup %u\n", vk.device_name.c_str(), vk.driver_name.c_str(), vk.subgroup_size);
  std::vector<uint32_t> tab(8452, 0);
  memcpy(&tab[0], qtab::iq2xxs_grid, sizeof(qtab::iq2xxs_grid));
  memcpy(&tab[512], qtab::iq2xs_grid, sizeof(qtab::iq2xs_grid));
  memcpy(&tab[1536], qtab::iq2s_grid, sizeof(qtab::iq2s_grid));
  memcpy(&tab[3584], qtab::iq3xxs_grid, sizeof(qtab::iq3xxs_grid));
  memcpy(&tab[3840], qtab::iq3s_grid, sizeof(qtab::iq3s_grid));
  memcpy(&tab[4352], qtab::iq1s_grid, sizeof(qtab::iq1s_grid));
  memcpy(&tab[8448], qtab::kvalues_iq4nl, sizeof(qtab::kvalues_iq4nl));
  VBuf dtab = vk.alloc(tab.size() * 4, Mem::Host);
  memcpy(dtab.map, tab.data(), tab.size() * 4);
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  bool all_ok = true;
  for (auto [type, t] : pick) {
    const int K = (int)t->ne[0];
    const int N = (int)std::min<int64_t>(t->rows(), 300);  // not a multiple of 16: partial workgroups
    const size_t rb = qtype_row_bytes(type, K);
    std::vector<float> W((size_t)N * K);
    for (int r = 0; r < N; r++) dequant_row_ref(type, t->data + r * rb, K, &W[(size_t)r * K]);
    VBuf dw = vk.alloc(rb * N, Mem::Host), dy = vk.alloc((size_t)N * K * 4, Mem::Host);
    memcpy(dw.map, t->data, rb * N);
    // dequant
    MvPC dq{dw.addr, 0, dy.addr, dtab.addr, (uint32_t)K, (uint32_t)N, 1, (uint32_t)rb, 0, 0, 0, 0, (uint32_t)(K / 32)};
    vk.begin();
    vk.dispatch((std::string("dq_") + tag(type)).c_str(), &dq, sizeof dq, (uint32_t)(((size_t)N * K / 32 + 63) / 64));
    vk.submit();
    double derr = 0, wmax = 1e-30;
    for (size_t i = 0; i < W.size(); i++) {
      derr = std::max(derr, (double)std::fabs(((float*)dy.map)[i] - W[i]));
      wmax = std::max(wmax, (double)std::fabs(W[i]));
    }
    // matvec / GEMM
    double worst = 0, worst16 = 0;
    for (int T : {1, 3, 8, 40}) {
      std::vector<float> x((size_t)T * K);
      for (auto& v : x) v = nd(rng);
      VBuf dx = vk.alloc(x.size() * 4, Mem::Host), dout = vk.alloc((size_t)T * N * 4, Mem::Host);
      memcpy(dx.map, x.data(), x.size() * 4);
      for (int wide = 0; wide < (T <= 8 ? 2 : 1); wide++) {
        MvPC pc{dw.addr, dx.addr, dout.addr, dtab.addr, (uint32_t)K, (uint32_t)N, (uint32_t)T, (uint32_t)rb, (uint32_t)K,
                (uint32_t)N, 0, (uint32_t)wide, (uint32_t)(K / 32)};
        vk.begin();
        PipeOpt po;
        po.spec2 = T > 1;
        if (T <= 8) vk.dispatch((std::string("mv_") + tag(type)).c_str(), &pc, sizeof pc, wide ? N : (N + 15) / 16, 1, 1, true, po);
        else vk.dispatch((std::string("mm_") + tag(type)).c_str(), &pc, sizeof pc, (N + 63) / 64, (T + 63) / 64);
        vk.submit();
        double rms = 0, e = 0;
        std::vector<double> ref((size_t)T * N);
        for (int b = 0; b < T; b++)
          for (int r = 0; r < N; r++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)W[(size_t)r * K + k] * x[(size_t)b * K + k];
            ref[(size_t)b * N + r] = s;
            rms += s * s;
          }
        rms = std::sqrt(rms / ref.size()) + 1e-30;
        for (size_t i = 0; i < ref.size(); i++) e = std::max(e, std::fabs(((float*)dout.map)[i] - ref[i]) / rms);
        // T in 2..8 stages the activations as f16 (verify passes): 4x looser than the fp32 paths
        (T > 1 && T <= 8 && !wide ? worst16 : worst) = std::max(T > 1 && T <= 8 && !wide ? worst16 : worst, e);
      }
      vk.free(dx);
      vk.free(dout);
    }
    const bool ok = derr / wmax < 1e-5 && worst < 1e-3 && worst16 < 4e-3;
    all_ok &= ok;
    printf("%-8s %-32s K=%5d  dequant rel=%.1e  matvec/gemm max_err/rms=%.1e (f16-staged T>1: %.1e)  %s\n",
           qtype_info(type).name, t->name.c_str(), K, derr / wmax, worst, worst16, ok ? "OK" : "FAIL");
    vk.free(dw);
    vk.free(dy);
  }
  // QE_VK_TEST_BENCH=1: matvec time vs batch size T (speculative verify cost), largest matrix per type
  if (getenv("QE_VK_TEST_BENCH")) {
    std::map<uint32_t, const GGUFTensor*> big;
    for (auto& t : f.tensors())
      if (t.ne.size() == 2 && t.ne[0] % 256 == 0 && tag(t.type) && t.type != QT_F32 && t.type != QT_BF16 &&
          t.name.find("token_embd") == std::string::npos && t.name.find("output") == std::string::npos &&
          (!big.count(t.type) || t.nbytes > big[t.type]->nbytes))
        big[t.type] = &t;
    for (auto [type, t] : big) {
      const int K = (int)t->ne[0], N = (int)t->rows();
      const size_t rb = qtype_row_bytes(type, K);
      VBuf dw = vk.alloc(rb * N, Mem::Device), dx = vk.alloc((size_t)8 * K * 4, Mem::Device), dout = vk.alloc((size_t)8 * N * 4, Mem::Device);
      VBuf st = vk.alloc(64 << 20, Mem::Host);
      for (size_t off = 0; off < rb * N; off += st.size) {
        const size_t k = std::min(st.size, rb * N - off);
        memcpy(st.map, t->data + off, k);
        vk.begin(); vk.copy(st, 0, dw, off, k); vk.submit();
      }
      printf("%-8s %-28s %6d x %5d (%.0f MiB):", qtype_info(type).name, t->name.c_str(), N, K, rb * N / 1048576.0);
      double t1 = 0;
      for (int T = 1; T <= 8; T++) {
        MvPC pc{dw.addr, dx.addr, dout.addr, dtab.addr, (uint32_t)K, (uint32_t)N, (uint32_t)T, (uint32_t)rb, (uint32_t)K,
                (uint32_t)N, 0, 0, (uint32_t)(K / 32)};
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
          vk.begin();
          PipeOpt po;
          po.spec2 = T > 1;
          int teams = 4, rr = 4;
          if (T > 1 && getenv("QE_VK_MULTI")) sscanf(getenv("QE_VK_MULTI"), "%d,%d", &teams, &rr);
          po.spec0 = rr;
          po.spec1 = 32 * teams;
          const int rows = teams * rr;
          for (int i = 0; i < 10; i++)
            vk.dispatch((std::string("mv_") + tag(type)).c_str(), &pc, sizeof pc, (N + rows - 1) / rows, 1, 1, true, po);
          const double t0 = qe_now();
          vk.submit();
          if (rep) best = std::min(best, (qe_now() - t0) / 10);
        }
        if (T == 1) t1 = best;
        printf("  T%d %.0fus(%.2fx)", T, best * 1e6, best / t1);
      }
      printf("   [%.0f GB/s at T=1]\n", rb * N / t1 / 1e9);
      vk.free(dw); vk.free(dx); vk.free(dout); vk.free(st);
    }
  }
  printf(all_ok ? "ALL OK\n" : "FAILURES\n");
  return all_ok ? 0 : 1;
}
