#include <set>
#include <thread>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)  // MSVC CRT has no POSIX mkdir
#endif
#include "model.h"
#include "requant.h"

#include <algorithm>
#include <atomic>
#include <immintrin.h>
#include <cmath>

#include "quant.h"
#include <cuda.h>

#define CUBLAS_CHECK(x)                                                \
  do {                                                                 \
    cublasStatus_t st_ = (x);                                          \
    if (st_ != CUBLAS_STATUS_SUCCESS) QE_FATAL("cuBLAS error %d at %s", (int)st_, #x); \
  } while (0)

static constexpr size_t MiB = 1024 * 1024;

// QE_PROFILE=1: synchronize around regions and accumulate wall time per category (debug only)
struct Prof {
  // QE_PROFILE=1: synchronizing region timers (prefill analysis); =2: event-only decode profile
  bool on = getenv("QE_PROFILE") && atoi(getenv("QE_PROFILE")) == 1;
  double t[16] = {};
  const char* names[16] = {"attn_layer", "gdn_layer", "ffn", "fa_kernel", "gemm", "dequant", "gdn_core", "gdn_rec"};
  ~Prof() {
    if (!on) return;
    for (int i = 0; i < 8; i++) fprintf(stderr, "[prof] %-12s %8.3f s\n", names[i], t[i]);
  }
} g_prof;
struct ProfScope {
  int cat; double t0; cudaStream_t s;
  ProfScope(int c, cudaStream_t st) : cat(c), s(st) {
    if (g_prof.on) { cudaStreamSynchronize(s); t0 = qe_now(); }
  }
  ~ProfScope() {
    if (g_prof.on) { cudaStreamSynchronize(s); g_prof.t[cat] += qe_now() - t0; }
  }
};

// Debug: QE_DEBUG_SUMS=1 prints sum and first/last 3 values of each row (same sampling as
// llama.cpp's eval-callback) for comparison with the reference implementation.
static void qe_debug_dump(const char* nm, int il, const float* p, int T, int W, cudaStream_t s) {
  static const bool dbg = getenv("QE_DEBUG_SUMS") != nullptr;
  if (!dbg) return;
  std::vector<float> h((size_t)T * W);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  double sum = 0;
  for (float v : h) sum += v;
  fprintf(stderr, "%s-%d %f\n", nm, il, sum);
  for (int t = 0; t < T; t++) {
    const float* r = h.data() + (size_t)t * W;
    fprintf(stderr, "ROW %s-%d %.4f %.4f %.4f %.4f %.4f %.4f\n", nm, il, r[0], r[1], r[2], r[W - 3], r[W - 2], r[W - 1]);
  }
}

Model::Model(const EngineOptions& opt) : opt_(opt) {
  g_verbose = opt.verbose;
  gguf_ = std::make_unique<GGUFFile>(opt.model_path);
  load_config();  // checks the architecture
  // don't burn a CPU core spinning in cudaStreamSynchronize: the cores belong to the CPU FFN pool.
  // Must be set before the primary context is created (i.e. before cudaSetDevice on CUDA 12+).
  if (cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) {
    QE_LOG("[warn] could not set blocking-sync device flag\n");
    cudaGetLastError();
  }
  CUDA_CHECK(cudaSetDevice(0));
  {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    if (p.major < 8)
      QE_FATAL("GPU %s (compute capability %d.%d) is not supported: needs Ampere (8.0) or newer", p.name, p.major, p.minor);
    qk::set_num_sms(p.multiProcessorCount);
    QE_LOG("[hw] GPU %s, compute %d.%d, %d SMs, %.1f GiB VRAM\n", p.name, p.major, p.minor, p.multiProcessorCount,
           p.totalGlobalMem / (1024.0 * 1024 * 1024));
  }
  calibrate_host();
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));
  plan_and_load();
}

// Host-side machine parameters the planner uses: physical cores, PCIe host->device bandwidth and DRAM
// read bandwidth, measured at start-up unless given explicitly (they decide the CPU/GPU split).
static int physical_cores() {
#ifdef _WIN32
  return (int)std::thread::hardware_concurrency();  // no /sys topology on Windows
#else
  std::set<std::pair<int, int>> cores;
  for (int cpu = 0; cpu < 1024; cpu++) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    FILE* f1 = fopen((base + "core_id").c_str(), "r");
    if (!f1) break;
    int core = -1, pkg = 0;
    if (fscanf(f1, "%d", &core) != 1) core = cpu;
    fclose(f1);
    if (FILE* f2 = fopen((base + "physical_package_id").c_str(), "r")) {
      if (fscanf(f2, "%d", &pkg) != 1) pkg = 0;
      fclose(f2);
    }
    cores.insert({pkg, core});
  }
  return cores.empty() ? (int)std::thread::hardware_concurrency() : (int)cores.size();
#endif
}

void Model::calibrate_host() {
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c") ||
      !__builtin_cpu_supports("bmi2"))
    QE_FATAL("CPU without AVX2/FMA/F16C/BMI2 (x86-64-v3) is not supported");
  // big cores only on hybrid CPUs (P-cores; the E-cores would hold every parallel matvec back)
  if (opt_.cpu_threads <= 0) opt_.cpu_threads = std::clamp(std::min(physical_cores(), cpu::fast_cores()), 1, 32);
  if (opt_.pcie_gbps <= 0) {
    const size_t n = 256ull << 20;
    void* h = nullptr;
    void* d = nullptr;
    double gbps = 25.0;
    if (cudaHostAlloc(&h, n, cudaHostAllocDefault) == cudaSuccess && cudaMalloc(&d, n) == cudaSuccess) {
      memset(h, 1, n);
      cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
      double best = 1e9;
      for (int i = 0; i < 3; i++) {
        const double t0 = qe_now();
        cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
        best = std::min(best, qe_now() - t0);
      }
      gbps = n / best / 1e9;
    }
    if (d) cudaFree(d);
    if (h) cudaFreeHost(h);
    cudaGetLastError();
    // the balancer was tuned with 1.5x the raw copy rate (the copies overlap compute imperfectly)
    opt_.pcie_gbps = gbps * 1.5;
    measured_pcie_ = gbps;
  }
  if (opt_.dram_gbps <= 0) {
    const size_t n = 512ull << 20, nt = std::max(1, opt_.cpu_threads);
    std::vector<uint64_t> buf(n / 8, 1);
    std::vector<uint64_t> sums(nt);
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
      const double t0 = qe_now();
      std::vector<std::thread> th;
      for (size_t t = 0; t < nt; t++)
        th.emplace_back([&, t] {
          const size_t a = buf.size() * t / nt, b = buf.size() * (t + 1) / nt;
          uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
          for (size_t i = a; i + 3 < b; i += 4) { s0 += buf[i]; s1 += buf[i + 1]; s2 += buf[i + 2]; s3 += buf[i + 3]; }
          sums[t] = s0 + s1 + s2 + s3;
        });
      for (auto& x : th) x.join();
      best = std::min(best, qe_now() - t0);
    }
    opt_.dram_gbps = n / best / 1e9;
  }
  QE_LOG("[hw] CPU %d threads, PCIe H2D %.1f GB/s (balancer %.1f), DRAM read %.1f GB/s\n", opt_.cpu_threads,
         measured_pcie_ > 0 ? measured_pcie_ : opt_.pcie_gbps, opt_.pcie_gbps, opt_.dram_gbps);
  QE_LOG("[hw] CPU kernels: %s (features: %s)\n", cpu::isa_name(cpu::isa()), cpu::cpu_features().c_str());
}

Model::~Model() {
  cudaStreamSynchronize(stream_);
  for (void* p : allocs_) cudaFree(p);
  for (void* p : host_allocs_) cudaFreeHost(p);
  if (cublas_) cublasDestroy(cublas_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Model::load_config() { cfg_ = load_qwen35_config(*gguf_); }

void Model::requant_mtp() {
  Layer& L = layers_[cfg_.n_layer];
  std::vector<Mat*> mats;
  static const bool ffn_only = !getenv("QE_MTP_REQ_ALL");
  for (Mat* m : {&L.wq, &L.wk, &L.wv, &L.wo, &L.ffn_gate, &L.ffn_up, &L.ffn_down, &L.eh_proj})
    if (m->valid() && m->type == QT_Q6_K && m->K % 256 == 0 &&
        (!ffn_only || m == &L.ffn_gate || m == &L.ffn_up || m == &L.ffn_down))
      mats.push_back(m);
  if (mats.empty()) return;
  size_t total = 0, before = 0;
  for (Mat* m : mats) { total += iq4_xs_row_bytes(m->K) * m->N; before += m->bytes; }
  // cache keyed by the model file identity
  struct stat st{};
  stat(opt_.model_path.c_str(), &st);
  const char* xdg = getenv("XDG_CACHE_HOME");
  const std::string dir = xdg && *xdg ? std::string(xdg) + "/qwen-hyperdrive" : std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/.cache/qwen-hyperdrive";
  const std::string path = dir + (ffn_only ? "/mtp-ffn-iq4xs-v1-" : "/mtp-iq4xs-v1-") + std::to_string((long long)st.st_size) + "-" + std::to_string((long long)st.st_mtime) + ".bin";
  requant_store_.emplace_back(total);
  uint8_t* buf = requant_store_.back().data();
  bool cached = false;
  if (FILE* f = fopen(path.c_str(), "rb")) {
    cached = fread(buf, 1, total, f) == total && fgetc(f) == EOF;
    fclose(f);
  }
  const double t0 = qe_now();
  size_t off = 0;
  for (Mat* m : mats) {
    const size_t rb = iq4_xs_row_bytes(m->K);
    if (!cached) requant_rows_iq4_xs(m->type, m->host, m->row_bytes, m->K, m->N, buf + off);
    m->type = QT_IQ4_XS;
    m->row_bytes = rb;
    m->bytes = rb * m->N;
    m->host = buf + off;
    off += m->bytes;
  }
  if (!cached) {
    mkdir(dir.c_str(), 0755);
    const std::string tmp = path + ".tmp";
    if (FILE* f = fopen(tmp.c_str(), "wb")) {
      const bool ok = fwrite(buf, 1, total, f) == total;
      fclose(f);
      if (ok) rename(tmp.c_str(), path.c_str());
    }
  }
  QE_LOG("[plan] MTP block Q6_K -> IQ4_XS: %.0f -> %.0f MiB (%s, %.1f s)\n", before / (double)MiB, total / (double)MiB,
         cached ? "cached" : "requantized", qe_now() - t0);
}

Mat Model::make_mat(const std::string& name, bool required) {
  Mat m;
  const GGUFTensor* t = gguf_->tensor(name);
  if (!t) {
    if (required) QE_FATAL("missing tensor %s", name.c_str());
    return m;
  }
  if (!qtype_supported(t->type)) QE_FATAL("tensor %s has unsupported type %s", name.c_str(), qtype_info(t->type).name);
  m.name = name;
  m.type = t->type;
  m.K = (int)t->ne[0];
  m.N = (int)t->rows();
  m.row_bytes = qtype_row_bytes(t->type, m.K);
  m.bytes = t->nbytes;
  m.host = t->data;
  return m;
}

// Bump allocator over one big cudaMalloc arena: hundreds of separate cudaMallocs waste up to
// 2 MiB each to allocation granularity, which matters when VRAM is planned to the last MiB.
// Measured CPU matvec throughput (GB/s of weights) per quant type with the configured thread count
// and the decode batch (2 when MTP verification is on).  Cached per type; ~20 ms per type.
double Model::cpu_bw(uint32_t type, bool r4) {
  const uint32_t key = type | (r4 ? 0x100u : 0u);
  auto it = cpu_bw_.find(key);
  if (it != cpu_bw_.end()) return it->second;
  if (!pool_) pool_ = std::make_unique<cpu::ThreadPool>(opt_.cpu_threads);
  const GGUFTensor* t = nullptr;
  for (auto& x : gguf_->tensors())
    if (x.type == type && x.name.find("ffn_gate") != std::string::npos) { t = &x; break; }
  if (!t)
    for (auto& x : gguf_->tensors())
      if (x.type == type && x.ne.size() == 2 && x.ne[0] % 256 == 0 && x.rows() >= 2048) { t = &x; break; }
  double bw = 15.0;
  if (t) {
    const int K = (int)t->ne[0];
    const int rows = (int)std::min<int64_t>(t->rows(), 4096);
    const size_t rb = qtype_row_bytes(type, K);
    std::vector<uint8_t> w(rb * rows);
    memcpy(w.data(), t->data, w.size());
    const int B = has_mtp() ? 2 : 1;
    cpu::Q8Act x;
    x.resize(K, B);
    std::vector<float> xf((size_t)K * B, 0.01f), y((size_t)rows * B);
    for (int b = 0; b < B; b++) cpu::quantize(xf.data(), K, b, x, 0, K / 256);
    const int n = pool_->size();
    std::vector<uint8_t> w4;
    if (r4) {
      w4.resize((size_t)rows * (K / 256) * cpu::R4_BLOCK_BYTES);
      cpu::r4_convert(type, w.data(), rb, K, rows, w4.data());
    }
    auto run = [&] {
      pool_->run([&](int tid) {
        if (r4) cpu::matvec_r4(cpu::r4_table(type), w4.data(), K, rows * tid / n, rows * (tid + 1) / n, x, B, y.data(), rows);
        else cpu::matvec(type, w.data(), rb, K, rows * tid / n, rows * (tid + 1) / n, x, B, y.data(), rows);
      });
    };
    run();
    const double t0 = qe_now();
    const int reps = 4;
    for (int i = 0; i < reps; i++) run();
    bw = w.size() * reps / (qe_now() - t0) / 1e9;
  }
  cpu_bw_[key] = bw;
  QE_DBG("[calib] cpu %s%s: %.1f GB/s (native-byte equivalent)\n", qtype_info(type).name, r4 ? " (R4)" : "", bw);
  return bw;
}

void* Model::dalloc(size_t bytes) {
  bytes = (std::max<size_t>(bytes, 16) + 255) / 256 * 256;
  if (vram_used_ + bytes > arena_bytes_)
    QE_FATAL("VRAM arena exhausted: need %.1f MiB more (arena %.0f MiB, used %.1f MiB)", bytes / (double)MiB,
             arena_bytes_ / (double)MiB, vram_used_ / (double)MiB);
  void* p = arena_ + vram_used_;
  vram_used_ += bytes;
  return p;
}

float* Model::upload_f32(const std::string& name) {
  const GGUFTensor& t = gguf_->need(name);
  std::vector<float> v(t.nelems());
  dequant_row_ref(t.type, t.data, t.nelems(), v.data());
  float* d = (float*)dalloc(v.size() * 4);
  CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  return d;
}

void Model::plan_and_load() {
  const Config& c = cfg_;
  const double t0 = qe_now();
  const int nL = c.n_layer + c.n_mtp;
  layers_.resize(nL);
  auto bn = [](int il, const char* n) { return "blk." + std::to_string(il) + "." + n; };

  tok_embd_ = make_mat("token_embd.weight");
  output_ = make_mat(gguf_->tensor("output.weight") ? "output.weight" : "token_embd.weight");  // tied head (small models)
  for (int il = 0; il < nL; il++) {
    Layer& L = layers_[il];
    L.attn = c.is_attn(il);
    if (c.aliased(il)) {  // second visit of a loop: takes the first visit's weights after planning
      L.ffn_skip = true;  // (keeps the planner from placing / streaming an FFN it does not own)
      continue;
    }
    if (L.attn) {
      L.wq = make_mat(bn(il, "attn_q.weight"));
      L.wk = make_mat(bn(il, "attn_k.weight"));
      L.wv = make_mat(bn(il, "attn_v.weight"));
      L.wo = make_mat(bn(il, "attn_output.weight"));
    } else {
      L.wqkv = make_mat(bn(il, "attn_qkv.weight"));
      L.wgate = make_mat(bn(il, "attn_gate.weight"));
      L.walpha = make_mat(bn(il, "ssm_alpha.weight"));
      L.wbeta = make_mat(bn(il, "ssm_beta.weight"));
      L.wssm_out = make_mat(bn(il, "ssm_out.weight"));
    }
    L.ffn_gate = make_mat(bn(il, "ffn_gate.weight"));
    L.ffn_up = make_mat(bn(il, "ffn_up.weight"));
    L.ffn_down = make_mat(bn(il, "ffn_down.weight"));
    if (il >= c.n_layer) L.eh_proj = make_mat(bn(il, "nextn.eh_proj.weight"));
  }

  // MTP block: Q6_K -> IQ4_XS (drafts only; ~1/3 fewer bytes per draft step and more VRAM for FFN)
  if (has_mtp() && !getenv("QE_MTP_Q6")) requant_mtp();

  // ---------------- memory plan ----------------
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  int64_t budget = (int64_t)free_b - (int64_t)opt_.vram_reserve_mb * MiB;
  if (opt_.vram_budget_mb > 0) budget = std::min<int64_t>(budget, opt_.vram_budget_mb * (int64_t)MiB);

  const size_t recb = qk::kv_bytes_per_token_head(opt_.kv_type, c.hd);
  const int n_attn = (int)std::count_if(layers_.begin(), layers_.end(), [](const Layer& L) { return L.attn; });
  n_kv_blocks_ = (opt_.ctx + qk::KV_BLOCK - 1) / qk::KV_BLOCK;
  kv_block_bytes_ = (size_t)n_attn * 2 * c.n_kv * qk::KV_BLOCK * recb;
  const size_t kv_bytes = (size_t)n_kv_blocks_ * kv_block_bytes_;
  const int n_gdn = nL - n_attn;
  state16_ = !getenv("QE_STATE16") || atoi(getenv("QE_STATE16")) != 0;  // fp16 state by default (PPL +0.06% in decode)
  size_t ssm_bytes = (size_t)n_gdn * (state_bytes() + 3 * (size_t)c.conv_dim() * 4);
  // speculative verify passes (MTP or n-gram drafts): keep their recurrence inputs for rollback
  spec_ = !getenv("QE_NO_SPEC");
  if (spec_) {
    opt_.draft_max = std::clamp(opt_.draft_max, 1, qk::kMaxB - 1);
    // saved recurrence inputs of the verified tokens (the accepted prefix is recomputed in place after
    // verification, so no second copy of the state is needed); legacy fp16-GEMM mode keeps a backup
    legacy16_ = getenv("QE_GEMM16") && atoi(getenv("QE_GEMM16")) == 1;
    ssm_bytes += (size_t)n_gdn * ((legacy16_ ? state_bytes() : 0) + (3 + qk::kMaxB) * (size_t)c.conv_dim() * 4 +
                                  (size_t)(opt_.draft_max + 1) * (c.conv_dim() + 2 * c.nv) * 4);
  }
  Tmax_ = opt_.ubatch;
  const int T = Tmax_;
  const int maxw = std::max({c.n_ff, c.n_head * c.hd * 2, c.conv_dim()});
  size_t ws = 0;
  ws += 2 * (size_t)T * c.n_embd * 4;          // x, xn
  ws += 2 * (size_t)T * maxw * 4;              // t0, t1
  ws += (size_t)T * std::max(c.d_inner, c.n_head * c.hd) * 4;  // t2
  ws += (size_t)T * c.conv_dim() * 4;          // t3
  ws += (size_t)T * c.n_embd * 4;              // t4
  ws += 2 * (size_t)T * c.nv * 4;              // g, beta
  ws += (size_t)T * maxw + (size_t)T * maxw / 32 * 4;  // xq, xd
  ws += (size_t)T * maxw * 2;                  // a16
  w16_elems_ = 48ull * MiB;                    // f16 dequant buffer for GEMM (96 MiB)
  ws += w16_elems_ * 2;
  ws += (size_t)T * maxw * 2;                  // c16 (fp16 GEMM output)
  const int attT = attn_dec_T();  // attn_decode handles verify passes up to this; larger T: prefill FA kernel
  attn_work_bytes_ = qk::attn_decode_work_bytes_auto(opt_.ctx, c.n_head, c.n_kv, c.hd, attT);
  ws += attn_work_bytes_;
  ws += (size_t)kCpuMaxB * c.n_embd * 4;
  ws += qk::attn_prefill_work_bytes(T, c.n_head, c.hd);
  ws += (size_t)qk::kMaxB * c.n_vocab * 4;    // logits
  ws += (size_t)T * c.n_embd * 4 + 2 * (size_t)c.n_embd * 4 + 512;  // hid, carry, mtp_h, argmax
  ws += 64 * MiB;                              // cuBLAS workspace & misc slack

  int64_t weight_budget = budget - (int64_t)kv_bytes - (int64_t)ssm_bytes - (int64_t)ws;
  QE_LOG("[plan] VRAM free %.0f MiB, budget %.0f MiB: KV %.0f MiB (%s, ctx %d), SSM %.0f MiB, workspace %.0f MiB -> weights %.0f MiB\n",
         free_b / (double)MiB, budget / (double)MiB, kv_bytes / (double)MiB,
         opt_.kv_type == qk::KV_Q4 ? "q4+WHT" : opt_.kv_type == qk::KV_Q8 ? "q8+WHT" : opt_.kv_type == qk::KV_Q3 ? "q3+WHT" : "f16", opt_.ctx,
         ssm_bytes / (double)MiB, ws / (double)MiB, weight_budget / (double)MiB);
  if (weight_budget < 0) QE_FATAL("context %d does not fit in VRAM", opt_.ctx);
  {
    arena_bytes_ = (size_t)budget;
    cudaError_t e = cudaMalloc((void**)&arena_, arena_bytes_);
    if (e != cudaSuccess) QE_FATAL("cannot reserve %.0f MiB VRAM arena: %s", arena_bytes_ / (double)MiB, cudaGetErrorString(e));
    allocs_.push_back(arena_);
  }

  // Priority: output head, attention/SSM projections (+MTP block), then FFN.
  std::vector<Mat*> prio, ffn;
  prio.push_back(&output_);
  for (auto& L : layers_) {
    for (Mat* m : {&L.wq, &L.wk, &L.wv, &L.wo, &L.wqkv, &L.wgate, &L.walpha, &L.wbeta, &L.wssm_out, &L.eh_proj})
      if (m->valid()) prio.push_back(m);
  }
  for (int il = nL - 1; il >= c.n_layer; il--)  // MTP FFN is used every draft step: keep resident
    for (Mat* m : {&layers_[il].ffn_gate, &layers_[il].ffn_up, &layers_[il].ffn_down}) prio.push_back(m);
  // lossy option: FFN blocks of the listed layers are dropped entirely (never loaded or streamed)
  for (int il : opt_.skip_ffn)
    if (il >= 0 && il < c.n_layer) layers_[il].ffn_skip = true;
  // reserve streaming double-buffer (largest per-layer FFN group) before deciding residency
  size_t max_ffn_layer = 0, tot_ffn = 0;
  for (int il = 0; il < c.n_layer; il++) {
    if (layers_[il].ffn_skip) continue;
    const size_t lb = layers_[il].ffn_gate.bytes + layers_[il].ffn_up.bytes + layers_[il].ffn_down.bytes;
    max_ffn_layer = std::max(max_ffn_layer, lb);
    tot_ffn += lb;
  }
  int64_t remaining = weight_budget;
  std::vector<Mat*> resident;
  for (Mat* m : prio) {
    if ((int64_t)m->bytes > remaining) QE_FATAL("not enough VRAM for %s", m->name.c_str());
    remaining -= m->bytes;
    resident.push_back(m);
  }
  const bool any_stream = (int64_t)tot_ffn > remaining;
  if (any_stream) remaining -= 2 * (int64_t)(max_ffn_layer + 4096);
  // FFN residency is decided per layer (all three matrices together).  Resident layers are spread
  // evenly over the depth so the streamed ones are too: the copy engine then always has an upcoming
  // streamed layer to prefetch instead of stalling on a full ring during a long resident stretch.
  int n_res_ffn = 0;
  {
    auto lbytes = [&](int il) {
      const Layer& L = layers_[il];
      return (int64_t)(L.ffn_gate.bytes + L.ffn_up.bytes + L.ffn_down.bytes);
    };
    std::vector<int> live;  // layers that have an FFN
    for (int il = 0; il < c.n_layer; il++)
      if (!layers_[il].ffn_skip) live.push_back(il);
    const int nlive = (int)live.size();
    std::vector<int> pick;
    for (int k = nlive; k >= 1; k--) {
      std::vector<int> cand;
      int64_t sum = 0;
      for (int i = 0; i < k; i++) {
        const int il = live[std::min(nlive - 1, (int)((i + 0.5) * nlive / k))];
        if (!cand.empty() && cand.back() == il) continue;
        cand.push_back(il);
        sum += lbytes(il);
      }
      if (sum <= remaining) { pick = cand; remaining -= sum; break; }
    }
    std::vector<char> res(c.n_layer, 0);
    for (int il : pick) res[il] = 1;
    for (int il = 0; il < c.n_layer; il++) {
      Layer& L = layers_[il];
      if (L.ffn_skip) continue;
      if (res[il]) {
        for (Mat* m : {&L.ffn_gate, &L.ffn_up, &L.ffn_down}) resident.push_back(m);
        n_res_ffn++;
      } else {
        L.ffn_streamed = true;
      }
    }
  }

  size_t res_bytes = 0, str_bytes = 0, cpu_bytes = 0, r4_bytes_total = 0;
  for (Mat* m : resident) {
    m->dev = dalloc(m->bytes);
    CUDA_CHECK(cudaMemcpy(m->dev, m->host, m->bytes, cudaMemcpyHostToDevice));
    res_bytes += m->bytes;
  }
  // Streamed layers: split FFN rows between CPU ([0,r)) and GPU ([r,n_ff)); see layer_ffn.
  // Pinned host block layout: [gate_g | up_g | down_g] [gate_c | up_c | down_c] (256-aligned).
  for (int il = 0; il < c.n_layer; il++) {
    Layer& L = layers_[il];
    if (!L.ffn_streamed) continue;
    const bool cpu_ok = opt_.cpu_frac != 0 && cpu::type_supported(L.ffn_gate.type) &&
                        cpu::type_supported(L.ffn_up.type) && cpu::type_supported(L.ffn_down.type);
    int r = 0;
    bool use_r4 = false;
    if (cpu_ok && opt_.cpu_frac < 0) {
      // auto: choose the CPU row fraction f (and CPU weight format) minimizing
      //   max( t_cpu = f*S/bw_cpu,  t_pcie = (1-f)*S/bw_pcie,  t_dram = (f*S*expand + (1-f)*S)/bw_dram )
      // where bw_cpu is in native-bytes/s (R4 expands the bytes the CPU reads).
      double S = 0, inv_nat = 0, inv_r4 = 0, r4_bytes = 0;
      bool r4_ok = opt_.cpu_r4;
      for (const Mat* m : {&L.ffn_gate, &L.ffn_up, &L.ffn_down}) {
        S += m->bytes;
        inv_nat += m->bytes / (cpu_bw(m->type) * 1e9);
        if (cpu::r4_supported(m->type)) {
          inv_r4 += m->bytes / (cpu_bw(m->type, true) * 1e9);
          r4_bytes += (double)m->N * (m->K / 256) * cpu::R4_BLOCK_BYTES;
        } else {
          inv_r4 += m->bytes / (cpu_bw(m->type) * 1e9);
          r4_bytes += m->bytes;
        }
      }
      if (r4_bytes == S) r4_ok = false;
      double best = 1e30;
      for (int fmt = 0; fmt < (r4_ok ? 2 : 1); fmt++) {
        const double inv_cpu = fmt ? inv_r4 : inv_nat;   // seconds for f = 1
        const double expand = fmt ? r4_bytes / S : 1.0;
        for (int rr = 0; rr <= c.n_ff; rr += 256) {
          const double f = (double)rr / c.n_ff;
          const double t = std::max({f * inv_cpu, (1 - f) * S / (opt_.pcie_gbps * 1e9),
                                     (f * S * expand + (1 - f) * S) / (opt_.dram_gbps * 1e9)});
          if (t < best - 1e-12) { best = t; r = rr; use_r4 = fmt == 1; }
        }
      }
    } else if (cpu_ok) {
      r = (int)std::lround(opt_.cpu_frac * c.n_ff / 256.0) * 256;
      use_r4 = opt_.cpu_r4;
    }
    L.cpu_r4 = use_r4;
    r = std::clamp(r, 0, c.n_ff);
    L.cpu_rows = r;
    auto rows_part = [&](const Mat& src, int r0, int r1, const char* sfx) {
      Mat m = src;
      m.name = src.name + sfx;
      m.N = r1 - r0;
      m.bytes = (size_t)m.N * m.row_bytes;
      m.host = src.host + (size_t)r0 * src.row_bytes;
      m.dev = nullptr;
      return m;
    };
    auto cols_part = [&](const Mat& src, int k0, int k1, const char* sfx) {
      Mat m = src;
      m.name = src.name + sfx;
      m.K = k1 - k0;
      m.row_bytes = qtype_row_bytes(src.type, m.K);
      m.bytes = (size_t)m.N * m.row_bytes;
      m.host = nullptr;  // needs repacking
      m.dev = nullptr;
      return m;
    };
    L.gate_g = rows_part(L.ffn_gate, r, c.n_ff, "[gpu]");
    L.up_g = rows_part(L.ffn_up, r, c.n_ff, "[gpu]");
    L.down_g = cols_part(L.ffn_down, r, c.n_ff, "[gpu]");
    L.gate_c = rows_part(L.ffn_gate, 0, r, "[cpu]");
    L.up_c = rows_part(L.ffn_up, 0, r, "[cpu]");
    L.down_c = cols_part(L.ffn_down, 0, r, "[cpu]");
    size_t off = 0;
    Mat* order[6] = {&L.gate_g, &L.up_g, &L.down_g, &L.gate_c, &L.up_c, &L.down_c};
    for (int i = 0; i < 6; i++) {
      if (i == 3) L.stream_bytes_gpu = off;
      order[i]->stream_slot_off = (int)off;
      off += (order[i]->bytes + 255) / 256 * 256;
    }
    L.stream_bytes_full = off;
    uint8_t* hp = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&hp, off, cudaHostAllocDefault));
    host_allocs_.push_back(hp);
    L.stream_host = hp;
    const Mat& D = L.ffn_down;
    for (int i = 0; i < 6; i++) {
      Mat* m = order[i];
      uint8_t* dst = hp + m->stream_slot_off;
      if (m->host) {
        if (m->bytes) memcpy(dst, m->host, m->bytes);
      } else {  // down: repack the K-column range of every row
        const int k0 = (m == &L.down_g) ? r : 0;
        const size_t skip = qtype_row_bytes(D.type, k0);
        for (int row = 0; row < D.N; row++) memcpy(dst + (size_t)row * m->row_bytes, D.host + (size_t)row * D.row_bytes + skip, m->row_bytes);
      }
      m->host = dst;
    }
    str_bytes += L.stream_bytes_gpu;
    cpu_bytes += L.stream_bytes_full - L.stream_bytes_gpu;
    for (Mat* m : {&L.gate_c, &L.up_c, &L.down_c}) {
      m->cpu_w = m->host;
      m->cpu_tab = nullptr;
      if (L.cpu_r4 && r > 0 && cpu::r4_supported(m->type)) {
        r4_store_.emplace_back((size_t)m->N * (m->K / 256) * cpu::R4_BLOCK_BYTES);
        cpu::r4_convert(m->type, m->host, m->row_bytes, m->K, m->N, r4_store_.back().data());
        m->cpu_w = r4_store_.back().data();
        m->cpu_tab = cpu::r4_table(m->type);
        r4_bytes_total += r4_store_.back().size();
      }
    }
  }
  if (any_stream) {
    stream_buf_bytes_ = (max_ffn_layer + 4096 + 255) / 256 * 256;
    stream_buf_ = (uint8_t*)dalloc(2 * stream_buf_bytes_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    for (int i = 0; i < kMaxSlots; i++) {
      CUDA_CHECK(cudaEventCreateWithFlags(&slot_ready_[i], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&slot_free_[i], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(slot_free_[i], stream_));
    }
    nslots_ = 0;
    for (int il = 0; il < c.n_layer; il++)
      if (layers_[il].ffn_streamed) streamed_layers_.push_back(il);
  }
  plan_res_mb_ = res_bytes / (double)MiB;
  plan_stream_mb_ = str_bytes / (double)MiB;
  plan_cpu_mb_ = cpu_bytes / (double)MiB;
  plan_res_ffn_ = n_res_ffn;
  QE_LOG("[plan] resident weights %.0f MiB (%d/%d FFN layers), host FFN: %.0f MiB streamed over PCIe + %.0f MiB computed on CPU (%s)\n",
         res_bytes / (double)MiB, n_res_ffn, c.n_layer, str_bytes / (double)MiB, cpu_bytes / (double)MiB,
         opt_.cpu_frac < 0 ? "auto-balanced per layer" : ("frac " + std::to_string(opt_.cpu_frac)).c_str());
  if (r4_bytes_total) QE_LOG("[plan] CPU part repacked to R4 nibbles: %.0f MiB\n", r4_bytes_total / (double)MiB);

  // CPU side of the hybrid FFN
  if (cpu_bytes > 0) {
    if (!pool_) pool_ = std::make_unique<cpu::ThreadPool>(opt_.cpu_threads);
    CUDA_CHECK(cudaHostAlloc((void**)&hx_in_, (size_t)kCpuMaxB * c.n_embd * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&hy_out_, (size_t)kCpuMaxB * c.n_embd * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&hflags_, 128, cudaHostAllocMapped));
    host_allocs_.push_back(hx_in_);
    host_allocs_.push_back(hy_out_);
    host_allocs_.push_back((void*)hflags_);
    memset((void*)hflags_, 0, 128);
    void* dp = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dp, (void*)hflags_, 0));
    dflags_ = (CUdeviceptr)dp;
    cx_.resize(c.n_embd, kCpuMaxB);
    ch_.resize(c.n_ff, kCpuMaxB);
    cg_.resize((size_t)kCpuMaxB * c.n_ff);
    cu_.resize((size_t)kCpuMaxB * c.n_ff);
  }

  // ---------------- small tensors, state ----------------
  // KV pool: block-major over all attention layers, so the unused tail can be lent to FFN weights
  kv_pool_bytes_ = (size_t)n_kv_blocks_ * kv_block_bytes_;
  kv_pool_ = (uint8_t*)dalloc(kv_pool_bytes_);
  borrow_floor_ = kv_pool_bytes_;
  blk_owner_.assign(n_kv_blocks_, -1);
  btab_dev_ = (int*)dalloc((size_t)n_kv_blocks_ * 4);
  int attn_idx = 0;
  output_norm_ = upload_f32("output_norm.weight");
  for (int il = 0; il < nL; il++) {
    Layer& L = layers_[il];
    const bool al = c.aliased(il);  // weights come from the first loop visit (below)
    if (!al) {
      L.attn_norm = upload_f32(bn(il, "attn_norm.weight"));
      L.post_norm = upload_f32(bn(il, "post_attention_norm.weight"));
      if (c.mhc(il)) {
        L.mhc_attn_w = upload_f32(bn(il, "mhc_attn.weight"));
        L.mhc_attn_b = upload_f32(bn(il, "mhc_attn.bias"));
        L.mhc_ffn_w = upload_f32(bn(il, "mhc_ffn.weight"));
        L.mhc_ffn_b = upload_f32(bn(il, "mhc_ffn.bias"));
      }
    }
    if (L.attn) {
      if (!al) {
        L.q_norm = upload_f32(bn(il, "attn_q_norm.weight"));
        L.k_norm = upload_f32(bn(il, "attn_k_norm.weight"));
      }
      L.kcache = kv_pool_ + (size_t)(2 * attn_idx) * c.n_kv * qk::KV_BLOCK * recb;
      L.vcache = kv_pool_ + (size_t)(2 * attn_idx + 1) * c.n_kv * qk::KV_BLOCK * recb;
      attn_idx++;
    } else {
      if (!al) {
        L.conv_w = upload_f32(bn(il, "ssm_conv1d.weight"));
        L.dt_bias = upload_f32(bn(il, "ssm_dt.bias"));
        L.ssm_a = upload_f32(bn(il, "ssm_a"));
        L.ssm_norm = upload_f32(bn(il, "ssm_norm.weight"));
      }
      L.conv_state = (float*)dalloc(3 * (size_t)c.conv_dim() * 4);
      L.ssm_state = (float*)dalloc(state_bytes());
      L.gdn_idx = n_gdn_++;
      if (spec_) {
        if (legacy16_) L.ssm_alt = (float*)dalloc(state_bytes());
        L.sv_qkv = (float*)dalloc((size_t)(opt_.draft_max + 1) * c.conv_dim() * 4);
        L.sv_g = (float*)dalloc((size_t)(opt_.draft_max + 1) * c.nv * 4);
        L.sv_beta = (float*)dalloc((size_t)(opt_.draft_max + 1) * c.nv * 4);
        L.conv_hist = (float*)dalloc((3 + qk::kMaxB) * (size_t)c.conv_dim() * 4);
      }
    }
    if (al) {  // own runtime state (KV / recurrent state above), the first visit's weights
      const Layer& S = layers_[c.wl(il)];
      L.attn_norm = S.attn_norm; L.post_norm = S.post_norm; L.q_norm = S.q_norm; L.k_norm = S.k_norm;
      L.conv_w = S.conv_w; L.dt_bias = S.dt_bias; L.ssm_a = S.ssm_a; L.ssm_norm = S.ssm_norm;
      L.mhc_attn_w = S.mhc_attn_w; L.mhc_attn_b = S.mhc_attn_b; L.mhc_ffn_w = S.mhc_ffn_w; L.mhc_ffn_b = S.mhc_ffn_b;
      L.wq = S.wq; L.wk = S.wk; L.wv = S.wv; L.wo = S.wo;
      L.wqkv = S.wqkv; L.wgate = S.wgate; L.walpha = S.walpha; L.wbeta = S.wbeta; L.wssm_out = S.wssm_out;
      L.ffn_gate = S.ffn_gate; L.ffn_up = S.ffn_up; L.ffn_down = S.ffn_down;
      L.ffn_skip = S.ffn_skip; L.ffn_streamed = S.ffn_streamed; L.cpu_rows = S.cpu_rows; L.cpu_r4 = S.cpu_r4;
      L.gate_g = S.gate_g; L.up_g = S.up_g; L.down_g = S.down_g; L.gate_c = S.gate_c; L.up_c = S.up_c; L.down_c = S.down_c;
      L.stream_bytes_gpu = S.stream_bytes_gpu; L.stream_bytes_full = S.stream_bytes_full; L.stream_host = S.stream_host;
      L.hyb_on = S.hyb_on;
    }
    if (il >= c.n_layer) {
      L.enorm = upload_f32(bn(il, "nextn.enorm.weight"));
      L.hnorm = upload_f32(bn(il, "nextn.hnorm.weight"));
      if (gguf_->tensor(bn(il, "nextn.shared_head_norm.weight")))
        L.shared_head_norm = upload_f32(bn(il, "nextn.shared_head_norm.weight"));
    }
  }

  // ---------------- workspace ----------------
  // Two workspace sets: prefill (Tmax rows, one contiguous arena that is lent to FFN weights while
  // decoding) and decode (kMaxB rows, permanent).  select_ws() points the working buffers at one.
  auto carve = [&](WSet& w, int rows, bool big) {
    size_t tot = 0;
    auto sz = [&](size_t b) { size_t r = (b + 255) / 256 * 256; tot += r; return r; };
    const size_t s_x = sz((size_t)rows * c.n_embd * 4), s_xn = sz((size_t)rows * c.n_embd * 4);
    const size_t s_t0 = sz((size_t)rows * maxw * 4), s_t1 = sz((size_t)rows * maxw * 4);
    const size_t s_t2 = sz((size_t)rows * std::max(c.d_inner, c.n_head * c.hd) * 4);
    const size_t s_t3 = sz((size_t)rows * c.conv_dim() * 4), s_t4 = sz((size_t)rows * c.n_embd * 4);
    const size_t s_g = sz((size_t)rows * c.nv * 4), s_b = sz((size_t)rows * c.nv * 4);
    const size_t s_xq = sz((size_t)rows * maxw), s_xd = sz((size_t)rows * maxw / 32 * 4);
    const size_t s_hid = sz((size_t)rows * c.n_embd * 4);
    const size_t s_a16 = big ? sz((size_t)rows * maxw * 2) : 0, s_c16 = big ? sz((size_t)rows * maxw * 2) : 0;
    const size_t s_w16 = big ? sz(w16_elems_ * 2) : 0;
    const size_t s_fa = big ? sz(qk::attn_prefill_work_bytes(rows, c.n_head, c.hd)) : 0;
    uint8_t* p = (uint8_t*)dalloc(tot);
    w.base = p;
    w.bytes = tot;
    auto take = [&](size_t b) { uint8_t* r = b ? p : nullptr; p += b; return r; };
    w.x = (float*)take(s_x); w.xn = (float*)take(s_xn); w.t0 = (float*)take(s_t0); w.t1 = (float*)take(s_t1);
    w.t2 = (float*)take(s_t2); w.t3 = (float*)take(s_t3); w.t4 = (float*)take(s_t4);
    w.g = (float*)take(s_g); w.beta = (float*)take(s_b); w.xq = (int8_t*)take(s_xq); w.xd = (float*)take(s_xd);
    w.hid = (float*)take(s_hid); w.a16 = (__half*)take(s_a16); w.c16 = (__half*)take(s_c16);
    w.w16 = (__half*)take(s_w16); w.fa = take(s_fa);
  };
  carve(ws_big_, T, true);
  carve(ws_small_, qk::kMaxB, false);
  select_ws(true);
  hsafe_ = (float*)dalloc((size_t)qk::kMaxB * c.n_embd * 4);
  // per-256 activations of the tensor-core small-batch matvec (verify passes)
  xq256_ = (int8_t*)dalloc((size_t)qk::kMaxB * kTqMaxK);
  xd256_ = (float*)dalloc((size_t)qk::kMaxB * (kTqMaxK / 256) * 4);
  if (spec_ && !getenv("QE_NO_BATCH_RB")) {
    std::vector<qk::GdnBatch> tab;
    for (auto& L : layers_)
      if (L.sv_qkv) tab.push_back({L.sv_qkv, L.sv_g, L.sv_beta, L.ssm_state, L.conv_state, L.conv_hist});
    if (!tab.empty()) {
      gdn_tab_ = (qk::GdnBatch*)dalloc(tab.size() * sizeof(qk::GdnBatch));
      CUDA_CHECK(cudaMemcpy(gdn_tab_, tab.data(), tab.size() * sizeof(qk::GdnBatch), cudaMemcpyHostToDevice));
      n_gdn_tab_ = (int)tab.size();
    }
  }
  // fp16-accumulate cuBLAS prefill (for the types GQ does not take), guarded: the recurrent state is
  // backed up to pinned host memory before each chunk and restored if a GEMM overflowed
  gemm16_ = !(getenv("QE_GEMM16") && atoi(getenv("QE_GEMM16")) == 0);
  if (gemm16_ && has_mtp() && !legacy16_) {
    const size_t sb = (size_t)n_gdn_ * state_bytes();
    if (cudaHostAlloc((void**)&state_bk_host_, sb, cudaHostAllocDefault) != cudaSuccess) gemm16_ = false;
    else host_allocs_.push_back(state_bk_host_);
  }
  use_mmq_ = !(getenv("QE_MMQ") && atoi(getenv("QE_MMQ")) == 0);
  mmq_gs_ = getenv("QE_MMQ_GS") ? atoi(getenv("QE_MMQ_GS")) : 32;
  use_gq_ = !getenv("QE_GQ") || atoi(getenv("QE_GQ")) != 0;
  zc_ = !getenv("QE_ZC") || atoi(getenv("QE_ZC")) != 0;
  fuse_ = (!getenv("QE_FUSE") || atoi(getenv("QE_FUSE")) != 0) && !getenv("QE_DEBUG_SUMS");
  mmq_max_T_ = getenv("QE_MMQ_MAXT") ? atoi(getenv("QE_MMQ_MAXT")) : 192;
  ovf_dev_ = (int*)dalloc(256);
  hcarry_bk_ = (float*)dalloc((size_t)c.n_embd * 4);
  attn_work_ = (float*)dalloc(attn_work_bytes_);
  t5_ = (float*)dalloc((size_t)kCpuMaxB * c.n_embd * 4);
  logits_dev_ = (float*)dalloc((size_t)qk::kMaxB * c.n_vocab * 4);
  topk_i_d_ = (int*)dalloc((size_t)(qk::kMaxB + 1) * kTopKMax * 4);
  topk_v_d_ = (float*)dalloc((size_t)(qk::kMaxB + 1) * kTopKMax * 4);
  topk_work_ = dalloc(qk::topk_work_bytes(qk::kMaxB));
  CUDA_CHECK(cudaHostAlloc((void**)&topk_i_h_, (size_t)(qk::kMaxB + 1) * kTopKMax * 4, cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void**)&topk_v_h_, (size_t)(qk::kMaxB + 1) * kTopKMax * 4, cudaHostAllocDefault));
  host_allocs_.push_back(topk_i_h_);
  host_allocs_.push_back(topk_v_h_);
  if (has_mtp() && opt_.mtp_vocab > 0) dv_dev_ = (int*)dalloc((size_t)kDraftVocabCap * 4);
  hcarry_ = (float*)dalloc((size_t)c.n_embd * 4);
  argmax_dev_ = (int*)dalloc(64 * 4);
  mtp_h_ = (float*)dalloc((size_t)c.n_embd * 4);
  conf_dev_ = (float*)dalloc(64 * 4);
  pos_dev_ = (int*)dalloc((size_t)T * 4);
  CUDA_CHECK(cudaHostAlloc((void**)&embd_host_, (size_t)T * c.n_embd * 4, cudaHostAllocDefault));
  host_allocs_.push_back(embd_host_);
  CUDA_CHECK(cudaEventCreate(&gp_start_));
  CUDA_CHECK(cudaEventCreate(&gp_end_));
  if (opt_.ck_slots > 0) {
    const size_t bytes = ck_carry_off() + (size_t)c.n_embd * 4;
    for (int i = 0; i < opt_.ck_slots; i++) {
      uint8_t* hp = nullptr;
      if (cudaHostAlloc((void**)&hp, bytes, cudaHostAllocMapped) != cudaSuccess) break;
      void* dp = nullptr;
      CUDA_CHECK(cudaHostGetDevicePointer(&dp, hp, 0));
      host_allocs_.push_back(hp);
      ck_host_.push_back(hp);
      ck_dev_.push_back((uint8_t*)dp);
    }
    QE_LOG("[plan] prompt cache: %zu state checkpoints of %.0f MiB (pinned host)\n", ck_host_.size(), bytes / (double)MiB);
  }
  // Arena tail: whatever the plan did not use (slack items, rounding of the evenly spaced resident
  // pick) permanently holds full copies of the largest streamed FFN layers that fit.
  if (copy_stream_ && !getenv("QE_NO_TAIL")) {
    std::vector<int> cand;
    for (int il = 0; il < c.n_layer; il++)
      if (layers_[il].ffn_streamed && !layers_[il].borrowed_dev && !layers_[il].ffn_skip && !c.aliased(il)) cand.push_back(il);
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return layers_[a].stream_bytes_full > layers_[b].stream_bytes_full; });
    int n = 0;
    size_t tb = 0;
    const size_t free0 = arena_bytes_ - vram_used_;
    for (int il : cand) {
      Layer& L = layers_[il];
      const size_t b = (L.stream_bytes_full + 255) / 256 * 256;
      if (vram_used_ + b > arena_bytes_) continue;
      L.borrowed_dev = (uint8_t*)dalloc(b);
      CUDA_CHECK(cudaMemcpy(L.borrowed_dev, L.stream_host, L.stream_bytes_full, cudaMemcpyHostToDevice));
      n++;
      tb += b;
    }
    for (int il = 0; il < c.n_layer; il++)
      if (c.aliased(il)) layers_[il].borrowed_dev = layers_[c.wl(il)].borrowed_dev;
    if (n) refresh_streamed();
    QE_LOG("[plan] arena tail %.0f MiB: %d streamed FFN layers made resident (%.0f MiB)\n", free0 / (double)MiB, n, tb / (double)MiB);
  }
  if (c.mhc_streams) {  // two residual streams, the routed sublayer input, router outputs (per token)
    for (float** p : {&mhc_a_, &mhc_b_, &mhc_branch_}) *p = (float*)dalloc((size_t)Tmax_ * c.n_embd * 4);
    mhc_pk_ = (float*)dalloc((size_t)Tmax_ * 8 * 4);
  }
  rope_table_ = (c.rope_mscale != 1.f || c.arch != "qwen35") && !getenv("QE_ROPE_PLAIN");  // QE_ROPE_PLAIN: tests
  if (rope_table_) qk::set_rope_table(c.rope_freq.data(), (int)c.rope_freq.size(), c.rope_mscale);
  reset();
  size_t fr = 0;
  CUDA_CHECK(cudaMemGetInfo(&fr, &total_b));
  QE_LOG("[plan] loaded in %.1f s, engine VRAM %.0f MiB, free now %.0f MiB\n", qe_now() - t0,
         vram_used_ / (double)MiB, fr / (double)MiB);
}

cudaEvent_t Model::gp_ev() {
  if (gp_n_ >= (int)gp_evs_.size()) {
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreate(&e));
    gp_evs_.push_back(e);
  }
  return gp_evs_[gp_n_++];
}
cudaEvent_t Model::sec_ev(int il, int k) {
  while ((int)sec_evs_.size() < (cfg_.n_layer + 1) * 3) {
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreate(&e));
    sec_evs_.push_back(e);
  }
  return sec_evs_[il * 3 + k];
}
cudaEvent_t Model::cp_ev() {
  if (cp_n_ >= (int)cp_evs_.size()) {
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreate(&e));
    cp_evs_.push_back(e);
  }
  return cp_evs_[cp_n_++];
}

void Model::reset() {
  const Config& c = cfg_;
  for (auto& L : layers_) {
    if (L.conv_state) CUDA_CHECK(cudaMemsetAsync(L.conv_state, 0, 3 * (size_t)c.conv_dim() * 4, stream_));
    if (L.ssm_state) CUDA_CHECK(cudaMemsetAsync(L.ssm_state, 0, state_bytes(), stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---------------------------------------------------------------------------------------------
// Streamed weights rotate through a ring of VRAM slots (2 full-layer slots for prefill; the same
// buffer split into more, smaller slots for decode where only the GPU part of a layer streams).
// Copies run on copy_stream_ ahead of compute, wrapping around to the first streamed layer of the
// next pass, so the copy engine never waits for compute unless the ring is full.
void Model::issue_copy(int slot, int li) {
  Layer& L = layers_[li];
  const size_t bytes = (stream_full_ || !L.hyb_on) ? L.stream_bytes_full : L.stream_bytes_gpu;
  QE_ASSERT(bytes <= slot_bytes_);
  CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, slot_free_[slot], 0));
  if (gprof_on_ && bytes) CUDA_CHECK(cudaEventRecord(cp_ev(), copy_stream_));
  if (bytes)
    CUDA_CHECK(cudaMemcpyAsync(stream_buf_ + (size_t)slot * slot_bytes_, L.stream_host, bytes,
                               cudaMemcpyHostToDevice, copy_stream_));
  if (gprof_on_ && bytes) { CUDA_CHECK(cudaEventRecord(cp_ev(), copy_stream_)); cp_bytes_ += bytes; }
  CUDA_CHECK(cudaEventRecord(slot_ready_[slot], copy_stream_));
  slot_layer_[slot] = li;
  slot_waited_[slot] = false;
  stream_total_ += bytes;
  last_pf_ = li;
}

int Model::next_streamed(int li) const {
  if (streamed_layers_.empty()) return -1;
  auto it = std::upper_bound(streamed_layers_.begin(), streamed_layers_.end(), li);
  return it == streamed_layers_.end() ? streamed_layers_.front() : *it;
}

int Model::find_slot(int li) const {
  for (int i = 0; i < nslots_; i++)
    if (slot_layer_[i] == li) return i;
  return -1;
}

void Model::ring_reset() {
  for (int i = 0; i < kMaxSlots; i++) {
    slot_layer_[i] = -1;
    slot_waited_[i] = false;
    if (copy_stream_) CUDA_CHECK(cudaEventRecord(slot_free_[i], stream_));
  }
  last_pf_ = -1;
}

const void* Model::mat_ptr(Layer* L, Mat& m, cudaStream_t s) {
  if (m.dev) return m.dev;
  QE_ASSERT(L && m.stream_slot_off >= 0);
  if (L->borrowed_dev) return L->borrowed_dev + m.stream_slot_off;
  const int li = (int)(L - layers_.data());
  int slot = find_slot(li);
  if (slot < 0) {
    // ring not primed for this layer: (re)fill every slot starting from li
    ring_reset();
    int layer = li;
    for (int k = 0; k < nslots_; k++) {
      if (k > 0) {
        layer = next_streamed(layer);
        if (layer == li || find_slot(layer) >= 0) break;
      }
      issue_copy(k, layer);
    }
    slot = 0;
  }
  if (!slot_waited_[slot]) {
    CUDA_CHECK(cudaStreamWaitEvent(s, slot_ready_[slot], 0));
    slot_waited_[slot] = true;
  }
  return stream_buf_ + (size_t)slot * slot_bytes_ + m.stream_slot_off;
}

void Model::set_stream_mode(bool full) {
  if (!copy_stream_) return;
  if (full == stream_full_ && nslots_ > 0 && !hyb_dirty_) return;
  hyb_dirty_ = false;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
  stream_full_ = full;
  const size_t total = 2 * stream_buf_bytes_;
  if (full) {
    nslots_ = 2;
    slot_bytes_ = stream_buf_bytes_;
  } else {
    size_t mx = 0;
    for (int il = 0; il < cfg_.n_layer; il++)
      if (layers_[il].ffn_streamed)
        mx = std::max(mx, layers_[il].hyb_on ? layers_[il].stream_bytes_gpu : layers_[il].stream_bytes_full);
    mx = (mx + 255) / 256 * 256;
    nslots_ = (int)std::clamp<size_t>(mx ? total / mx : 2, 2, kMaxSlots);
    slot_bytes_ = total / nslots_ / 256 * 256;
  }
  ring_reset();
  QE_DBG("[stream] mode %s: %d slots of %.0f MiB\n", full ? "full" : "decode", nslots_, slot_bytes_ / (double)MiB);
}

// called after the last use of layer li's streamed weights in this pass: refill its slot with the
// next layer after the furthest one already prefetched
void Model::stream_release(int li) {
  if (!layers_[li].ffn_streamed || layers_[li].borrowed_dev) return;
  const int slot = find_slot(li);
  if (slot < 0) return;
  CUDA_CHECK(cudaEventRecord(slot_free_[slot], stream_));
  const int want = next_streamed(last_pf_);
  if (want >= 0 && find_slot(want) < 0) issue_copy(slot, want);
  else slot_waited_[slot] = false;  // everything fits in the ring: keep it resident in its slot
}

// Activation formats: decode matvec = int8 per 32; prefill MMQ = int8 per 256; cuBLAS = f16.
// The two prefill formats can be live at the same time (BF16 alpha/beta use cuBLAS).
void Model::prepare_input(const float* x, int K, int T, cudaStream_t s, int fmt) {
  if (fmt == 2) {
    if (prep16_ == x && prep16_K_ == K && prep16_T_ == T) return;
    qk::f32_to_f16(x, a16_, (int64_t)T * K, s);
    prep16_ = x; prep16_K_ = K; prep16_T_ = T;
    return;
  }
  if (prepared_ == x && prepared_K_ == K && prepared_T_ == T) return;
  if (T <= qk::kMaxB || mmq_gs_ == 32) qk::quantize_q8(x, xq_, xd_, K, T, s);
  else qk::quantize_q8_256(x, xq_, xd_, K, T, s);
  prep256_ = nullptr; prepared_ = x;
  prepared_K_ = K;
  prepared_T_ = T;
}

void Model::linear(Layer* L, Mat& m, const float* x, int T, float* y, int ldy, bool acc,
                   cudaStream_t s) {
  const void* W = mat_ptr(L, m, s);
  const bool is_float = m.type == QT_F32 || m.type == QT_F16 || m.type == QT_BF16;
  if (T <= qk::kMaxB) {
    if (!is_float && use_tq(m, T)) {
      static const int gs = getenv("QE_TQ_GS") && atoi(getenv("QE_TQ_GS")) == 256 ? 256 : 32;
      if (gs == 32) {  // the dp4a kernels' activations: same numbers as the T = 1 path
        prepare_input(x, m.K, T, s);
        qk::matvec_tq(m.type, W, m.row_bytes, m.K, m.N, xq_, xd_, T, y, ldy, acc, s, 32);
        return;
      }
      if (prep256_ != x || prep256_K_ != m.K || prep256_T_ != T) {
        qk::quantize_q8_256(x, xq256_, xd256_, m.K, T, s);
        prep256_ = x; prep256_K_ = m.K; prep256_T_ = T;
      }
      qk::matvec_tq(m.type, W, m.row_bytes, m.K, m.N, xq256_, xd256_, T, y, ldy, acc, s, 256);
      return;
    }
    if (!is_float) prepare_input(x, m.K, T, s);
    qk::matvec(m.type, W, m.row_bytes, m.K, m.N, xq_, xd_, x, T, y, ldy, acc, s);
    return;
  }
  // small prefill chunks: MMQ (no f16 weight expansion, cost independent of the chunk being short);
  // large chunks: f16 expansion + fp16-accumulate cuBLAS amortizes better
  if (use_mmq_ && qk::mmq_supported(m.type) && T < mmq_max_T_) {
    ProfScope ps(4, s);
    prepare_input(x, m.K, T, s, 1);
    qk::mmq(m.type, W, m.row_bytes, m.K, m.N, xq_, xd_, T, y, ldy, acc, mmq_gs_, s);
    return;
  }
  if (use_gq_ && qk::gq_supported(m.type) && m.K % 256 == 0) {
    // decode-once int8 GEMM: activations int8 per 256 (+ 16-group sums for MIN types) in the f16
    // activation buffer, weights decoded chunk-wise into the f16 dequant buffer
    const bool xsum = qk::gq_needs_xsum(m.type);
    auto al = [](size_t b) { return (b + 255) / 256 * 256; };
    int8_t* gx = (int8_t*)a16_;
    float* gxd = (float*)(gx + al((size_t)T * m.K));
    int16_t* gxs = (int16_t*)((uint8_t*)gxd + al((size_t)T * (m.K / 256) * 4));
    if (gqp_ != x || gqp_K_ != m.K || gqp_T_ != T || (xsum && !gqp_xs_)) {
      prep16_ = nullptr;  // a16_ is overwritten
      { ProfScope ps(5, s); qk::quantize_q8_256(x, gx, gxd, m.K, T, s); }
      if (xsum) qk::gq_xsum(gx, m.K, T, gxs, s);
      gqp_ = x; gqp_K_ = m.K; gqp_T_ = T; gqp_xs_ = xsum;
    }
    const size_t per_row = (size_t)m.K + 2 * (m.K / 16) + 2 * (m.K / 256) * 4;
    int rows_per = (int)std::min<size_t>(m.N, (w16_elems_ * 2 - 4096) / per_row);
    if (rows_per < m.N) rows_per = std::max(64, rows_per / 64 * 64);
    for (int r0 = 0; r0 < m.N; r0 += rows_per) {
      const int nr = std::min(rows_per, m.N - r0);
      { ProfScope ps(5, s); qk::gq_decode(m.type, (const uint8_t*)W + (size_t)r0 * m.row_bytes, m.row_bytes, m.K, nr, w16_, s); }
      ProfScope ps(4, s);
      qk::gq_gemm(m.type, w16_, m.K, nr, gx, gxd, gxs, T, y + r0, ldy, acc, s);
    }
    return;
  }
  prepare_input(x, m.K, T, s, 2);
  gqp_ = nullptr;  // a16_ is overwritten
  // chunk rows so the f16 dequant buffer suffices
  const int rows_per = std::max(1, (int)std::min<size_t>(m.N, w16_elems_ / m.K));
  const float alpha = 1.f, beta = acc ? 1.f : 0.f;
  for (int r0 = 0; r0 < m.N; r0 += rows_per) {
    const int nr = std::min(rows_per, m.N - r0);
    { ProfScope ps(5, s); qk::dequant_f16(m.type, (const uint8_t*)W + (size_t)r0 * m.row_bytes, m.row_bytes, m.K, nr, w16_, s); }
    ProfScope ps(4, s);
    // C[T][N] row-major = (col-major N x T); C = W(nr x K) * X^T
    if (use16_) {
      // fp16-accumulate tensor cores (2x the fp32-accumulate rate on GeForce Ampere), then widen
      const __half ha = __float2half(1.f), hb = __float2half(0.f);
      CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, nr, T, m.K, &ha, w16_, CUDA_R_16F, m.K,
                                a16_, CUDA_R_16F, m.K, &hb, c16_, CUDA_R_16F, nr,
                                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
      qk::h2f_accum(c16_, nr, T, nr, y + r0, ldy, acc, ovf_dev_, s);
    } else {
      CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, nr, T, m.K, &alpha, w16_, CUDA_R_16F, m.K,
                                a16_, CUDA_R_16F, m.K, &beta, y + r0, CUDA_R_32F, ldy,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
  }
}

void Model::embed(const int* tokens, int n, float* dst) {
  const Config& c = cfg_;
  for (int t = 0; t < n; t++) {
    QE_ASSERT(tokens[t] >= 0 && tokens[t] < c.n_vocab);
    dequant_row_ref(tok_embd_.type, tok_embd_.host + (size_t)tokens[t] * tok_embd_.row_bytes, c.n_embd,
                    embd_host_ + (size_t)t * c.n_embd);
  }
  if (n <= qk::kMaxB && zc_) qk::copy_f32_zc(dst, embd_host_, (int64_t)n * c.n_embd, stream_);
  else CUDA_CHECK(cudaMemcpyAsync(dst, embd_host_, (size_t)n * c.n_embd * 4, cudaMemcpyHostToDevice, stream_));
}

void Model::layer_attn(int il, int T, int pos0) {
  const Config& c = cfg_;
  Layer& L = layers_[il];
  cudaStream_t s = stream_;
  const int qw = c.n_head * c.hd * 2, kvw = c.n_kv * c.hd;
  norm_in(L.attn_norm, xn_, T, true);
  float* qg = t0_;
  float* k = t3_;
  float* v = t3_ + (size_t)T * kvw;
  linear(&L, L.wq, xn_, T, qg, qw, false, s);
  linear(&L, L.wk, xn_, T, k, kvw, false, s);
  linear(&L, L.wv, xn_, T, v, kvw, false, s);
  if (!fuse_ || !qk::qk_norm_rope(qg, L.q_norm, 2 * c.hd, qw, c.n_head, k, L.k_norm, c.hd, kvw, c.n_kv, pos_dev_, c.hd, T,
                                  c.rope_dims, rope_table_ ? -1.f : c.rope_base, c.eps, s)) {
    qk::rmsnorm_heads(qg, L.q_norm, c.hd, c.n_head, 2 * c.hd, T, qw, c.eps, s);
    qk::rmsnorm_heads(k, L.k_norm, c.hd, c.n_kv, c.hd, T, kvw, c.eps, s);
    const float rb = rope_table_ ? -1.f : c.rope_base;  // < 0: this model's YaRN table
    qk::rope_neox(qg, pos_dev_, c.hd, c.n_head, 2 * c.hd, T, qw, c.rope_dims, rb, s);
    qk::rope_neox(k, pos_dev_, c.hd, c.n_kv, c.hd, T, kvw, c.rope_dims, rb, s);
  }
  qk::kv_store(opt_.kv_type, L.kcache, L.vcache, k, v, pos_dev_, c.n_kv, c.hd, T, kvw, kvw, kvmap(), s);
  float* o = t2_;
  const float scale = 1.f / std::sqrt((float)c.hd);
  const int attT = attn_dec_T();  // attn_decode handles verify passes up to this; larger T: prefill FA kernel
  if (T <= attT && !getenv("QE_FORCE_FA")) {
    qk::attn_decode(opt_.kv_type, L.kcache, L.vcache, qg, 2 * c.hd, qw, o, c.n_head, c.n_kv, c.hd, pos0, T,
                    scale, attn_work_, attn_work_bytes_, kvmap(), s);
  } else if (T <= qk::kMaxB || getenv("QE_SLOW_ATTN")) {
    // small batches run with the prefill workspace lent to FFN weights (fa_work_ lives there), so they
    // use the decode kernel in chunks of attT; also the reference path for testing the prefill kernel
    for (int t0 = 0; t0 < T; t0 += attT) {
      const int nt = std::min(attT, T - t0);
      qk::attn_decode(opt_.kv_type, L.kcache, L.vcache, qg + (size_t)t0 * qw, 2 * c.hd, qw,
                      o + (size_t)t0 * c.n_head * c.hd, c.n_head, c.n_kv, c.hd, pos0 + t0, nt, scale,
                      attn_work_, attn_work_bytes_, kvmap(), s);
    }
  } else {
    ProfScope ps(3, s);
    qk::attn_prefill(opt_.kv_type, L.kcache, L.vcache, qg, 2 * c.hd, qw, o, c.n_head, c.n_kv, c.hd, pos0, T,
                     scale, fa_work_, kvmap(), s);
  }
  if (fuse_ && T <= qk::kMaxB) {
    qk::gate_sigmoid(o, qg, c.hd, c.n_head, T, s, nullptr, xq_, xd_);
    prep256_ = nullptr; prepared_ = o; prepared_K_ = c.n_head * c.hd; prepared_T_ = T; prep16_ = nullptr; gqp_ = nullptr;
  } else {
    qk::gate_sigmoid(o, qg, c.hd, c.n_head, T, s);
    prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
  }
  linear(&L, L.wo, o, T, t4_, c.n_embd, false, s);
  residual(t4_, T);
}

// Layer input norm.  On the decode path the previous layer's residual add is deferred into it and
// the result is quantized for the matvec kernels in the same kernel.
void Model::norm_in(const float* w, float* y, int T, bool quant) {
  const Config& c = cfg_;
  if (mhc_on_) {  // mHC sublayer: the input is the routed mix of the two streams, no pending residual
    if (fuse_ && T <= qk::kMaxB) {
      qk::add_rmsnorm_q8(mhc_branch_, nullptr, w, y, c.n_embd, T, c.eps, quant ? xq_ : nullptr, quant ? xd_ : nullptr, stream_);
      prep16_ = nullptr; gqp_ = nullptr;
      if (quant) { prep256_ = nullptr; prepared_ = y; prepared_K_ = c.n_embd; prepared_T_ = T; }
      else prep256_ = nullptr; prepared_ = nullptr;
    } else {
      qk::rmsnorm(mhc_branch_, w, y, c.n_embd, T, c.eps, stream_);
      prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
    }
    return;
  }
  if (fuse_ && T <= qk::kMaxB) {
    qk::add_rmsnorm_q8(x_, pend_, w, y, c.n_embd, T, c.eps, quant ? xq_ : nullptr, quant ? xd_ : nullptr, stream_);
    pend_ = nullptr;
    prep16_ = nullptr; gqp_ = nullptr;
    if (quant) { prep256_ = nullptr; prepared_ = y; prepared_K_ = c.n_embd; prepared_T_ = T; }
    else prep256_ = nullptr; prepared_ = nullptr;
    return;
  }
  if (pend_) { qk::add_inplace(x_, pend_, (int64_t)T * c.n_embd, stream_); pend_ = nullptr; }
  qk::rmsnorm(x_, w, y, c.n_embd, T, c.eps, stream_);
  prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
}
void Model::residual(const float* src, int T) {
  if (mhc_on_) { mhc_out_ = src; return; }  // merged into the streams by the caller
  QE_ASSERT(!pend_);
  if (fuse_ && T <= qk::kMaxB) pend_ = src;
  else qk::add_inplace(x_, src, (int64_t)T * cfg_.n_embd, stream_);
}

void Model::layer_gdn(int il, int T) {
  const Config& c = cfg_;
  Layer& L = layers_[il];
  cudaStream_t s = stream_;
  const int C = c.conv_dim();
  norm_in(L.attn_norm, xn_, T, true);
  float* qkv_raw = t0_;
  float* z = t1_;
  float* ab = t4_;  // alpha [T][nv] then beta [T][nv] (t4 is free until the output projection)
  linear(&L, L.wqkv, xn_, T, qkv_raw, C, false, s);
  linear(&L, L.wgate, xn_, T, z, c.d_inner, false, s);
  if (fuse_ && T <= qk::kMaxB && L.walpha.dev && L.wbeta.dev && L.walpha.type == QT_BF16 && L.wbeta.type == QT_BF16 &&
      L.walpha.row_bytes == L.wbeta.row_bytes && c.nv <= 256 && L.walpha.row_bytes % 16 == 0) {
    qk::matvec_float_pair(QT_BF16, L.walpha.dev, L.wbeta.dev, L.walpha.row_bytes, c.n_embd, c.nv, xn_, T, ab,
                          ab + (size_t)T * c.nv, c.nv, s);
  } else {
    linear(&L, L.walpha, xn_, T, ab, c.nv, false, s);
    linear(&L, L.wbeta, xn_, T, ab + (size_t)T * c.nv, c.nv, false, s);
  }
  qe_debug_dump("linear_attn_qkv_mixed", il, qkv_raw, T, C, s);
  qe_debug_dump("z", il, z, T, c.d_inner, s);
  float* qkv = t3_;
  const bool snap = snapshot_ && L.sv_qkv;
  if (snap) qkv = L.sv_qkv;  // keep the recurrence inputs of the verified tokens for a rollback
  float* gg = snap ? L.sv_g : g_;
  float* bb = snap ? L.sv_beta : beta_;
  // checkpoint capture: stage in device memory (alt state / conv history), DMA to the host slot after
  // the chunk -- zero-copy kernel writes over PCIe are far too slow for 150 MB
  float* ck_conv = nullptr;
  float* ck_state = nullptr;
  float* o = t2_;
  {
  ProfScope psc(6, s);
  qk::ssm_conv(qkv_raw, C, L.conv_state, L.conv_w, qkv, C, T, snap ? L.conv_hist : nullptr, s, snap_t_, ck_conv);
  qk::gdn_prep(qkv, C, ab, ab + (size_t)T * c.nv, c.nv, L.dt_bias, L.ssm_a, gg, bb, c.nk, c.nv, c.dk, T,
               c.eps, s);
  qe_debug_dump("conv_output_silu", il, qkv, T, C, s);
  // verify pass: read state A, write B (A survives for a rollback); otherwise in place
  {
    ProfScope psr(7, s);
    // verify pass: the state is only read; rollback() recomputes the accepted prefix in place
    qk::gdn_recurrence(qkv, C, gg, bb, L.ssm_state, snap ? nullptr : L.ssm_state, o, c.nk, c.nv, c.dk, T, s,
                       -1, ck_state, state16_);
  }
  qe_debug_dump("attn_output", il, o, T, c.d_inner, s);
  const bool gq8 = fuse_ && T <= qk::kMaxB && c.dv % 32 == 0;
  qk::gdn_gated_norm(o, z, c.d_inner, L.ssm_norm, c.nv, c.dv, T, c.eps, s, nullptr, gq8 ? xq_ : nullptr, gq8 ? xd_ : nullptr);
  }
  qe_debug_dump("final_output", il, o, T, c.d_inner, s);
  prep16_ = nullptr; gqp_ = nullptr;
  if (fuse_ && T <= qk::kMaxB && c.dv % 32 == 0) { prep256_ = nullptr; prepared_ = o; prepared_K_ = c.d_inner; prepared_T_ = T; }  // quantized by gated_norm
  else prep256_ = nullptr; prepared_ = nullptr;
  linear(&L, L.wssm_out, o, T, t4_, c.n_embd, false, s);
  qe_debug_dump("linear_attn_out", il, t4_, T, c.n_embd, s);
  residual(t4_, T);
}

void Model::layer_ffn(int il, int T) {
  const Config& c = cfg_;
  Layer& L = layers_[il];
  cudaStream_t s = stream_;
  if (L.ffn_skip) return;  // dropped by --skip-ffn (lossy)
  norm_in(L.post_norm, xn_, T, true);
  if (!L.ffn_streamed) {
    linear(&L, L.ffn_gate, xn_, T, t0_, c.n_ff, false, s);
    linear(&L, L.ffn_up, xn_, T, t1_, c.n_ff, false, s);
    if (fuse_ && T <= qk::kMaxB && c.n_ff % 32 == 0) {
      qk::silu_mul_q8(t0_, t1_, t0_, (int64_t)T * c.n_ff, xq_, xd_, s);
      prep256_ = nullptr; prepared_ = t0_; prepared_K_ = c.n_ff; prepared_T_ = T; prep16_ = nullptr; gqp_ = nullptr;
    } else {
      qk::silu_mul(t0_, t1_, t0_, (int64_t)T * c.n_ff, s);
      prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
    }
    linear(&L, L.ffn_down, t0_, T, t4_, c.n_embd, false, s);
    residual(t4_, T);
    return;
  }
  const int r = L.cpu_rows, Ng = c.n_ff - r;
  const bool hybrid = !stream_full_ && r > 0 && !L.borrowed_dev && L.hyb_on;
  if (hybrid) {  // hand the normalized input to the CPU workers
    CUDA_CHECK(cudaMemcpyAsync(hx_in_, xn_, (size_t)T * c.n_embd * 4, cudaMemcpyDeviceToHost, s));
    if (cuStreamWriteValue32((CUstream)s, dflags_, L.job_seq, CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS)
      QE_FATAL("cuStreamWriteValue32 failed");
  }
  float* hg = t0_;
  float* ug = t1_;
  if (Ng > 0) {
    linear(&L, L.gate_g, xn_, T, hg, Ng, false, s);
    linear(&L, L.up_g, xn_, T, ug, Ng, false, s);
    qk::silu_mul(hg, ug, hg, (int64_t)T * Ng, s);
  }
  if (!hybrid && r > 0) {  // prefill: CPU part streamed too and computed on the GPU
    float* hc = t0_ + (size_t)T * Ng;
    float* uc = t1_ + (size_t)T * Ng;
    linear(&L, L.gate_c, xn_, T, hc, r, false, s);
    linear(&L, L.up_c, xn_, T, uc, r, false, s);
    qk::silu_mul(hc, uc, hc, (int64_t)T * r, s);
    prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
    linear(&L, L.down_c, hc, T, t4_, c.n_embd, false, s);
    if (Ng > 0) {
      prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
      linear(&L, L.down_g, hg, T, t4_, c.n_embd, true, s);
    }
  } else {
    prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
    if (Ng > 0) linear(&L, L.down_g, hg, T, t4_, c.n_embd, false, s);
    else CUDA_CHECK(cudaMemsetAsync(t4_, 0, (size_t)T * c.n_embd * 4, s));
  }
  if (hybrid) {  // wait for the CPU partial result and add it
    if (gprof_on_) CUDA_CHECK(cudaEventRecord(gp_ev(), s));
    if (cuStreamWaitValue32((CUstream)s, dflags_ + 64, L.job_seq, CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS)
      QE_FATAL("cuStreamWaitValue32 failed");
    if (gprof_on_) CUDA_CHECK(cudaEventRecord(gp_ev(), s));
    if (zc_) qk::copy_f32_zc(t5_, hy_out_, (int64_t)T * c.n_embd, s);
    else CUDA_CHECK(cudaMemcpyAsync(t5_, hy_out_, (size_t)T * c.n_embd * 4, cudaMemcpyHostToDevice, s));
    qk::add_inplace(t4_, t5_, (int64_t)T * c.n_embd, s);
  }
  residual(t4_, T);
  stream_release(il);
}

static void cpu_mv(const Mat& m, int r0, int r1, const cpu::Q8Act& x, int T, float* y, int ldy) {
  if (m.cpu_tab) cpu::matvec_r4(m.cpu_tab, m.cpu_w, m.K, r0, r1, x, T, y, ldy);
  else cpu::matvec(m.type, m.cpu_w, m.row_bytes, m.K, r0, r1, x, T, y, ldy);
}

// CPU worker side of the hybrid FFN for one forward pass: for every hybrid layer (in order) wait
// for the GPU to publish the normalized input, compute the CPU row-slice of gate/up/down, publish.
void Model::cpu_ffn_job(const std::vector<std::pair<int, uint32_t>>& jobs, int T, int tid) {
  const Config& c = cfg_;
  const int n = pool_->size();
  static const bool prof = getenv("QE_PROFILE") != nullptr;
  double tw = 0, tc = 0, t_start = prof ? qe_now() : 0;
  (void)t_start;
  for (auto [li, seq] : jobs) {
    Layer& L = layers_[li];
    const int r = L.cpu_rows;
    double t0 = 0;
    if (tid == 0) {
      if (prof) t0 = qe_now();
      while (hflags_[0] < seq) _mm_pause();
      std::atomic_thread_fence(std::memory_order_acquire);
      if (prof) { tw += qe_now() - t0; t0 = qe_now(); }
    }
    pool_->barrier();
    double tp = prof && tid == 0 ? qe_now() : 0;
    auto phase = [&](int k) {
      if (prof && tid == 0) { const double t = qe_now(); cpu_phase_[k] += t - tp; tp = t; }
    };
    {
      const int nb = c.n_embd / 256;
      for (int b = 0; b < T; b++) cpu::quantize(hx_in_, c.n_embd, b, cx_, nb * tid / n, nb * (tid + 1) / n);
    }
    pool_->barrier();
    phase(0);
    {
      const int r0 = (int)((int64_t)r * tid / n), r1 = (int)((int64_t)r * (tid + 1) / n);
      cpu_mv(L.gate_c, r0, r1, cx_, T, cg_.data(), r);
      cpu_mv(L.up_c, r0, r1, cx_, T, cu_.data(), r);
      for (int b = 0; b < T; b++)
        for (int i = r0; i < r1; i++) {
          const float g = cg_[(size_t)b * r + i];
          cg_[(size_t)b * r + i] = g / (1.f + std::exp(-g)) * cu_[(size_t)b * r + i];
        }
    }
    pool_->barrier();
    phase(1);
    {
      const int nb = r / 256;
      for (int b = 0; b < T; b++) cpu::quantize(cg_.data(), r, b, ch_, nb * tid / n, nb * (tid + 1) / n);
    }
    pool_->barrier();
    phase(2);
    {
      const int r0 = c.n_embd * tid / n, r1 = c.n_embd * (tid + 1) / n;
      cpu_mv(L.down_c, r0, r1, ch_, T, hy_out_, c.n_embd);
    }
    pool_->barrier();
    phase(3);
    if (prof && tid == 0) cpu_phase_bytes_ += (double)(L.stream_bytes_full - L.stream_bytes_gpu);
    if (tid == 0) {
      std::atomic_thread_fence(std::memory_order_release);
      hflags_[16] = seq;
      if (prof) tc += qe_now() - t0;
    }
  }
  if (prof && tid == 0) {
    cpu_prof_wait_ += tw;
    cpu_prof_comp_ += tc;
    cpu_prof_total_ += qe_now() - t_start;
    cpu_prof_n_++;
    if (cpu_prof_n_ % 50 == 0)
      fprintf(stderr, "[prof] cpu ffn per pass: wait-for-gpu %.1f ms, compute %.1f ms, pass %.1f ms (%zu layers)\n"
                      "[prof] cpu phases per pass: quant-x %.2f, gate/up %.2f, quant-h %.2f, down %.2f ms; %.0f MB -> %.1f GB/s in matvecs\n",
              cpu_prof_wait_ / cpu_prof_n_ * 1e3, cpu_prof_comp_ / cpu_prof_n_ * 1e3, cpu_prof_total_ / cpu_prof_n_ * 1e3, jobs.size(),
              cpu_phase_[0] / cpu_prof_n_ * 1e3, cpu_phase_[1] / cpu_prof_n_ * 1e3, cpu_phase_[2] / cpu_prof_n_ * 1e3,
              cpu_phase_[3] / cpu_prof_n_ * 1e3, cpu_phase_bytes_ / cpu_prof_n_ / 1e6,
              cpu_phase_bytes_ / (cpu_phase_[1] + cpu_phase_[3]) / 1e9);
  }
}

// MTP (nextn) block: x = eh_proj([enorm(emb(x_p)), hnorm(h_{p-1})]); one full-attention decoder
// layer with its own KV cache; final norm.  Leaves the normed result rows in xn_.
void Model::mtp_block(const float* hrows, const int* tokens, int T, int pos0) {
  const Config& c = cfg_;
  const int il = c.n_layer;
  Layer& L = layers_[il];
  cudaStream_t s = stream_;
  qk::iota_i32(pos_dev_, pos0, T, s);
  float* e = t1_;
  float* hn = t2_;
  embed(tokens, T, e);
  qk::rmsnorm(e, L.enorm, e, c.n_embd, T, c.eps, s);
  qk::rmsnorm(hrows, L.hnorm, hn, c.n_embd, T, c.eps, s);
  float* cat = t0_;  // [T][2*n_embd]
  CUDA_CHECK(cudaMemcpy2DAsync(cat, 2 * c.n_embd * 4, e, c.n_embd * 4, c.n_embd * 4, T, cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpy2DAsync(cat + c.n_embd, 2 * c.n_embd * 4, hn, c.n_embd * 4, c.n_embd * 4, T, cudaMemcpyDeviceToDevice, s));
  prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
  linear(&L, L.eh_proj, cat, T, x_, c.n_embd, false, s);
  const bool snap_save = snapshot_;
  snapshot_ = false;
  layer_attn(il, T, pos0);
  layer_ffn(il, T);
  snapshot_ = snap_save;
  norm_in(L.shared_head_norm ? L.shared_head_norm : output_norm_, xn_, T, false);
}

// MTP draft head.  Drafts only need the likely tokens, so the logits are computed over a vocabulary
// subset: the first `mtp_vocab` ids (BPE ids are roughly frequency ordered) plus the special-token
// tail.  The draft distribution is defined on that subset, which keeps speculative sampling exact.
// MTP draft head.  With a draft vocabulary (opt.mtp_vocab > 0) the logits are computed only for the
// first mtp_vocab ids, the special-token tail and every token seen in the conversation (gathered rows
// of the output head: ~1/8 of the bytes).  Drafts outside the subset are simply never proposed;
// verification still uses the full head, so this only affects the acceptance rate.

void Model::draft_vocab_add(const int* ids, int n) {
  const Config& c = cfg_;
  if (opt_.mtp_vocab <= 0) return;
  if (dv_has_.empty()) {
    dv_has_.assign(c.n_vocab, 0);
    const int head = std::min(opt_.mtp_vocab, c.n_vocab);
    for (int i = 0; i < head; i++) { dv_has_[i] = 1; dv_host_.push_back(i); }
    for (int i = std::max(head, c.n_vocab - 320); i < c.n_vocab; i++) { dv_has_[i] = 1; dv_host_.push_back(i); }
    dv_cls_.assign(c.n_vocab, 0);
    if (const GGUFValue* tk = gguf_->get("tokenizer.ggml.tokens"))
      for (int i = 0; i < c.n_vocab && i < (int)tk->arr.size(); i++) dv_cls_[i] = (uint8_t)token_script(tk->arr[i].as_str());
    dv_dirty_ = true;
  }
  auto add = [&](int t) {
    if (dv_has_[t] || (int)dv_host_.size() >= kDraftVocabCap) return;
    dv_has_[t] = 1;
    dv_host_.push_back(t);
    dv_dirty_ = true;
  };
  for (int i = 0; i < n; i++) {
    const int t = ids[i];
    if (t < 0 || t >= c.n_vocab) continue;
    const int cl = dv_cls_[t];
    if (cl && !(dv_cls_on_ & (1u << cl))) {  // a script appears: every token of it becomes draftable
      dv_cls_on_ |= 1u << cl;
      if (cl == 2) dv_cls_on_ |= 1u << 1;  // CJK text: kana too (Japanese)
      if (cl == 1) dv_cls_on_ |= 1u << 2;
      for (int j = 0; j < c.n_vocab; j++)
        if (dv_cls_[j] && (dv_cls_on_ & (1u << dv_cls_[j]))) add(j);
      QE_DBG("[mtp] draft vocabulary: script %d enabled, %zu tokens\n", cl, dv_host_.size());
    }
    add(t);
  }
}

void Model::topk_to_host(const float* dev_logits, int V, int rows, int host_row0) {
  qk::topk_rows(dev_logits, V, rows, topk_K_, topk_v_d_ + (size_t)host_row0 * kTopKMax, topk_i_d_ + (size_t)host_row0 * kTopKMax,
                topk_work_, stream_);
  // rows are packed with stride K on the device; host rows use stride kTopKMax
  CUDA_CHECK(cudaMemcpy2DAsync(topk_v_h_ + (size_t)host_row0 * kTopKMax, kTopKMax * 4, topk_v_d_ + (size_t)host_row0 * kTopKMax,
                               topk_K_ * 4, topk_K_ * 4, rows, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpy2DAsync(topk_i_h_ + (size_t)host_row0 * kTopKMax, kTopKMax * 4, topk_i_d_ + (size_t)host_row0 * kTopKMax,
                               topk_K_ * 4, topk_K_ * 4, rows, cudaMemcpyDeviceToHost, stream_));
}

int Model::mtp_finish(int n, float* conf, float* logits_host) {
  const Config& c = cfg_;
  // keep the normed MTP output of the last row for chaining
  CUDA_CHECK(cudaMemcpyAsync(mtp_h_, xn_ + (size_t)(n - 1) * c.n_embd, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
  prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
  const float* hrow = xn_ + (size_t)(n - 1) * c.n_embd;
  const bool sub = !dv_host_.empty() && output_.dev;
  const int nl = sub ? (int)dv_host_.size() : c.n_vocab;
  if (sub) {
    if (dv_dirty_) {
      CUDA_CHECK(cudaMemcpyAsync(dv_dev_, dv_host_.data(), dv_host_.size() * 4, cudaMemcpyHostToDevice, stream_));
      dv_dirty_ = false;
    }
    prepare_input(hrow, c.n_embd, 1, stream_);
    qk::matvec_rows(output_.type, output_.dev, output_.row_bytes, c.n_embd, dv_dev_, nl, xq_, xd_, logits_dev_, stream_);
  } else {
    linear(nullptr, output_, hrow, 1, logits_dev_, c.n_vocab, false, stream_);
  }
  int d = 0;
  const bool tk = logits_host && topk_K_ > 0;
  if (tk) {
    topk_to_host(logits_dev_, nl, 1, qk::kMaxB);
  } else if (logits_host) {
    if (sub) {
      dv_tmp_.resize(nl);
      CUDA_CHECK(cudaMemcpyAsync(dv_tmp_.data(), logits_dev_, (size_t)nl * 4, cudaMemcpyDeviceToHost, stream_));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(logits_host, logits_dev_, (size_t)c.n_vocab * 4, cudaMemcpyDeviceToHost, stream_));
    }
  }
  if (conf) {
    qk::argmax_prob(logits_dev_, nl, argmax_dev_, conf_dev_, stream_);
    CUDA_CHECK(cudaMemcpyAsync(conf, conf_dev_, 4, cudaMemcpyDeviceToHost, stream_));
  } else {
    qk::argmax_rows(logits_dev_, nl, 1, argmax_dev_, stream_);
  }
  CUDA_CHECK(cudaMemcpyAsync(&d, argmax_dev_, 4, cudaMemcpyDeviceToHost, stream_));
  static const bool m2 = getenv("QE_MEASURE_TOP2") != nullptr;
  if (m2) {  // experiment: second-best MTP candidate
    qk::topk_rows(logits_dev_, nl, 1, 4, topk_v_d_, topk_i_d_, topk_work_, stream_);
    CUDA_CHECK(cudaMemcpyAsync(top2_, topk_i_d_, 16, cudaMemcpyDeviceToHost, stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (m2 && sub) for (int& t : top2_) t = dv_host_[t];
  if (sub && tk) {
    int* ti = topk_i_h_ + (size_t)qk::kMaxB * kTopKMax;
    for (int i = 0; i < topk_K_; i++) ti[i] = dv_host_[ti[i]];
  }
  if (sub) {
    if (logits_host && !tk) {
      std::fill(logits_host, logits_host + c.n_vocab, -INFINITY);
      for (int i = 0; i < nl; i++) logits_host[dv_host_[i]] = dv_tmp_[i];
    }
    d = dv_host_[d];
  }
  return d;
}

int Model::mtp_draft(int hrow0, const int* tokens, int n, int pos0, float* conf, float* logits_host, bool head) {
  const Config& c = cfg_;
  QE_ASSERT(has_mtp() && n >= 1 && n <= qk::kMaxB);
  ensure_kv(pos0 + n + 1);
  const float* src = hrow0 < 0 ? hcarry_ : hid_ + (size_t)hrow0 * c.n_embd;
  // mtp_block clobbers t0..t4/x/xn: stage the h rows in the tail of hid_ (unused by decode)
  float* hsafe = hsafe_;
  CUDA_CHECK(cudaMemcpyAsync(hsafe, src, (size_t)n * c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
  mtp_block(hsafe, tokens, n, pos0);
  if (!head) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return -1;
  }
  return mtp_finish(n, conf, logits_host);
}

int Model::mtp_chain(int token, int pos, float* conf, float* logits_host) {
  const Config& c = cfg_;
  static const bool tprof = getenv("QE_MTP_PROF") != nullptr;  // wall time per chained draft step
  static double t_acc = 0, t_blk = 0;
  static long t_n = 0;
  const double t0 = tprof ? qe_now() : 0;
  ensure_kv(pos + 2);
  float* hsafe = hsafe_;
  CUDA_CHECK(cudaMemcpyAsync(hsafe, mtp_h_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
  mtp_block(hsafe, &token, 1, pos);
  if (tprof) { CUDA_CHECK(cudaStreamSynchronize(stream_)); t_blk += qe_now() - t0; }
  const int d = mtp_finish(1, conf, logits_host);
  if (tprof) {
    t_acc += qe_now() - t0;
    if (++t_n % 200 == 0)
      fprintf(stderr, "[mtp] chained step %.3f ms (block %.3f ms, head+sync %.3f ms) at pos %d\n", t_acc / t_n * 1e3,
              t_blk / t_n * 1e3, (t_acc - t_blk) / t_n * 1e3, pos);
  }
  return d;
}

// ---------------------------------------------------------------------------------------------
// Elastic KV: FFN layers that normally stream from host RAM can live in the unused tail of the KV
// pool while the context is short.  Borrowed layers are stacked downward from the pool top; when the
// context grows into them they are evicted (LIFO) and simply fall back to streaming.
int Model::kv_limit() const {
  return cur_seq_ < 0 ? 0 : (int)seqs_.at(cur_seq_).blocks.size() * qk::KV_BLOCK;
}

void Model::refresh_streamed() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (copy_stream_) CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
  streamed_layers_.clear();
  for (int il = 0; il < cfg_.n_layer; il++)
    if (layers_[il].ffn_streamed && !layers_[il].borrowed_dev) streamed_layers_.push_back(il);
  ring_reset();
}

void Model::upload_btab() {
  std::vector<int> t(n_kv_blocks_, 0);
  if (cur_seq_ >= 0) {
    const auto& bl = seqs_[cur_seq_].blocks;
    for (size_t i = 0; i < bl.size(); i++) t[i] = bl[i];
  }
  CUDA_CHECK(cudaMemcpy(btab_dev_, t.data(), t.size() * 4, cudaMemcpyHostToDevice));
}

int Model::max_used_block() const {
  for (int b = n_kv_blocks_ - 1; b >= 0; b--)
    if (blk_owner_[b] >= 0) return b;
  return -1;
}

// Physical KV block for the current sequence: lowest free block below the lent region; otherwise
// take back lent FFN layers (LIFO), and as a last resort drop the least recently used other sequence.
int Model::alloc_block() {
  for (;;) {
    const int floor_blk = (int)(borrow_floor_ / kv_block_bytes_);
    for (int b = 0; b < floor_blk; b++)
      if (blk_owner_[b] < 0) {
        blk_owner_[b] = cur_seq_;
        QE_DBG("[kv] seq %d <- block %d (floor %d)\n", cur_seq_, b, floor_blk);
        return b;
      }
    if (!borrowed_.empty()) {
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      Layer& L = layers_[borrowed_.back()];
      QE_DBG("[kv] evict borrowed layer %d\n", borrowed_.back());
      borrowed_.pop_back();
      L.borrowed_dev = nullptr;
      borrow_floor_ = borrowed_.empty() ? kv_pool_bytes_ : (size_t)(layers_[borrowed_.back()].borrowed_dev - kv_pool_);
      refresh_streamed();
      continue;
    }
    int victim = -1;
    uint64_t best = UINT64_MAX;
    for (auto& [id, si] : seqs_)
      if (id != cur_seq_ && !si.blocks.empty() && si.used < best) { best = si.used; victim = id; }
    if (victim < 0) QE_FATAL("KV pool exhausted");
    QE_DBG("[kv] evicting cached sequence %d (%zu blocks)\n", victim, seqs_[victim].blocks.size());
    seq_drop(victim);
  }
}

void Model::ensure_kv(int npos) {
  if (npos > opt_.ctx + 8) QE_FATAL("context overflow (%d > %d)", npos, opt_.ctx);
  if (cur_seq_ < 0) seq_select(seq_create());
  const int nb = std::min(n_kv_blocks_, (npos + qk::KV_BLOCK - 1) / qk::KV_BLOCK);
  bool changed = false;
  while ((int)seqs_[cur_seq_].blocks.size() < nb) {
    const int b = alloc_block();
    seqs_[cur_seq_].blocks.push_back(b);
    changed = true;
  }
  if (changed) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    upload_btab();
  }
}

int Model::seq_create() {
  const int id = next_seq_++;
  seqs_[id].used = ++clock_;
  return id;
}

void Model::seq_select(int id) {
  QE_ASSERT(seqs_.count(id));
  seqs_[id].used = ++clock_;
  if (id == cur_seq_) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  cur_seq_ = id;
  if (!seqs_[id].host.empty()) seq_restore(id);
  upload_btab();
}

// Inactive conversations keep their KV in host RAM instead of VRAM (where the space is lent to FFN
// weights and speeds up every token); switching back copies it in again (~25 GB/s).
void Model::seq_offload(int id) {
  SeqInfo& si = seqs_[id];
  if (si.blocks.empty()) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int b : si.blocks) {
    si.host.emplace_back(kv_block_bytes_);
    CUDA_CHECK(cudaMemcpy(si.host.back().data(), kv_pool_ + (size_t)b * kv_block_bytes_, kv_block_bytes_, cudaMemcpyDeviceToHost));
    blk_owner_[b] = -1;
    host_kv_bytes_ += kv_block_bytes_;
  }
  QE_DBG("[kv] offloaded sequence %d (%zu blocks) to host RAM\n", id, si.blocks.size());
  si.blocks.clear();
}

void Model::seq_restore(int id) {
  SeqInfo& si = seqs_[id];
  QE_ASSERT(id == cur_seq_);
  std::vector<std::vector<uint8_t>> host = std::move(si.host);
  si.host.clear();
  for (auto& h : host) {
    const int b = alloc_block();
    CUDA_CHECK(cudaMemcpy(kv_pool_ + (size_t)b * kv_block_bytes_, h.data(), kv_block_bytes_, cudaMemcpyHostToDevice));
    si.blocks.push_back(b);
    host_kv_bytes_ -= kv_block_bytes_;
  }
  QE_DBG("[kv] restored sequence %d (%zu blocks) from host RAM\n", id, si.blocks.size());
}

void Model::seq_drop(int id) {
  auto it = seqs_.find(id);
  if (it == seqs_.end()) return;
  for (int b : it->second.blocks) blk_owner_[b] = -1;
  host_kv_bytes_ -= it->second.host.size() * kv_block_bytes_;
  seqs_.erase(it);
  if (cur_seq_ == id) cur_seq_ = -1;
}

size_t Model::ck_state_off(int gi) const { return (size_t)gi * state_bytes(); }
size_t Model::ck_conv_off(int gi) const { return (size_t)n_gdn_ * state_bytes() + (size_t)gi * 3 * cfg_.conv_dim() * 4; }
size_t Model::ck_carry_off() const { return ck_conv_off(n_gdn_); }

void Model::ck_save_live(int slot) {
  const Config& c = cfg_;
  for (auto& L : layers_) {
    if (L.gdn_idx < 0) continue;
    CUDA_CHECK(cudaMemcpyAsync(ck_host_[slot] + ck_state_off(L.gdn_idx), L.ssm_state, state_bytes(), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(ck_host_[slot] + ck_conv_off(L.gdn_idx), L.conv_state, 3 * (size_t)c.conv_dim() * 4, cudaMemcpyDeviceToHost, stream_));
  }
  CUDA_CHECK(cudaMemcpyAsync(ck_host_[slot] + ck_carry_off(), hcarry_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Model::ck_restore(int slot) {
  const Config& c = cfg_;
  for (auto& L : layers_) {
    if (L.gdn_idx < 0) continue;
    CUDA_CHECK(cudaMemcpyAsync(L.ssm_state, ck_host_[slot] + ck_state_off(L.gdn_idx), state_bytes(), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(L.conv_state, ck_host_[slot] + ck_conv_off(L.gdn_idx), 3 * (size_t)c.conv_dim() * 4, cudaMemcpyHostToDevice, stream_));
  }
  CUDA_CHECK(cudaMemcpyAsync(hcarry_, ck_host_[slot] + ck_carry_off(), (size_t)c.n_embd * 4, cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// Cached (not current) sequences hold KV blocks that could otherwise hold lent FFN weights, which
// speed up every decoded token.  Keep only the most recently used few (bounded in blocks) and pack
// all live blocks to the bottom of the pool so the freed space can be lent again.
void Model::trim_cache() {
  std::vector<std::pair<uint64_t, int>> others;
  for (auto& [id, si] : seqs_)
    if (id != cur_seq_) others.emplace_back(si.used, id);
  std::sort(others.rbegin(), others.rend());
  int kept = 0;
  size_t kept_blocks = 0;
  for (auto [u, id] : others) {
    SeqInfo& si = seqs_[id];
    const size_t nb = si.blocks.size();
    if (nb == 0) continue;
    if (kept < opt_.cache_keep && kept_blocks + nb <= (size_t)opt_.cache_blocks) {
      kept++;
      kept_blocks += nb;
    } else {
      seq_offload(id);
    }
  }
  // host budget: drop the least recently used offloaded conversations
  for (auto it = others.rbegin(); it != others.rend() && host_kv_bytes_ > (size_t)opt_.cache_host_mb * MiB; ++it) {
    if (!seqs_.count(it->second) || seqs_[it->second].host.empty()) continue;
    QE_DBG("[kv] dropping cached sequence %d (host budget)\n", it->second);
    seq_drop(it->second);
  }
  // compaction: move the highest used block into the lowest free one until no gap remains
  bool moved = false;
  for (;;) {
    int lo = -1, hi = -1;
    for (int b = 0; b < n_kv_blocks_; b++)
      if (blk_owner_[b] < 0) { lo = b; break; }
    for (int b = n_kv_blocks_ - 1; b >= 0; b--)
      if (blk_owner_[b] >= 0) { hi = b; break; }
    if (lo < 0 || hi < 0 || lo > hi) break;
    const int owner = blk_owner_[hi];
    auto& bl = seqs_[owner].blocks;
    for (auto& x : bl)
      if (x == hi) x = lo;
    CUDA_CHECK(cudaMemcpyAsync(kv_pool_ + (size_t)lo * kv_block_bytes_, kv_pool_ + (size_t)hi * kv_block_bytes_,
                               kv_block_bytes_, cudaMemcpyDeviceToDevice, stream_));
    blk_owner_[lo] = owner;
    blk_owner_[hi] = -1;
    moved = true;
  }
  if (moved) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    upload_btab();
  }
}

void Model::prepare_context(int npos, bool small_prefill) {
  npos = std::min(npos, opt_.ctx);
  // re-plan lending after the KV borrow (decode will lend the workspace again); a prefill of only a
  // few tokens runs with the small workspace, so the lent layers (855 MiB of copies) can stay
  if (!small_prefill) ws_reclaim();
  if (opt_.elastic_kv) trim_cache();
  ensure_kv(npos);
  if (!opt_.elastic_kv) return;
  const size_t need = (size_t)(max_used_block() + 1) * kv_block_bytes_;
  // Borrow order: van der Corput (bit-reversed) order over the streamed layers, so any prefix of
  // borrowed layers -- and the remaining streamed ones -- stay evenly spread over the depth.
  std::vector<std::pair<double, int>> cand;
  {
    std::vector<int> st;
    for (int il = 0; il < cfg_.n_layer; il++)
      if (layers_[il].ffn_streamed && !layers_[il].borrowed_dev) st.push_back(il);
    const int m = (int)st.size();
    std::vector<std::pair<double, int>> order;
    for (int i = 0; i < m; i++) {
      // position i/m mapped through the radical inverse of its rank in a power-of-two grid
      int bits = 0;
      while ((1 << bits) < m) bits++;
      unsigned rev = 0;
      for (int b = 0; b < bits; b++)
        if (i & (1 << b)) rev |= 1u << (bits - 1 - b);
      order.emplace_back(-(double)rev, st[i]);  // sorted descending below -> ascending rev
    }
    std::sort(order.rbegin(), order.rend());
    cand = order;
  }
  int added = 0;
  for (auto [cost, il] : cand) {
    Layer& L = layers_[il];
    const size_t bytes = (L.stream_bytes_full + 255) / 256 * 256;
    if (borrow_floor_ < bytes || borrow_floor_ - bytes < need) continue;
    borrow_floor_ -= bytes;
    L.borrowed_dev = kv_pool_ + borrow_floor_;
    CUDA_CHECK(cudaMemcpyAsync(L.borrowed_dev, L.stream_host, L.stream_bytes_full, cudaMemcpyHostToDevice, stream_));
    borrowed_.push_back(il);
    added++;
  }
  if (added) {
    refresh_streamed();
    QE_DBG("[kv] context reserve %d: borrowed %d more FFN layers into the KV pool (%zu total, KV limit %d)\n", npos, added,
           borrowed_.size(), kv_limit());
  }
}

KvInfo Model::kv_info() const {
  KvInfo k;
  k.blocks_total = n_kv_blocks_;
  for (int b = 0; b < n_kv_blocks_; b++) k.blocks_used += blk_owner_[b] >= 0;
  k.blocks_lendable = (int)(borrow_floor_ / std::max<size_t>(1, kv_block_bytes_));
  k.seqs = (int)seqs_.size();
  for (auto& [id, si] : seqs_) k.seqs_offloaded += si.blocks.empty() && !si.host.empty();
  k.cur_positions = cur_seq_ >= 0 && seqs_.count(cur_seq_) ? (int)seqs_.at(cur_seq_).blocks.size() * qk::KV_BLOCK : 0;
  k.borrowed_kv = (int)borrowed_.size();
  k.borrowed_ws = (int)ws_borrowed_.size();
  int streamed = 0;
  for (int il = 0; il < cfg_.n_layer; il++) streamed += layers_[il].ffn_streamed && !layers_[il].borrowed_dev;
  k.streamed = streamed;
  k.host_mb = host_kv_bytes_ / (double)MiB;
  k.block_mb = kv_block_bytes_ / (double)MiB;
  return k;
}

int Model::borrowed_layers() const { return (int)(borrowed_.size() + ws_borrowed_.size()); }

void Model::select_ws(bool big) {
  const WSet& w = big ? ws_big_ : ws_small_;
  if (x_ == w.x) return;
  x_ = w.x; xn_ = w.xn; t0_ = w.t0; t1_ = w.t1; t2_ = w.t2; t3_ = w.t3; t4_ = w.t4;
  g_ = w.g; beta_ = w.beta; xq_ = w.xq; xd_ = w.xd; hid_ = w.hid;
  if (big) { a16_ = w.a16; c16_ = w.c16; w16_ = w.w16; fa_work_ = w.fa; }
  prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
  ws_is_big_ = big;
}

// While decoding, the prefill workspace arena holds streamed FFN layers (copied from their pinned
// host blocks); they are dropped again (back to streaming) before the next prefill.
void Model::ws_lend() {
  static const bool no_lend = getenv("QE_NO_WSLEND") != nullptr;
  if (!opt_.elastic_kv || ws_lent_ || !copy_stream_ || no_lend) return;
  ws_lent_ = true;
  std::vector<int> st;
  for (int il = 0; il < cfg_.n_layer; il++)
    if (layers_[il].ffn_streamed && !layers_[il].borrowed_dev) st.push_back(il);
  // spread over the remaining streamed layers (every k-th)
  size_t used = 0;
  std::vector<int> pick;
  const size_t cap = ws_big_.bytes;
  for (int stride = 2; stride <= (int)st.size() && pick.empty(); stride++) {
    used = 0;
    std::vector<int> cand;
    for (size_t i = stride / 2; i < st.size(); i += stride) {
      const size_t b = (layers_[st[i]].stream_bytes_full + 255) / 256 * 256;
      if (used + b > cap) continue;
      used += b;
      cand.push_back(st[i]);
    }
    if (!cand.empty()) pick = cand;
  }
  // greedily fill what is left
  for (int il : st) {
    if (std::find(pick.begin(), pick.end(), il) != pick.end()) continue;
    const size_t b = (layers_[il].stream_bytes_full + 255) / 256 * 256;
    if (used + b <= cap) { used += b; pick.push_back(il); }
  }
  if (pick.empty()) return;
  size_t off = 0;
  for (int il : pick) {
    Layer& L = layers_[il];
    L.borrowed_dev = ws_big_.base + off;
    off += (L.stream_bytes_full + 255) / 256 * 256;
    CUDA_CHECK(cudaMemcpyAsync(L.borrowed_dev, L.stream_host, L.stream_bytes_full, cudaMemcpyHostToDevice, stream_));
    ws_borrowed_.push_back(il);
  }
  refresh_streamed();
  QE_DBG("[ws] lent prefill workspace (%.0f MiB) to %zu FFN layers\n", ws_big_.bytes / (1024.0 * 1024), ws_borrowed_.size());
}

void Model::ws_reclaim() {
  if (!ws_lent_) return;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int il : ws_borrowed_) layers_[il].borrowed_dev = nullptr;
  const bool any = !ws_borrowed_.empty();
  ws_borrowed_.clear();
  ws_lent_ = false;
  if (any) refresh_streamed();
}

// recurrent state (DeltaNet + conv history of every GDN layer) <-> one device backup (draft models)
bool Model::state_save() {
  const size_t cb = 3 * (size_t)cfg_.conv_dim() * 4, per = state_bytes() + cb;
  if (!state_bk_dev_) {
    if (cudaMalloc((void**)&state_bk_dev_, per * std::max(1, n_gdn_)) != cudaSuccess) { cudaGetLastError(); return false; }
    allocs_.push_back(state_bk_dev_);
  }
  for (auto& L : layers_) {
    if (!L.ssm_state) continue;
    uint8_t* d = state_bk_dev_ + (size_t)L.gdn_idx * per;
    CUDA_CHECK(cudaMemcpyAsync(d, L.ssm_state, state_bytes(), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d + state_bytes(), L.conv_state, cb, cudaMemcpyDeviceToDevice, stream_));
  }
  return true;
}
void Model::state_restore() {
  const size_t cb = 3 * (size_t)cfg_.conv_dim() * 4, per = state_bytes() + cb;
  QE_ASSERT(state_bk_dev_);
  for (auto& L : layers_) {
    if (!L.ssm_state) continue;
    const uint8_t* d = state_bk_dev_ + (size_t)L.gdn_idx * per;
    CUDA_CHECK(cudaMemcpyAsync(L.ssm_state, d, state_bytes(), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(L.conv_state, d + state_bytes(), cb, cudaMemcpyDeviceToDevice, stream_));
  }
}

void Model::set_carry_row(int row) {
  CUDA_CHECK(cudaMemcpyAsync(hcarry_, hid_ + (size_t)row * cfg_.n_embd, (size_t)cfg_.n_embd * 4,
                             cudaMemcpyDeviceToDevice, stream_));
}

void Model::rollback(int n_keep) {
  const Config& c = cfg_;
  QE_ASSERT(snap_valid_ && n_keep >= 1 && n_keep <= snap_T_);
  if (gdn_tab_) {  // one launch for all layers (state traffic at full bandwidth instead of 96 small launches)
    qk::gdn_rollback_batched(gdn_tab_, n_gdn_tab_, c.conv_dim(), c.nk, c.nv, c.dk, n_keep, c.conv_dim(), n_keep < snap_T_, stream_,
                             state16_);
  } else {
    for (auto& L : layers_) {
      if (!L.sv_qkv) continue;
      // advance the (untouched) pre-pass state by the accepted tokens, in place
      qk::gdn_recurrence(L.sv_qkv, c.conv_dim(), L.sv_g, L.sv_beta, L.ssm_state, L.ssm_state, nullptr, c.nk, c.nv,
                         c.dk, n_keep, stream_, -1, nullptr, state16_);
      if (n_keep < snap_T_) qk::ssm_conv_rollback(L.conv_state, L.conv_hist, c.conv_dim(), n_keep, stream_);
    }
  }
  snap_valid_ = false;  // no sync: the next MTP step is enqueued behind it on the same stream
}

void Model::forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax,
                    bool snapshot, bool mtp_prefill) {
  const Config& c = cfg_;
  static const bool hprof = getenv("QE_HPROF") != nullptr;
  static double h_launch = 0, h_sync = 0, h_tail = 0, h_n = 0;
  const double h0 = hprof ? qe_now() : 0;
  double h1 = 0, h2 = 0, hl = 0;
  static double h_layers = 0;
  QE_ASSERT(n_logits <= n);
  QE_ASSERT(!snapshot || n <= opt_.draft_max + 1);
  snapshot_ = snapshot && spec_ && n > 1;
  snap_valid_ = snapshot_;
  snap_T_ = n;
  ensure_kv(pos0 + n + 4);
  if (n > qk::kMaxB) {
    ws_reclaim();       // prefill needs the big workspace back
    select_ws(true);
  } else {
    select_ws(false);
    ws_lend();          // decode: lend the idle prefill workspace to FFN weights
  }
  // CPU/GPU split of the streamed FFN layers, chosen online: hybrid (CPU computes its row slice) wins
  // when PCIe is the bottleneck (long context), GPU-only (whole layers streamed) when the GPU is.
  // Both are timed with an EMA; the slower one is re-measured for a few passes every 64 passes.
  bool use_cpu = pool_ && n <= kCpuMaxB;
  int dm = -1;
  if (use_cpu) {
    // level k: k/4 of the hybrid-capable streamed layers keep their CPU row slice (evenly spread),
    // the rest stream whole layers; the fastest level is kept, its neighbours re-measured now and then
    static const int force = getenv("QE_CPU_MODE") ? atoi(getenv("QE_CPU_MODE")) : -1;
    if (force == 0) dm = 0;
    else if (force == 1) dm = kDmLevels - 1;
    else if (force >= 2 && force < 2 + kDmLevels) dm = force - 2;
    else {
      int best = -1;
      for (int k = 0; k < kDmLevels && best < 0; k++)
        if (dm_cnt_[k] < 3 && (k == 0 || k == kDmLevels - 1 || dm_cnt_[0] >= 3)) best = k;
      if (best < 0) {
        best = 0;
        for (int k = 1; k < kDmLevels; k++)
          if (dm_cnt_[k] >= 3 && dm_ema_[k] < dm_ema_[best]) best = k;
      }
      if (++dm_pass_ % 96 == 0) dm_explore_ = 3, dm_dir_ = -dm_dir_;
      dm = dm_explore_ > 0 ? std::clamp(best + dm_dir_, 0, kDmLevels - 1) : best;
      if (dm_explore_ > 0) dm_explore_--;
    }
    if (dm != dm_last_) {
      int nh = 0, idx = 0;
      for (int il = 0; il < c.n_layer; il++) nh += layers_[il].ffn_streamed && layers_[il].cpu_rows > 0;
      const int want = (nh * dm + (kDmLevels - 1) / 2) / (kDmLevels - 1);
      for (int il = 0; il < c.n_layer; il++) {
        Layer& L = layers_[il];
        if (!(L.ffn_streamed && L.cpu_rows > 0)) continue;
        // evenly spread: layer idx is hybrid iff floor((idx+1)*want/nh) > floor(idx*want/nh)
        const bool on = nh > 0 && (idx + 1) * want / nh > idx * want / nh;
        if (on != L.hyb_on) { L.hyb_on = on; hyb_dirty_ = true; }
        idx++;
      }
      QE_DBG("[auto] decode CPU level %d/%d: %d of %d streamed layers hybrid (ema ms: %.1f %.1f %.1f %.1f %.1f, n %d %d %d %d %d)\n",
             dm, kDmLevels - 1, want, nh, dm_ema_[0] * 1e3, dm_ema_[1] * 1e3, dm_ema_[2] * 1e3, dm_ema_[3] * 1e3,
             dm_ema_[4] * 1e3, dm_cnt_[0], dm_cnt_[1], dm_cnt_[2], dm_cnt_[3], dm_cnt_[4]);
    }
    use_cpu = dm > 0;
  }
  const bool dm_skip = dm != dm_last_;  // the first pass after a switch refills the ring: not timed
  dm_last_ = dm;
  const double dm_t0 = dm >= 0 ? qe_now() : 0;
  static const bool gprof = getenv("QE_PROFILE") != nullptr;
  gprof_on_ = gprof && n <= kCpuMaxB;
  if (gprof_on_) { gp_n_ = cp_n_ = 0; cp_bytes_ = 0; CUDA_CHECK(cudaEventRecord(gp_start_, stream_)); }
  if (cap_slot_ >= 0 && cap_pos_ == pos0) ck_save_live(cap_slot_);
  if (n > Tmax_ - qk::kMaxB && mtp_prefill) QE_ASSERT(Tmax_ > qk::kMaxB);
  set_stream_mode(!use_cpu);
  std::vector<std::pair<int, uint32_t>> jobs;
  if (use_cpu) {
    for (int il = 0; il < c.n_layer; il++)
      if (layers_[il].ffn_streamed && layers_[il].cpu_rows > 0 && !layers_[il].borrowed_dev && layers_[il].hyb_on) {
        layers_[il].job_seq = ++seq_;
        jobs.emplace_back(il, layers_[il].job_seq);
      }
    pool_->start([this, &jobs, n](int tid) { cpu_ffn_job(jobs, n, tid); });
  }
  for (int b0 = 0, T = 0; b0 < n; b0 += T) {
    if (b0 > 0 && cancel_ && cancel_->load(std::memory_order_relaxed)) break;  // cancelled prefill
    T = std::min(Tmax_, n - b0);
    const int p0 = pos0 + b0;
    // prompt-cache checkpoint inside this chunk: end the chunk there and save the live state after it
    if (cap_slot_ >= 0 && cap_pos_ > p0 && cap_pos_ < p0 + T) T = cap_pos_ - p0;
    if (p0 + T > opt_.ctx) QE_FATAL("context overflow (%d > %d)", p0 + T, opt_.ctx);
    // fp16-accumulate GEMMs for prefill chunks, guarded: on any non-finite GEMM output the chunk
    // is recomputed with fp32 accumulation from the saved recurrent state.
    const bool g16 = gemm16_ && T > qk::kMaxB && has_mtp();
    if (g16) {
      for (auto& L : layers_) {
        if (!L.conv_hist) continue;
        if (L.ssm_alt) CUDA_CHECK(cudaMemcpyAsync(L.ssm_alt, L.ssm_state, state_bytes(), cudaMemcpyDeviceToDevice, stream_));
        else CUDA_CHECK(cudaMemcpyAsync((float*)((uint8_t*)state_bk_host_ + (size_t)L.gdn_idx * state_bytes()), L.ssm_state, state_bytes(), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(L.conv_hist, L.conv_state, 3 * (size_t)c.conv_dim() * 4, cudaMemcpyDeviceToDevice, stream_));
      }
      CUDA_CHECK(cudaMemcpyAsync(hcarry_bk_, hcarry_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemsetAsync(ovf_dev_, 0, 4, stream_));
    }
    snap_t_ = -1;
    for (int attempt = 0; attempt < 2; attempt++) {
    use16_ = g16 && attempt == 0;
    qk::iota_i32(pos_dev_, p0, T, stream_);
    embed(tokens + b0, T, x_);
    auto dsum = [&](const char* nm, int il, const float* p, size_t cnt) { qe_debug_dump(nm, il, p, T, (int)(cnt / T), stream_); };
    for (int il = 0; il < c.n_layer; il++) {
      if (gprof_on_) CUDA_CHECK(cudaEventRecord(sec_ev(il, 0), stream_));
      static const int skip_lo = getenv("QE_SKIP_BLOCKS") ? atoi(getenv("QE_SKIP_BLOCKS")) : -1;  // experiments: "lo"
      static const int skip_hi = getenv("QE_SKIP_BLOCKS_HI") ? atoi(getenv("QE_SKIP_BLOCKS_HI")) : -1;  // .. "hi" excl.
      if (il >= skip_lo && il < skip_hi) continue;
      const int kind = c.mhc(il);
      if (kind) {
        // two residual streams (sensenova_loop): each sublayer reads a routed mix of them and its
        // output is merged back with the router's gates and 2x2 mixing matrix
        Layer& L = layers_[il];
        const size_t nb = (size_t)T * c.n_embd * 4;
        if (kind == 1 || kind == 3) {  // enter: both streams start as the single residual
          if (pend_) { qk::add_inplace(x_, pend_, (int64_t)T * c.n_embd, stream_); pend_ = nullptr; }
          CUDA_CHECK(cudaMemcpyAsync(mhc_a_, x_, nb, cudaMemcpyDeviceToDevice, stream_));
          CUDA_CHECK(cudaMemcpyAsync(mhc_b_, x_, nb, cudaMemcpyDeviceToDevice, stream_));
        }
        for (int sub = 0; sub < 2; sub++) {
          qk::mhc_route(mhc_a_, mhc_b_, sub ? L.mhc_ffn_w : L.mhc_attn_w, sub ? L.mhc_ffn_b : L.mhc_attn_b, c.n_embd, T,
                        mhc_branch_, mhc_pk_, stream_);
          mhc_on_ = true;
          mhc_out_ = nullptr;
          if (sub) { ProfScope ps(2, stream_); layer_ffn(il, T); }
          else if (L.attn) { ProfScope ps(0, stream_); layer_attn(il, T, p0); }
          else { ProfScope ps(1, stream_); layer_gdn(il, T); }
          mhc_on_ = false;
          qk::mhc_merge(mhc_a_, mhc_b_, mhc_out_, mhc_pk_, c.n_embd, T, stream_);
        }
        if (kind == 2 || kind == 3) qk::mhc_leave(x_, mhc_a_, mhc_b_, (int64_t)T * c.n_embd, stream_);
        else CUDA_CHECK(cudaMemcpyAsync(x_, mhc_a_, nb, cudaMemcpyDeviceToDevice, stream_));  // stream 0 = the residual
        if (gprof_on_) { CUDA_CHECK(cudaEventRecord(sec_ev(il, 1), stream_)); CUDA_CHECK(cudaEventRecord(sec_ev(il, 2), stream_)); }
        dsum("l_out", il, x_, (size_t)T * c.n_embd);
        continue;
      }
      if (layers_[il].attn) { ProfScope ps(0, stream_); layer_attn(il, T, p0); }
      else { ProfScope ps(1, stream_); layer_gdn(il, T); }
      dsum("attn_residual", il, x_, (size_t)T * c.n_embd);
      if (gprof_on_) CUDA_CHECK(cudaEventRecord(sec_ev(il, 1), stream_));
      { ProfScope ps(2, stream_); layer_ffn(il, T); }
      if (gprof_on_) CUDA_CHECK(cudaEventRecord(sec_ev(il, 2), stream_));
      dsum("l_out", il, x_, (size_t)T * c.n_embd);
    }
    if (hprof) hl = qe_now();
    // final norm for all rows (MTP needs every row's hidden state)
    norm_in(output_norm_, hid_, T, false);
    // logits for rows that fall in this chunk
    const int first_logit_row = n - n_logits;  // global index
    const int r_begin = std::max(first_logit_row, b0), r_end = b0 + T;
    if ((logits || argmax) && r_begin < r_end) {
      for (int r = r_begin; r < r_end; r += qk::kMaxB) {
        const int nb = std::min(qk::kMaxB, r_end - r);
        prep256_ = nullptr; prepared_ = nullptr; prep16_ = nullptr; gqp_ = nullptr;
        linear(nullptr, output_, hid_ + (size_t)(r - b0) * c.n_embd, nb, logits_dev_, c.n_vocab, false, stream_);
        if (r + nb == n) {
          qe_debug_dump("result_norm", -1, hid_ + (size_t)(r + nb - 1 - b0) * c.n_embd, 1, c.n_embd, stream_);
          qe_debug_dump("result_output", -1, logits_dev_ + (size_t)(nb - 1) * c.n_vocab, 1, c.n_vocab, stream_);
        }
        if (logits && topk_K_ > 0)
          topk_to_host(logits_dev_, c.n_vocab, nb, r - first_logit_row);
        else if (logits)
          CUDA_CHECK(cudaMemcpyAsync(logits + (size_t)(r - first_logit_row) * c.n_vocab, logits_dev_,
                                     (size_t)nb * c.n_vocab * 4, cudaMemcpyDeviceToHost, stream_));
        if (argmax) {
          qk::argmax_rows(logits_dev_, c.n_vocab, nb, argmax_dev_, stream_);
          CUDA_CHECK(cudaMemcpyAsync(argmax + (r - first_logit_row), argmax_dev_, nb * 4, cudaMemcpyDeviceToHost, stream_));
        }
      }
    }
    if (mtp_prefill && has_mtp()) {
      // pairs (h_{p-1}, x_p) for p in [p0, p0+T): rows [carry, hid_0 .. hid_{T-2}]
      float* hm = t3_;  // [T][n_embd] (t3_ is free between layers)
      // position 0 has no h_{-1}: run it with a zero hidden row so its MTP KV slot is written (the MTP
      // attention reads every position; an unwritten slot holds stale bytes, e.g. lent FFN weights)
      const int skip = 0;
      if (p0 == 0) CUDA_CHECK(cudaMemsetAsync(hm, 0, (size_t)c.n_embd * 4, stream_));
      else CUDA_CHECK(cudaMemcpyAsync(hm, hcarry_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
      if (T > 1) CUDA_CHECK(cudaMemcpyAsync(hm + (size_t)c.n_embd, hid_, (size_t)(T - 1) * c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(hcarry_, hid_ + (size_t)(T - 1) * c.n_embd, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
      if (T - skip > 0) {
        // hm rows must survive mtp_block's use of t3_: copy them into hid_ region after saving carry
        CUDA_CHECK(cudaMemcpyAsync(hid_, hm + (size_t)skip * c.n_embd, (size_t)(T - skip) * c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
        mtp_block(hid_, tokens + b0 + skip, T - skip, p0 + skip);
      }
    }
    if (use16_) {
      int ovf = 0;
      CUDA_CHECK(cudaMemcpyAsync(&ovf, ovf_dev_, 4, cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      static const bool test_ovf = getenv("QE_TEST_OVF") != nullptr;  // exercise the fallback path
      if (ovf || test_ovf) {
        if (!test_ovf) QE_LOG("[warn] fp16 GEMM overflow in chunk at %d: recomputing with fp32 accumulation\n", p0);
        for (auto& L : layers_) {
          if (!L.conv_hist) continue;
          if (L.ssm_alt) CUDA_CHECK(cudaMemcpyAsync(L.ssm_state, L.ssm_alt, state_bytes(), cudaMemcpyDeviceToDevice, stream_));
          else CUDA_CHECK(cudaMemcpyAsync(L.ssm_state, (float*)((uint8_t*)state_bk_host_ + (size_t)L.gdn_idx * state_bytes()), state_bytes(), cudaMemcpyHostToDevice, stream_));
          CUDA_CHECK(cudaMemcpyAsync(L.conv_state, L.conv_hist, 3 * (size_t)c.conv_dim() * 4, cudaMemcpyDeviceToDevice, stream_));
        }
        CUDA_CHECK(cudaMemcpyAsync(hcarry_, hcarry_bk_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
        continue;
      }
    }
    if (hprof) h1 = qe_now();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (hprof) h2 = qe_now();
    CUDA_CHECK(cudaGetLastError());
    break;
    }  // attempt
    if (cap_slot_ >= 0 && cap_pos_ == p0 + T && p0 + T < pos0 + n) ck_save_live(cap_slot_);
  }
  if (use_cpu) pool_->wait();
  { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) QE_FATAL("forward: %s", cudaGetErrorString(e)); }
  if (dm >= 0 && !dm_skip) {
    const double dt = qe_now() - dm_t0;
    dm_ema_[dm] = dm_cnt_[dm] ? 0.8 * dm_ema_[dm] + 0.2 * dt : dt;
    dm_cnt_[dm]++;
  }
  if (hprof && n <= qk::kMaxB && h1 > 0) {
    const double h3 = qe_now();
    h_launch += h1 - h0; h_sync += h2 - h1; h_tail += h3 - h2; h_layers += hl - h0; h_n += 1;
    if ((int)h_n % 100 == 0)
      fprintf(stderr, "[hprof] forward(T<=8): layers enqueued after %.2f ms, head+rest enqueue until %.2f ms, wait GPU %.2f ms, tail %.2f ms\n",
              h_layers / h_n * 1e3, h_launch / h_n * 1e3, h_sync / h_n * 1e3, h_tail / h_n * 1e3);
  }
  snap_t_ = -1;
  cap_slot_ = -1;
  if (gprof_on_) {
    CUDA_CHECK(cudaEventRecord(gp_end_, stream_));
    CUDA_CHECK(cudaEventSynchronize(gp_end_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_ ? copy_stream_ : stream_));
    float ms = 0, tot = 0, wait = 0, copy = 0;
    cudaEventElapsedTime(&tot, gp_start_, gp_end_);
    for (int i = 0; i + 1 < gp_n_; i += 2) { cudaEventElapsedTime(&ms, gp_evs_[i], gp_evs_[i + 1]); wait += ms; }
    for (int i = 0; i + 1 < cp_n_; i += 2) { cudaEventElapsedTime(&ms, cp_evs_[i], cp_evs_[i + 1]); copy += ms; }
    gp_acc_[0] += tot; gp_acc_[1] += wait; gp_acc_[2] += copy; gp_acc_[3] += cp_bytes_ / 1e6; gp_cnt_++;
    {  // per-section GPU time: attention mixer, GDN mixer, FFN (resident/borrowed), FFN (hybrid/streamed)
      for (int il = 0; il < c.n_layer; il++) {
        float a = 0, f = 0;
        cudaEventElapsedTime(&a, sec_evs_[il * 3], sec_evs_[il * 3 + 1]);
        cudaEventElapsedTime(&f, sec_evs_[il * 3 + 1], sec_evs_[il * 3 + 2]);
        sec_acc_[layers_[il].attn ? 0 : 1] += a;
        const bool str = layers_[il].ffn_streamed && !layers_[il].borrowed_dev;
        sec_acc_[str ? 3 : 2] += f;
      }
      float tail = 0;
      cudaEventElapsedTime(&tail, sec_evs_[(c.n_layer - 1) * 3 + 2], gp_end_);
      sec_acc_[4] += tail;
    }
    if (gp_cnt_ % 50 == 0)
      fprintf(stderr, "[prof] gpu pass %.1f ms: waiting for CPU %.1f ms, copy engine busy %.1f ms (%.0f MB, %.1f GB/s)\n",
              gp_acc_[0] / gp_cnt_, gp_acc_[1] / gp_cnt_, gp_acc_[2] / gp_cnt_, gp_acc_[3] / gp_cnt_,
              gp_acc_[3] / gp_acc_[2]);
    if (gp_cnt_ % 50 == 0)
      fprintf(stderr, "[prof] gpu sections: attn %.1f ms, gdn %.1f ms, ffn(gpu-only) %.1f ms, ffn(streamed, incl. waits) %.1f ms, head+tail %.1f ms\n",
              sec_acc_[0] / gp_cnt_, sec_acc_[1] / gp_cnt_, sec_acc_[2] / gp_cnt_, sec_acc_[3] / gp_cnt_, sec_acc_[4] / gp_cnt_);
  }
}
