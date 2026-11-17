#include "vk_model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "common.h"
#include "gguf.h"
#include "quant.h"

namespace {

// push-constant blocks (must match the shaders; 64-bit addresses are uvec2 in GLSL)
struct NormPC { uint64_t x, res, w, y; uint32_t n; float eps; };
struct HeadNormPC { uint64_t x, w; uint32_t hd, nh, hs, ts, mode; float eps; };
struct RopePC { uint64_t x, cs; uint32_t nh, hs, ts, T, half; };
struct KvPC { uint64_t k, v, kc, vc; uint32_t nkv, hd, T, pos0, kts, q8; };
struct AttnPC { uint64_t q, kc, vc, o, part; uint32_t T, pos0, nh, nkv, qts, nsplit, chunk; float scale; };
struct CombPC { uint64_t part, o; uint32_t rows, nsplit; };
struct ElemPC { uint64_t a, b, c, d, e, f; uint32_t n, p0, p1; };
struct ConvPC { uint64_t x, st, w, y; uint32_t C, T, xts; };
struct RecPC { uint64_t qkv, g, beta, st, o; uint32_t nk, nv, T, C; };
struct GNormPC { uint64_t o, z, w; uint32_t nv, dv, zts; float eps; };
struct ArgPC { uint64_t x, out; uint32_t n; };

constexpr int kMaxSplit = 64;

const char* type_tag(uint32_t t) {
  switch (t) {
    case QT_F32: return "f32";
    case QT_F16: return "f16";
    case QT_BF16: return "bf16";
    case QT_Q8_0: return "q8_0";
    case QT_Q2_K: return "q2_k";
    case QT_Q3_K: return "q3_k";
    case QT_Q4_K: return "q4_k";
    case QT_Q5_K: return "q5_k";
    case QT_Q6_K: return "q6_k";
    case QT_IQ2_XXS: return "iq2_xxs";
    case QT_IQ2_XS: return "iq2_xs";
    case QT_IQ2_S: return "iq2_s";
    case QT_IQ3_XXS: return "iq3_xxs";
    case QT_IQ3_S: return "iq3_s";
    case QT_IQ1_S: return "iq1_s";
    case QT_IQ1_M: return "iq1_m";
    case QT_IQ4_XS: return "iq4_xs";
    default: return nullptr;
  }
}

// 1-D element grids beyond 65535 workgroups: (x, y) with x = 65535 (shaders flatten y * 65535 * 256 + x)
void grid1d(uint64_t threads, uint32_t& gx, uint32_t& gy) {
  const uint64_t g = (threads + 255) / 256;
  gx = (uint32_t)std::min<uint64_t>(g, 65535);
  gy = (uint32_t)((g + 65534) / 65535);
}
void grid_wg(uint64_t groups, uint32_t& gx, uint32_t& gy) {  // one workgroup per item, flattened
  gx = (uint32_t)std::min<uint64_t>(groups, 65535);
  gy = (uint32_t)((groups + gx - 1) / gx);
}

std::string bn(int il, const char* s) { return "blk." + std::to_string(il) + "." + s; }
uint16_t f2h(float f) { return _cvtss_sh(f, 0); }  // F16C (x86-64-v3 baseline)

}  // namespace

void VkModel::QE_ASSERT_SEQ(int id) const {
  if (id != cur_seq_) QE_FATAL("vulkan backend: sequence %d is not the live one", id);
}

VkModel::VkModel(const EngineOptions& opt) : opt_(opt) {
  g_verbose = opt.verbose;
  const double t_start = qe_now();
  gguf_ = std::make_unique<GGUFFile>(opt.model_path);
  cfg_ = load_qwen35_config(*gguf_);
  const Config& c = cfg_;
  if (c.n_head / c.n_kv > 32) QE_FATAL("vulkan backend: GQA group of %d heads > 32", c.n_head / c.n_kv);
  vk_ = std::make_unique<VkCtx>();
  const size_t MiB = 1 << 20;
  arch_ = arch_profile(*vk_);
  // workgroups to keep in flight (split-K GEMM, split attention): ~6 per CU / SM when the driver
  // reports the count (RDNA3 7900 XTX: 96 CUs), a small iGPU (Arc 130V: 8 Xe cores) needs few
  wg_target_ = vk_->compute_units ? std::max(128u, 6 * vk_->compute_units) : vk_->integrated ? 128u : 512u;
  QE_LOG("[hw] Vulkan device %s (%s), profile %s, subgroup %u%s, %s %.1f GiB%s, %s%s\n", vk_->device_name.c_str(),
         vk_->driver_name.c_str(), arch_.c_str(), vk_->subgroup_size,
         vk_->sg_control ? (" (" + std::to_string(vk_->sg_min) + "-" + std::to_string(vk_->sg_max) + ")").c_str() : "",
         vk_->integrated ? "shared system memory" : "VRAM", vk_->vram_total() / 1073741824.0,
         vk_->compute_units ? (", " + std::to_string(vk_->compute_units) + " CUs").c_str() : "",
         vk_->rebar ? "resizable BAR / mapped VRAM, " : "",
         vk_->coopmat ? "cooperative matrices" : "no cooperative matrices");
  ub_ = std::clamp(opt.ubatch, 64, 512);
  // weights this backend will place on the GPU (trunk layers, output head; not token_embd / MTP)
  size_t wbytes = 0, ffn_bytes = 0;
  for (const auto& t : gguf_->tensors()) {
    int il = -1;
    if (sscanf(t.name.c_str(), "blk.%d.", &il) == 1 && il >= c.n_layer) continue;
    if (t.name == "token_embd.weight" && gguf_->tensor("output.weight")) continue;
    const bool ffn = t.name.find(".ffn_") != std::string::npos;
    if (ffn && std::find(opt.skip_ffn.begin(), opt.skip_ffn.end(), il) != opt.skip_ffn.end()) continue;
    wbytes += t.nbytes;
    if (ffn) ffn_bytes += t.nbytes;
  }
  // memory kept free: on an iGPU the "VRAM" is the system RAM the OS and the server also need
  // (Arc 130V with 32 GB: >= 4 GiB); on a dGPU only a little headroom
  reserve_ = (size_t)opt.vram_reserve_mb * MiB +
             (vk_->integrated ? std::max<size_t>(4096 * MiB, vk_->vram_total() / 8) : 256 * MiB);
  const int n_attn_l = [&] { int n = 0; for (int il = 0; il < c.n_layer; il++) n += c.is_attn(il); return n; }();
  // KV format: f16 / q8 as asked; q4 / q3 (CUDA formats) -> f16 when VRAM easily holds weights + f16
  // KV (e.g. 24 GB RX 7900 XTX: fewer dequant ops in attention, exact), else q8 (half the bytes; an
  // iGPU is bandwidth-bound, so q8 there)
  int kvt = opt.kv_type;
  if (kvt >= 2) {
    const size_t f16_kv = (size_t)opt.ctx * n_attn_l * 2 * c.n_kv * c.hd * 2;
    kvt = (!vk_->integrated && wbytes + f16_kv + reserve_ + 1024 * MiB < vk_->vram_free()) ? 0 : 1;
    QE_LOG("[plan] KV cache q4/q3 are CUDA formats: using %s\n", kvt == 0 ? "f16 (fits in VRAM)" : "q8");
  }
  kv_q8_ = kvt != 0;
  kv_row_bytes_ = kv_q8_ ? (size_t)c.hd + c.hd / 8 : (size_t)c.hd * 2;
  drop_pages_ = vk_->integrated || getenv("QE_VK_DROP_PAGES");
  if (drop_pages_) model_fd_ = open(opt.model_path.c_str(), O_RDONLY);

  // ---- activations, state, small tensors (always device-local)
  const int E = c.n_embd, C = c.conv_dim(), qw = 2 * c.n_head * c.hd, kvw = c.n_kv * c.hd;
  auto dev = [&](size_t floats) { return vk_->alloc(floats * 4, Mem::Device, true); };
  const size_t U = (size_t)ub_;
  x_ = dev(U * E);
  xn_ = dev(U * E);
  t0_ = dev(U * std::max({qw, C, c.n_ff}));
  t1_ = dev(U * std::max(c.d_inner, c.n_ff));
  t2_ = dev(U * std::max(c.n_head * c.hd, c.d_inner));
  t3_ = dev(U * std::max(2 * kvw, C));
  t4_ = dev(U * std::max(E, 2 * c.nv));
  g_ = dev(U * c.nv);
  beta_ = dev(U * c.nv);
  logits_ = dev((size_t)kLogitRows * c.n_vocab);
  part_ = dev((size_t)kMaxSplit * kMaxTok * c.n_head * (c.hd + 2));
  argmax_ = dev(kLogitRows);
  ksplit_ = dev(8u << 20);  // 32 MiB split-K scratch
  {  // f16 copy of the largest layer matrix (two-stage prefill GEMM)
    size_t mx = 0;
    for (int il = 0; il < c.n_layer; il++) {
      const bool at = c.is_attn(il);
      const char* names[] = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
                             "attn_gate.weight", "ssm_out.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
      for (const char* nm : names) {
        const GGUFTensor* t = gguf_->tensor(bn(il, nm));
        if (t && (at || strncmp(nm, "attn_q.", 7))) mx = std::max(mx, (size_t)t->nelems() * 2);
      }
    }
    w16_ = vk_->alloc(mx, Mem::Device, true);
    if (vk_->coopmat) {
      const int kmax = std::max({c.n_embd, c.n_ff, c.d_inner, c.n_head * c.hd});
      x16_ = vk_->alloc((size_t)ub_ * kmax * 2, Mem::Device, true);
    }
  }
  emb_h_ = vk_->alloc(U * E * 4, Mem::Host);
  rope_h_ = vk_->alloc(U * (c.rope_dims / 2) * 2 * 4, Mem::Host);
  logits_h_ = vk_->alloc((size_t)kLogitRows * c.n_vocab * 4, Mem::Readback);
  argmax_h_ = vk_->alloc(kLogitRows * 4, Mem::Readback);
  staging_ = vk_->alloc(64 * MiB, Mem::Host);
  int n_gdn = 0;
  for (int il = 0; il < c.n_layer; il++) n_gdn += !c.is_attn(il);
  ssm_bytes_ = (size_t)n_gdn * c.nv * c.dk * c.dv * 4;
  conv_bytes_ = (size_t)n_gdn * 3 * C * 4;
  ssm_ = vk_->alloc(ssm_bytes_, Mem::Device, true);
  conv_ = vk_->alloc(conv_bytes_, Mem::Device, true);

  {  // IQ codebooks (word layout as in vk/common.glsl)
    std::vector<uint32_t> tab(8452, 0);
    memcpy(&tab[0], qtab::iq2xxs_grid, sizeof(qtab::iq2xxs_grid));
    memcpy(&tab[512], qtab::iq2xs_grid, sizeof(qtab::iq2xs_grid));
    memcpy(&tab[1536], qtab::iq2s_grid, sizeof(qtab::iq2s_grid));
    memcpy(&tab[3584], qtab::iq3xxs_grid, sizeof(qtab::iq3xxs_grid));
    memcpy(&tab[3840], qtab::iq3s_grid, sizeof(qtab::iq3s_grid));
    memcpy(&tab[4352], qtab::iq1s_grid, sizeof(qtab::iq1s_grid));
    memcpy(&tab[8448], qtab::kvalues_iq4nl, sizeof(qtab::kvalues_iq4nl));
    tab_ = vk_->alloc(tab.size() * 4, Mem::Device, true);
    upload(tab_, tab.data(), tab.size() * 4);
  }

  // ---- weights: small tensors, then everything but the FFN, KV cache, then FFN layers (VRAM first,
  // system RAM for what does not fit)
  layers_.resize(c.n_layer);
  output_norm_ = load_f32("output_norm.weight");
  output_ = load_mat(gguf_->tensor("output.weight") ? "output.weight" : "token_embd.weight", true);
  {
    const GGUFTensor& t = gguf_->need("token_embd.weight");
    tok_embd_.type = t.type;
    tok_embd_.K = (int)t.ne[0];
    tok_embd_.N = (int)t.ne[1];
    tok_embd_.rb = qtype_row_bytes(t.type, tok_embd_.K);
  }
  size_t gdn_i = 0;
  for (int il = 0; il < c.n_layer; il++) {
    VLayer& L = layers_[il];
    L.attn = c.is_attn(il);
    L.attn_norm = load_f32(bn(il, "attn_norm.weight"));
    L.post_norm = load_f32(bn(il, "post_attention_norm.weight"));
    if (L.attn) {
      L.wq = load_mat(bn(il, "attn_q.weight"), true);
      L.wk = load_mat(bn(il, "attn_k.weight"), true);
      L.wv = load_mat(bn(il, "attn_v.weight"), true);
      L.wo = load_mat(bn(il, "attn_output.weight"), true);
      L.q_norm = load_f32(bn(il, "attn_q_norm.weight"));
      L.k_norm = load_f32(bn(il, "attn_k_norm.weight"));
    } else {
      L.wqkv = load_mat(bn(il, "attn_qkv.weight"), true);
      L.wgate = load_mat(bn(il, "attn_gate.weight"), true);
      L.walpha = load_mat(bn(il, "ssm_alpha.weight"), true);
      L.wbeta = load_mat(bn(il, "ssm_beta.weight"), true);
      L.wssm_out = load_mat(bn(il, "ssm_out.weight"), true);
      L.conv_w = load_f32(bn(il, "ssm_conv1d.weight"));
      L.dt_bias = load_f32(bn(il, "ssm_dt.bias"));
      L.ssm_a = load_f32(bn(il, "ssm_a"));
      L.ssm_norm = load_f32(bn(il, "ssm_norm.weight"));
      L.ssm_off = gdn_i * c.nv * c.dk * c.dv * 4;
      L.conv_off = gdn_i * 3 * C * 4;
      gdn_i++;
    }
  }
  // KV cache: the full context if it fits next to ~1 GiB of other data, else shrink the context
  {
    const int n_attn = (int)std::count_if(layers_.begin(), layers_.end(), [](const VLayer& L) { return L.attn; });
    const size_t per_pos = (size_t)n_attn * 2 * c.n_kv * kv_row_bytes_;
    // iGPU: the FFN weights still to come have no system-RAM fallback (it is the same memory)
    const size_t keep = reserve_ + (vk_->integrated ? ffn_bytes : 0);
    const size_t avail = vk_->vram_free() > keep ? vk_->vram_free() - keep : 0;
    ctx_len_ = opt.ctx;
    if ((size_t)ctx_len_ * per_pos > avail) {
      const int fit = (int)std::max<size_t>(avail / per_pos / 1024 * 1024, 4096);
      QE_LOG("[plan] KV cache for %d positions (%.0f MiB) does not fit in GPU memory: context %d\n", ctx_len_,
             ctx_len_ * per_pos / (double)MiB, std::min(fit, ctx_len_));
      ctx_len_ = std::min(fit, ctx_len_);
    }
    for (auto& L : layers_)
      if (L.attn) {
        L.kc = vk_->alloc((size_t)ctx_len_ * c.n_kv * kv_row_bytes_, Mem::Device);
        L.vc = vk_->alloc((size_t)ctx_len_ * c.n_kv * kv_row_bytes_, Mem::Device);
      }
    QE_LOG("[plan] KV cache %s, %d positions: %.0f MiB\n", kv_q8_ ? "q8" : "f16", ctx_len_, ctx_len_ * per_pos / (double)MiB);
  }
  size_t host_ffn = 0, dev_ffn = 0;
  int res_ffn = 0;
  for (int il = 0; il < c.n_layer; il++) {
    VLayer& L = layers_[il];
    const bool skip = std::find(opt.skip_ffn.begin(), opt.skip_ffn.end(), il) != opt.skip_ffn.end();
    if (skip) continue;
    L.ffn_gate = load_mat(bn(il, "ffn_gate.weight"), true);
    L.ffn_up = load_mat(bn(il, "ffn_up.weight"), true);
    L.ffn_down = load_mat(bn(il, "ffn_down.weight"), true);
    bool on_dev = true;
    for (VMat* m : {&L.ffn_gate, &L.ffn_up, &L.ffn_down}) {
      (m->buf.device_local ? dev_ffn : host_ffn) += m->buf.size;
      on_dev &= m->buf.device_local;
    }
    res_ffn += on_dev;
  }
  plan_.res_mb = vk_->vram_used() / (double)MiB;
  plan_.stream_mb = host_ffn / (double)MiB;
  plan_.cpu_mb = 0;
  plan_.res_ffn = res_ffn;
  QE_LOG("[plan] VRAM used %.0f MiB (%d/%d FFN layers resident)%s\n", vk_->vram_used() / (double)MiB, res_ffn,
         c.n_layer, host_ffn ? "" : ", all weights in VRAM");
  if (host_ffn)
    QE_LOG("[plan] %.0f MiB of FFN weights read from system RAM over PCIe each token (decode will be slower)\n",
           host_ffn / (double)MiB);
  if (model_fd_ >= 0) { close(model_fd_); model_fd_ = -1; }
  vk_->warm_pipelines();
  autotune();
  if (const char* g = getenv("QE_VK_GEMM")) tune_.gemm = g;  // fused | gemm16 | cm_gemm_MxNxK
  reset();
  QE_LOG("[plan] loaded in %.1f s\n", qe_now() - t_start);
}

VkModel::~VkModel() {
  vk_.reset();  // device teardown frees every allocation
}

// iGPU: once a tensor is in GPU memory (= system RAM), drop the model file's page-cache copy of it so
// the weights are not held twice (32 GB Arc 130V laptops); token_embd stays mapped for embedding
void VkModel::release_pages(const void* p, size_t n) {
  if (model_fd_ < 0 || !n) return;
  const uintptr_t pg = 4096, a = ((uintptr_t)p + pg - 1) & ~(pg - 1), e = ((uintptr_t)p + n) & ~(pg - 1);
  if (e > a) madvise((void*)a, e - a, MADV_DONTNEED);
  posix_fadvise(model_fd_, (off_t)((const uint8_t*)p - gguf_->base()), (off_t)n, POSIX_FADV_DONTNEED);
}

// architecture profile (logged; the per-device numbers come from the autotuner)
std::string VkModel::arch_profile(const VkCtx& v) {
  const std::string& n = v.device_name;
  switch (v.vendor_id) {
    case 0x1002:  // AMD: Navi 3x (RDNA3) device ids 0x7440-0x74ff, also matched by name
      if ((v.device_id >= 0x7440 && v.device_id <= 0x74ff) || n.find("RX 7") != std::string::npos) return "amd-rdna3";
      return "amd";
    case 0x8086:  // Intel: Lunar Lake Xe2 iGPUs (Arc 130V / 140V), Battlemage (B5xx) dGPUs, Alchemist
      if (v.device_id == 0x6420 || v.device_id == 0x64a0 || v.device_id == 0x64b0) return "intel-xe2-lunarlake";
      if ((v.device_id & 0xfff0) == 0xe200) return "intel-xe2-battlemage";
      return v.integrated ? "intel-igpu" : "intel-arc";
    case 0x10de: return "nvidia";
    default: return v.device_name.find("llvmpipe") != std::string::npos ? "cpu-lavapipe" : "generic";
  }
}

void VkModel::upload(const VBuf& dst, const void* src, size_t n) {
  if (dst.map) { memcpy(dst.map, src, n); return; }
  const uint8_t* s = (const uint8_t*)src;
  for (size_t off = 0; off < n; off += staging_.size) {
    const size_t k = std::min(staging_.size, n - off);
    memcpy(staging_.map, s + off, k);
    vk_->begin();
    vk_->copy(staging_, 0, dst, off, k);
    vk_->submit();
  }
}

VkModel::VMat VkModel::load_mat(const std::string& name, bool prefer_device) {
  const GGUFTensor& t = gguf_->need(name);
  VMat m;
  m.type = t.type;
  m.K = (int)t.ne[0];
  m.N = (int)t.rows();
  m.rb = qtype_row_bytes(t.type, m.K);
  if (!type_tag(m.type)) QE_FATAL("vulkan backend: %s has unsupported type %s", name.c_str(), qtype_info(t.type).name);
  if (m.K % 256) QE_FATAL("vulkan backend: %s row length %d is not a multiple of 256", name.c_str(), m.K);
  if (m.rb % 4) QE_FATAL("vulkan backend: %s rows are not 4-byte aligned", name.c_str());
  const size_t bytes = m.rb * m.N;
  const bool fits = vk_->vram_free() > bytes + reserve_;
  if (!fits && vk_->integrated)
    QE_FATAL("not enough memory for %s (%.0f MiB free after the %.0f MiB reserve): use a smaller model or context",
             name.c_str(), (vk_->vram_free() > reserve_ ? vk_->vram_free() - reserve_ : 0) / 1048576.0, reserve_ / 1048576.0);
  // mapped device memory (ReBAR dGPU, any iGPU): the upload is a memcpy, no staging copy
  m.buf = vk_->alloc(bytes, prefer_device && fits ? Mem::DeviceMapped : Mem::Host);
  upload(m.buf, t.data, bytes);
  release_pages(t.data, bytes);
  return m;
}

VBuf VkModel::load_f32(const std::string& name) {
  const GGUFTensor& t = gguf_->need(name);
  std::vector<float> v(t.nelems());
  dequant_row_ref(t.type, t.data, t.nelems(), v.data());
  VBuf b = vk_->alloc(v.size() * 4, Mem::DeviceMapped, true);
  upload(b, v.data(), v.size() * 4);
  return b;
}

// one GEMM through a cooperative-matrix variant vs an fp64 host reference (wrong layouts / driver bugs
// show up as large errors; the variant is then not used)
bool VkModel::cm_selftest(const std::string& pipe, uint32_t M, uint32_t N) {
  const int K = 512, NR = 128, T = 70;
  if (NR % (8 * N)) return false;
  std::vector<uint16_t> w((size_t)NR * K);
  std::vector<float> x((size_t)T * K), wf((size_t)NR * K);
  uint32_t s = 12345;
  auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 9) & 0xffff) / 32768.f - 1.f; };
  for (size_t i = 0; i < w.size(); i++) { w[i] = f2h(rnd()); wf[i] = qe_h2f(w[i]); }
  for (auto& v : x) v = rnd();
  VBuf bw = vk_->alloc(w.size() * 2, Mem::Host), bx = vk_->alloc(x.size() * 4, Mem::Host);
  VBuf by = vk_->alloc((size_t)T * NR * 4, Mem::Host);
  memcpy(bw.map, w.data(), w.size() * 2);
  memcpy(bx.map, x.data(), x.size() * 4);
  vk_->begin();
  const uint32_t tp = (T + 63) / 64 * 64;
  ElemPC ep{bx.addr, x16_.addr, 0, 0, 0, 0, (uint32_t)(tp * K / 2), (uint32_t)(T * K / 2), 0};
  uint32_t ex, ey;
  grid1d(ep.n, ex, ey);
  vk_->dispatch("to_f16", &ep, sizeof ep, ex, ey);
  unsigned m_ = M, n_ = N, k_ = 0;
  sscanf(pipe.c_str(), "cm_gemm_%ux%ux%u", &m_, &n_, &k_);
  MvPC pc{bw.addr, x16_.addr, by.addr, tab_.addr, (uint32_t)K, (uint32_t)NR, (uint32_t)T, 0, (uint32_t)K, (uint32_t)NR,
          0, 0, (uint32_t)(K / k_)};
  vk_->dispatch(pipe.c_str(), &pc, sizeof pc, NR / (8 * N), (T + 4 * M - 1) / (4 * M), 1);
  vk_->submit();
  double e = 0, rms = 0;
  for (int t = 0; t < T; t++)
    for (int n = 0; n < NR; n++) {
      double r = 0;
      for (int k = 0; k < K; k++) r += (double)wf[(size_t)n * K + k] * qe_h2f(f2h(x[(size_t)t * K + k]));
      rms += r * r;
      e = std::max(e, std::fabs(((float*)by.map)[(size_t)t * NR + n] - r));
    }
  rms = std::sqrt(rms / (T * NR));
  vk_->free(bw); vk_->free(bx); vk_->free(by);
  const bool ok = std::isfinite(e) && e / rms < 1e-3;
  QE_DBG("[tune] %s self-test: max err / rms = %.2e %s\n", pipe.c_str(), e / rms, ok ? "ok" : "FAILED");
  return ok;
}

// wall time of one recorded sequence (min over reps, after a warm-up run)
double VkModel::bench(const std::function<void()>& rec, int reps) {
  double best = 1e30;
  for (int r = 0; r <= reps; r++) {
    vk_->begin();
    rec();
    const double t0 = qe_now();
    vk_->submit();
    if (r) best = std::min(best, qe_now() - t0);
  }
  return best;
}

// Per-device tuning, measured on this GPU with this model's matrices and cached per
// (vendor, device, driver, matrix shape): matvec workgroup shape and subgroup size for decode,
// the prefill GEMM method (cooperative-matrix shape / scalar f16 / fused).
void VkModel::autotune() {
  const Config& c = cfg_;
  const VLayer* L0 = nullptr;
  for (auto& L : layers_)
    if (L.ffn_gate.valid() && L.ffn_gate.buf.device_local && L.ffn_down.buf.device_local) { L0 = &L; break; }
  const VMat* mats[2] = {L0 ? &L0->ffn_gate : &output_, L0 ? &L0->ffn_down : &output_};
  char key[256];
  snprintf(key, sizeof key, "vk-tune-v1-%x-%x-%x-%s%dx%d-%s%dx%d-ub%d", vk_->vendor_id, vk_->device_id,
           vk_->driver_version, type_tag(mats[0]->type), mats[0]->N, mats[0]->K, type_tag(mats[1]->type), mats[1]->N,
           mats[1]->K, ub_);
  const char* home = getenv("HOME");
  const std::string path = home ? std::string(home) + "/.cache/qwen-hyperdrive/" + key + ".txt" : "";
  // cooperative-matrix shapes this device offers (preferred order per vendor: Intel XMX is 8 x 16 x 16)
  std::vector<std::string> cms;
  {
    std::vector<std::array<uint32_t, 3>> pref = {{16, 16, 16}, {16, 8, 16}, {8, 16, 16}, {16, 8, 8}, {8, 8, 16}, {8, 8, 8}};
    if (vk_->vendor_id == 0x8086) pref = {{8, 16, 16}, {8, 8, 16}, {16, 16, 16}, {16, 8, 16}, {16, 8, 8}, {8, 8, 8}};
    for (auto& p : pref)
      if (vk_->coopmat && x16_.buf && vk_->has_cm(p[0], p[1], p[2])) {
        char nm[64];
        snprintf(nm, sizeof nm, "cm_gemm_%ux%ux%u", p[0], p[1], p[2]);
        if (vk_->has_pipeline(nm)) cms.push_back(nm);
      }
  }
  if (const char* f = getenv("QE_VK_TUNE")) {  // "teams,rows,subgroup,gemm"
    char g[64] = "";
    sscanf(f, "%d,%d,%u,%63s", &tune_.mv_teams, &tune_.mv_r, &tune_.mv_sg, g);
    if (*g) tune_.gemm = g;
    QE_LOG("[tune] forced: matvec %d teams x %d rows, subgroup %u; prefill GEMM %s\n", tune_.mv_teams, tune_.mv_r,
           tune_.mv_sg, tune_.gemm.c_str());
    return;
  }
  if (!path.empty() && !getenv("QE_VK_RETUNE")) {
    if (FILE* f = fopen(path.c_str(), "r")) {
      char g[64] = "";
      Tune t;
      const int n = fscanf(f, "%d %d %u %63s", &t.mv_teams, &t.mv_r, &t.mv_sg, g);
      fclose(f);
      t.gemm = g;
      const bool cm_ok = t.gemm.rfind("cm_", 0) != 0 || std::find(cms.begin(), cms.end(), t.gemm) != cms.end();
      if (n == 4 && t.mv_teams > 0 && t.mv_r > 0 && cm_ok) {
        tune_ = t;
        QE_LOG("[tune] cached: matvec %d teams x %d rows, subgroup %s; prefill GEMM %s\n", tune_.mv_teams, tune_.mv_r,
               tune_.mv_sg ? std::to_string(tune_.mv_sg).c_str() : "default", tune_.gemm.c_str());
        return;
      }
    }
  }
  const double t_start = qe_now();
  // ---- decode matvec
  std::vector<uint32_t> sgs = {0};
  std::vector<uint32_t> want;
  if (vk_->vendor_id == 0x1002) want = {32, 64};       // RDNA: wave32 / wave64
  else if (vk_->vendor_id == 0x8086) want = {16, 32};  // Xe: SIMD16 / SIMD32 (SIMD8 is too narrow here)
  for (uint32_t w : want)
    if (vk_->sg_control && w >= vk_->sg_min && w <= vk_->sg_max) sgs.push_back(w);
  const int shapes[][2] = {{4, 4}, {2, 4}, {4, 2}, {8, 2}, {8, 1}, {2, 8}, {4, 1}};
  double best = 1e30;
  Tune bt = tune_;
  for (uint32_t sg : sgs)
    for (auto& sh : shapes) {
      tune_.mv_teams = sh[0];
      tune_.mv_r = sh[1];
      tune_.mv_sg = sg;
      const double t = bench([&] {
        for (int i = 0; i < 8; i++)
          for (const VMat* m : mats) {
            MvPC pc{m->buf.addr, m == mats[0] ? xn_.addr : t1_.addr, t0_.addr, tab_.addr, (uint32_t)m->K,
                    (uint32_t)m->N, 1, (uint32_t)m->rb, (uint32_t)m->K, (uint32_t)m->N, 0, 0, (uint32_t)(m->K / 32)};
            matvec(*m, pc, true);
          }
      }, 3);
      QE_DBG("[tune] matvec %d teams x %d rows, subgroup %u: %.1f us\n", sh[0], sh[1], sg, t / 16 * 1e6);
      if (t < best * 0.98) { best = t; bt = tune_; }
    }
  tune_ = bt;
  const double mv_us = best / 16 * 1e6;
  // ---- prefill GEMM
  std::vector<std::string> methods = {"fused", "gemm16"};
  for (auto& nm : cms) {
    unsigned M, N, K;
    sscanf(nm.c_str(), "cm_gemm_%ux%ux%u", &M, &N, &K);
    if (cm_selftest(nm, M, N)) methods.push_back(nm);
  }
  best = 1e30;
  std::string bg = "gemm16";
  for (auto& meth : methods) {
    const double t = bench([&] {
      for (const VMat* m : mats) {
        MvPC pc{m->buf.addr, m == mats[0] ? xn_.addr : t1_.addr, t0_.addr, tab_.addr, (uint32_t)m->K, (uint32_t)m->N,
                (uint32_t)ub_, (uint32_t)m->rb, (uint32_t)m->K, (uint32_t)m->N, 0, 0, (uint32_t)(m->K / 32)};
        gemm(*m, pc, meth, true);
      }
    }, 2);
    QE_DBG("[tune] prefill GEMM %s: %.2f ms\n", meth.c_str(), t * 1e3);
    if (t < best * 0.98) { best = t; bg = meth; }
  }
  tune_.gemm = bg;
  QE_LOG("[tune] measured in %.1f s: matvec %d teams x %d rows, subgroup %s (%.0f us/matrix); prefill GEMM %s (%.1f ms per %d-token pair)\n",
         qe_now() - t_start, tune_.mv_teams, tune_.mv_r, tune_.mv_sg ? std::to_string(tune_.mv_sg).c_str() : "default",
         mv_us, tune_.gemm.c_str(), best * 1e3, ub_);
  if (!path.empty()) {
    (void)!system(("mkdir -p '" + path.substr(0, path.rfind('/')) + "'").c_str());
    if (FILE* f = fopen(path.c_str(), "w")) {
      fprintf(f, "%d %d %u %s\n", tune_.mv_teams, tune_.mv_r, tune_.mv_sg, tune_.gemm.c_str());
      fclose(f);
    }
  }
}

void VkModel::reset() {
  vk_->begin();
  vk_->fill(ssm_, 0, ssm_bytes_);
  vk_->fill(conv_, 0, conv_bytes_);
  vk_->submit();
}

void VkModel::prepare_context(int npos, bool) {
  if (npos > ctx_len_) QE_LOG("[warn] context %d exceeds the KV cache (%d positions)\n", npos, ctx_len_);
}

void VkModel::ck_save_live(int slot) {
  if (slot < 0 || slot >= kCkSlots) return;
  if ((int)ck_.size() < kCkSlots) ck_.resize(kCkSlots);
  if (!ck_[slot].buf) ck_[slot] = vk_->alloc(ssm_bytes_ + conv_bytes_, Mem::Host);
  vk_->begin();
  vk_->copy(ssm_, 0, ck_[slot], 0, ssm_bytes_);
  vk_->copy(conv_, 0, ck_[slot], ssm_bytes_, conv_bytes_);
  vk_->submit();
}

void VkModel::ck_restore(int slot) {
  if (slot < 0 || slot >= (int)ck_.size() || !ck_[slot].buf) QE_FATAL("vulkan: restore of an empty checkpoint slot %d", slot);
  vk_->begin();
  vk_->copy(ck_[slot], 0, ssm_, 0, ssm_bytes_);
  vk_->copy(ck_[slot], ssm_bytes_, conv_, 0, conv_bytes_);
  vk_->submit();
}

KvInfo VkModel::kv_info() const {
  KvInfo k;
  k.cur_positions = max_pos_;
  k.seqs = cur_seq_ >= 0;
  k.block_mb = 0;
  return k;
}

// y[T][N] (+)= W x[T][K]
void VkModel::matvec(const VMat& m, const MvPC& pc0, bool bar) {
  MvPC pc = pc0;
  pc.wide = m.N < 2048;  // few rows: one row per workgroup so the GPU is filled
  const uint32_t rows = (uint32_t)(tune_.mv_teams * tune_.mv_r);
  uint32_t gx, gy;
  grid_wg(pc.wide ? m.N : (m.N + rows - 1) / rows, gx, gy);
  PipeOpt po;
  po.spec0 = tune_.mv_r;
  po.spec1 = 32 * tune_.mv_teams;
  po.subgroup = tune_.mv_sg;
  vk_->dispatch((std::string("mv_") + type_tag(m.type)).c_str(), &pc, sizeof pc, gx, gy, 1, bar, po);
}

// y[T][N] (+)= W x[T][K]
void VkModel::linear(const VMat& m, VkDeviceAddress x, int T, VkDeviceAddress y, bool acc, bool bar) {
  MvPC pc{m.buf.addr, x, y, tab_.addr, (uint32_t)m.K, (uint32_t)m.N, (uint32_t)T, (uint32_t)m.rb, (uint32_t)m.K,
          (uint32_t)m.N, acc ? 1u : 0u, 0u, (uint32_t)(m.K / 32)};
  if (T <= kMaxTok) { matvec(m, pc, bar); return; }
  gemm(m, pc, tune_.gemm, bar);
}

// prefill GEMM by method: "fused" (dequantize inside the tiles), "gemm16" (coalesced dequant to f16,
// scalar f16 GEMM) or "cm_gemm_MxNxK" (dequant to f16, cooperative-matrix GEMM)
void VkModel::gemm(const VMat& m, MvPC pc, const std::string& method, bool bar) {
  const int T = (int)pc.T;
  const VkDeviceAddress x = pc.x, y = pc.y;
  const bool acc = pc.acc != 0;
  const std::string tag = type_tag(m.type);
  const bool two = method != "fused" && (size_t)m.N * m.K * 2 <= w16_.size;  // output head: fused
  unsigned M = 0, N = 0, K = 0;
  const bool cm = two && sscanf(method.c_str(), "cm_gemm_%ux%ux%u", &M, &N, &K) == 3 && m.N % (8 * N) == 0 &&
                  x16_.buf;
  if (two) {
    MvPC dq = pc;
    dq.y = w16_.addr;
    const uint64_t groups = (uint64_t)m.N * (m.K / 32);
    const uint32_t dx = (uint32_t)std::min<uint64_t>((groups + 63) / 64, 65535);
    const uint32_t dy = (uint32_t)(((groups + 63) / 64 + 65534) / 65535);
    vk_->dispatch(("dq16_" + tag).c_str(), &dq, sizeof dq, dx, dy);
    pc.w = w16_.addr;
    bar = true;  // the f16 scratch is reused by the next projection
  }
  uint32_t gx, gy, nsteps, min_steps, align;
  std::string pipe;
  if (cm) {  // activations to f16, rows padded to 64 (a multiple of every tile height 4M)
    const uint32_t tp = (T + 63) / 64 * 64;
    ElemPC ep{x, x16_.addr, 0, 0, 0, 0, (uint32_t)((size_t)tp * m.K / 2), (uint32_t)((size_t)T * m.K / 2), 0};
    uint32_t ex, ey;
    grid1d(ep.n, ex, ey);
    vk_->dispatch("to_f16", &ep, sizeof ep, ex, ey);
    pc.x = x16_.addr;
    gx = m.N / (8 * N);
    gy = (T + 4 * M - 1) / (4 * M);
    nsteps = m.K / K;
    min_steps = 256 / K;
    align = 1;
    pipe = method;
  } else {
    gx = (m.N + 63) / 64;
    gy = (T + 63) / 64;
    nsteps = m.K / 32;
    min_steps = 8;
    align = two ? 1 : 2;  // the fused shader steps two groups at a time
    pipe = two ? "gemm16" : "mm_" + tag;
  }
  // split K while the tile grid alone leaves the GPU idle (scratch-bounded)
  uint32_t ks = 1;
  while (gx * gy * ks * 2 <= wg_target_ && nsteps / (ks * 2) >= min_steps &&
         (size_t)ks * 2 * T * m.N * 4 <= ksplit_.size)
    ks *= 2;
  if (ks == 1) {
    pc.kg = nsteps;
    vk_->dispatch(pipe.c_str(), &pc, sizeof pc, gx, gy, 1, bar);
    return;
  }
  pc.kg = (nsteps / ks + align - 1) / align * align;
  ks = (nsteps + pc.kg - 1) / pc.kg;
  MvPC p2 = pc;
  p2.y = ksplit_.addr;
  vk_->dispatch(pipe.c_str(), &p2, sizeof p2, gx, gy, ks);
  ElemPC ep{ksplit_.addr, y, 0, 0, 0, 0, (uint32_t)(T * m.N), ks, acc ? 1u : 0u};
  uint32_t ex, ey;
  grid1d(ep.n, ex, ey);
  vk_->dispatch("ksum", &ep, sizeof ep, ex, ey, 1, bar);
}

// xn = rmsnorm(x += pending residual) * w
void VkModel::norm_in(const VBuf& w, int T) {
  NormPC pc{x_.addr, pend_ ? t4_.addr : 0, w.addr, xn_.addr, (uint32_t)cfg_.n_embd, cfg_.eps};
  vk_->dispatch("rmsnorm", &pc, sizeof pc, T);
  pend_ = false;
}

void VkModel::layer_attn(VLayer& L, int T, int pos0) {
  const Config& c = cfg_;
  const int qw = 2 * c.n_head * c.hd, kvw = c.n_kv * c.hd;
  norm_in(L.attn_norm, T);
  const VkDeviceAddress qg = t0_.addr, k = t3_.addr, v = t3_.addr + (size_t)T * kvw * 4;
  linear(L.wq, xn_.addr, T, qg, false, false);
  linear(L.wk, xn_.addr, T, k, false, false);
  linear(L.wv, xn_.addr, T, v);
  uint32_t gx, gy;
  {
    HeadNormPC pq{qg, L.q_norm.addr, (uint32_t)c.hd, (uint32_t)c.n_head, (uint32_t)(2 * c.hd), (uint32_t)qw, 0, c.eps};
    grid_wg((uint64_t)T * c.n_head, gx, gy);
    vk_->dispatch("headnorm", &pq, sizeof pq, gx, gy, 1, false);
    HeadNormPC pk{k, L.k_norm.addr, (uint32_t)c.hd, (uint32_t)c.n_kv, (uint32_t)c.hd, (uint32_t)kvw, 0, c.eps};
    grid_wg((uint64_t)T * c.n_kv, gx, gy);
    vk_->dispatch("headnorm", &pk, sizeof pk, gx, gy);
  }
  {
    const uint32_t half = c.rope_dims / 2;
    RopePC rq{qg, rope_h_.addr, (uint32_t)c.n_head, (uint32_t)(2 * c.hd), (uint32_t)qw, (uint32_t)T, half};
    grid1d((uint64_t)T * c.n_head * half, gx, gy);
    vk_->dispatch("rope", &rq, sizeof rq, gx, gy, 1, false);
    RopePC rk{k, rope_h_.addr, (uint32_t)c.n_kv, (uint32_t)c.hd, (uint32_t)kvw, (uint32_t)T, half};
    grid1d((uint64_t)T * c.n_kv * half, gx, gy);
    vk_->dispatch("rope", &rk, sizeof rk, gx, gy);
  }
  {
    KvPC kp{k, v, L.kc.addr, L.vc.addr, (uint32_t)c.n_kv, (uint32_t)c.hd, (uint32_t)T, (uint32_t)pos0, (uint32_t)kvw,
            kv_q8_ ? 1u : 0u};
    grid1d((uint64_t)T * c.n_kv * (c.hd / 32), gx, gy);
    vk_->dispatch("kv_store", &kp, sizeof kp, (T * c.n_kv * (c.hd / 32) + 63) / 64);
  }
  {
    const uint32_t R = c.n_head / c.n_kv, TB = 32 / R;
    const uint32_t L_keys = (uint32_t)(pos0 + T);
    uint32_t nsplit = 1, chunk = L_keys;
    if (T <= kMaxTok) {  // decode: split the keys so every compute unit has work
      const uint32_t want = std::max<uint32_t>(1, wg_target_ / (c.n_kv * ((T + TB - 1) / TB)));
      nsplit = std::clamp<uint32_t>(std::min(want, (L_keys + 63) / 64), 1, kMaxSplit);
      chunk = ((L_keys + nsplit - 1) / nsplit + 31) / 32 * 32;
      nsplit = (L_keys + chunk - 1) / chunk;
    }
    AttnPC ap{qg, L.kc.addr, L.vc.addr, t2_.addr, part_.addr, (uint32_t)T, (uint32_t)pos0, (uint32_t)c.n_head,
              (uint32_t)c.n_kv, (uint32_t)qw, nsplit, chunk, 1.f / std::sqrt((float)c.hd)};
    vk_->dispatch(kv_q8_ ? "attn_q8" : "attn_f16", &ap, sizeof ap, (T + TB - 1) / TB, c.n_kv, nsplit);
    if (nsplit > 1) {
      CombPC cp{part_.addr, t2_.addr, (uint32_t)(T * c.n_head), nsplit};
      vk_->dispatch("attn_combine", &cp, sizeof cp, T * c.n_head);
    }
  }
  {
    ElemPC ep{t2_.addr, qg, 0, 0, 0, 0, (uint32_t)(T * c.n_head * c.hd), (uint32_t)c.hd, (uint32_t)c.n_head};
    grid1d(ep.n, gx, gy);
    vk_->dispatch("gate", &ep, sizeof ep, gx, gy);
  }
  linear(L.wo, t2_.addr, T, t4_.addr);
  pend_ = true;
}

void VkModel::layer_gdn(VLayer& L, int T) {
  const Config& c = cfg_;
  const int C = c.conv_dim();
  norm_in(L.attn_norm, T);
  const VkDeviceAddress ab = t4_.addr;  // alpha [T][nv], beta [T][nv]; t4 is free until ssm_out
  linear(L.wqkv, xn_.addr, T, t0_.addr, false, false);
  linear(L.wgate, xn_.addr, T, t1_.addr, false, false);
  linear(L.walpha, xn_.addr, T, ab, false, false);
  linear(L.wbeta, xn_.addr, T, ab + (size_t)T * c.nv * 4);
  uint32_t gx, gy;
  {
    ConvPC cp{t0_.addr, conv_.addr + L.conv_off, L.conv_w.addr, t3_.addr, (uint32_t)C, (uint32_t)T, (uint32_t)C};
    vk_->dispatch("conv", &cp, sizeof cp, (C + 255) / 256);
  }
  {
    HeadNormPC hp{t3_.addr, 0, (uint32_t)c.dk, (uint32_t)(2 * c.nk), (uint32_t)c.dk, (uint32_t)C, 1, c.eps};
    grid_wg((uint64_t)T * 2 * c.nk, gx, gy);
    vk_->dispatch("headnorm", &hp, sizeof hp, gx, gy, 1, false);
    ElemPC ep{ab, ab + (size_t)T * c.nv * 4, L.dt_bias.addr, L.ssm_a.addr, g_.addr, beta_.addr, (uint32_t)(T * c.nv),
              (uint32_t)c.nv, 0};
    grid1d(ep.n, gx, gy);
    vk_->dispatch("gdn_gb", &ep, sizeof ep, gx, gy);
  }
  {
    RecPC rp{t3_.addr, g_.addr, beta_.addr, ssm_.addr + L.ssm_off, t2_.addr, (uint32_t)c.nk, (uint32_t)c.nv, (uint32_t)T,
             (uint32_t)C};
    vk_->dispatch("gdn_rec", &rp, sizeof rp, c.nv, c.dv / 16);
  }
  {
    GNormPC gp{t2_.addr, t1_.addr, L.ssm_norm.addr, (uint32_t)c.nv, (uint32_t)c.dv, (uint32_t)c.d_inner, c.eps};
    grid_wg((uint64_t)T * c.nv, gx, gy);
    vk_->dispatch("gated_norm", &gp, sizeof gp, gx, gy);
  }
  linear(L.wssm_out, t2_.addr, T, t4_.addr);
  pend_ = true;
}

void VkModel::layer_ffn(VLayer& L, int T) {
  const Config& c = cfg_;
  if (!L.ffn_gate.valid()) return;  // --skip-ffn
  norm_in(L.post_norm, T);
  linear(L.ffn_gate, xn_.addr, T, t0_.addr, false, false);
  linear(L.ffn_up, xn_.addr, T, t1_.addr);
  uint32_t gx, gy;
  ElemPC ep{t0_.addr, t1_.addr, 0, 0, 0, 0, (uint32_t)(T * c.n_ff), 0, 0};
  grid1d(ep.n, gx, gy);
  vk_->dispatch("silu_mul", &ep, sizeof ep, gx, gy);
  linear(L.ffn_down, t0_.addr, T, t4_.addr);
  pend_ = true;
}

// record one chunk of <= ub_ tokens (embedding, all layers, final norm); not submitted
void VkModel::forward_chunk(const int* tokens, int T, int pos0) {
  const Config& c = cfg_;
  float* emb = (float*)emb_h_.map;
  for (int t = 0; t < T; t++) {
    if (tokens[t] < 0 || tokens[t] >= tok_embd_.N) QE_FATAL("token id %d out of range", tokens[t]);
    const GGUFTensor& te = gguf_->need("token_embd.weight");
    dequant_row_ref(te.type, te.data + (size_t)tokens[t] * tok_embd_.rb, c.n_embd, emb + (size_t)t * c.n_embd);
  }
  {  // RoPE angles in double precision (positions up to 1e5+ rad)
    const int half = c.rope_dims / 2;
    float* cs = (float*)rope_h_.map;
    for (int t = 0; t < T; t++)
      for (int i = 0; i < half; i++) {
        const double freq = std::pow((double)c.rope_base, -2.0 * i / c.rope_dims);
        const double a = (double)(pos0 + t) * freq;
        cs[2 * (t * half + i)] = (float)std::cos(a);
        cs[2 * (t * half + i) + 1] = (float)std::sin(a);
      }
  }
  vk_->begin();
  vk_->copy(emb_h_, 0, x_, 0, (size_t)T * c.n_embd * 4);
  pend_ = false;
  for (auto& L : layers_) {
    if (L.attn) layer_attn(L, T, pos0);
    else layer_gdn(L, T);
    layer_ffn(L, T);
  }
  norm_in(output_norm_, T);
}

void VkModel::forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax, bool, bool) {
  // checkpoint capture inside this pass: split it at the capture position
  if (cap_slot_ >= 0 && cap_pos_ == pos0) {
    ck_save_live(cap_slot_);
    cap_slot_ = -1;
  } else if (cap_slot_ >= 0 && cap_pos_ > pos0 && cap_pos_ < pos0 + n && n - (cap_pos_ - pos0) >= n_logits) {
    const int k = cap_pos_ - pos0;
    forward_impl(tokens, k, pos0, nullptr, 0, nullptr);
    ck_save_live(cap_slot_);
    cap_slot_ = -1;
    forward_impl(tokens + k, n - k, cap_pos_, logits, n_logits, argmax);
    return;
  }
  forward_impl(tokens, n, pos0, logits, n_logits, argmax);
}

void VkModel::forward_impl(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax) {
  const Config& c = cfg_;
  QE_ASSERT(n_logits <= n);
  if (pos0 + n > ctx_len_) QE_FATAL("context overflow: %d positions > KV cache %d", pos0 + n, ctx_len_);
  const int first = n - n_logits;
  for (int b0 = 0; b0 < n; b0 += ub_) {
    const int T = std::min(ub_, n - b0);
    forward_chunk(tokens + b0, T, pos0 + b0);
    const int r_begin = std::max(first, b0), r_end = b0 + T;
    if ((logits || argmax) && r_begin < r_end) {
      for (int r = r_begin; r < r_end; r += kLogitRows) {
        const int nb = std::min(kLogitRows, r_end - r);
        vk_->begin();
        linear(output_, xn_.addr + (size_t)(r - b0) * c.n_embd * 4, nb, logits_.addr);
        if (logits) vk_->copy(logits_, 0, logits_h_, 0, (size_t)nb * c.n_vocab * 4);
        if (argmax) {
          ArgPC ap{logits_.addr, argmax_.addr, (uint32_t)c.n_vocab};
          vk_->dispatch("argmax", &ap, sizeof ap, nb);
          vk_->copy(argmax_, 0, argmax_h_, 0, nb * 4);
        }
        vk_->submit();
        if (logits) memcpy(logits + (size_t)(r - first) * c.n_vocab, logits_h_.map, (size_t)nb * c.n_vocab * 4);
        if (argmax) memcpy(argmax + (r - first), argmax_h_.map, nb * 4);
      }
    }
    vk_->submit();
  }
  max_pos_ = std::max(max_pos_, pos0 + n);
}
