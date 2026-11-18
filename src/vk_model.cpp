#include "vk_model.h"

#include <fcntl.h>
#include <immintrin.h>
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
struct ConvPC { uint64_t x, st, w, y; uint32_t C, T, xts, wr; };
struct RecPC { uint64_t qkv, g, beta, st, o; uint32_t nk, nv, T, C, wr; };
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
  if (c.mhc_streams || c.rope_mscale != 1.f)
    QE_FATAL("%s (looped blocks / mHC residual streams / YaRN) runs on the CUDA backend only for now", c.arch.c_str());
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
  submit_layers_ = vk_->integrated ? 1 : 8;
  if (const char* e = getenv("QE_VK_SUBMIT_LAYERS")) submit_layers_ = std::max(1, atoi(e));
  // weights this backend will place on the GPU (trunk layers, output head; not token_embd / MTP)
  size_t wbytes = 0, ffn_bytes = 0;
  for (const auto& t : gguf_->tensors()) {
    int il = -1;
    if (sscanf(t.name.c_str(), "blk.%d.", &il) == 1 && il >= c.n_layer && !(opt.use_mtp && il == c.n_layer)) continue;
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
  mtp_on_ = c.n_mtp > 0 && opt.use_mtp;
  const int n_attn_l = [&] { int n = mtp_on_; for (int il = 0; il < c.n_layer; il++) n += c.is_attn(il); return n; }();
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
  // unified memory: use the weights in place from the mapped model file (QE_VK_ZEROCOPY=1, experimental)
  // opt-in only: Mesa ANV cannot pin read-only file pages and loses the device when the import fails
  zero_copy_ = vk_->host_import && getenv("QE_VK_ZEROCOPY") && atoi(getenv("QE_VK_ZEROCOPY")) != 0;
  if (zero_copy_) zc_plan();
  drop_pages_ = !zero_copy_ && (vk_->integrated || getenv("QE_VK_DROP_PAGES"));
  if (drop_pages_) model_fd_ = open(opt.model_path.c_str(), O_RDONLY);

  // ---- activations, state, small tensors (always device-local)
  const int E = c.n_embd, C = c.conv_dim(), qw = 2 * c.n_head * c.hd, kvw = c.n_kv * c.hd;
  auto dev = [&](size_t floats) { return vk_->alloc(floats * 4, Mem::Device, true); };
  const size_t U = (size_t)ub_;
  // CPU / GPU co-execution: possible on unified memory (QE_VK_HYBRID=0/1 overrides; the CPU kernels
  // must handle the FFN gate / up formats, checked when the weights are known)
  hyb_ = getenv("QE_VK_HYBRID") ? atoi(getenv("QE_VK_HYBRID")) != 0 : vk_->integrated;
  if (const char* e = getenv("QE_VK_HYBRID_LEVEL")) hyb_force_ = std::clamp(atoi(e), 0, kHybLevels - 1);
  auto devm = [&](size_t floats) { return vk_->alloc(floats * 4, hyb_ ? Mem::DeviceMapped : Mem::Device, true); };
  x_ = dev(U * E);
  xn_ = devm(U * E);
  t0_ = devm(U * std::max({qw, C, c.n_ff}));
  t1_ = devm(U * std::max(c.d_inner, c.n_ff));
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
    const int kmax = std::max({2 * c.n_embd, c.n_ff, c.d_inner, c.n_head * c.hd});
    if (vk_->coopmat) x16_ = vk_->alloc((size_t)ub_ * kmax * 2, Mem::Device, true);
    if (vk_->int_dot) {
      xq8_ = vk_->alloc((size_t)kMaxTok * kmax, Mem::Device, true);
      xd8_ = vk_->alloc((size_t)kMaxTok * kmax / 32 * 4, Mem::Device, true);
    }
  }
  emb_h_ = vk_->alloc(U * E * 4, Mem::Host);
  if (mtp_on_) {  // MTP: trunk hidden rows of the last pass, carried row, chained MTP output, staging
    hid_ = dev((size_t)kMaxTok * E);
    carry_ = dev(E);
    mtp_h_ = dev(E);
    hsafe_ = dev((size_t)kMaxTok * E);
    hm_ = dev(U * E);
    emb2_h_ = vk_->alloc(U * E * 4, Mem::Host);
    rope2_h_ = vk_->alloc(U * (c.rope_dims / 2) * 2 * 4, Mem::Host);
  }
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
      load_attn(L, il);
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
      if (mtp_on_) L.sv = dev((size_t)kMaxTok * (2 * C + 2 * c.nv));
    }
  }
  if (mtp_on_) {  // MTP (nextn) block: one full-attention layer + FFN, eh_proj, its norms
    const int il = c.n_layer;
    mtp_.attn = true;
    mtp_.attn_norm = load_f32(bn(il, "attn_norm.weight"));
    mtp_.post_norm = load_f32(bn(il, "post_attention_norm.weight"));
    load_attn(mtp_, il);
    mtp_.eh_proj = load_mat(bn(il, "nextn.eh_proj.weight"), true);
    mtp_.enorm = load_f32(bn(il, "nextn.enorm.weight"));
    mtp_.hnorm = load_f32(bn(il, "nextn.hnorm.weight"));
    if (gguf_->tensor(bn(il, "nextn.shared_head_norm.weight"))) mtp_.shnorm = load_f32(bn(il, "nextn.shared_head_norm.weight"));
  }
  // KV cache: the full context if it fits next to ~1 GiB of other data, else shrink the context
  {
    const int n_attn = (int)std::count_if(layers_.begin(), layers_.end(), [](const VLayer& L) { return L.attn; }) + mtp_on_;
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
    if (mtp_on_) {
      mtp_.kc = vk_->alloc((size_t)ctx_len_ * c.n_kv * kv_row_bytes_, Mem::Device);
      mtp_.vc = vk_->alloc((size_t)ctx_len_ * c.n_kv * kv_row_bytes_, Mem::Device);
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
  if (mtp_on_) {
    mtp_.ffn_gate = load_mat(bn(c.n_layer, "ffn_gate.weight"), true);
    mtp_.ffn_up = load_mat(bn(c.n_layer, "ffn_up.weight"), true);
    mtp_.ffn_down = load_mat(bn(c.n_layer, "ffn_down.weight"), true);
  }
  plan_.res_mb = vk_->vram_used() / (double)MiB;
  plan_.stream_mb = host_ffn / (double)MiB;
  plan_.cpu_mb = 0;
  plan_.res_ffn = res_ffn;
  QE_LOG("[plan] VRAM used %.0f MiB (%d/%d FFN layers resident)%s\n", vk_->vram_used() / (double)MiB, res_ffn,
         c.n_layer, host_ffn ? "" : ", all weights in VRAM");
  if (zero_copy_ && zc_bytes_)
    QE_LOG("[plan] unified memory: %.0f MiB of weights used in place from the mapped model file (no copy)\n",
           zc_bytes_ / (double)MiB);
  if (host_ffn)
    QE_LOG("[plan] %.0f MiB of FFN weights read from system RAM over PCIe each token (decode will be slower)\n",
           host_ffn / (double)MiB);
  if (model_fd_ >= 0) { close(model_fd_); model_fd_ = -1; }
  if (hyb_) {
    for (auto& L : layers_)
      if (L.ffn_gate.valid() && (!cpu::type_supported(L.ffn_gate.type) || !cpu::type_supported(L.ffn_up.type) || c.n_embd % 256))
        hyb_ = false;
    if (!hyb_) QE_LOG("[plan] CPU / GPU co-execution off: FFN formats without CPU kernels\n");
  }
  if (hyb_ && (!xn_.map || !t0_.map || !t1_.map)) {
    hyb_ = false;
    QE_LOG("[plan] CPU / GPU co-execution off: activations are not CPU-mappable on this device\n");
  }
  if (hyb_) {
    const int nt = opt.cpu_threads > 0 ? opt.cpu_threads : cpu::fast_cores();
    pool_ = std::make_unique<cpu::ThreadPool>(nt);
    for (size_t i = 0; i < layers_.size(); i++) hyb_ev_.push_back(vk_->make_event());
    hyb_flags_ = vk_->alloc(layers_.size() * 4, Mem::Readback);  // GPU writes, CPU polls
    memset(hyb_flags_.map, 0, layers_.size() * 4);
    hyb_q_.resize(c.n_embd, kMaxTok);
    QE_LOG("[plan] CPU / GPU co-execution of the FFN gate/up rows on unified memory: %d %s threads (%s), share chosen online\n",
           nt, cpu::fast_cores() < (int)cpu::core_order().size() ? "big-core" : "", cpu::isa_name(cpu::isa()));
  }
  vk_->warm_pipelines();
  autotune();
  if (const char* g = getenv("QE_VK_GEMM")) tune_.gemm = g;  // fused | gemm16 | cm_gemm_MxNxK
  if (const char* f = getenv("QE_VK_FORCE_COOP")) {  // 0 generic, 1 cooperative, 2 cooperative + int8 dot
    tune_.coop = atoi(f) >= 1;
    tune_.i8 = atoi(f) >= 2 && vk_->int_dot;
  }
  reset();
  QE_LOG("[plan] loaded in %.1f s\n", qe_now() - t_start);
}

VkModel::~VkModel() {
  vk_.reset();  // device teardown frees every allocation
}

void VkModel::zc_plan() {
  std::vector<const GGUFTensor*> ts;
  for (const auto& t : gguf_->tensors()) ts.push_back(&t);
  std::sort(ts.begin(), ts.end(), [](const GGUFTensor* a, const GGUFTensor* b) { return a->data < b->data; });
  const size_t al = vk_->import_align, base = (size_t)gguf_->base();
  const size_t limit = std::min<size_t>(vk_->max_buffer, (size_t)3 << 30) / al * al;
  for (const GGUFTensor* t : ts) {
    const size_t s = ((size_t)t->data - base) / al * al, e = ((size_t)t->data - base + t->nbytes + al - 1) / al * al;
    if (zc_chunks_.empty() || e - zc_chunks_.back().begin > limit) zc_chunks_.push_back({s, e});
    else zc_chunks_.back().end = std::max(zc_chunks_.back().end, e);
    zc_index_.push_back({t->data, (int)zc_chunks_.size() - 1});
  }
  std::sort(zc_index_.begin(), zc_index_.end());
}

bool VkModel::zc_view(const uint8_t* data, size_t bytes, VBuf& out) {
  auto it = std::lower_bound(zc_index_.begin(), zc_index_.end(), std::make_pair(data, -1));
  if (it == zc_index_.end() || it->first != data) return false;
  ZcChunk& c = zc_chunks_[it->second];
  const size_t file_end = gguf_->size();
  if (!c.tried) {
    c.tried = true;
    // the mapping covers whole pages, so the last chunk may round past the file end: clamp to the map
    const size_t end = std::min(c.end, (file_end + vk_->import_align - 1) / vk_->import_align * vk_->import_align);
    c.buf = vk_->import_host(gguf_->base() + c.begin, end - c.begin);
    if (!c.buf.buf) {
      QE_LOG("[plan] host-memory import failed: copying the weights instead\n");
      zero_copy_ = false;
      return false;
    }
    zc_bytes_ += c.buf.size;
  }
  if (!c.buf.buf) return false;
  out = c.buf;
  out.addr = c.buf.addr + (size_t)(data - gguf_->base() - c.begin);
  out.size = bytes;
  out.map = const_cast<uint8_t*>(data);
  return true;
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
  if (zero_copy_ && prefer_device && zc_view(t.data, bytes, m.buf)) return m;
  const bool fits = vk_->vram_free() > bytes + reserve_;
  if (!fits && vk_->integrated)
    QE_FATAL("not enough memory for %s (%.0f MiB free after the %.0f MiB reserve): use a smaller model or context",
             name.c_str(), (vk_->vram_free() > reserve_ ? vk_->vram_free() - reserve_ : 0) / 1048576.0, reserve_ / 1048576.0);
  // mapped device memory (ReBAR dGPU, any iGPU): the upload is a memcpy, no staging copy
  m.buf = vk_->alloc(bytes, prefer_device && fits ? Mem::DeviceMapped : Mem::Host);
  upload(m.buf, t.data, bytes);
  // co-execution reads the first rows of gate / up from the mapped file: keep those pages
  const bool cpu_rows = hyb_ && (name.find("ffn_gate") != std::string::npos || name.find("ffn_up") != std::string::npos);
  const size_t keep = cpu_rows ? (size_t)(m.N / 2) * m.rb : 0;
  release_pages(t.data + keep, bytes - keep);
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
  const bool staged = pipe.rfind("cms_", 0) == 0;
  sscanf(pipe.c_str(), staged ? "cms_gemm_%ux%ux%u" : "cm_gemm_%ux%ux%u", &m_, &n_, &k_);
  MvPC pc{bw.addr, x16_.addr, by.addr, tab_.addr, (uint32_t)K, (uint32_t)NR, (uint32_t)T, 0, (uint32_t)K, (uint32_t)NR,
          0, 0, (uint32_t)(K / (staged ? 32 : k_))};
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
  snprintf(key, sizeof key, "vk-tune-v4-%x-%x-%x-%s%dx%d-%s%dx%d-ub%d", vk_->vendor_id, vk_->device_id,
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
  if (const char* f = getenv("QE_VK_TUNE")) {  // "teams,rows,subgroup,multi_teams,multi_rows,gemm"
    char g[64] = "";
    sscanf(f, "%d,%d,%u,%d,%d,%63s", &tune_.mv_teams, &tune_.mv_r, &tune_.mv_sg, &tune_.mm_teams, &tune_.mm_r, g);
    if (*g) tune_.gemm = g;
    QE_LOG("[tune] forced: matvec %d teams x %d rows, subgroup %u; prefill GEMM %s\n", tune_.mv_teams, tune_.mv_r,
           tune_.mv_sg, tune_.gemm.c_str());
    return;
  }
  if (!path.empty() && !getenv("QE_VK_RETUNE")) {
    if (FILE* f = fopen(path.c_str(), "r")) {
      char g[64] = "";
      Tune t;
      int coop = 0, i8 = 0;
      const int n = fscanf(f, "%d %d %u %d %d %d %d %d %d %d %d %63s", &t.mv_teams, &t.mv_r, &t.mv_sg, &t.mm_teams, &t.mm_r,
                           &coop, &t.c_teams, &t.c_r, &t.cm_teams, &t.cm_r, &i8, g);
      fclose(f);
      t.gemm = g;
      t.coop = coop != 0;
      t.i8 = i8 != 0 && vk_->int_dot;
      const std::string base = t.gemm.rfind("cms_", 0) == 0 ? "cm" + t.gemm.substr(3) : t.gemm;
      const bool cm_ok = base.rfind("cm_", 0) != 0 || std::find(cms.begin(), cms.end(), base) != cms.end();
      if (n == 12 && t.mv_teams > 0 && t.mv_r > 0 && t.mm_teams > 0 && t.mm_r > 0 && cm_ok) {
        tune_ = t;
        QE_LOG("[tune] cached: matvec %d teams x %d rows, subgroup %s, T=4 %d x %d; cooperative K-quant kernel %s "
               "(%d x %d, T=4 %d x %d); prefill GEMM %s\n", tune_.mv_teams, tune_.mv_r,
               tune_.mv_sg ? std::to_string(tune_.mv_sg).c_str() : "default", tune_.mm_teams, tune_.mm_r,
               tune_.coop ? (tune_.i8 ? "on, int8 dot" : "on") : "off", tune_.c_teams, tune_.c_r, tune_.cm_teams, tune_.cm_r, tune_.gemm.c_str());
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
  // ---- T = 4 matvec (speculative verify): its own workgroup shape (holds R dequantized rows)
  best = 1e30;
  for (auto& sh : shapes) {
    tune_.mm_teams = sh[0];
    tune_.mm_r = sh[1];
    const double t = bench([&] {
      for (int i = 0; i < 4; i++)
        for (const VMat* m : mats) {
          MvPC pc{m->buf.addr, m == mats[0] ? xn_.addr : t1_.addr, t0_.addr, tab_.addr, (uint32_t)m->K, (uint32_t)m->N,
                  4, (uint32_t)m->rb, (uint32_t)m->K, (uint32_t)m->N, 0, 0, (uint32_t)(m->K / 32)};
          matvec(*m, pc, true);
        }
    }, 3);
    QE_DBG("[tune] T=4 matvec %d teams x %d rows: %.1f us\n", sh[0], sh[1], t / 8 * 1e6);
    if (t < best * 0.98) { best = t; bt.mm_teams = sh[0]; bt.mm_r = sh[1]; }
  }
  tune_ = bt;
  const double mm_us = best / 8 * 1e6;
  // ---- cooperative super-block kernel (K-quants / Q8_0): its own shapes; kept when it beats the
  // generic kernel on this device (Intel Xe2: scattered per-lane block reads are slow there)
  bool any_c = false;
  for (const VMat* m : mats) any_c |= vk_->has_pipeline((std::string("mvc_") + type_tag(m->type)).c_str());
  double c1 = 1e30, c4 = 1e30;
  if (any_c) {
    const int cshapes[][2] = {{16, 1}, {8, 2}, {32, 1}, {8, 1}, {16, 2}, {4, 2}};
    tune_.coop = true;
    for (int multi = 0; multi < 2; multi++)
      for (auto& sh : cshapes) {
        const size_t shmem = 8u * sh[0] * sh[1] * 33 * 4;
        if (shmem > vk_->max_shared || 32u * sh[0] > vk_->max_wg_invocations) continue;
        (multi ? tune_.cm_teams : tune_.c_teams) = sh[0];
        (multi ? tune_.cm_r : tune_.c_r) = sh[1];
        const uint32_t T = multi ? 4 : 1;
        const double t = bench([&] {
          for (int i = 0; i < (multi ? 4 : 8); i++)
            for (const VMat* m : mats) {
              MvPC pc{m->buf.addr, m == mats[0] ? xn_.addr : t1_.addr, t0_.addr, tab_.addr, (uint32_t)m->K,
                      (uint32_t)m->N, T, (uint32_t)m->rb, (uint32_t)m->K, (uint32_t)m->N, 0, 0, (uint32_t)(m->K / 32)};
              matvec(*m, pc, true);
            }
        }, 3) / (multi ? 8 : 16) * 1e6;
        QE_DBG("[tune] cooperative %s %d teams x %d rows: %.1f us\n", multi ? "T=4" : "T=1", sh[0], sh[1], t);
        double& b = multi ? c4 : c1;
        if (t < b * 0.98) {
          b = t;
          if (multi) { bt.cm_teams = sh[0]; bt.cm_r = sh[1]; }
          else { bt.c_teams = sh[0]; bt.c_r = sh[1]; }
        }
        (multi ? tune_.cm_teams : tune_.c_teams) = multi ? bt.cm_teams : bt.c_teams;
        (multi ? tune_.cm_r : tune_.c_r) = multi ? bt.cm_r : bt.c_r;
      }
    // decode is ~all T = 1 and verify passes (T ~ 3-4) are its other half: weigh both
    bt.coop = c1 + c4 < (mv_us + mm_us) * 0.97;
    // int8 activations + packed integer dot products with the shapes found above
    bool any_i = false;
    for (const VMat* m : mats) any_i |= vk_->has_pipeline((std::string("mvci_") + type_tag(m->type)).c_str());
    if (bt.coop && any_i && xq8_.buf) {
      tune_ = bt;
      tune_.i8 = true;
      double i1 = 0, i4 = 0;
      for (int multi = 0; multi < 2; multi++) {
        const uint32_t T = multi ? 4 : 1;
        (multi ? i4 : i1) = bench([&] {
          for (int i = 0; i < (multi ? 4 : 8); i++)
            for (const VMat* m : mats) {
              MvPC pc{m->buf.addr, m == mats[0] ? xn_.addr : t1_.addr, t0_.addr, tab_.addr, (uint32_t)m->K,
                      (uint32_t)m->N, T, (uint32_t)m->rb, (uint32_t)m->K, (uint32_t)m->N, 0, 0, (uint32_t)(m->K / 32)};
              xq_mark_ = ~0ull;  // quantize every time: as in decode, where each input is new
              matvec(*m, pc, true);
            }
        }, 3) / (multi ? 8 : 16) * 1e6;
      }
      QE_DBG("[tune] cooperative int8 dot: T=1 %.1f us, T=4 %.1f us (float: %.1f / %.1f)\n", i1, i4, c1, c4);
      bt.i8 = i1 + i4 < (c1 + c4) * 0.97;
      if (bt.i8) { c1 = i1; c4 = i4; }
    }
  }
  tune_ = bt;
  // ---- prefill GEMM
  std::vector<std::string> methods = {"fused", "gemm16"};
  for (auto& nm : cms) {
    unsigned M, N, K;
    sscanf(nm.c_str(), "cm_gemm_%ux%ux%u", &M, &N, &K);
    if (cm_selftest(nm, M, N)) methods.push_back(nm);
    // shared-memory staged variant (operands from on-chip memory), when its tiles fit
    const std::string st = "cms" + nm.substr(2);
    const size_t shm = (size_t)(4 * M + 8 * N) * 40 * 2 + (size_t)4 * M * 8 * N * 4;
    if (vk_->has_pipeline(st.c_str()) && shm <= vk_->max_shared && cm_selftest(st, M, N)) methods.push_back(st);
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
  QE_LOG("[tune] measured in %.1f s: matvec %d teams x %d rows, subgroup %s (%.0f us/matrix), T=4 %d x %d (%.0f us); "
         "cooperative K-quant kernel %s; prefill GEMM %s (%.1f ms per %d-token pair)\n",
         qe_now() - t_start, tune_.mv_teams, tune_.mv_r, tune_.mv_sg ? std::to_string(tune_.mv_sg).c_str() : "default",
         mv_us, tune_.mm_teams, tune_.mm_r, mm_us,
         !any_c ? "n/a (no K-quant / Q8_0 matrices)"
                : (std::string(tune_.coop ? (tune_.i8 ? "on, int8 dot" : "on") : "off") + " (" + std::to_string(tune_.c_teams) + " x " +
                   std::to_string(tune_.c_r) + ": " + std::to_string((int)c1) + " us, T=4 " + std::to_string(tune_.cm_teams) + " x " +
                   std::to_string(tune_.cm_r) + ": " + std::to_string((int)c4) + " us)").c_str(),
         tune_.gemm.c_str(), best * 1e3, ub_);
  if (!path.empty()) {
    (void)!system(("mkdir -p '" + path.substr(0, path.rfind('/')) + "'").c_str());
    if (FILE* f = fopen(path.c_str(), "w")) {
      fprintf(f, "%d %d %u %d %d %d %d %d %d %d %d %s\n", tune_.mv_teams, tune_.mv_r, tune_.mv_sg, tune_.mm_teams, tune_.mm_r,
              tune_.coop ? 1 : 0, tune_.c_teams, tune_.c_r, tune_.cm_teams, tune_.cm_r, tune_.i8 ? 1 : 0, tune_.gemm.c_str());
      fclose(f);
    }
  }
}

void VkModel::load_attn(VLayer& L, int il) {
  L.wq = load_mat(bn(il, "attn_q.weight"), true);
  L.wk = load_mat(bn(il, "attn_k.weight"), true);
  L.wv = load_mat(bn(il, "attn_v.weight"), true);
  L.wo = load_mat(bn(il, "attn_output.weight"), true);
  L.q_norm = load_f32(bn(il, "attn_q_norm.weight"));
  L.k_norm = load_f32(bn(il, "attn_k_norm.weight"));
}

// MTP (nextn) block: x = eh_proj([enorm(emb(x_p)), hnorm(h_{p-1})]); one full-attention decoder layer
// with its own KV cache; final norm.  Records into the current command buffer; result rows in xn_.
void VkModel::mtp_block(VkDeviceAddress hrows, const int* tokens, int T, int pos0) {
  const Config& c = cfg_;
  const int E = c.n_embd;
  fill_inputs(tokens, T, pos0, emb2_h_, rope2_h_);
  rope_addr_ = rope2_h_.addr;
  vk_->begin();
  vk_->copy(emb2_h_, 0, t1_, 0, (size_t)T * E * 4);
  NormPC ne{t1_.addr, 0, mtp_.enorm.addr, t1_.addr, (uint32_t)E, c.eps};
  vk_->dispatch("rmsnorm", &ne, sizeof ne, T, 1, 1, false);
  NormPC nh{hrows, 0, mtp_.hnorm.addr, t2_.addr, (uint32_t)E, c.eps};
  vk_->dispatch("rmsnorm", &nh, sizeof nh, T);
  std::vector<VkBufferCopy> r1, r2;  // cat [T][2E] = [enorm(e) | hnorm(h)]
  for (int t = 0; t < T; t++) {
    r1.push_back({(VkDeviceSize)t * E * 4, (VkDeviceSize)t * 2 * E * 4, (VkDeviceSize)E * 4});
    r2.push_back({(VkDeviceSize)t * E * 4, (VkDeviceSize)(t * 2 + 1) * E * 4, (VkDeviceSize)E * 4});
  }
  vk_->copy_regions(t1_, t0_, r1);
  vk_->copy_regions(t2_, t0_, r2);
  linear(mtp_.eh_proj, t0_.addr, T, x_.addr);
  pend_ = false;
  const bool snap = snapshot_;
  snapshot_ = false;
  layer_attn(mtp_, T, pos0);
  layer_ffn(mtp_, T);
  snapshot_ = snap;
  norm_in(mtp_.shnorm.buf ? mtp_.shnorm : output_norm_, T);
}

// MTP head over the draft vocabulary: greedy draft (and its logits for sampled drafts)
int VkModel::mtp_finish(int n, float* conf, float* logits_host) {
  const Config& c = cfg_;
  const size_t rb = (size_t)c.n_embd * 4;
  vk_->copy(xn_, (n - 1) * rb, mtp_h_, 0, rb);  // chained drafts continue from this row
  const VMat* head = &output_;
  if (!dv_host_.empty()) {
    if (!dvhead_.buf.buf) {
      dvhead_ = output_;
      dvhead_.buf = vk_->alloc((size_t)kDraftVocabCap * output_.rb, output_.buf.device_local ? Mem::Device : Mem::Host);
      dv_uploaded_ = 0;
    }
    if (dv_uploaded_ < dv_host_.size()) {  // gather the new rows of the output head
      std::vector<VkBufferCopy> r;
      for (size_t i = dv_uploaded_; i < dv_host_.size(); i++)
        r.push_back({(VkDeviceSize)dv_host_[i] * output_.rb, (VkDeviceSize)i * output_.rb, (VkDeviceSize)output_.rb});
      vk_->copy_regions(output_.buf, dvhead_.buf, r);
      dv_uploaded_ = dv_host_.size();
    }
    dvhead_.N = (int)dv_host_.size();
    head = &dvhead_;
  }
  const int nl = head->N;
  linear(*head, xn_.addr + (n - 1) * rb, 1, logits_.addr);
  ArgPC ap{logits_.addr, argmax_.addr, (uint32_t)nl};
  vk_->dispatch("argmax", &ap, sizeof ap, 1);
  vk_->copy(argmax_, 0, argmax_h_, 0, 4);
  if (logits_host || conf) vk_->copy(logits_, 0, logits_h_, 0, (size_t)nl * 4);
  vk_->submit();
  int d = ((const int*)argmax_h_.map)[0];
  const float* lg = (const float*)logits_h_.map;
  if (conf) {
    double s = 0;
    for (int i = 0; i < nl; i++) s += std::exp((double)lg[i] - lg[d]);
    *conf = (float)(1.0 / s);
  }
  if (head == &dvhead_) {
    if (logits_host) {
      std::fill(logits_host, logits_host + c.n_vocab, -INFINITY);
      for (int i = 0; i < nl; i++) logits_host[dv_host_[i]] = lg[i];
    }
    d = dv_host_[d];
  } else if (logits_host) {
    memcpy(logits_host, lg, (size_t)nl * 4);
  }
  return d;
}

int VkModel::mtp_draft(int hrow0, const int* tokens, int n, int pos0, float* conf, float* logits_host, bool head) {
  const Config& c = cfg_;
  QE_ASSERT(mtp_on_ && n >= 1 && n <= kMaxTok);
  if (pos0 + n > ctx_len_) QE_FATAL("context overflow: %d positions > KV cache %d", pos0 + n, ctx_len_);
  vk_->begin();
  // mtp_block clobbers the activations: stage the h rows
  vk_->copy(hrow0 < 0 ? carry_ : hid_, hrow0 < 0 ? 0 : (size_t)hrow0 * c.n_embd * 4, hsafe_, 0, (size_t)n * c.n_embd * 4);
  mtp_block(hsafe_.addr, tokens, n, pos0);
  if (!head) { vk_->submit(); return -1; }
  return mtp_finish(n, conf, logits_host);
}

int VkModel::mtp_chain(int token, int pos, float* conf, float* logits_host) {
  if (pos + 1 > ctx_len_) QE_FATAL("context overflow: %d positions > KV cache %d", pos + 1, ctx_len_);
  vk_->begin();
  vk_->copy(mtp_h_, 0, hsafe_, 0, (size_t)cfg_.n_embd * 4);
  mtp_block(hsafe_.addr, &token, 1, pos);
  return mtp_finish(1, conf, logits_host);
}

void VkModel::set_carry_row(int row) {
  vk_->begin();  // recorded; runs with the next submission (mtp_draft / forward)
  vk_->copy(hid_, (size_t)row * cfg_.n_embd * 4, carry_, 0, (size_t)cfg_.n_embd * 4);
}

// commit the first n_keep tokens of the last verify pass: advance the untouched pre-pass conv and
// recurrent states by the saved inputs (every GDN layer in one go, no barriers between layers)
void VkModel::rollback(int n_keep) {
  const Config& c = cfg_;
  QE_ASSERT(snap_valid_ && n_keep >= 1 && n_keep <= snap_T_);
  const int C = c.conv_dim();
  vk_->begin();
  for (auto& L : layers_) {
    if (!L.sv.buf) continue;
    const VkDeviceAddress raw = L.sv.addr, qkv = L.sv.addr + (size_t)kMaxTok * C * 4;
    const VkDeviceAddress gb = L.sv.addr + (size_t)kMaxTok * 2 * C * 4, bb = gb + (size_t)kMaxTok * c.nv * 4;
    RecPC rp{qkv, gb, bb, ssm_.addr + L.ssm_off, 0, (uint32_t)c.nk, (uint32_t)c.nv, (uint32_t)n_keep, (uint32_t)C, 1};
    vk_->dispatch("gdn_rec", &rp, sizeof rp, c.nv, c.dv / 16, 1, false);
    ConvPC cp{raw, conv_.addr + L.conv_off, L.conv_w.addr, 0, (uint32_t)C, (uint32_t)n_keep, (uint32_t)C, 1};
    vk_->dispatch("conv", &cp, sizeof cp, (C + 255) / 256, 1, 1, false);
  }
  vk_->barrier();
  snap_valid_ = false;  // not submitted: runs ahead of the next MTP step / pass in the same submission
}

// draft vocabulary: the first mtp_vocab ids, the special-token tail, every token seen, and every token
// of a script (kana / CJK / hangul / ...) once that script appears (same policy as the CUDA backend)
void VkModel::draft_vocab_add(const int* ids, int n) {
  const Config& c = cfg_;
  if (!mtp_on_ || opt_.mtp_vocab <= 0) return;
  if (dv_has_.empty()) {
    dv_has_.assign(c.n_vocab, 0);
    const int head = std::min(opt_.mtp_vocab, c.n_vocab);
    for (int i = 0; i < head; i++) { dv_has_[i] = 1; dv_host_.push_back(i); }
    for (int i = std::max(head, c.n_vocab - 320); i < c.n_vocab; i++) { dv_has_[i] = 1; dv_host_.push_back(i); }
    dv_cls_.assign(c.n_vocab, 0);
    if (const GGUFValue* tk = gguf_->get("tokenizer.ggml.tokens"))
      for (int i = 0; i < c.n_vocab && i < (int)tk->arr.size(); i++) dv_cls_[i] = (uint8_t)token_script(tk->arr[i].as_str());
  }
  auto add = [&](int t) {
    if (dv_has_[t] || (int)dv_host_.size() >= kDraftVocabCap) return;
    dv_has_[t] = 1;
    dv_host_.push_back(t);
  };
  for (int i = 0; i < n; i++) {
    const int t = ids[i];
    if (t < 0 || t >= c.n_vocab) continue;
    const int cl = dv_cls_[t];
    if (cl && !(dv_cls_on_ & (1u << cl))) {
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

// submit the recorded pass; with co-execution, run the CPU's share of every FFN layer meanwhile
void VkModel::submit_pass() {
  if (!pass_hyb_) { vk_->submit(); return; }
  const double t0 = qe_now();
  vk_->submit_nowait();
  hyb_host_loop();
  vk_->finish();
  pass_hyb_ = false;
  const double dt = qe_now() - t0;
  double& e = hyb_ema_[hyb_level_];
  e = hyb_cnt_[hyb_level_]++ ? 0.8 * e + 0.2 * dt : dt;
}

void VkModel::hyb_host_loop() {
  const Config& c = cfg_;
  const int T = hyb_T_, r = hyb_rows_, n = pool_->size();
  volatile const uint32_t* flags = (volatile const uint32_t*)hyb_flags_.map;
  const float* xn = (const float*)xn_.map;
  float* g = (float*)t0_.map;
  float* u = (float*)t1_.map;
  for (int il = 0; il < (int)layers_.size(); il++) {
    VLayer& L = layers_[il];
    if (!L.ffn_gate.valid()) continue;
    const double tw = qe_now();
    while (flags[il] != hyb_serial_) {
      _mm_pause();
      if (qe_now() - tw > 30.0) QE_FATAL("co-execution: GPU did not reach layer %d", il);
    }
    for (int t = 0; t < T; t++) cpu::quantize(xn, c.n_embd, t, hyb_q_, 0, c.n_embd / 256);
    const GGUFTensor& tg = gguf_->need(bn(il, "ffn_gate.weight"));
    const GGUFTensor& tu = gguf_->need(bn(il, "ffn_up.weight"));
    pool_->run([&](int i) {  // 2r rows (gate then up) split evenly
      const int a = 2 * r * i / n, b = 2 * r * (i + 1) / n;
      if (a < r) cpu::matvec(L.ffn_gate.type, tg.data, L.ffn_gate.rb, c.n_embd, a, std::min(b, r), hyb_q_, T, g, c.n_ff);
      if (b > r) cpu::matvec(L.ffn_up.type, tu.data, L.ffn_up.rb, c.n_embd, std::max(a, r) - r, b - r, hyb_q_, T, u, c.n_ff);
    });
    vk_->set_event(hyb_ev_[il]);
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
  const size_t cb = mtp_on_ ? (size_t)cfg_.n_embd * 4 : 0;  // + MTP carry row
  if (!ck_[slot].buf) ck_[slot] = vk_->alloc(ssm_bytes_ + conv_bytes_ + cb, Mem::Host);
  vk_->begin();
  vk_->copy(ssm_, 0, ck_[slot], 0, ssm_bytes_);
  vk_->copy(conv_, 0, ck_[slot], ssm_bytes_, conv_bytes_);
  if (cb) vk_->copy(carry_, 0, ck_[slot], ssm_bytes_ + conv_bytes_, cb);
  vk_->submit();
}

void VkModel::ck_restore(int slot) {
  if (slot < 0 || slot >= (int)ck_.size() || !ck_[slot].buf) QE_FATAL("vulkan: restore of an empty checkpoint slot %d", slot);
  vk_->begin();
  vk_->copy(ck_[slot], 0, ssm_, 0, ssm_bytes_);
  vk_->copy(ck_[slot], ssm_bytes_, conv_, 0, conv_bytes_);
  if (mtp_on_) vk_->copy(ck_[slot], ssm_bytes_ + conv_bytes_, carry_, 0, (size_t)cfg_.n_embd * 4);
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
  const bool multi = pc.T > 1;
  const std::string ctag = std::string("mvc_") + type_tag(m.type);
  // the cooperative kernel also serves the small K/V projections (1024 rows), which the generic kernel
  // runs in its slow one-row-per-workgroup mode
  const std::string itag = std::string("mvci_") + type_tag(m.type);
  if (tune_.coop && tune_.i8 && m.N >= 256 && xq8_.buf && vk_->has_pipeline(itag.c_str())) {
    // int8 activations: quantize once per input (q/k/v, gate/up share it), then DP4A
    if (!(pc.x == xq_src_ && pc.T == xq_T_ && pc.K == xq_K_ && vk_->ops() == xq_mark_)) {
      ElemPC ep{pc.x, xq8_.addr, xd8_.addr, 0, 0, 0, pc.T * pc.K / 32, 0, 0};
      uint32_t ex, ey;
      grid1d(ep.n, ex, ey);
      vk_->dispatch("quant_q8", &ep, sizeof ep, ex, ey);
      xq_src_ = pc.x; xq_T_ = pc.T; xq_K_ = pc.K;
    }
    pc.wide = 0;
    pc.x = xq8_.addr;
    pc.tab = xd8_.addr;
    const int teams = multi ? tune_.cm_teams : tune_.c_teams, rr = multi ? tune_.cm_r : tune_.c_r;
    const uint32_t rows = (uint32_t)(teams * rr);
    uint32_t gx, gy;
    grid_wg((m.N + rows - 1) / rows, gx, gy);
    PipeOpt po;
    po.spec0 = rr;
    po.spec1 = 32 * teams;
    po.spec2 = multi ? 1 : 0;
    po.subgroup = tune_.mv_sg;
    vk_->dispatch(itag.c_str(), &pc, sizeof pc, gx, gy, 1, bar, po);
    xq_mark_ = vk_->ops();  // only further matvecs on the same input may reuse the codes
    return;
  }
  if (tune_.coop && m.N >= 256 && vk_->has_pipeline(ctag.c_str())) {
    pc.wide = 0;
    const int teams = multi ? tune_.cm_teams : tune_.c_teams, rr = multi ? tune_.cm_r : tune_.c_r;
    const uint32_t rows = (uint32_t)(teams * rr);
    uint32_t gx, gy;
    grid_wg((m.N + rows - 1) / rows, gx, gy);
    PipeOpt po;
    po.spec0 = rr;
    po.spec1 = 32 * teams;
    po.spec2 = multi ? 1 : 0;
    po.subgroup = tune_.mv_sg;
    vk_->dispatch(ctag.c_str(), &pc, sizeof pc, gx, gy, 1, bar, po);
    return;
  }
  const int teams = multi ? tune_.mm_teams : tune_.mv_teams, rr = multi ? tune_.mm_r : tune_.mv_r;
  const uint32_t rows = (uint32_t)(teams * rr);
  uint32_t gx, gy;
  grid_wg(pc.wide ? m.N : (m.N + rows - 1) / rows, gx, gy);
  PipeOpt po;
  po.spec0 = rr;
  po.spec1 = 32 * teams;
  po.subgroup = tune_.mv_sg;
  po.spec2 = multi ? 1 : 0;
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
  const bool staged = method.rfind("cms_", 0) == 0;  // shared-memory staged variant: 32-wide k steps
  const bool cm = two && sscanf(method.c_str(), staged ? "cms_gemm_%ux%ux%u" : "cm_gemm_%ux%ux%u", &M, &N, &K) == 3 &&
                  m.N % (8 * N) == 0 && x16_.buf;
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
    nsteps = m.K / (staged ? 32 : K);
    min_steps = 256 / (staged ? 32 : K);
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
    RopePC rq{qg, rope_addr_, (uint32_t)c.n_head, (uint32_t)(2 * c.hd), (uint32_t)qw, (uint32_t)T, half};
    grid1d((uint64_t)T * c.n_head * half, gx, gy);
    vk_->dispatch("rope", &rq, sizeof rq, gx, gy, 1, false);
    RopePC rk{k, rope_addr_, (uint32_t)c.n_kv, (uint32_t)c.hd, (uint32_t)kvw, (uint32_t)T, half};
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
  // speculative verify: keep this pass' recurrence inputs (rollback commits the accepted prefix) and
  // leave the conv / recurrent state untouched
  const bool snap = snapshot_ && L.sv.buf;
  const VkDeviceAddress raw = snap ? L.sv.addr : t0_.addr;
  const VkDeviceAddress qkv = snap ? L.sv.addr + (size_t)kMaxTok * C * 4 : t3_.addr;
  const VkDeviceAddress gb = snap ? L.sv.addr + (size_t)kMaxTok * 2 * C * 4 : g_.addr;
  const VkDeviceAddress bb = snap ? gb + (size_t)kMaxTok * c.nv * 4 : beta_.addr;
  linear(L.wqkv, xn_.addr, T, raw, false, false);
  linear(L.wgate, xn_.addr, T, t1_.addr, false, false);
  linear(L.walpha, xn_.addr, T, ab, false, false);
  linear(L.wbeta, xn_.addr, T, ab + (size_t)T * c.nv * 4);
  uint32_t gx, gy;
  {
    ConvPC cp{raw, conv_.addr + L.conv_off, L.conv_w.addr, qkv, (uint32_t)C, (uint32_t)T, (uint32_t)C, snap ? 0u : 1u};
    vk_->dispatch("conv", &cp, sizeof cp, (C + 255) / 256);
  }
  {
    HeadNormPC hp{qkv, 0, (uint32_t)c.dk, (uint32_t)(2 * c.nk), (uint32_t)c.dk, (uint32_t)C, 1, c.eps};
    grid_wg((uint64_t)T * 2 * c.nk, gx, gy);
    vk_->dispatch("headnorm", &hp, sizeof hp, gx, gy, 1, false);
    ElemPC ep{ab, ab + (size_t)T * c.nv * 4, L.dt_bias.addr, L.ssm_a.addr, gb, bb, (uint32_t)(T * c.nv),
              (uint32_t)c.nv, 0};
    grid1d(ep.n, gx, gy);
    vk_->dispatch("gdn_gb", &ep, sizeof ep, gx, gy);
  }
  {
    RecPC rp{qkv, gb, bb, ssm_.addr + L.ssm_off, t2_.addr, (uint32_t)c.nk, (uint32_t)c.nv, (uint32_t)T, (uint32_t)C,
             snap ? 0u : 1u};
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
  const int il = (int)(&L - layers_.data());
  if (pass_hyb_ && il >= 0 && il < (int)layers_.size() && hyb_rows_ > 0) {
    // the CPU computes rows [0, r) of gate / up into t0_ / t1_ (it waits for this flag), the GPU
    // the rest; SwiGLU + down wait for the CPU's event
    const int r = hyb_rows_;
    vk_->signal_host(hyb_flags_, (size_t)il * 4, hyb_serial_);
    VMat g = L.ffn_gate, u = L.ffn_up;
    g.buf.addr += (size_t)r * g.rb; g.N -= r;
    u.buf.addr += (size_t)r * u.rb; u.N -= r;
    MvPC pg{g.buf.addr, xn_.addr, t0_.addr + (size_t)r * 4, tab_.addr, (uint32_t)g.K, (uint32_t)g.N, (uint32_t)T,
            (uint32_t)g.rb, (uint32_t)g.K, (uint32_t)c.n_ff, 0, 0, (uint32_t)(g.K / 32)};
    matvec(g, pg, false);
    MvPC pu = pg;
    pu.w = u.buf.addr;
    pu.y = t1_.addr + (size_t)r * 4;
    matvec(u, pu, true);
    vk_->wait_event(hyb_ev_[il]);
  } else {
    linear(L.ffn_gate, xn_.addr, T, t0_.addr, false, false);
    linear(L.ffn_up, xn_.addr, T, t1_.addr);
  }
  uint32_t gx, gy;
  ElemPC ep{t0_.addr, t1_.addr, 0, 0, 0, 0, (uint32_t)(T * c.n_ff), 0, 0};
  grid1d(ep.n, gx, gy);
  vk_->dispatch("silu_mul", &ep, sizeof ep, gx, gy);
  linear(L.ffn_down, t0_.addr, T, t4_.addr);
  pend_ = true;
}

// host-side inputs of a pass: embedding rows (CPU dequant of token_embd) and RoPE angles (fp64)
void VkModel::fill_inputs(const int* tokens, int T, int pos0, const VBuf& embb, const VBuf& rope) {
  const Config& c = cfg_;
  float* emb = (float*)embb.map;
  const GGUFTensor& te = gguf_->need("token_embd.weight");
  for (int t = 0; t < T; t++) {
    if (tokens[t] < 0 || tokens[t] >= tok_embd_.N) QE_FATAL("token id %d out of range", tokens[t]);
    dequant_row_ref(te.type, te.data + (size_t)tokens[t] * tok_embd_.rb, c.n_embd, emb + (size_t)t * c.n_embd);
  }
  const int half = c.rope_dims / 2;
  float* cs = (float*)rope.map;
  for (int t = 0; t < T; t++)
    for (int i = 0; i < half; i++) {
      const double freq = std::pow((double)c.rope_base, -2.0 * i / c.rope_dims);
      const double a = (double)(pos0 + t) * freq;
      cs[2 * (t * half + i)] = (float)std::cos(a);
      cs[2 * (t * half + i) + 1] = (float)std::sin(a);
    }
}

// record one chunk of <= ub_ tokens (embedding, all layers, final norm); not submitted
void VkModel::forward_chunk(const int* tokens, int T, int pos0) {
  const Config& c = cfg_;
  pass_hyb_ = false;
  if (hyb_ && T <= kMaxTok) {
    // share for this pass: measured levels first, then the fastest, neighbours re-measured now and then
    int lv;
    if (hyb_force_ >= 0) lv = hyb_force_;
    else {
      lv = -1;
      for (int k = 0; k < kHybLevels && lv < 0; k++)
        if (hyb_cnt_[k] < 3) lv = k;
      if (lv < 0) {
        int best = 0;
        for (int k = 1; k < kHybLevels; k++)
          if (hyb_ema_[k] < hyb_ema_[best]) best = k;
        if (++hyb_pass_ % 96 == 0) hyb_explore_ = 3, hyb_dir_ = -hyb_dir_;
        lv = hyb_explore_ > 0 ? std::clamp(best + hyb_dir_, 0, kHybLevels - 1) : best;
        if (hyb_explore_ > 0) hyb_explore_--;
      }
    }
    if (lv != hyb_level_) QE_DBG("[auto] CPU share of the FFN gate/up rows: %d/8\n", lv);
    hyb_level_ = lv;
    hyb_rows_ = std::min(c.n_ff / 2, (int)std::lround(lv / 8.0 * c.n_ff / 256.0) * 256);
    pass_hyb_ = hyb_rows_ > 0;
    if (pass_hyb_) {
      hyb_serial_++;
      hyb_T_ = T;
      for (VkEvent e : hyb_ev_) vk_->reset_event(e);
    }
  }
  fill_inputs(tokens, T, pos0, emb_h_, rope_h_);
  rope_addr_ = rope_h_.addr;
  vk_->begin();
  vk_->copy(emb_h_, 0, x_, 0, (size_t)T * c.n_embd * 4);
  pend_ = false;
  // prefill: split the pass into several submissions so no single one runs for seconds (GPU drivers
  // reset jobs that exceed their timeout: Xe / i915 / amdgpu; an iGPU runs a 27B layer in ~0.1 s)
  const int every = T > kMaxTok ? submit_layers_ : 1 << 30;
  int nl = 0;
  for (auto& L : layers_) {
    if (L.attn) layer_attn(L, T, pos0);
    else layer_gdn(L, T);
    layer_ffn(L, T);
    if (++nl % every == 0 && nl < (int)layers_.size()) {
      vk_->submit();
      vk_->begin();
    }
  }
  norm_in(output_norm_, T);
  if (mtp_on_ && T <= kMaxTok) vk_->copy(xn_, 0, hid_, 0, (size_t)T * c.n_embd * 4);  // rows for mtp_draft
}

void VkModel::forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax, bool snapshot,
                      bool mtp_prefill) {
  snapshot_ = snapshot && mtp_on_ && n > 1 && n <= kMaxTok;
  snap_valid_ = snapshot_;
  snap_T_ = n;
  // checkpoint capture inside this pass: split it at the capture position
  if (cap_slot_ >= 0 && cap_pos_ == pos0) {
    ck_save_live(cap_slot_);
    cap_slot_ = -1;
  } else if (cap_slot_ >= 0 && !snapshot_ && cap_pos_ > pos0 && cap_pos_ < pos0 + n && n - (cap_pos_ - pos0) >= n_logits) {
    const int k = cap_pos_ - pos0;
    forward_impl(tokens, k, pos0, nullptr, 0, nullptr, mtp_prefill);
    ck_save_live(cap_slot_);
    cap_slot_ = -1;
    forward_impl(tokens + k, n - k, cap_pos_, logits, n_logits, argmax, mtp_prefill);
    return;
  }
  forward_impl(tokens, n, pos0, logits, n_logits, argmax, mtp_prefill);
  snapshot_ = false;
}

void VkModel::forward_impl(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax,
                           bool mtp_prefill) {
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
        submit_pass();
        if (logits) memcpy(logits + (size_t)(r - first) * c.n_vocab, logits_h_.map, (size_t)nb * c.n_vocab * 4);
        if (argmax) memcpy(argmax + (r - first), argmax_h_.map, nb * 4);
      }
    }
    submit_pass();
    if (mtp_prefill && mtp_on_) {
      // MTP KV for this chunk: pairs (h_{p-1}, x_p) = rows [carry, xn_0 .. xn_{T-2}]; carry := xn_{T-1}.
      // Position 0 has no h_{-1}: a zero row (its MTP KV slot must still be written)
      const size_t rb = (size_t)c.n_embd * 4;
      vk_->begin();
      if (pos0 + b0 == 0) vk_->fill(hm_, 0, rb);
      else vk_->copy(carry_, 0, hm_, 0, rb);
      if (T > 1) vk_->copy(xn_, 0, hm_, rb, (T - 1) * rb);
      vk_->copy(xn_, (T - 1) * rb, carry_, 0, rb);
      mtp_block(hm_.addr, tokens + b0, T, pos0 + b0);
      vk_->submit();
    }
  }
  max_pos_ = std::max(max_pos_, pos0 + n);
}
