#include "vk_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "common.h"
#include "gguf.h"
#include "quant.h"

namespace {

// push-constant blocks (must match the shaders; 64-bit addresses are uvec2 in GLSL)
struct MvPC { uint64_t w, x, y, tab; uint32_t K, N, T, rb, ldx, ldy, acc, wide, kg; };
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
  QE_LOG("[hw] Vulkan device %s (%s), subgroup %u, VRAM %.1f GiB%s\n", vk_->device_name.c_str(),
         vk_->driver_name.c_str(), vk_->subgroup_size, vk_->vram_total() / 1073741824.0,
         vk_->integrated ? " (integrated: shared memory)" : "");
  kv_q8_ = opt.kv_type != 0;
  if (opt.kv_type == 2 || opt.kv_type == 3) QE_LOG("[plan] vulkan: KV cache q4/q3 not implemented, using q8\n");
  kv_row_bytes_ = kv_q8_ ? (size_t)c.hd + c.hd / 8 : (size_t)c.hd * 2;
  ub_ = std::clamp(opt.ubatch, 64, 512);
  mv_max_t_ = getenv("QE_VK_MV_MAXT") ? atoi(getenv("QE_VK_MV_MAXT")) : 8;
  const size_t MiB = 1 << 20;

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
  use_w16_ = !(getenv("QE_VK_GEMM") && !strcmp(getenv("QE_VK_GEMM"), "fused"));
  if (use_w16_) {  // f16 copy of the largest layer matrix (two-stage prefill GEMM)
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
      int kmax = std::max({c.n_embd, c.n_ff, c.d_inner, c.n_head * c.hd});
      x16_ = vk_->alloc((size_t)ub_ * kmax * 2, Mem::Device, true);
      use_cm_ = cm_selftest();
      QE_LOG("[hw] Vulkan cooperative matrices (16x16x16 f16): %s\n", use_cm_ ? "on" : "self-test failed, off");
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
    const size_t reserve = (size_t)opt.vram_reserve_mb * MiB + 256 * MiB;
    const size_t avail = vk_->vram_free() > reserve ? vk_->vram_free() - reserve : 0;
    ctx_len_ = opt.ctx;
    if ((size_t)ctx_len_ * per_pos > avail && !vk_->integrated) {
      const int fit = (int)std::max<size_t>(avail / per_pos / 1024 * 1024, 4096);
      QE_LOG("[plan] KV cache for %d positions (%.0f MiB) does not fit in VRAM: context %d\n", ctx_len_,
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
  vk_->warm_pipelines();
  reset();
  QE_LOG("[plan] loaded in %.1f s\n", qe_now() - t_start);
}

VkModel::~VkModel() {
  vk_.reset();  // device teardown frees every allocation
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
  const size_t reserve = ((size_t)opt_.vram_reserve_mb << 20) + (128u << 20);
  const bool fits = vk_->integrated || vk_->vram_free() > bytes + reserve;
  m.buf = vk_->alloc(bytes, prefer_device && fits ? Mem::Device : Mem::Host);
  upload(m.buf, t.data, bytes);
  return m;
}

VBuf VkModel::load_f32(const std::string& name) {
  const GGUFTensor& t = gguf_->need(name);
  std::vector<float> v(t.nelems());
  dequant_row_ref(t.type, t.data, t.nelems(), v.data());
  VBuf b = vk_->alloc(v.size() * 4, Mem::Device, true);
  upload(b, v.data(), v.size() * 4);
  return b;
}

// run one GEMM through the cooperative-matrix path and the scalar f16 path and compare
bool VkModel::cm_selftest() {
  const int K = 512, N = 128, T = 70;
  std::vector<uint16_t> w((size_t)N * K);
  std::vector<float> x((size_t)T * K), wf((size_t)N * K);
  uint32_t s = 12345;
  auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 9) & 0xffff) / 32768.f - 1.f; };
  for (size_t i = 0; i < w.size(); i++) { w[i] = f2h(rnd()); wf[i] = qe_h2f(w[i]); }
  for (auto& v : x) v = rnd();
  VBuf bw = vk_->alloc(w.size() * 2, Mem::Host), bx = vk_->alloc(x.size() * 4, Mem::Host);
  VBuf by = vk_->alloc((size_t)T * N * 4, Mem::Host);
  memcpy(bw.map, w.data(), w.size() * 2);
  memcpy(bx.map, x.data(), x.size() * 4);
  // shaped like the f16 scratch path: W already f16, so call cm_gemm directly after to_f16
  vk_->begin();
  const uint32_t tp = (T + 63) / 64 * 64;
  ElemPC ep{bx.addr, x16_.addr, 0, 0, 0, 0, (uint32_t)(tp * K / 2), (uint32_t)(T * K / 2), 0};
  uint32_t ex, ey;
  grid1d(ep.n, ex, ey);
  vk_->dispatch("to_f16", &ep, sizeof ep, ex, ey);
  MvPC pc{bw.addr, x16_.addr, by.addr, tab_.addr, (uint32_t)K, (uint32_t)N, (uint32_t)T, 0, (uint32_t)K, (uint32_t)N, 0, 0,
          (uint32_t)(K / 16)};
  vk_->dispatch("cm_gemm", &pc, sizeof pc, N / 128, tp / 64, 1);
  vk_->submit();
  double e = 0, rms = 0;
  for (int t = 0; t < T; t++)
    for (int n = 0; n < N; n++) {
      double r = 0;
      for (int k = 0; k < K; k++) r += (double)wf[(size_t)n * K + k] * qe_h2f(f2h(x[(size_t)t * K + k]));
      rms += r * r;
      e = std::max(e, std::fabs(((float*)by.map)[(size_t)t * N + n] - r));
    }
  rms = std::sqrt(rms / (T * N));
  vk_->free(bw); vk_->free(bx); vk_->free(by);
  QE_DBG("[vk] cooperative-matrix self-test: max err / rms = %.2e\n", e / rms);
  return std::isfinite(e) && e / rms < 1e-3;
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
void VkModel::linear(const VMat& m, VkDeviceAddress x, int T, VkDeviceAddress y, bool acc, bool bar) {
  MvPC pc{m.buf.addr, x, y, tab_.addr, (uint32_t)m.K, (uint32_t)m.N, (uint32_t)T, (uint32_t)m.rb, (uint32_t)m.K,
          (uint32_t)m.N, acc ? 1u : 0u, 0u, (uint32_t)(m.K / 32)};
  const std::string tag = type_tag(m.type);
  if (T <= kMaxTok) {
    uint32_t gx, gy;
    pc.wide = m.N < 2048;  // few rows: one row per workgroup so the GPU is filled
    grid_wg(pc.wide ? m.N : (m.N + 15) / 16, gx, gy);
    vk_->dispatch(("mv_" + tag).c_str(), &pc, sizeof pc, gx, gy, 1, bar);  // 16 rows per workgroup
  } else if (T <= mv_max_t_) {
    // short prompts: 8-token matvec slices beat a mostly empty 64-token GEMM tile
    uint32_t gx, gy;
    pc.wide = m.N < 2048;
    grid_wg(pc.wide ? m.N : (m.N + 15) / 16, gx, gy);
    for (int t0 = 0; t0 < T; t0 += kMaxTok) {
      MvPC p2 = pc;
      p2.T = (uint32_t)std::min(kMaxTok, T - t0);
      p2.x = x + (size_t)t0 * m.K * 4;
      p2.y = y + (size_t)t0 * m.N * 4;
      vk_->dispatch(("mv_" + tag).c_str(), &p2, sizeof p2, gx, gy, 1, bar && t0 + kMaxTok >= T);
    }
  } else {
    // two-stage (coalesced dequant to f16 once, then an f16 GEMM) unless the matrix is too big for
    // the scratch (the output head, which only ever sees <= 16 rows) or QE_VK_GEMM=fused
    const bool two = use_w16_ && (size_t)m.N * m.K * 2 <= w16_.size;
    if (two) {
      MvPC dq = pc;
      dq.y = w16_.addr;
      uint32_t dx, dy;
      const uint64_t groups = (uint64_t)m.N * (m.K / 32);
      dx = (uint32_t)std::min<uint64_t>((groups + 63) / 64, 65535);
      dy = (uint32_t)(((groups + 63) / 64 + 65534) / 65535);
      vk_->dispatch(("dq16_" + tag).c_str(), &dq, sizeof dq, dx, dy);
      pc.w = w16_.addr;
    }
    if (two && use_cm_ && m.N % 128 == 0) {  // matrix hardware: activations to f16 (rows padded to 64)
      const uint32_t tp = (T + 63) / 64 * 64;
      ElemPC ep{x, x16_.addr, 0, 0, 0, 0, (uint32_t)((size_t)tp * m.K / 2), (uint32_t)((size_t)T * m.K / 2), 0};
      uint32_t ex, ey;
      grid1d(ep.n, ex, ey);
      vk_->dispatch("to_f16", &ep, sizeof ep, ex, ey);
      pc.x = x16_.addr;
      const uint32_t gx = m.N / 128, gy = tp / 64, nk = m.K / 16;
      uint32_t ks = 1;
      while (gx * gy * ks * 2 <= 512 && nk / (ks * 2) >= 16 && (size_t)ks * 2 * T * m.N * 4 <= ksplit_.size) ks *= 2;
      pc.kg = (nk + ks - 1) / ks;
      ks = (nk + pc.kg - 1) / pc.kg;
      if (ks == 1) {
        vk_->dispatch("cm_gemm", &pc, sizeof pc, gx, gy, 1, true);
      } else {
        MvPC p2 = pc;
        p2.y = ksplit_.addr;
        vk_->dispatch("cm_gemm", &p2, sizeof p2, gx, gy, ks);
        ElemPC ep2{ksplit_.addr, y, 0, 0, 0, 0, (uint32_t)(T * m.N), ks, acc ? 1u : 0u};
        grid1d(ep2.n, ex, ey);
        vk_->dispatch("ksum", &ep2, sizeof ep2, ex, ey, 1, true);
      }
      return;
    }
    const char* gemm = two ? "gemm16" : nullptr;
    const std::string fused = "mm_" + tag;
    if (!gemm) gemm = fused.c_str();
    const uint32_t gx = (m.N + 63) / 64, gy = (T + 63) / 64, ng = m.K / 32;
    // split K until ~512 workgroups are in flight (keeps >= 8 groups per slice, scratch-bounded)
    uint32_t ks = 1;
    while (gx * gy * ks * 2 <= 512 && ng / (ks * 2) >= 8 && (size_t)ks * 2 * T * m.N * 4 <= ksplit_.size) ks *= 2;
    if (two) bar = true;  // the f16 scratch is reused by the next projection
    if (ks == 1) {
      vk_->dispatch(gemm, &pc, sizeof pc, gx, gy, 1, bar);
    } else {
      pc.kg = (ng / ks + 1) & ~1u;  // even: the fused shader steps two groups at a time
      ks = (ng + pc.kg - 1) / pc.kg;
      MvPC p2 = pc;
      p2.y = ksplit_.addr;
      vk_->dispatch(gemm, &p2, sizeof p2, gx, gy, ks);
      ElemPC ep{ksplit_.addr, y, 0, 0, 0, 0, (uint32_t)(T * m.N), ks, acc ? 1u : 0u};
      uint32_t ex, ey;
      grid1d(ep.n, ex, ey);
      vk_->dispatch("ksum", &ep, sizeof ep, ex, ey, 1, bar);
    }
  }
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
      nsplit = std::clamp<uint32_t>((L_keys + 255) / 256, 1, kMaxSplit);
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
