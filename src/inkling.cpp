// Inkling 2 (inkling2_loop) CUDA runtime.  Reference: llama.cpp src/models/inkling2.cpp, inkling2-loop.cpp.
#include "inkling.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.h"
#include "inkling_kernels.h"
#include "kernels.h"
#include "quant.h"

#define CUBLAS_CHECK(x)                                                                     \
  do {                                                                                      \
    cublasStatus_t st_ = (x);                                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS) QE_FATAL("cuBLAS error %d at %s", (int)st_, #x);      \
  } while (0)

bool is_inkling_arch(const std::string& arch) { return arch == "inkling2_loop"; }

namespace {
std::string bn(int il, const char* s) { return "blk." + std::to_string(il) + "." + s; }
int first_int(const GGUFValue* v, int def) {
  if (!v) return def;
  if (v->is_array()) return v->arr.empty() ? def : (int)v->arr[0].as_int();
  return (int)v->as_int();
}
}  // namespace

void* InklingModel::dalloc(size_t bytes) {
  void* p = nullptr;
  if (bytes == 0) bytes = 16;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  allocs_.push_back(p);
  vram_used_ += bytes;
  return p;
}

float* InklingModel::upload_f32(const std::string& name, bool required) {
  const GGUFTensor* t = gguf_->tensor(name);
  if (!t) {
    if (required) QE_FATAL("missing tensor %s", name.c_str());
    return nullptr;
  }
  std::vector<float> h(t->nelems());
  if (t->type == QT_F32) memcpy(h.data(), t->data, h.size() * 4);
  else if (t->type == QT_F16) for (size_t i = 0; i < h.size(); i++) h[i] = qe_h2f(((const uint16_t*)t->data)[i]);
  else if (t->type == QT_BF16) for (size_t i = 0; i < h.size(); i++) h[i] = qe_bf2f(((const uint16_t*)t->data)[i]);
  else {
    const int64_t K = t->ne[0];
    const size_t rb = qtype_row_bytes(t->type, K);
    for (int64_t r = 0; r < t->rows(); r++) dequant_row_ref(t->type, t->data + r * rb, K, h.data() + r * K);
  }
  float* d = (float*)dalloc(h.size() * 4);
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  return d;
}

InklingModel::W InklingModel::upload_w(const std::string& name, bool required) {
  W w;
  const GGUFTensor* t = gguf_->tensor(name);
  if (!t) {
    if (required) QE_FATAL("missing tensor %s", name.c_str());
    return w;
  }
  w.K = (int)t->ne[0];
  w.N = (int)t->rows();
  if (t->type == QT_F32 || t->type == QT_F16 || t->type == QT_BF16) {
    w.type = QT_F32;
    w.rb = (size_t)w.K * 4;
    w.dev = upload_f32(name);
  } else {
    w.type = t->type;
    w.rb = qtype_row_bytes(t->type, w.K);
    void* d = dalloc(t->nbytes);
    CUDA_CHECK(cudaMemcpy(d, t->data, t->nbytes, cudaMemcpyHostToDevice));
    w.dev = d;
  }
  return w;
}

InklingModel::Experts InklingModel::load_experts(const std::string& name, const std::string& scale_name) {
  Experts x;
  const GGUFTensor& t = gguf_->need(name);
  QE_ASSERT(t.ne.size() == 3 && t.ne[2] == n_exp_);
  x.type = t.type;
  x.K = (int)t.ne[0];
  x.N = (int)t.ne[1];
  x.rb = qtype_row_bytes(t.type, x.K);
  x.stride = x.rb * x.N;
  QE_ASSERT(x.stride * n_exp_ == t.nbytes);
  if (gguf_->tensor(scale_name)) x.scale = upload_f32(scale_name);
  return x;
}

InklingModel::InklingModel(const EngineOptions& opt) : opt_(opt) {
  gguf_ = std::make_unique<GGUFFile>(opt.model_path);
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking));
  for (int b = 0; b < 2; b++) {
    CUDA_CHECK(cudaEventCreateWithFlags(&stg_ready_[b], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stg_free_[b], cudaEventDisableTiming));
  }
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, s_));
  int dev = 0, nsm = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  qk::set_num_sms(nsm);
  load();
}

InklingModel::~InklingModel() {
  if (s_) cudaStreamSynchronize(s_);
  for (auto& g : graph_)
    if (g) cudaGraphExecDestroy(g);
  for (void* p : allocs_) cudaFree(p);
  for (void* p : host_allocs_) cudaFreeHost(p);
  if (cublas_) cublasDestroy(cublas_);
  if (s_) cudaStreamDestroy(s_);
}

void InklingModel::load() {
  const GGUFFile& g = *gguf_;
  const std::string a = g.get_str("general.architecture", "");
  if (!is_inkling_arch(a)) QE_FATAL("not an inkling2_loop model: %s", a.c_str());
  auto key = [&](const char* k) { return a + "." + k; };
  auto need_int = [&](const char* k) {
    const GGUFValue* v = g.get(key(k));
    if (!v) QE_FATAL("missing GGUF key %s", key(k).c_str());
    return first_int(v, 0);
  };
  auto need_f = [&](const char* k) {
    const GGUFValue* v = g.get(key(k));
    if (!v) QE_FATAL("missing GGUF key %s", key(k).c_str());
    return (float)v->as_float();
  };
  n_phys_ = need_int("block_count");
  E_ = need_int("embedding_length");
  nh_ = need_int("attention.head_count");
  nkv_ = need_int("attention.head_count_kv");
  hd_ = need_int("attention.key_length");
  n_exp_ = need_int("expert_count");
  k_exp_ = need_int("expert_used_count");
  n_ff_ = need_int("expert_feed_forward_length");
  core_start_ = need_int("loop.start");
  core_end_ = need_int("loop.end");
  n_pass_ = need_int("loop.count");
  gate_rank_ = need_int("loop.gate_rank");
  n_vocab_pad_ = need_int("vocab_size");
  n_vocab_ = need_int("vocab_size_unpadded");
  swa_ = need_int("attention.sliding_window");
  n_conv_ = need_int("shortconv.l_cache");
  block_ = need_int("attn_res.block_size");
  n_rel_ = need_int("attention.relative_length");
  rel_extent_ = need_int("attention.relative_extent");
  eps_ = need_f("attention.layer_norm_rms_epsilon");
  qk_clip_ = need_f("attention.qk_clip");
  rel_clip_ = need_f("attention.relative_clip");
  log_floor_ = need_f("attention.log_scale_floor");
  log_alpha_ = need_f("attention.log_scale_alpha");
  route_scale_ = (float)g.get_float(key("expert_weights_scale"), 1.0);
  situ_b_ = need_f("activation.situ_beta");
  situ_lb_ = need_f("activation.situ_linear_beta");
  softcap_ = (float)g.get_float(key("final_logit_softcapping"), 0.0);
  if (g.get(key("loop.varying")) && g.get(key("loop.varying"))->as_int()) QE_FATAL("inkling2_loop: varying loops are not supported");
  if (hd_ != 128 || n_rel_ != 16 || n_conv_ != 4 || nh_ % nkv_ || n_exp_ > 256 || k_exp_ > 16)
    QE_FATAL("inkling2_loop: unsupported dimensions (head %d, rel %d, conv %d, heads %d/%d, experts %d/%d)", hd_, n_rel_,
             n_conv_, nh_, nkv_, n_exp_, k_exp_);
  const GGUFValue* pat = g.get(key("attention.sliding_window_pattern"));
  if (!pat || (int)pat->arr.size() != n_phys_) QE_FATAL("inkling2_loop: missing sliding_window_pattern");

  cfg_.arch = a;
  cfg_.n_embd = E_;
  cfg_.n_layer = n_phys_ + (n_pass_ - 1) * (core_end_ - core_start_);
  cfg_.n_ff = n_ff_;
  cfg_.n_head = nh_;
  cfg_.n_kv = nkv_;
  cfg_.hd = hd_;
  cfg_.n_vocab = n_vocab_;
  cfg_.eps = eps_;
  cfg_.attn_interval = 1;

  U_ = std::clamp(opt_.ubatch, 64, 1024);
  ctx_ = opt_.ctx;

  // ---- dense weights
  L_.resize(n_phys_);
  for (int il = 0; il < n_phys_; il++) {
    PLayer& L = L_[il];
    L.idx = il;
    L.swa = pat->arr[il].as_int() != 0;
    L.q = upload_w(bn(il, "attn_q.weight"));
    L.k = upload_w(bn(il, "attn_k.weight"));
    L.v = upload_w(bn(il, "attn_v.weight"));
    L.o = upload_w(bn(il, "attn_output.weight"));
    L.gate = upload_w(bn(il, "attn_gate.weight"));
    L.rel = upload_w(bn(il, "attn_rel.weight"));
    L.sink = upload_w(bn(il, "attn_sink_proj.weight"));
    L.q_clip = upload_f32(bn(il, "attn_q_clip.weight"));
    L.rel_clip = upload_f32(bn(il, "attn_rel_clip.weight"));
    L.rel_table = upload_f32(bn(il, "attn_rel_bias.weight"));
    L.k_conv = upload_f32(bn(il, "attn_k_conv.weight"));
    L.v_conv = upload_f32(bn(il, "attn_v_conv.weight"));
    L.ao_conv = upload_f32(bn(il, "attn_post_conv.weight"));
    L.fo_conv = upload_f32(bn(il, "ffn_post_conv.weight"));
    if (L.swa) {
      L.attn_norm = upload_f32(bn(il, "attn_norm.weight"));
    } else {
      L.qn = upload_f32(bn(il, "attn_q_inp_norm.weight"));
      L.kn = upload_f32(bn(il, "attn_k_inp_norm.weight"));
      L.vn = upload_f32(bn(il, "attn_v_inp_norm.weight"));
      L.kcp = upload_w(bn(il, "attn_k_conv_proj.weight"));
      L.vcp = upload_w(bn(il, "attn_v_conv_proj.weight"));
    }
    if (il > 0) {
      const GGUFTensor& rq = g.need(bn(il, "attn_res_query.weight"));
      L.attn_nq = rq.ne.size() > 1 ? (int)rq.ne[1] : 1;
      L.attn_rq = upload_f32(bn(il, "attn_res_query.weight"));
    }
    L.ffn_rq = upload_f32(bn(il, "ffn_res_query.weight"));
    L.ffn_norm = upload_f32(bn(il, "ffn_norm.weight"));
    L.gate_inp = upload_w(bn(il, "ffn_gate_inp.weight"));
    L.exp_b = upload_f32(bn(il, "exp_probs_b.bias"), false);
    L.gu = load_experts(bn(il, "ffn_gate_up_exps.weight"), bn(il, "ffn_gate_up_exps.scale"));
    L.dn = load_experts(bn(il, "ffn_down_exps.weight"), bn(il, "ffn_down_exps.scale"));
    QE_ASSERT(L.gu.N == 2 * n_ff_ && L.gu.K == E_ && L.dn.K == n_ff_ && L.dn.N == E_);
    if (il >= core_start_ && il < core_end_) {
      L.msg_q = upload_f32(bn(il, "loop_msg_query.weight"));
      static const char* nm[4] = {"q", "k", "v", "ffn"};
      for (int c = 0; c < 4; c++) {
        L.cdown[c] = upload_w(bn(il, (std::string("loop_") + nm[c] + "_ctrl_down.weight").c_str()));
        // gain and forget share their input: one stacked [2E][rank] f32 matrix
        const std::string gn = bn(il, (std::string("loop_") + nm[c] + "_ctrl_gain.weight").c_str());
        const std::string fn = bn(il, (std::string("loop_") + nm[c] + "_ctrl_forget.weight").c_str());
        const GGUFTensor& tg = g.need(gn);
        const GGUFTensor& tf = g.need(fn);
        QE_ASSERT(tg.type == QT_F32 && tf.type == QT_F32 && tg.ne[0] == tf.ne[0] && tg.rows() == tf.rows());
        W gf;
        gf.type = QT_F32;
        gf.K = (int)tg.ne[0];
        gf.N = (int)(tg.rows() + tf.rows());
        gf.rb = (size_t)gf.K * 4;
        uint8_t* d = (uint8_t*)dalloc(tg.nbytes + tf.nbytes);
        CUDA_CHECK(cudaMemcpy(d, tg.data, tg.nbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d + tg.nbytes, tf.data, tf.nbytes, cudaMemcpyHostToDevice));
        gf.dev = d;
        L.cgf[c] = gf;
      }
    }
  }
  tok_embd_ = &g.need("token_embd.weight");
  tok_norm_ = upload_f32("token_embd_norm.weight");
  output_norm_ = upload_f32("output_norm.weight");
  output_rq_ = upload_f32("output_res_query.weight");
  {
    // only the unpadded vocabulary rows of the head
    const GGUFTensor& t = g.need("output.weight");
    output_.type = t.type;
    output_.K = (int)t.ne[0];
    output_.N = n_vocab_;
    output_.rb = qtype_row_bytes(t.type, output_.K);
    const size_t bytes = output_.rb * n_vocab_;
    void* d = dalloc(bytes);
    CUDA_CHECK(cudaMemcpy(d, t.data, bytes, cudaMemcpyHostToDevice));
    output_.dev = d;
  }
  const double dense_mb = vram_used_ / 1048576.0;

  // ---- cache slots: KV + convolution state per layer application
  const int n_slot = cfg_.n_layer;
  slots_.resize(n_slot);
  // KV format: --kv f16 / q8 / q4 (q3 -> q4)
  kv_bits_ = opt_.kv_type == 0 ? 16 : opt_.kv_type == 1 ? 8 : 4;
  if (const char* e = getenv("QE_INK_KV")) kv_bits_ = atoi(e);
  kv_rowb_ = ik::kv_row_bytes(nkv_ * hd_, kv_bits_);
  const int kvsz = nkv_ * hd_;
  const size_t hist_floats = 3 * (size_t)(2 * kvsz + 2 * E_);
  size_t kv_bytes = 0;
  for (int s = 0; s < n_slot; s++) {
    Slot& S = slots_[s];
    S.phys = s < n_phys_ ? s : core_start_ + (s - n_phys_) % (core_end_ - core_start_);
    const bool swa = L_[S.phys].swa;
    S.ring = swa ? swa_ + U_ : 0;
    const size_t rows = swa ? (size_t)S.ring : (size_t)ctx_;
    S.kc = (uint8_t*)dalloc(rows * kv_rowb_);
    S.vc = (uint8_t*)dalloc(rows * kv_rowb_);
    kv_bytes += 2 * rows * kv_rowb_;
    S.hist = (float*)dalloc(hist_floats * 4);
  }
  {
    // speculative verify: per slot and conv, [3 + kMaxTok][C] saved inputs and the rollback table
    std::vector<ik::ConvSave> tab;
    save_.resize(n_slot);
    const int C[4] = {kvsz, kvsz, E_, E_};
    const size_t hoff[4] = {0, 3 * (size_t)kvsz, 6 * (size_t)kvsz, 6 * (size_t)kvsz + 3 * (size_t)E_};
    for (int s = 0; s < n_slot; s++)
      for (int p = 0; p < 4; p++) {
        save_[s][p] = (float*)dalloc((size_t)(3 + kMaxTok) * C[p] * 4);
        tab.push_back({save_[s][p], slots_[s].hist + hoff[p], C[p]});
      }
    n_csave_ = (int)tab.size();
    csave_d_ = (ik::ConvSave*)dalloc(tab.size() * sizeof(ik::ConvSave));
    CUDA_CHECK(cudaMemcpy(csave_d_, tab.data(), tab.size() * sizeof(ik::ConvSave), cudaMemcpyHostToDevice));
  }

  // ---- workspace
  const size_t UE = (size_t)U_ * E_;
  for (int i = 0; i < 10; i++) pool_.push_back((float*)dalloc(UE * 4));
  for (int j = 0; j < 3; j++) {
    mixed_[j] = (float*)dalloc(UE * 4);
    ins_[j] = (float*)dalloc(UE * 4);
    xn_[j] = (float*)dalloc(UE * 4);
  }
  for (int j = 0; j < 4; j++) msgs_[j] = (float*)dalloc(UE * 4);
  const size_t qsz = (size_t)U_ * nh_ * hd_;
  q_ = (float*)dalloc(qsz * 4);
  g_ = (float*)dalloc(qsz * 4);
  att_ = (float*)dalloc(qsz * 4);
  kp_ = (float*)dalloc((size_t)U_ * kvsz * 4);
  vp_ = (float*)dalloc((size_t)U_ * kvsz * 4);
  k_ = (float*)dalloc((size_t)U_ * kvsz * 4);
  v_ = (float*)dalloc((size_t)U_ * kvsz * 4);
  r_ = (float*)dalloc((size_t)U_ * nh_ * n_rel_ * 4);
  sink_ = (float*)dalloc((size_t)U_ * nh_ * 4);
  kd_ = (float*)dalloc((size_t)U_ * nkv_ * n_conv_ * 4);
  vd_ = (float*)dalloc((size_t)U_ * nkv_ * n_conv_ * 4);
  o_ = (float*)dalloc(UE * 4);
  o2_ = (float*)dalloc(UE * 4);
  gcat_ = (float*)dalloc(2 * UE * 4);
  gd_ = (float*)dalloc((size_t)U_ * gate_rank_ * 4);
  gg_ = (float*)dalloc(2 * UE * 4);
  gf_ = (float*)dalloc(UE * 4);
  attn_work_bytes_ = 16;
  for (int t = 1; t <= kMaxTok; t++) attn_work_bytes_ = std::max(attn_work_bytes_, ik::attn_work_bytes(t, nh_, nkv_, ctx_));
  attn_work_ = dalloc(attn_work_bytes_);
  const size_t A = (size_t)U_ * k_exp_;
  logit_e_ = (float*)dalloc((size_t)U_ * n_exp_ * 4);
  w_e_ = (float*)dalloc(A * 4);
  coef_ = (float*)dalloc(A * 4);
  ids_ = (int*)dalloc(A * 4);
  tok_d_ = (int*)dalloc(A * 4);
  eid_d_ = (int*)dalloc(A * 4);
  inv_d_ = (int*)dalloc(A * 4);
  goff_d_ = (int*)dalloc((n_exp_ + 1) * 4);
  {
    // prefill-only buffers in one region (lent to the expert cache between prefills)
    w16_elems_ = (size_t)std::max(nh_ * hd_, E_) * std::max(nh_ * hd_, E_);
    a16_elems_ = std::max(A * n_ff_, (size_t)U_ * nh_ * hd_);
    const size_t parts[] = {A * 2 * n_ff_ * 4, A * E_ * 4, A * n_ff_ * 4, A * E_, A * E_ / 32 * 4, A * n_ff_, A * n_ff_ / 32 * 4,
                            w16_elems_ * 2, a16_elems_ * 2};
    size_t off[9], b = 0;
    for (int i = 0; i < 9; i++) {
      off[i] = b;
      b += (parts[i] + 255) & ~(size_t)255;
    }
    big_bytes_ = b;
    big_ = (uint8_t*)dalloc(b);
    ygu_ = (float*)(big_ + off[0]);
    yd_ = (float*)(big_ + off[1]);
    h_ = (float*)(big_ + off[2]);
    xq_ = (int8_t*)(big_ + off[3]);
    xd_ = (float*)(big_ + off[4]);
    xq2_ = (int8_t*)(big_ + off[5]);
    xd2_ = (float*)(big_ + off[6]);
    w16_ = (__half*)(big_ + off[7]);
    a16_ = (__half*)(big_ + off[8]);
    // small-batch copies (decode / verify: <= kMaxTok tokens, kMaxTok * k_exp_ assignments)
    const size_t As = (size_t)kMaxTok * k_exp_;
    const size_t xs = std::max((size_t)kMaxTok * std::max(nh_ * hd_, 2 * E_), As * n_ff_);
    xqs_ = (int8_t*)dalloc(xs);
    xds_ = (float*)dalloc(xs / 32 * 4);
    ygus_ = (float*)dalloc(As * 2 * n_ff_ * 4);
    yds_ = (float*)dalloc(As * E_ * 4);
    hs_ = (float*)dalloc(As * n_ff_ * 4);
  }
  logits_d_ = (float*)dalloc((size_t)kMaxTok * n_vocab_ * 4);
  argmax_d_ = (int*)dalloc(kMaxTok * 4);
  CUDA_CHECK(cudaHostAlloc((void**)&ids_h_, 4 * A * 4, cudaHostAllocDefault));
  host_allocs_.push_back(ids_h_);
  perm_h_ = ids_h_ + A;
  CUDA_CHECK(cudaHostAlloc((void**)&embd_h_, UE * 4, cudaHostAllocDefault));
  host_allocs_.push_back(embd_h_);
  CUDA_CHECK(cudaHostAlloc((void**)&out_h_, (size_t)kMaxTok * n_vocab_ * 4 + 64 * 4, cudaHostAllocDefault));
  host_allocs_.push_back(out_h_);
  argmax_h_ = (int*)(out_h_ + (size_t)kMaxTok * n_vocab_);
  pos_h_ = argmax_h_ + 16;
  pos_d_ = (int*)dalloc(64);
  size_t max_gu = 0, max_dn = 0;
  for (auto& L : L_) {
    max_gu = std::max(max_gu, L.gu.stride);
    max_dn = std::max(max_dn, L.dn.stride);
  }
  stg_slots_ = 2 * k_exp_;
  stg_gu_ = (uint8_t*)dalloc(max_gu * stg_slots_);
  stg_dn_ = (uint8_t*)dalloc(max_dn * stg_slots_);
  const double ws_mb = vram_used_ / 1048576.0 - dense_mb - kv_bytes / 1048576.0;

  // checkpoints: every slot's conv state + the last swa_ positions of each local slot's KV ring
  {
    size_t b = 0;
    for (auto& S : slots_) b += hist_floats * 4 + (S.ring ? 2 * (size_t)swa_ * kv_rowb_ : 0);
    ck_bytes_ = b;
    const int n = std::clamp(opt_.ck_slots, 0, 4);
    for (int i = 0; i < n; i++) {
      Ck c;
      CUDA_CHECK(cudaHostAlloc((void**)&c.host, ck_bytes_, cudaHostAllocDefault));
      host_allocs_.push_back(c.host);
      cks_.push_back(c);
    }
  }
  plan_experts();
  QE_LOG("[inkling] %d layers (%d applied: core [%d,%d) x %d), %d experts top-%d, vocab %d, ctx %d, ubatch %d, KV %s\n", n_phys_,
         cfg_.n_layer, core_start_, core_end_, n_pass_, n_exp_, k_exp_, n_vocab_, ctx_, U_,
         kv_bits_ == 16 ? "f16" : kv_bits_ == 8 ? "q8" : "q4");
  QE_LOG("[inkling] VRAM: dense %.0f MiB, KV %.0f MiB, workspace %.0f MiB, experts resident in %d/%d layers, expert cache %d "
         "slots (total %.0f MiB)\n",
         dense_mb, kv_bytes / 1048576.0, ws_mb, ndev_layers_, n_phys_, cache_slots_, vram_used_ / 1048576.0);
  reset();
}

void InklingModel::plan_experts() {
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  const size_t reserve = ((size_t)opt_.vram_reserve_mb + 192) << 20;
  size_t budget = free_b > reserve ? free_b - reserve : 0;
  if (opt_.vram_budget_mb >= 0) {
    const size_t cap = (size_t)opt_.vram_budget_mb << 20;
    budget = cap > vram_used_ ? std::min(budget, cap - vram_used_) : 0;
  }
  // core layers run once per pass: keep them resident first
  std::vector<int> order;
  for (int il = core_start_; il < core_end_; il++) order.push_back(il);
  for (int il = 0; il < n_phys_; il++)
    if (il < core_start_ || il >= core_end_) order.push_back(il);
  if (const char* e = getenv("QE_INK_DEV_LAYERS")) order.resize(std::min<size_t>(order.size(), atoi(e)));
  for (int il : order) {
    PLayer& L = L_[il];
    const size_t bytes = (L.gu.stride + L.dn.stride) * n_exp_;
    if (bytes > budget) continue;
    budget -= bytes;
    L.resident = true;
    ndev_layers_++;
  }
  // what is left (plus the idle KV tail and the prefill workspace): the elastic cache of host-layer experts
  {
    size_t gmax = 0, dmax = 0;
    int n_host = 0;
    for (auto& L : L_)
      if (!L.resident) {
        n_host++;
        gmax = std::max(gmax, L.gu.stride);
        dmax = std::max(dmax, L.dn.stride);
      }
    for (auto& L : L_) {
      L.slot_of.assign(n_exp_, -1);
      L.freq.assign(n_exp_, 0.0);
    }
    static const bool no_cache = getenv("QE_INK_CACHE") && atoi(getenv("QE_INK_CACHE")) == 0;
    if (n_host > 0 && !no_cache) {
      const size_t g16 = (gmax + 255) & ~(size_t)255, sz = g16 + ((dmax + 255) & ~(size_t)255);
      auto carve = [&](uint8_t* base, size_t bytes, int row_bytes, bool from_end) {
        for (size_t k = 0; (k + 1) * sz <= bytes; k++) {
          const size_t o = from_end ? bytes - (k + 1) * sz : k * sz;
          CSlot c;
          c.gu = base + o;
          c.dn = base + o + g16;
          c.kv_row = row_bytes > 0 ? (int)(o / row_bytes) : -1;
          cslots_.push_back(c);
          if (row_bytes > 0) cache_kv_slots_++;
        }
      };
      // dedicated slots from the leftover budget (at most what the host layers can use)
      const size_t need = (size_t)n_host * n_exp_;
      const size_t ded = std::min(need, budget / sz);
      if (ded > 0) carve((uint8_t*)dalloc(ded * sz), ded * sz, 0, false);
      const size_t n_ded = cslots_.size();
      for (auto& c : cslots_) c.kv_row = INT32_MAX;  // always valid
      carve(big_, big_bytes_, 0, false);
      // global layers' K and V rows beyond the context in use, from the end
      for (auto& S : slots_)
        if (!S.ring) {
          const size_t bytes = (size_t)ctx_ * kv_rowb_;
          carve(S.kc, bytes, (int)kv_rowb_, true);
          carve(S.vc, bytes, (int)kv_rowb_, true);
        }
      cache_slots_ = (int)cslots_.size();
      QE_LOG("[inkling] elastic expert cache: %zu dedicated + %zu workspace + %d KV-tail slots of %.1f MiB (host layers hold %zu "
             "experts)\n", n_ded, cslots_.size() - n_ded - cache_kv_slots_, cache_kv_slots_, sz / 1048576.0, need);
    }
    const size_t tab = (size_t)2 * n_phys_ * n_exp_;
    eptr_d_ = (const uint8_t**)dalloc(tab * sizeof(void*));
    CUDA_CHECK(cudaHostAlloc((void**)&eptr_h_, tab * sizeof(void*), cudaHostAllocDefault));
    host_allocs_.push_back((void*)eptr_h_);
    memset((void*)eptr_h_, 0, tab * sizeof(void*));
    CUDA_CHECK(cudaMemcpy(eptr_d_, eptr_h_, tab * sizeof(void*), cudaMemcpyHostToDevice));
    cnt_h_pending_.assign((size_t)n_phys_ * n_exp_, 0);
    cnt_d_ = (int*)dalloc((size_t)n_phys_ * n_exp_ * 4);
    CUDA_CHECK(cudaMemset(cnt_d_, 0, (size_t)n_phys_ * n_exp_ * 4));
    CUDA_CHECK(cudaHostAlloc((void**)&cnt_h_, (size_t)n_phys_ * n_exp_ * 4, cudaHostAllocDefault));
    host_allocs_.push_back(cnt_h_);
    ptr_gu_ = (const uint8_t**)dalloc((size_t)kMaxTok * k_exp_ * sizeof(void*));
    ptr_dn_ = (const uint8_t**)dalloc((size_t)kMaxTok * k_exp_ * sizeof(void*));
    slot_d_ = (int*)dalloc((size_t)2 * kMaxTok * k_exp_ * sizeof(int));
    grp_d_ = dalloc(qk::moe_group_bytes(kMaxTok * k_exp_));
    ngrp_d_ = (int*)dalloc(16);
  }
  const GGUFFile& g = *gguf_;
  for (int il = 0; il < n_phys_; il++) {
    PLayer& L = L_[il];
    for (Experts* x : {&L.gu, &L.dn}) {
      const GGUFTensor& t = g.need(bn(il, x == &L.gu ? "ffn_gate_up_exps.weight" : "ffn_down_exps.weight"));
      if (L.resident) {
        void* d = dalloc(t.nbytes);
        CUDA_CHECK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
        x->dev = (const uint8_t*)d;
      } else {
        void* h = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h, t.nbytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        host_allocs_.push_back(h);
        memcpy(h, t.data, t.nbytes);
        x->host = (const uint8_t*)h;
      }
    }
  }
}

void InklingModel::upload_eptr() {
  CUDA_CHECK(cudaMemcpyAsync(eptr_d_, eptr_h_, (size_t)2 * n_phys_ * n_exp_ * sizeof(void*), cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));  // the host table may change right after
}

// drop the cached experts of slots that are about to be overwritten (KV rows below kv_hi_, the workspace)
void InklingModel::cache_evict(bool kv, bool ws) {
  bool dirty = false;
  for (auto& c : cslots_) {
    if (c.layer < 0 || slot_valid(c)) continue;
    if (!((kv && c.kv_row >= 0) || (ws && c.kv_row < 0))) continue;
    eptr_h_[(size_t)c.layer * n_exp_ + c.expert] = nullptr;
    eptr_h_[(size_t)(n_phys_ + c.layer) * n_exp_ + c.expert] = nullptr;
    L_[c.layer].slot_of[c.expert] = -1;
    c.layer = c.expert = -1;
    dirty = true;
  }
  if (dirty) upload_eptr();
}

void InklingModel::rebalance() {
  if (cslots_.empty()) return;
  CUDA_CHECK(cudaMemcpyAsync(cnt_h_, cnt_d_, (size_t)n_phys_ * n_exp_ * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaMemsetAsync(cnt_d_, 0, (size_t)n_phys_ * n_exp_ * 4, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  // routing frequencies (decayed) of every host-layer expert
  std::vector<std::pair<double, int>> cand;  // (freq, layer * n_exp + expert)
  for (PLayer& L : L_) {
    if (L.resident) continue;
    for (int e = 0; e < n_exp_; e++) {
      const size_t i = (size_t)L.idx * n_exp_ + e;
      if (L.slot_of[e] >= 0) cache_hits_ += cnt_h_[i];
      cache_uses_ += cnt_h_[i];
      L.freq[e] = L.freq[e] * 0.9 + cnt_h_[i] + cnt_h_pending_[i];
      if (L.freq[e] > 0) cand.push_back({L.freq[e], (int)i});
    }
  }
  std::fill(cnt_h_pending_.begin(), cnt_h_pending_.end(), 0);
  int nvalid = 0;
  for (auto& c : cslots_) nvalid += slot_valid(c);
  std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
  if ((int)cand.size() > nvalid) cand.resize(nvalid);
  std::vector<char> want((size_t)n_phys_ * n_exp_, 0);
  for (auto& [f, i] : cand) want[i] = 1;
  // free slots: valid ones that are empty or hold an expert no longer wanted
  std::vector<int> fr;
  for (int k = 0; k < (int)cslots_.size(); k++) {
    const CSlot& c = cslots_[k];
    if (!slot_valid(c)) continue;
    if (c.layer < 0 || !want[(size_t)c.layer * n_exp_ + c.expert]) fr.push_back(k);
  }
  bool dirty = false;
  size_t fi = 0, copied = 0;
  static const size_t max_copy = 192;  // per call (~600 MB): the rest follows at the next rebalances
  for (auto& [f, i] : cand) {
    const int il = i / n_exp_, e = i % n_exp_;
    PLayer& L = L_[il];
    if (L.slot_of[e] >= 0) continue;
    if (fi >= fr.size() || copied >= max_copy) break;
    const int k = fr[fi++];
    CSlot& c = cslots_[k];
    if (c.layer >= 0) {
      eptr_h_[(size_t)c.layer * n_exp_ + c.expert] = nullptr;
      eptr_h_[(size_t)(n_phys_ + c.layer) * n_exp_ + c.expert] = nullptr;
      L_[c.layer].slot_of[c.expert] = -1;
    }
    CUDA_CHECK(cudaMemcpyAsync(c.gu, L.gu.host + (size_t)e * L.gu.stride, L.gu.stride, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(c.dn, L.dn.host + (size_t)e * L.dn.stride, L.dn.stride, cudaMemcpyHostToDevice, s_));
    c.layer = il;
    c.expert = e;
    L.slot_of[e] = k;
    eptr_h_[(size_t)il * n_exp_ + e] = c.gu;
    eptr_h_[(size_t)(n_phys_ + il) * n_exp_ + e] = c.dn;
    copied++;
    dirty = true;
  }
  if (dirty) upload_eptr();
}

// ring rows of positions [pos - swa, pos) as at most two contiguous row ranges
static int ring_ranges(int pos, int swa, int ring, int (&r0)[2], int (&n)[2]) {
  const int lo = std::max(0, pos - swa);
  int k = 0, p = lo;
  while (p < pos) {
    const int row = p % ring;
    const int len = std::min(pos - p, ring - row);
    r0[k] = row;
    n[k] = len;
    k++;
    p += len;
  }
  return k;
}

void InklingModel::ck_save(int slot, int pos) {
  if (slot < 0 || slot >= (int)cks_.size()) return;
  const int kvsz = nkv_ * hd_;
  const size_t hb = 3 * (size_t)(2 * kvsz + 2 * E_) * 4;
  uint8_t* h = cks_[slot].host;
  for (auto& S : slots_) {
    CUDA_CHECK(cudaMemcpyAsync(h, S.hist, hb, cudaMemcpyDeviceToHost, s_));
    h += hb;
    if (!S.ring) continue;
    int r0[2], n[2];
    const int k = ring_ranges(pos, swa_, S.ring, r0, n);
    size_t o = 0;
    for (int i = 0; i < k; i++) {
      const size_t bytes = (size_t)n[i] * kv_rowb_;
      CUDA_CHECK(cudaMemcpyAsync(h + o, S.kc + (size_t)r0[i] * kv_rowb_, bytes, cudaMemcpyDeviceToHost, s_));
      CUDA_CHECK(cudaMemcpyAsync(h + (size_t)swa_ * kv_rowb_ + o, S.vc + (size_t)r0[i] * kv_rowb_, bytes, cudaMemcpyDeviceToHost, s_));
      o += bytes;
    }
    h += 2 * (size_t)swa_ * kv_rowb_;
  }
  cks_[slot].pos = pos;
  CUDA_CHECK(cudaStreamSynchronize(s_));
}

void InklingModel::ck_restore(int slot) {
  if (slot < 0 || slot >= (int)cks_.size() || cks_[slot].pos < 0) return;
  const int pos = cks_[slot].pos;
  const int kvsz = nkv_ * hd_;
  const size_t hb = 3 * (size_t)(2 * kvsz + 2 * E_) * 4;
  const uint8_t* h = cks_[slot].host;
  for (auto& S : slots_) {
    CUDA_CHECK(cudaMemcpyAsync(S.hist, h, hb, cudaMemcpyHostToDevice, s_));
    h += hb;
    if (!S.ring) continue;
    int r0[2], n[2];
    const int k = ring_ranges(pos, swa_, S.ring, r0, n);
    size_t o = 0;
    for (int i = 0; i < k; i++) {
      const size_t bytes = (size_t)n[i] * kv_rowb_;
      CUDA_CHECK(cudaMemcpyAsync(S.kc + (size_t)r0[i] * kv_rowb_, h + o, bytes, cudaMemcpyHostToDevice, s_));
      CUDA_CHECK(cudaMemcpyAsync(S.vc + (size_t)r0[i] * kv_rowb_, h + (size_t)swa_ * kv_rowb_ + o, bytes, cudaMemcpyHostToDevice, s_));
      o += bytes;
    }
    h += 2 * (size_t)swa_ * kv_rowb_;
  }
  pos_ = pos;
  CUDA_CHECK(cudaStreamSynchronize(s_));
}

void InklingModel::reset() {
  pos_ = 0;
  kv_hi_ = 0;  // a new sequence: the KV rows are free again (refilled by rebalance)
  for (auto& S : slots_)
    CUDA_CHECK(cudaMemsetAsync(S.hist, 0, 3 * (size_t)(2 * nkv_ * hd_ + 2 * E_) * 4, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
}
void InklingModel::prepare_context(int npos, bool small_prefill) {
  (void)npos;
  (void)small_prefill;
}
int InklingModel::seq_create() {
  seq_ = next_seq_++;
  return seq_;
}
void InklingModel::seq_drop(int id) {
  if (id == seq_) seq_ = 0;
}
void InklingModel::sync() { CUDA_CHECK(cudaStreamSynchronize(s_)); }
KvInfo InklingModel::kv_info() const {
  KvInfo k;
  k.seqs = seq_ ? 1 : 0;
  return k;
}

// ---------------------------------------------------------------------------------------------
void InklingModel::lin(const W& w, const float* x, int T, float* y, int ldy, bool acc, bool reuse) {
  if (ldy < 0) ldy = w.N;
  if (T <= 0) return;
  // reuse: x was the input of the previous quantizing lin (same contents): its xq_/xd_ are still valid
  int8_t* xq = T <= kMaxTok ? xqs_ : xq_;
  float* xd = T <= kMaxTok ? xds_ : xd_;
  auto quant = [&] {
    if (reuse && qx_ == x && qK_ == w.K && qT_ == T) return;
    qk::quantize_q8(x, xq, xd, w.K, T, s_);
    qx_ = x;
    qK_ = w.K;
    qT_ = T;
  };
  if (w.type == QT_F32) {
    if (T <= kMaxTok) {
      qk::matvec(QT_F32, w.dev, w.rb, w.K, w.N, nullptr, nullptr, x, T, y, ldy, acc, s_);
    } else {
      const float alpha = 1.f, beta = acc ? 1.f : 0.f;
      CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, w.N, T, w.K, &alpha, (const float*)w.dev, w.K, x, w.K,
                               &beta, y, ldy));
    }
    return;
  }
  if (T <= kMaxTok) {
    quant();
    qk::matvec(w.type, w.dev, w.rb, w.K, w.N, xq, xd, nullptr, T, y, ldy, acc, s_);
    return;
  }
  static const bool no_q8mmq = getenv("QE_INK_NO_Q8MMQ") != nullptr;
  const bool fsc = (w.type == QT_Q8_0 && !no_q8mmq) || w.type == QT_IQ4_NL || w.type == QT_NVFP4;  // K % 32 suffices
  if ((qk::mmq_supported(w.type) && w.K % 256 == 0) || (fsc && w.K % 32 == 0)) {
    quant();
    qk::mmq(w.type, w.dev, w.rb, w.K, w.N, xq_, xd_, T, y, ldy, acc, 32, s_);
    return;
  }
  QE_ASSERT((size_t)T * w.K <= a16_elems_);
  qx_ = nullptr;
  qk::f32_to_f16(x, a16_, (int64_t)T * w.K, s_);
  const int rows_per = (int)std::min<size_t>(w.N, w16_elems_ / w.K);
  const float alpha = 1.f, beta = acc ? 1.f : 0.f;
  for (int r0 = 0; r0 < w.N; r0 += rows_per) {
    const int nr = std::min(rows_per, w.N - r0);
    qk::dequant_f16(w.type, (const uint8_t*)w.dev + (size_t)r0 * w.rb, w.rb, w.K, nr, w16_, s_);
    CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, nr, T, w.K, &alpha, w16_, CUDA_R_16F, w.K, a16_,
                              CUDA_R_16F, w.K, &beta, y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
}

void InklingModel::control(const float* cur, const float* msg, int c, const PLayer& L, float* out, int T) {
  ik::concat_rms(cur, msg, gcat_, E_, T, s_);
  lin(L.cdown[c], gcat_, T, gd_);
  lin(L.cgf[c], gd_, T, gg_);
  ik::ctrl_combine(out, cur, msg, gg_, E_, T, s_);
}

void InklingModel::moe(const PLayer& L, const float* x, int T, float* out) {
  qx_ = nullptr;  // the MoE paths overwrite xq_
  lin(L.gate_inp, x, T, logit_e_);
  ik::moe_route(logit_e_, L.exp_b, n_exp_, k_exp_, route_scale_, T, ids_, w_e_, s_);
  prof(10);
  if (T <= kMaxTok) moe_small(L, x, T, out);
  else moe_large(L, x, T, out);
}

// decode / short verify: per-assignment matvecs (host experts are first gathered into staging slots)
void InklingModel::moe_small(const PLayer& L, const float* x, int T, float* out) {
  const int A = T * k_exp_;
  const uint8_t *gu = L.gu.dev, *dn = L.dn.dev;
  const int* ids = ids_;
  const uint8_t *const *pg = nullptr, *const *pd = nullptr;
  if (!L.resident) {
    // cached experts are read in place, the others are copied from host memory into staging slots
    // distinct experts resolved once: cached in place, misses into the staging slots (beyond: read from host)
    qk::gather_experts_small(L.gu.host, L.gu.stride, ids_, A, eptr_d_ + (size_t)L.idx * n_exp_, stg_gu_, stg_slots_,
                             ptr_gu_, slot_d_, cnt_d_ + (size_t)L.idx * n_exp_, s_);
    qk::gather_experts_small(L.dn.host, L.dn.stride, ids_, A, eptr_d_ + (size_t)(n_phys_ + L.idx) * n_exp_, stg_dn_,
                             stg_slots_, ptr_dn_, slot_d_ + A, nullptr, s_);
    pg = ptr_gu_;
    prof(11);
    pd = ptr_dn_;
  }
  qk::quantize_q8(x, xqs_, xds_, E_, T, s_);
  static const bool grouped = !getenv("QE_INK_MOE_UNGROUPED");
  if (grouped && T > 1) {
    // assignments sharing an expert read its weights once
    qk::moe_group(ids_, A, grp_d_, ngrp_d_, s_);
    qk::moe_matvec_grouped(L.gu.type, gu, L.gu.stride, L.gu.rb, L.gu.K, L.gu.N, A, grp_d_, ngrp_d_, L.gu.scale, xqs_, xds_,
                           k_exp_, ygus_, s_, pg);
  } else {
    qk::moe_matvec(L.gu.type, gu, L.gu.stride, L.gu.rb, L.gu.K, L.gu.N, A, ids, ids_, L.gu.scale, xqs_, xds_, k_exp_, ygus_, s_, pg);
  }
  prof(12);
  ik::situ(ygus_, A, n_ff_, situ_b_, situ_lb_, nullptr, nullptr, hs_, s_);
  qk::quantize_q8(hs_, xqs_, xds_, n_ff_, A, s_);
  prof(13);
  if (grouped && T > 1)
    qk::moe_matvec_grouped(L.dn.type, dn, L.dn.stride, L.dn.rb, L.dn.K, L.dn.N, A, grp_d_, ngrp_d_, L.dn.scale, xqs_, xds_, 1,
                           yds_, s_, pd);
  else
    qk::moe_matvec(L.dn.type, dn, L.dn.stride, L.dn.rb, L.dn.K, L.dn.N, A, ids, ids_, L.dn.scale, xqs_, xds_, 1, yds_, s_, pd);
  ik::moe_combine(yds_, nullptr, w_e_, T, k_exp_, E_, out, s_);
}

// prefill: assignments sorted by expert, one GEMM per expert
void InklingModel::moe_large(const PLayer& L, const float* x, int T, float* out) {
  const int A = T * k_exp_;
  CUDA_CHECK(cudaMemcpyAsync(ids_h_, ids_, (size_t)A * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  std::vector<int> cnt(n_exp_ + 1, 0), off(n_exp_ + 1, 0);
  for (int a = 0; a < A; a++) cnt[ids_h_[a]]++;
  if (!L.resident)
    for (int e = 0; e < n_exp_; e++) cnt_h_pending_[(size_t)L.idx * n_exp_ + e] += cnt[e];
  for (int e = 0; e < n_exp_; e++) off[e + 1] = off[e] + cnt[e];
  int* tok = perm_h_;          // sorted row i: token
  int* eid = perm_h_ + A;      // sorted row i: expert
  int* inv = perm_h_ + 2 * A;  // assignment a -> sorted row
  {
    std::vector<int> fill(off.begin(), off.end() - 1);
    for (int a = 0; a < A; a++) {
      const int e = ids_h_[a];
      const int i = fill[e]++;
      tok[i] = a / k_exp_;
      eid[i] = e;
      inv[a] = i;
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(tok_d_, tok, (size_t)A * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(eid_d_, eid, (size_t)A * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(inv_d_, inv, (size_t)A * 4, cudaMemcpyHostToDevice, s_));
  // gate_up
  ik::gather_rows(x, tok_d_, A, E_, yd_, s_);
  const bool gu_mmq = qk::mmq_supported(L.gu.type) && L.gu.K % 256 == 0;
  const bool dn_mmq = qk::mmq_supported(L.dn.type) && L.dn.K % 256 == 0;
  auto expert_ptr = [&](const Experts& X, int e, uint8_t* stg) -> const uint8_t* {
    if (X.dev) return X.dev + (size_t)e * X.stride;
    if (L.slot_of[e] >= 0) return &X == &L.gu ? cslots_[L.slot_of[e]].gu : cslots_[L.slot_of[e]].dn;
    CUDA_CHECK(cudaMemcpyAsync(stg, X.host + (size_t)e * X.stride, X.stride, cudaMemcpyHostToDevice, s_));
    return stg;
  };
  auto run = [&](const Experts& X, bool use_mmq, const float* src, int K, float* dst, uint8_t* stg) {
    if (use_mmq) qk::quantize_q8(src, xq_, xd_, K, A, s_);
    else qk::f32_to_f16(src, a16_, (int64_t)A * K, s_);
    const float alpha = 1.f, beta = 0.f;
    for (int e = 0; e < n_exp_; e++) {
      const int n = cnt[e];
      if (n == 0) continue;
      const uint8_t* wp = expert_ptr(X, e, stg);
      const int o = off[e];
      if (use_mmq) {
        qk::mmq(X.type, wp, X.rb, X.K, X.N, xq_ + (size_t)o * K, xd_ + (size_t)o * (K / 32), n, dst + (size_t)o * X.N, X.N,
                false, 32, s_);
      } else {
        QE_ASSERT((size_t)X.N * X.K <= w16_elems_);
        qk::dequant_f16(X.type, wp, X.rb, X.K, X.N, w16_, s_);
        CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, X.N, n, X.K, &alpha, w16_, CUDA_R_16F, X.K,
                                  a16_ + (size_t)o * K, CUDA_R_16F, K, &beta, dst + (size_t)o * X.N, CUDA_R_32F, X.N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    }
  };
  static const bool mprof = getenv("QE_INK_PROF") && getenv("QE_INK_PROF")[0] == '3';
  cudaEvent_t ev[6];
  auto mark = [&](int i) {
    if (!mprof) return;
    CUDA_CHECK(cudaEventCreate(&ev[i]));
    CUDA_CHECK(cudaEventRecord(ev[i], s_));
  };
  mark(0);
  auto grouped_ok = [](uint32_t t) { return t == QT_NVFP4 || t == QT_IQ4_NL || qk::mmq_supported(t); };
  static const bool no_stream = getenv("QE_INK_NOGROUP") != nullptr;
  if (!L.resident && gu_mmq && grouped_ok(L.dn.type) && !no_stream) {
    // host layer: groups of G experts streamed into the two halves of the staging slots (copy stream) while the
    // previous group runs as grouped GEMMs
    const int G = stg_slots_ / 2;
    CUDA_CHECK(cudaMemcpyAsync(goff_d_, off.data(), (n_exp_ + 1) * 4, cudaMemcpyHostToDevice, s_));
    qk::quantize_q8(yd_, xq_, xd_, E_, A, s_);
    for (int b = 0; b < 2; b++) CUDA_CHECK(cudaEventRecord(stg_free_[b], s_));
    for (int e0 = 0, gi = 0; e0 < n_exp_; e0 += G, gi++) {
      const int e1 = std::min(n_exp_, e0 + G), b = gi & 1;
      const int r0 = off[e0], nr = off[e1] - off[e0];
      if (nr == 0) continue;
      uint8_t* sg = stg_gu_ + (size_t)b * G * L.gu.stride;
      uint8_t* sd = stg_dn_ + (size_t)b * G * L.dn.stride;
      CUDA_CHECK(cudaStreamWaitEvent(cs_, stg_free_[b], 0));
      for (int e = e0; e < e1; e++) {
        if (cnt[e] == 0) continue;
        const int sl = L.slot_of[e];
        const uint8_t* srcg = sl >= 0 ? cslots_[sl].gu : L.gu.host + (size_t)e * L.gu.stride;
        const uint8_t* srcd = sl >= 0 ? cslots_[sl].dn : L.dn.host + (size_t)e * L.dn.stride;
        const cudaMemcpyKind k = sl >= 0 ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpyAsync(sg + (size_t)(e - e0) * L.gu.stride, srcg, L.gu.stride, k, cs_));
        CUDA_CHECK(cudaMemcpyAsync(sd + (size_t)(e - e0) * L.dn.stride, srcd, L.dn.stride, k, cs_));
      }
      CUDA_CHECK(cudaEventRecord(stg_ready_[b], cs_));
      CUDA_CHECK(cudaStreamWaitEvent(s_, stg_ready_[b], 0));
      const int mx = *std::max_element(cnt.begin() + e0, cnt.begin() + e1);
      qk::mmq_grouped(L.gu.type, sg, L.gu.stride, L.gu.rb, L.gu.K, L.gu.N, goff_d_ + e0, e1 - e0, mx, xq_, xd_, ygu_,
                      L.gu.N, s_);
      ik::situ(ygu_ + (size_t)r0 * 2 * n_ff_, nr, n_ff_, situ_b_, situ_lb_, eid_d_ + r0, L.gu.scale, h_ + (size_t)r0 * n_ff_, s_);
      qk::quantize_q8(h_ + (size_t)r0 * n_ff_, xq2_ + (size_t)r0 * n_ff_, xd2_ + (size_t)r0 * (n_ff_ / 32), n_ff_, nr, s_);
      qk::mmq_grouped(L.dn.type, sd, L.dn.stride, L.dn.rb, L.dn.K, L.dn.N, goff_d_ + e0, e1 - e0, mx, xq2_, xd2_, yd_,
                      L.dn.N, s_);
      CUDA_CHECK(cudaEventRecord(stg_free_[b], s_));
    }
    ik::moe_coef(w_e_, ids_, L.dn.scale, A, coef_, s_);
    ik::moe_combine(yd_, inv_d_, coef_, T, k_exp_, E_, out, s_);
    return;
  }
  QE_ASSERT(gu_mmq || (size_t)A * E_ <= a16_elems_);
  QE_ASSERT(dn_mmq || (size_t)A * n_ff_ <= a16_elems_);
  if (L.resident && gu_mmq && !getenv("QE_INK_NOGROUP")) {
    // one launch for every expert
    const int mx = *std::max_element(cnt.begin(), cnt.end());
    CUDA_CHECK(cudaMemcpyAsync(goff_d_, off.data(), (n_exp_ + 1) * 4, cudaMemcpyHostToDevice, s_));
    qk::quantize_q8(yd_, xq_, xd_, E_, A, s_);
    qk::mmq_grouped(L.gu.type, L.gu.dev, L.gu.stride, L.gu.rb, L.gu.K, L.gu.N, goff_d_, n_exp_, mx, xq_, xd_, ygu_, L.gu.N, s_);
  } else {
    run(L.gu, gu_mmq, yd_, E_, ygu_, stg_gu_);
  }
  mark(1);
  ik::situ(ygu_, A, n_ff_, situ_b_, situ_lb_, eid_d_, L.gu.scale, h_, s_);
  mark(2);
  if (L.resident && !getenv("QE_INK_NOGROUP") && (L.dn.type == QT_NVFP4 || L.dn.type == QT_IQ4_NL || qk::mmq_supported(L.dn.type))) {
    const int mx = *std::max_element(cnt.begin(), cnt.end());
    CUDA_CHECK(cudaMemcpyAsync(goff_d_, off.data(), (n_exp_ + 1) * 4, cudaMemcpyHostToDevice, s_));
    qk::quantize_q8(h_, xq_, xd_, n_ff_, A, s_);
    qk::mmq_grouped(L.dn.type, L.dn.dev, L.dn.stride, L.dn.rb, L.dn.K, L.dn.N, goff_d_, n_exp_, mx, xq_, xd_, yd_, L.dn.N, s_);
  } else {
    run(L.dn, dn_mmq, h_, n_ff_, yd_, stg_dn_);
  }
  mark(3);
  ik::moe_coef(w_e_, ids_, L.dn.scale, A, coef_, s_);
  ik::moe_combine(yd_, inv_d_, coef_, T, k_exp_, E_, out, s_);
  mark(4);
  if (mprof) {
    CUDA_CHECK(cudaEventSynchronize(ev[4]));
    float t[4];
    for (int i = 0; i < 4; i++) CUDA_CHECK(cudaEventElapsedTime(&t[i], ev[i], ev[i + 1]));
    fprintf(stderr, "[moe] layer %2d %s: gate_up %.2f  situ %.2f  down %.2f  combine %.2f ms\n", L.idx,
            L.resident ? "dev " : "host", t[0], t[1], t[2], t[3]);
    for (int i = 0; i < 5; i++) cudaEventDestroy(ev[i]);
  }
}

// QE_INK_PROF=1: per-section GPU time of small-batch steps (events; graphs off)
static bool ink_prof() {
  static const bool on = getenv("QE_INK_PROF") && atoi(getenv("QE_INK_PROF")) != 0;
  return on;
}
void InklingModel::prof(int k) {
  if (!ink_prof() || (prof_T_ <= kMaxTok) != (getenv("QE_INK_PROF")[0] == '1')) return;
  static const int only_t = getenv("QE_INK_PROF_T") ? atoi(getenv("QE_INK_PROF_T")) : 0;
  if (only_t && prof_T_ != only_t) return;
  cudaEvent_t e;
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(e, s_));
  prof_ev_.push_back({k, e});
}
void InklingModel::prof_flush() {
  if (prof_ev_.empty()) return;
  CUDA_CHECK(cudaStreamSynchronize(s_));
  for (size_t i = 1; i < prof_ev_.size(); i++) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, prof_ev_[i - 1].second, prof_ev_[i].second));
    prof_ms_[prof_ev_[i].first] += ms;
  }
  for (auto& [k, e] : prof_ev_) cudaEventDestroy(e);
  prof_ev_.clear();
  if (++prof_n_ % (getenv("QE_INK_PROF_T") ? 10 : 50) == 0 || getenv("QE_INK_PROF")[0] == '2') {
    static const char* nm[14] = {"start", "mix+ctrl+norm", "qkv proj", "conv/clip/kv", "attention", "out proj+conv",
                                 "ffn mix+norm", "moe down+combine", "ffn conv+add", "head", "moe router", "moe gather",
                                 "moe gate_up", "moe situ+quant"};
    double tot = 0;
    for (int i = 0; i < 14; i++) tot += prof_ms_[i];
    fprintf(stderr, "[prof] per step (%d steps, T=%d): ", prof_n_, prof_T_);
    for (int i = 1; i < 14; i++)
      if (prof_ms_[i] > 0) fprintf(stderr, "%s %.2f  ", nm[i], prof_ms_[i] / prof_n_);
    fprintf(stderr, "| total %.2f ms\n", tot / prof_n_);
  }
}

// ---------------------------------------------------------------------------------------------
void InklingModel::layer(int slot, int ip, const float* inp, float* out, std::vector<const float*>& blocks,
                         const float* const* msgs, int T, int pos0) {
  const PLayer& L = L_[ip];
  Slot& S = slots_[slot];
  const int kvsz = nkv_ * hd_, qsz = nh_ * hd_;
  const float* ins[3] = {inp, inp, inp};
  prof(0);
  auto states = [&](const float* cur) {
    ik::MixSrc st{};
    QE_ASSERT(blocks.size() + 1 <= 8);
    for (const float* b : blocks) st.p[st.n++] = b;
    st.p[st.n++] = cur;
    return st;
  };
  if (L.attn_rq) {
    ik::MixDst d{{mixed_[0], mixed_[1], mixed_[2], nullptr}};
    ik::mix(states(inp), L.attn_rq, L.attn_nq, E_, T, d, s_);
    for (int j = 0; j < 3; j++) ins[j] = mixed_[L.swa ? 0 : j];
  }
  if (msgs) {
    for (int j = 0; j < 3; j++) {
      control(ins[j], msgs[j], j, L, ins_[j], T);
      ins[j] = ins_[j];
    }
  }
  const float* x[3];
  if (L.swa) {
    const int nn = msgs ? 3 : 1;
    for (int j = 0; j < nn; j++) ik::rms_rows(ins[j], L.attn_norm, xn_[j], E_, T, eps_, s_);
    x[0] = xn_[0];
    x[1] = xn_[nn > 1 ? 1 : 0];
    x[2] = xn_[nn > 1 ? 2 : 0];
  } else {
    ik::rms_rows(ins[0], L.qn, xn_[0], E_, T, eps_, s_);
    ik::rms_rows(ins[1], L.kn, xn_[1], E_, T, eps_, s_);
    ik::rms_rows(ins[2], L.vn, xn_[2], E_, T, eps_, s_);
    x[0] = xn_[0];
    x[1] = xn_[1];
    x[2] = xn_[2];
  }
  prof(1);
  // inputs quantized once per distinct tensor (reuse); the float projections do not touch the quantized copy
  qx_ = nullptr;
  lin(L.q, x[0], T, q_);
  lin(L.gate, x[0], T, g_, -1, false, true);
  lin(L.rel, x[0], T, r_, -1, false, true);
  lin(L.sink, x[0], T, sink_);
  lin(L.k, x[1], T, kp_, -1, false, true);
  if (!L.swa) lin(L.kcp, x[1], T, kd_);
  lin(L.v, x[2], T, vp_, -1, false, true);
  if (!L.swa) lin(L.vcp, x[2], T, vd_);
  prof(2);
  const float inv_s = 1.f / sqrtf((float)n_conv_ + 1), inv_d = 1.f / sqrtf(2.f * n_conv_ + 1);
  float* hist = S.hist;
  float* const* sv = snap_ ? save_[slot].data() : nullptr;  // speculative verify: keep the conv inputs (rollback)
  ik::conv(kp_, T, kvsz, L.k_conv, L.swa ? nullptr : kd_, nkv_ * n_conv_, hd_, hist, k_, L.swa ? inv_s : inv_d, s_,
           sv ? sv[0] : nullptr);
  ik::conv(vp_, T, kvsz, L.v_conv, L.swa ? nullptr : vd_, nkv_ * n_conv_, hd_, hist + 3 * kvsz, v_,
           L.swa ? inv_s : inv_d, s_, sv ? sv[1] : nullptr);
  ik::clip_heads(q_, qsz, L.q_clip, nh_, hd_, T, qk_clip_, !L.swa, pos_d_, log_alpha_, log_floor_, s_);
  ik::clip_heads(k_, kvsz, nullptr, nkv_, hd_, T, qk_clip_, false, pos_d_, 0.f, 1.f, s_);
  ik::head_rms(v_, hd_, nkv_, kvsz, T, eps_, s_);
  ik::clip_heads(r_, nh_ * n_rel_, L.rel_clip, nh_, n_rel_, T, rel_clip_, !L.swa, pos_d_, log_alpha_, log_floor_, s_);
  ik::kv_store(k_, kvsz, v_, kvsz, S.kc, S.vc, kvsz, kv_bits_, pos_d_, T, S.ring, s_);
  prof(3);
  // key bound per query tile: small batches use the worst case (the same launch for every position: CUDA graphs)
  const int max_keys = T <= kMaxTok ? (L.swa ? swa_ + T : ctx_) : (L.swa ? std::min(pos0 + T, swa_ + T) : pos0 + T);
  ik::attn(S.kc, S.vc, kv_bits_, S.ring, q_, qsz, r_, sink_, L.rel_table, L.swa ? swa_ : rel_extent_, L.swa ? swa_ : 0, att_, pos_d_,
           max_keys, T, nh_, nkv_, hd_, attn_work_, attn_work_bytes_, s_);
  prof(4);
  ik::mul_sigmoid(att_, g_, (int64_t)T * qsz, s_);
  lin(L.o, att_, T, o_);
  ik::conv(o_, T, E_, L.ao_conv, nullptr, 0, 1, hist + 6 * kvsz, o2_, inv_s, s_, sv ? sv[2] : nullptr);
  // the first layer of a block starts a new residual stream; its input joins the mixed blocks
  if (ip % block_ == 0) {
    blocks.push_back(inp);
    CUDA_CHECK(cudaMemcpyAsync(out, o2_, (size_t)T * E_ * 4, cudaMemcpyDeviceToDevice, s_));
  } else {
    ik::add(out, inp, o2_, (int64_t)T * E_, s_);
  }
  prof(5);
  // MoE FFN
  {
    ik::MixDst d{{mixed_[0], nullptr, nullptr, nullptr}};
    ik::mix(states(out), L.ffn_rq, 1, E_, T, d, s_);
  }
  const float* fx = mixed_[0];
  if (msgs) {
    control(fx, msgs[3], 3, L, ins_[0], T);
    fx = ins_[0];
  }
  ik::rms_rows(fx, L.ffn_norm, xn_[0], E_, T, eps_, s_);
  prof(6);
  moe(L, xn_[0], T, o_);
  prof(7);
  ik::conv(o_, T, E_, L.fo_conv, nullptr, 0, 1, hist + 6 * kvsz + 3 * E_, o2_, inv_s, s_, sv ? sv[3] : nullptr);
  ik::add(out, out, o2_, (int64_t)T * E_, s_);
  prof(8);
}

float* InklingModel::fresh(const std::vector<const float*>& live) {
  for (float* p : pool_)
    if (std::find(live.begin(), live.end(), p) == live.end()) return p;
  QE_FATAL("inkling: hidden-state buffer pool exhausted");
}

// host side of a chunk: token embeddings (rows dequantized on the CPU) and the position, into pinned buffers
void InklingModel::embed(const int* tokens, int n, int pos0) {
  CUDA_CHECK(cudaStreamSynchronize(s_));  // the pinned buffers are reused
  const size_t rb = qtype_row_bytes(tok_embd_->type, E_);
  for (int t = 0; t < n; t++) {
    QE_ASSERT(tokens[t] >= 0 && tokens[t] < n_vocab_pad_);
    const uint8_t* row = tok_embd_->data + (size_t)tokens[t] * rb;
    float* y = embd_h_ + (size_t)t * E_;
    if (tok_embd_->type == QT_F32) memcpy(y, row, E_ * 4);
    else dequant_row_ref(tok_embd_->type, row, E_, y);
  }
  pos_h_[0] = pos0;
}

void InklingModel::chunk(const int* tokens, int T, int pos0, int n_out, float* logits_host, int* argmax_host) {
  if (pos0 + T > ctx_) QE_FATAL("inkling: context overflow (%d > %d)", pos0 + T, ctx_);
  embed(tokens, T, pos0);
  // decode steps replay a CUDA graph of the whole step (the per-step inputs live in pinned / device buffers)
  static const bool use_graph = (!getenv("QE_INK_GRAPH") || atoi(getenv("QE_INK_GRAPH")) != 0) && !ink_prof();
  prof_T_ = T;
  const int gv = logits_host ? 1 : 0;
  if (use_graph && T == 1 && n_out == 1) {
    if (!graph_[gv] && graph_warm_[gv]++ >= 1) {
      cudaGraph_t g = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal));
      body(1, pos0, 1, gv != 0, true);
      CUDA_CHECK(cudaStreamEndCapture(s_, &g));
      CUDA_CHECK(cudaGraphInstantiate(&graph_[gv], g, 0));
      CUDA_CHECK(cudaGraphDestroy(g));
    }
    if (graph_[gv]) CUDA_CHECK(cudaGraphLaunch(graph_[gv], s_));
    else body(1, pos0, 1, gv != 0, true);
  } else {
    big_out_ = n_out > kMaxTok ? logits_host : nullptr;
    QE_ASSERT(n_out <= kMaxTok || (logits_host && !argmax_host));
    body(T, pos0, n_out, logits_host != nullptr, argmax_host != nullptr);
    if (n_out > kMaxTok) return;
  }
  CUDA_CHECK(cudaStreamSynchronize(s_));
  prof_flush();
  if (n_out > 0 && argmax_host) memcpy(argmax_host, argmax_h_, n_out * 4);
  if (n_out > 0 && logits_host) memcpy(logits_host, out_h_, (size_t)n_out * n_vocab_ * 4);
}

void InklingModel::body(int T, int pos0, int n_out, bool want_logits, bool want_argmax) {
  CUDA_CHECK(cudaMemcpyAsync(pos_d_, pos_h_, 4, cudaMemcpyHostToDevice, s_));
  std::vector<const float*> blocks, live;
  float* x0 = pool_[0];
  CUDA_CHECK(cudaMemcpyAsync(x0, embd_h_, (size_t)T * E_ * 4, cudaMemcpyHostToDevice, s_));
  ik::rms_rows(x0, tok_norm_, x0, E_, T, eps_, s_);
  const float* cur = x0;
  auto live_set = [&](std::initializer_list<const float*> extra, const std::vector<const float*>& more) {
    live.assign(blocks.begin(), blocks.end());
    live.insert(live.end(), extra.begin(), extra.end());
    live.insert(live.end(), more.begin(), more.end());
    return live;
  };
  const std::vector<const float*> none;
  for (int il = 0; il < core_start_; il++) {
    float* out = fresh(live_set({cur}, none));
    layer(il, il, cur, out, blocks, nullptr, T, pos0);
    cur = out;
  }
  const float* prelude = cur;
  const std::vector<const float*> prelude_blocks = blocks;
  const int ncore = core_end_ - core_start_;
  std::vector<const float*> bank;
  for (int pass = 0; pass < n_pass_; pass++) {
    const float* bank_out = nullptr;
    if (pass > 0) {
      bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
      bank_out = cur;
      cur = prelude;
      blocks = prelude_blocks;
    }
    for (int ip = core_start_; ip < core_end_; ip++) {
      const int slot = pass == 0 ? ip : n_phys_ + (pass - 1) * ncore + (ip - core_start_);
      const float* const* msgs = nullptr;
      if (pass > 0) {
        ik::MixSrc st{};
        for (const float* b : bank) st.p[st.n++] = b;
        st.p[st.n++] = bank_out;
        ik::MixDst d{{msgs_[0], msgs_[1], msgs_[2], msgs_[3]}};
        ik::mix(st, L_[ip].msg_q, 4, E_, T, d, s_);
        msgs = msgs_;
      }
      std::vector<const float*> keep = bank;
      keep.push_back(prelude);
      if (bank_out) keep.push_back(bank_out);
      keep.insert(keep.end(), prelude_blocks.begin(), prelude_blocks.end());
      float* out = fresh(live_set({cur}, keep));
      layer(slot, ip, cur, out, blocks, msgs, T, pos0);
      cur = out;
    }
  }
  for (int il = core_end_; il < n_phys_; il++) {
    float* out = fresh(live_set({cur}, none));
    layer(il, il, cur, out, blocks, nullptr, T, pos0);
    cur = out;
  }
  if (n_out <= 0) return;
  prof(0);
  // head: depth mix of the blocks, norm, logits of the last n_out rows
  ik::MixSrc st{};
  for (const float* b : blocks) st.p[st.n++] = b;
  st.p[st.n++] = cur;
  ik::MixSrc sl = st;
  for (int i = 0; i < sl.n; i++) sl.p[i] += (size_t)(T - n_out) * E_;  // mix only the rows needed
  {
    ik::MixDst d{{mixed_[0], nullptr, nullptr, nullptr}};
    ik::mix(sl, output_rq_, 1, E_, n_out, d, s_);
    ik::rms_rows(mixed_[0], output_norm_, xn_[0], E_, n_out, eps_, s_);
  }
  if (n_out > kMaxTok) {  // every row (perplexity): groups of kMaxTok rows straight to the caller
    for (int r0 = 0; r0 < n_out; r0 += kMaxTok) {
      const int nr = std::min(kMaxTok, n_out - r0);
      lin(output_, xn_[0] + (size_t)r0 * E_, nr, logits_d_);
      if (softcap_ > 0) ik::softcap(logits_d_, (int64_t)nr * n_vocab_, softcap_, s_);
      CUDA_CHECK(cudaMemcpyAsync(big_out_ + (size_t)r0 * n_vocab_, logits_d_, (size_t)nr * n_vocab_ * 4,
                                 cudaMemcpyDeviceToHost, s_));
    }
    CUDA_CHECK(cudaStreamSynchronize(s_));
    return;
  }
  lin(output_, xn_[0], n_out, logits_d_);
  if (softcap_ > 0) ik::softcap(logits_d_, (int64_t)n_out * n_vocab_, softcap_, s_);
  prof(9);
  if (want_argmax) {
    qk::argmax_rows(logits_d_, n_vocab_, n_out, argmax_d_, s_);
    CUDA_CHECK(cudaMemcpyAsync(argmax_h_, argmax_d_, n_out * 4, cudaMemcpyDeviceToHost, s_));
  }
  if (want_logits)
    CUDA_CHECK(cudaMemcpyAsync(out_h_, logits_d_, (size_t)n_out * n_vocab_ * 4, cudaMemcpyDeviceToHost, s_));
}

void InklingModel::rollback(int n_keep) {
  if (!snap_) return;
  ik::conv_rollback(csave_d_, n_csave_, n_keep, s_);
  pos_ = snap_pos0_ + n_keep;
  snap_ = false;
}

void InklingModel::forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax, bool snapshot,
                           bool mtp_prefill) {
  (void)mtp_prefill;
  QE_ASSERT(n > 0 && n_logits <= n && (n_logits <= kMaxTok || n_logits <= U_));
  // elastic cache: the context grows into KV rows (rounded up), larger batches need the prefill workspace
  if (pos0 + n > kv_hi_) {
    kv_hi_ = std::min(ctx_, (pos0 + n + 2047) / 2048 * 2048);
    cache_evict(true, false);
  }
  const bool big = n > kMaxTok;
  snap_ = snapshot && n <= kMaxTok;
  snap_pos0_ = pos0;
  if (big) {
    ws_busy_ = true;
    cache_evict(false, true);
  }
  int off = 0;
  const int cap = cap_pos_ - pos0;  // capture the state after this many tokens of the call
  while (off < n) {
    int T = std::min(U_, n - off);
    if (n - off - T > 0 && n - off - T < n_logits) T -= n_logits;  // the last chunk holds every logits row
    if (cap > off && cap < off + T) T = cap - off;                  // a chunk ends at the capture point
    const bool last = off + T == n;
    chunk(tokens + off, T, pos0 + off, last ? n_logits : 0, last ? logits : nullptr, last ? argmax : nullptr);
    off += T;
    pos_ = pos0 + off;
    if (cap == off && cap_slot_ >= 0) ck_save(cap_slot_, pos_);
    if (!last && cancel_ && cancel_->load(std::memory_order_relaxed)) break;
  }
  cap_pos_ = cap_slot_ = -1;
  ws_busy_ = false;
  // the expert cache follows the routing: after every prompt and every 32 decode calls
  if (big || ++steps_since_rb_ >= 32) {
    rebalance();
    steps_since_rb_ = 0;
    if (opt_.verbose && cache_uses_)
      QE_LOG("[inkling] expert cache hit rate %.1f%%\n", 100.0 * cache_hits_ / cache_uses_);
  }
}
