// DSpark drafter for Inkling 2 (GGUF architecture dflash, Inkling variant; reference: llama.cpp PR #32218 at
// 78d7794, as shipped in Unsloth's builds).
//
// Context: for every committed target token the features f_j = target hidden states at the layer visits
// `targets` (the last one is the final normed output) are mixed per draft layer, rms(sum_j softmax(fuse)_j f_j) *
// norm, and projected to that layer's K / V (K head-normed and roped) -- the draft's KV cache.  A second mix,
// f_rnn, feeds the Markov head.
// Draft: a block [anchor, mask x 7] at positions pos..pos+7 runs through 3 transformer layers (non-causal inside
// the block, sliding window on the local layers); base logits = the target's lm head on the outputs.  Block slot
// i >= 1 drafts the token at pos+i: a GRU-like state, fed [state, w1[prev], hidden_i, f_rnn(pos-1)], adds a
// correction (through the lm head again) to the base logits, softcapped.  Here the correction is evaluated on the
// top-K base candidates only (the draft distribution q lives on them: verification stays exact).
#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.h"
#include "inkling.h"
#include "inkling_kernels.h"
#include "kernels.h"
#include "quant.h"

namespace {
std::string bn(int il, const char* s) { return "blk." + std::to_string(il) + "." + s; }
void softmax4(const float* x, int n, float* y) {
  float m = -INFINITY, z = 0.f;
  for (int i = 0; i < n; i++) m = std::max(m, x[i]);
  for (int i = 0; i < n; i++) z += std::exp(x[i] - m);
  for (int i = 0; i < n; i++) y[i] = std::exp(x[i] - m) / z;
}
}  // namespace

void InklingModel::ds_load(const std::string& path) {
  auto D = std::make_unique<DSpark>();
  D->g = std::make_unique<GGUFFile>(path);
  const GGUFFile& g = *D->g;
  if (g.get_str("general.architecture", "") != "dflash" || !g.tensor("dspark_fuse.weight"))
    QE_FATAL("%s: not an Inkling 2 DSpark drafter", path.c_str());
  auto gi = [&](const char* k) {
    const GGUFValue* v = g.get(std::string("dflash.") + k);
    if (!v) QE_FATAL("DSpark: missing dflash.%s", k);
    return v->is_array() ? (int)v->arr[0].as_int() : (int)v->as_int();
  };
  D->n_layer = gi("block_count");
  D->nh = gi("attention.head_count");
  D->nkv = gi("attention.head_count_kv");
  D->hd = gi("attention.key_length");
  D->n_ff = gi("feed_forward_length");
  D->block = gi("block_size");
  D->rank = gi("dspark.markov_rank");
  D->swa = (int)g.get_int("dflash.attention.sliding_window", 0);
  D->theta = (float)g.get_float("dflash.rope.freq_base", 1e6);
  D->eps = (float)g.get_float("dflash.attention.layer_norm_rms_epsilon", 1e-6);
  D->cap = (float)g.get_float("dflash.final_logit_softcapping", 0.0);
  if (gi("embedding_length") != E_ || D->hd != 128 || D->nh % D->nkv || D->block > kMaxTok || D->block < 2 ||
      gi("vocab_size_unpadded") != n_vocab_ || D->rank > 256)
    QE_FATAL("DSpark: drafter does not match the target (embd %d, head %d, block %d, vocab %d)", gi("embedding_length"),
             D->hd, D->block, gi("vocab_size_unpadded"));
  for (auto& v : g.get("dflash.target_layers")->arr) D->targets.push_back((int)v.as_int());
  if (D->targets.size() > 4 || D->targets.back() != cfg_.n_layer) QE_FATAL("DSpark: unexpected target layers");
  const int nf = (int)D->targets.size();
  const GGUFValue* pat = g.get("dflash.attention.sliding_window_pattern");
  const GGUFTensor& fuse = g.need("dspark_fuse.weight");  // [nf][n_layer + 1]
  QE_ASSERT(fuse.type == QT_F32 && fuse.ne[0] == nf && fuse.ne[1] == D->n_layer + 1);
  const float* fz = (const float*)fuse.data;
  const size_t v0 = vram_used_;
  D->L.resize(D->n_layer);
  const int kvsz = D->nkv * D->hd;
  for (int il = 0; il < D->n_layer; il++) {
    DsLayer& L = D->L[il];
    L.swa = pat && il < (int)pat->arr.size() && pat->arr[il].as_int() && D->swa > 0;
    L.q = upload_w(bn(il, "attn_q.weight"), true, &g);
    L.k = upload_w(bn(il, "attn_k.weight"), true, &g);
    L.v = upload_w(bn(il, "attn_v.weight"), true, &g);
    L.o = upload_w(bn(il, "attn_output.weight"), true, &g);
    L.kctx = upload_w(bn(il, "attn_k_ctx.weight"), true, &g);
    L.vctx = upload_w(bn(il, "attn_v_ctx.weight"), true, &g);
    L.gate = upload_w(bn(il, "ffn_gate.weight"), true, &g);
    L.up = upload_w(bn(il, "ffn_up.weight"), true, &g);
    L.down = upload_w(bn(il, "ffn_down.weight"), true, &g);
    L.attn_norm = upload_f32(bn(il, "attn_norm.weight"), true, &g);
    L.ffn_norm = upload_f32(bn(il, "ffn_norm.weight"), true, &g);
    L.ctx_norm = upload_f32(bn(il, "dflare_norm.weight"), true, &g);
    L.q_norm = upload_f32(bn(il, "attn_q_norm.weight"), true, &g);
    L.k_norm = upload_f32(bn(il, "attn_k_norm.weight"), true, &g);
    L.kctx_norm = upload_f32(bn(il, "attn_k_ctx_norm.weight"), true, &g);
    softmax4(fz + (size_t)il * nf, nf, L.fuse);
    L.ring = L.swa ? D->swa + U_ : 0;
    const size_t rows = L.swa ? (size_t)L.ring : (size_t)ctx_;
    L.kc = (uint8_t*)dalloc(rows * kv_rowb_);
    L.vc = (uint8_t*)dalloc(rows * kv_rowb_);
  }
  softmax4(fz + (size_t)D->n_layer * nf, nf, D->frnn_fuse);
  D->joint = upload_w("joint_proj.weight", true, &g);
  D->joint_b = upload_f32("joint_proj.bias", true, &g);
  D->hidden_out = upload_w("dspark_hidden_out.weight", true, &g);
  D->hidden_out_b = upload_f32("dspark_hidden_out.bias", true, &g);
  D->mask = upload_f32("mask_embd.weight", true, &g);
  D->out_norm = upload_f32("output_norm.weight", true, &g);
  D->frnn_norm = upload_f32("dspark_norm.weight", true, &g);
  D->w1 = &g.need("markov_w1.weight");
  QE_ASSERT(D->w1->type == QT_BF16 && D->w1->ne[0] == D->rank && D->joint.K == 2 * D->rank + 2 * E_ &&
            D->joint.N == 3 * D->rank && D->hidden_out.K == D->rank && D->hidden_out.N == E_);
  // buffers
  const int B = D->block;
  D->feat = (float*)dalloc((size_t)U_ * nf * E_ * 4);
  D->x = (float*)dalloc((size_t)B * E_ * 4);
  D->h = (float*)dalloc((size_t)B * E_ * 4);
  D->g1 = (float*)dalloc((size_t)B * D->n_ff * 4);
  D->u1 = (float*)dalloc((size_t)B * D->n_ff * 4);
  D->cur = (float*)dalloc((size_t)B * E_ * 4);
  D->base = (float*)dalloc((size_t)B * n_vocab_ * 4);
  D->frnn_ring = D->swa + U_;
  D->frnn = (float*)dalloc((size_t)D->frnn_ring * E_ * 4);
  D->cand_v = (float*)dalloc((size_t)B * D->K * 4);
  D->cand_i = (int*)dalloc((size_t)B * D->K * 4);
  D->cand_l = (float*)dalloc((size_t)D->K * 4);
  D->bias = (float*)dalloc((size_t)D->K * 4);
  D->joint_in = (float*)dalloc((size_t)D->joint.K * 4);
  D->jout = (float*)dalloc((size_t)D->joint.N * 4);
  D->state = (float*)dalloc((size_t)D->rank * 4);
  D->latent = (float*)dalloc((size_t)D->rank * 4);
  D->corr = (float*)dalloc((size_t)E_ * 4);
  D->pos_d = (int*)dalloc(16);
  D->posv_d = (int*)dalloc((size_t)U_ * 4);
  D->tok_d = (int*)dalloc(16);
  if (const char* e = getenv("QE_DS_K")) D->K = std::clamp(atoi(e), 1, 128);
  D->topk_work = dalloc(qk::topk_work_bytes(B));
  CUDA_CHECK(cudaHostAlloc((void**)&D->pos_h, 64, cudaHostAllocDefault));
  host_allocs_.push_back(D->pos_h);
  CUDA_CHECK(cudaHostAlloc((void**)&D->w1_h, ((size_t)D->rank + 2 * D->K + 16 + E_) * 4, cudaHostAllocDefault));
  host_allocs_.push_back(D->w1_h);
  D->cand_h = D->w1_h + D->rank;
  D->cand_i_h = (int*)(D->cand_h + D->K);
  (void)kvsz;
  QE_LOG("[inkling] DSpark drafter %s: %d layers, block %d, rank %d, targets", path.c_str(), D->n_layer, D->block, D->rank);
  for (int t : D->targets) QE_LOG(" %d", t);
  QE_LOG(", %.0f MiB\n", (vram_used_ - v0) / 1048576.0);
  ds_ = std::move(D);
}

void InklingModel::ds_capture(int k, const float* rows, int T) {
  DSpark& D = *ds_;
  const int nf = (int)D.targets.size();
  CUDA_CHECK(cudaMemcpy2DAsync(D.feat + (size_t)k * E_, (size_t)nf * E_ * 4, rows, (size_t)E_ * 4, (size_t)E_ * 4, T,
                               cudaMemcpyDeviceToDevice, s_));
}

void InklingModel::ds_inject(int row0, int n, int pos0) {
  DSpark& D = *ds_;
  if (n <= 0) return;
  const int nf = (int)D.targets.size(), kvsz = D.nkv * D.hd;
  qx_ = nullptr;
  CUDA_CHECK(cudaStreamSynchronize(s_));  // pos_h is reused
  D.pos_h[0] = pos0;
  CUDA_CHECK(cudaMemcpyAsync(D.pos_d, D.pos_h, 4, cudaMemcpyHostToDevice, s_));
  qk::iota_i32(D.posv_d, pos0, n, s_);
  const float* feat = D.feat + (size_t)row0 * nf * E_;
  float* ctx = mixed_[0];
  for (int il = 0; il < D.n_layer; il++) {
    DsLayer& L = D.L[il];
    ik::mix_feat(feat, nf, E_, L.fuse, L.ctx_norm, D.eps, ctx, n, s_);
    lin(L.kctx, ctx, n, k_);
    lin(L.vctx, ctx, n, v_, -1, false, true);
    qk::rmsnorm_heads(k_, L.kctx_norm, D.hd, D.nkv, D.hd, n, kvsz, D.eps, s_);
    qk::rope_neox(k_, D.posv_d, D.hd, D.nkv, D.hd, n, kvsz, D.hd, D.theta, s_);
    ik::kv_store(k_, kvsz, v_, kvsz, L.kc, L.vc, kvsz, kv_bits_, D.pos_d, n, L.ring, s_);
  }
  // f_rnn of these positions into its ring
  ik::mix_feat(feat, nf, E_, D.frnn_fuse, D.frnn_norm, D.eps, mixed_[1], n, s_);
  for (int r = 0; r < n;) {
    const int row = (pos0 + r) % D.frnn_ring;
    const int len = std::min(n - r, D.frnn_ring - row);
    CUDA_CHECK(cudaMemcpyAsync(D.frnn + (size_t)row * E_, mixed_[1] + (size_t)r * E_, (size_t)len * E_ * 4,
                               cudaMemcpyDeviceToDevice, s_));
    r += len;
  }
}

void InklingModel::ds_block(int anchor, int pos) {
  DSpark& D = *ds_;
  const int B = D.block, kvsz = D.nkv * D.hd, qsz = D.nh * D.hd;
  qx_ = nullptr;
  CUDA_CHECK(cudaStreamSynchronize(s_));
  D.pos_h[0] = pos;
  CUDA_CHECK(cudaMemcpyAsync(D.pos_d, D.pos_h, 4, cudaMemcpyHostToDevice, s_));
  qk::iota_i32(D.posv_d, pos, B, s_);
  // block input: the anchor's normed target embedding, then the mask embedding
  {
    const size_t rb = qtype_row_bytes(tok_embd_->type, E_);
    float* y = D.cand_h + 2 * D.K + 16;  // E floats of pinned scratch
    dequant_row_ref(tok_embd_->type, tok_embd_->data + (size_t)anchor * rb, E_, y);
    CUDA_CHECK(cudaMemcpyAsync(D.x, y, (size_t)E_ * 4, cudaMemcpyHostToDevice, s_));
    ik::rms_rows(D.x, tok_norm_, D.x, E_, 1, eps_, s_);
    for (int i = 1; i < B; i++)
      CUDA_CHECK(cudaMemcpyAsync(D.x + (size_t)i * E_, D.mask, (size_t)E_ * 4, cudaMemcpyDeviceToDevice, s_));
  }
  for (int il = 0; il < D.n_layer; il++) {
    DsLayer& L = D.L[il];
    ik::rms_rows(D.x, L.attn_norm, D.h, E_, B, D.eps, s_);
    lin(L.q, D.h, B, q_);
    lin(L.k, D.h, B, k_, -1, false, true);
    lin(L.v, D.h, B, v_, -1, false, true);
    qk::rmsnorm_heads(q_, L.q_norm, D.hd, D.nh, D.hd, B, qsz, D.eps, s_);
    qk::rmsnorm_heads(k_, L.k_norm, D.hd, D.nkv, D.hd, B, kvsz, D.eps, s_);
    qk::rope_neox(q_, D.posv_d, D.hd, D.nh, D.hd, B, qsz, D.hd, D.theta, s_);
    qk::rope_neox(k_, D.posv_d, D.hd, D.nkv, D.hd, B, kvsz, D.hd, D.theta, s_);
    ik::kv_store(k_, kvsz, v_, kvsz, L.kc, L.vc, kvsz, kv_bits_, D.pos_d, B, L.ring, s_);
    ik::attn(L.kc, L.vc, kv_bits_, L.ring, q_, qsz, nullptr, nullptr, nullptr, 0, L.swa ? D.swa : 0, att_, D.pos_d,
             L.swa ? D.swa + B : ctx_, B, D.nh, D.nkv, D.hd, attn_work_, attn_work_bytes_, s_, true);
    lin(L.o, att_, B, D.h);
    ik::add(D.x, D.x, D.h, (int64_t)B * E_, s_);
    ik::rms_rows(D.x, L.ffn_norm, D.h, E_, B, D.eps, s_);
    lin(L.gate, D.h, B, D.g1);
    lin(L.up, D.h, B, D.u1, -1, false, true);
    qk::silu_mul(D.g1, D.u1, D.g1, (int64_t)B * D.n_ff, s_);
    lin(L.down, D.g1, B, D.h);
    ik::add(D.x, D.x, D.h, (int64_t)B * E_, s_);
  }
  ik::rms_rows(D.x, D.out_norm, D.cur, E_, B, D.eps, s_);
  lin(output_, D.cur, B, D.base);
  // top-K base candidates of the draft slots 1..B-1
  qk::topk_rows(D.base + (size_t)n_vocab_, n_vocab_, B - 1, D.K, D.cand_v, D.cand_i, D.topk_work, s_);
  CUDA_CHECK(cudaMemsetAsync(D.state, 0, (size_t)D.rank * 4, s_));
  D.anchor_pos = pos;
}

int InklingModel::ds_head(int prev, int slot, float* logits_host) {
  DSpark& D = *ds_;
  const int R = D.rank, K = D.K;
  qx_ = nullptr;
  // joint input [state | w1[prev] | hidden(slot) | f_rnn(anchor - 1)]
  CUDA_CHECK(cudaStreamSynchronize(s_));  // w1_h is reused
  const uint16_t* w1 = (const uint16_t*)D.w1->data + (size_t)prev * R;
  for (int i = 0; i < R; i++) D.w1_h[i] = qe_bf2f(w1[i]);
  CUDA_CHECK(cudaMemcpyAsync(D.joint_in, D.state, (size_t)R * 4, cudaMemcpyDeviceToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(D.joint_in + R, D.w1_h, (size_t)R * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(D.joint_in + 2 * R, D.cur + (size_t)slot * E_, (size_t)E_ * 4, cudaMemcpyDeviceToDevice, s_));
  const int fr = ((D.anchor_pos - 1) % D.frnn_ring + D.frnn_ring) % D.frnn_ring;
  CUDA_CHECK(cudaMemcpyAsync(D.joint_in + 2 * R + E_, D.frnn + (size_t)fr * E_, (size_t)E_ * 4, cudaMemcpyDeviceToDevice, s_));
  lin(D.joint, D.joint_in, 1, D.jout);
  ik::add(D.jout, D.jout, D.joint_b, 3 * R, s_);
  ik::gru(D.jout, D.state, D.latent, R, s_);
  lin(D.hidden_out, D.latent, 1, D.corr);
  ik::add(D.corr, D.corr, D.hidden_out_b, E_, s_);
  // the correction through the lm head, on the candidates only
  const int* ci = D.cand_i + (size_t)(slot - 1) * K;
  qk::quantize_q8(D.corr, xqs_, xds_, E_, 1, s_);
  qk::matvec_rows(output_.type, output_.dev, output_.rb, E_, ci, K, xqs_, xds_, D.bias, s_);
  static const bool no_markov = getenv("QE_DS_NO_MARKOV") != nullptr;
  if (no_markov) CUDA_CHECK(cudaMemsetAsync(D.bias, 0, (size_t)K * 4, s_));
  ik::cand_logits(D.cand_v + (size_t)(slot - 1) * K, D.bias, ci, K, D.cap, D.cand_l, D.tok_d, s_);
  CUDA_CHECK(cudaMemcpyAsync(D.cand_i_h + K, D.tok_d, 4, cudaMemcpyDeviceToHost, s_));
  if (logits_host) {
    CUDA_CHECK(cudaMemcpyAsync(D.cand_h, D.cand_l, (size_t)K * 4, cudaMemcpyDeviceToHost, s_));
    CUDA_CHECK(cudaMemcpyAsync(D.cand_i_h, ci, (size_t)K * 4, cudaMemcpyDeviceToHost, s_));
  }
  CUDA_CHECK(cudaStreamSynchronize(s_));
  if (logits_host) {
    std::fill(logits_host, logits_host + n_vocab_, -INFINITY);
    for (int i = 0; i < K; i++) logits_host[D.cand_i_h[i]] = D.cand_h[i];
  }
  D.slot = slot;
  return D.cand_i_h[K];
}

int InklingModel::mtp_draft(int hrow0, const int* tokens, int n, int pos0, float* conf, float* logits_host, bool head) {
  if (!ds_) return -1;
  if (conf) *conf = 1.f;
  if (hrow0 >= 0) ds_inject(hrow0, n, pos0 - 1);  // the committed rows of the last forward
  if (!head) return -1;
  const int anchor = tokens[n - 1], pos = pos0 + n - 1;
  if (pos < 1) return -1;
  ds_block(anchor, pos);
  return ds_head(anchor, 1, logits_host);
}

int InklingModel::mtp_chain(int token, int pos, float* conf, float* logits_host) {
  if (!ds_) return -1;
  if (conf) *conf = 1.f;
  const int slot = pos - ds_->anchor_pos + 1;
  if (slot < 2 || slot >= ds_->block || slot != ds_->slot + 1) return -1;
  return ds_head(token, slot, logits_host);
}
