// Inkling 2 (GGUF architecture inkling2_loop) CUDA runtime: a looped MoE transformer.
//   24 physical layers; layers [core_start, core_end) run num_loops times with shared weights (each pass has
//   its own KV / convolution state, later passes get cross-pass "messages" through gated controllers).
//   Attention: no RoPE; a learned relative-position bias (r . table[distance]) over the last rel_extent
//   positions, per-query sink logits, L2-clipped q/k, RMS-normed v, sliding window (1024) on local layers.
//   Short causal convolutions on K, V (input-dependent taps on global layers), attention and FFN outputs.
//   Residuals: "block attention residuals" mix the starts of the 6-layer blocks by learned depth queries.
//   FFN: 128 experts, top-8 sigmoid routing, SITU activation, per-expert scales (NVFP4 down projections).
#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <utility>
#include <string>
#include <vector>

#include "gguf.h"
#include "imodel.h"
#include "inkling_kernels.h"

bool is_inkling_arch(const std::string& arch);

class InklingModel : public IModel {
 public:
  explicit InklingModel(const EngineOptions& opt);
  ~InklingModel() override;

  const Config& cfg() const override { return cfg_; }
  const GGUFFile& gguf() const override { return *gguf_; }
  const EngineOptions& options() const override { return opt_; }
  bool has_mtp() const override { return false; }
  const char* backend() const override { return "cuda"; }
  void forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax = nullptr,
               bool snapshot = false, bool mtp_prefill = false) override;
  void reset() override;
  void set_cancel(const std::atomic<bool>* c) override { cancel_ = c; }
  // speculative decoding (n-gram / draft-model drafts): verify passes of up to draft_max + 1 tokens, then rollback
  int draft_max() const override {
    static const bool off = getenv("QE_INK_SPEC") && atoi(getenv("QE_INK_SPEC")) == 0;
    return off ? 0 : std::min(opt_.draft_max, kMaxTok - 1);
  }
  void rollback(int n_keep) override;
  void prepare_context(int npos, bool small_prefill = false) override;
  int seq_create() override;
  void seq_select(int id) override { (void)id; }
  bool seq_alive(int id) const override { return id == seq_; }
  void seq_drop(int id) override;
  void sync() override;
  KvInfo kv_info() const override;
  // prompt cache: state checkpoints (conv states + the sliding-window KV of the local layers) in pinned host memory
  int ck_slots() const override { return (int)cks_.size(); }
  void ck_save_live(int slot) override { ck_save(slot, pos_); }
  void ck_restore(int slot) override;
  void set_capture(int pos, int slot) override {
    cap_pos_ = pos;
    cap_slot_ = slot;
  }

 private:
  // a weight matrix [N][K]: quantized (GGUF type) or float (converted to f32 on the device)
  struct W {
    uint32_t type = 0;
    int K = 0, N = 0;
    size_t rb = 0;
    const void* dev = nullptr;
    bool valid() const { return K > 0; }
  };
  struct Experts {
    uint32_t type = 0;
    int K = 0, N = 0;
    size_t rb = 0, stride = 0;     // row bytes, bytes per expert
    const uint8_t* dev = nullptr;   // resident copy (or null)
    const uint8_t* host = nullptr;  // pinned, mapped host copy (when not resident)
    float* scale = nullptr;         // per-expert output scale [n_exp] (device)
  };
  struct PLayer {
    bool swa = false;
    W q, k, v, o, gate, rel, sink, kcp, vcp, gate_inp;
    float *q_clip = nullptr, *rel_clip = nullptr, *rel_table = nullptr;
    float *k_conv = nullptr, *v_conv = nullptr, *ao_conv = nullptr, *fo_conv = nullptr;
    float *attn_norm = nullptr, *qn = nullptr, *kn = nullptr, *vn = nullptr;
    float *attn_rq = nullptr, *ffn_rq = nullptr, *ffn_norm = nullptr, *exp_b = nullptr;
    int attn_nq = 0;
    Experts gu, dn;
    bool resident = false;
    int idx = 0;
    // host layers: experts cached in the elastic VRAM cache (slot_of[e] = cache slot or -1), routing frequencies
    std::vector<int> slot_of;
    std::vector<double> freq;
    // loop controllers (core layers): message queries [4][E], per controller down [64][2E], gain / forget [E][64]
    float* msg_q = nullptr;
    W cdown[4], cgf[4];  // down [rank][2E]; gain | forget stacked [2E][rank]
  };
  struct Slot {
    int phys = 0;
    uint8_t *kc = nullptr, *vc = nullptr;  // KV rows of kv_rowb_ bytes
    int ring = 0;           // 0: linear (global layer), else ring size (local layer)
    float* hist = nullptr;  // conv state: K [3][kv], V [3][kv], attn out [3][E], ffn out [3][E]
  };

  void load();
  W upload_w(const std::string& name, bool required = true);
  float* upload_f32(const std::string& name, bool required = true);
  Experts load_experts(const std::string& name, const std::string& scale_name);
  void* dalloc(size_t bytes);
  void plan_experts();
  // Elastic expert cache: host-layer experts copied into VRAM that is idle right now -- the KV rows beyond the
  // current context of the global layers and the prefill-only workspace.  Being a cache it can be dropped at any
  // time: slots are evicted before the context grows into them or a prefill needs the workspace.
  struct CSlot {
    uint8_t *gu = nullptr, *dn = nullptr;
    int kv_row = -1;  // lowest KV row it covers (-1: workspace slot)
    int layer = -1, expert = -1;
  };
  std::vector<CSlot> cslots_;
  const uint8_t** eptr_d_ = nullptr;  // [2][n_phys][n_exp] gate_up / down pointer of each cached expert (null: not)
  const uint8_t** eptr_h_ = nullptr;  // pinned host mirror
  int kv_hi_ = 0;                     // KV rows that may hold keys / values (slots below are invalid)
  bool ws_busy_ = false;              // the prefill workspace is in use
  bool slot_valid(const CSlot& c) const { return c.kv_row < 0 ? !ws_busy_ : c.kv_row >= kv_hi_; }
  void cache_evict(bool kv, bool ws);
  void upload_eptr();
  void rebalance();  // refresh the expert cache from the routing counts
  int* cnt_d_ = nullptr;  // [n_phys][n_exp] uses of host-layer experts since the last rebalance
  int* cnt_h_ = nullptr;
  std::vector<int> cnt_h_pending_;  // prefill counts (known on the host)
  const uint8_t **ptr_gu_ = nullptr, **ptr_dn_ = nullptr;
  int steps_since_rb_ = 0;
  int cache_slots_ = 0, cache_kv_slots_ = 0;
  uint64_t cache_hits_ = 0, cache_uses_ = 0;

  void lin(const W& w, const float* x, int T, float* y, int ldy = -1, bool acc = false, bool reuse = false);
  const float* qx_ = nullptr;  // input currently quantized in xq_ / xd_ (with qK_, qT_)
  int qK_ = 0, qT_ = 0;
  void layer(int slot, int ip, const float* inp, float* out, std::vector<const float*>& blocks,
             const float* const* msgs, int T, int pos0);
  void control(const float* cur, const float* msg, int ctrl, const PLayer& L, float* out, int T);
  void moe(const PLayer& L, const float* x, int T, float* out);
  void moe_small(const PLayer& L, const float* x, int T, float* out);
  void moe_large(const PLayer& L, const float* x, int T, float* out);
  void embed(const int* tokens, int n, int pos0);
  void body(int T, int pos0, int n_out, bool want_logits, bool want_argmax);
  cudaGraphExec_t graph_[2] = {};
  void prof(int k);
  void prof_flush();
  std::vector<std::pair<int, cudaEvent_t>> prof_ev_;
  double prof_ms_[16] = {};
  int prof_n_ = 0, prof_T_ = 0;
  int graph_warm_[2] = {};
  float* out_h_ = nullptr;
  float* big_out_ = nullptr;
  int *argmax_h_ = nullptr, *pos_h_ = nullptr, *pos_d_ = nullptr;
  void ck_save(int slot, int pos);
  struct Ck {
    uint8_t* host = nullptr;
    int pos = -1;
  };
  std::vector<Ck> cks_;
  size_t ck_bytes_ = 0;
  int cap_pos_ = -1, cap_slot_ = -1, pos_ = 0;
  // speculative verify: conv inputs saved per slot and conv (K, V, attention out, FFN out) for rollback()
  std::vector<std::array<float*, 4>> save_;
  ik::ConvSave* csave_d_ = nullptr;
  int n_csave_ = 0;
  bool snap_ = false;
  int snap_pos0_ = 0;
  int* slot_d_ = nullptr;
  void* grp_d_ = nullptr;  // distinct experts of a small batch (moe_group)
  int* ngrp_d_ = nullptr;
  void chunk(const int* tokens, int T, int pos0, int n_out, float* logits_host, int* argmax_host);
  float* fresh(const std::vector<const float*>& live);

  EngineOptions opt_;
  Config cfg_{};
  std::unique_ptr<GGUFFile> gguf_;
  int E_ = 0, nh_ = 0, nkv_ = 0, hd_ = 128, n_rel_ = 16, rel_extent_ = 0, swa_ = 0, n_exp_ = 0, k_exp_ = 0;
  int n_ff_ = 0, n_conv_ = 4, block_ = 6, n_phys_ = 0, core_start_ = 0, core_end_ = 0, n_pass_ = 0, gate_rank_ = 0;
  int n_vocab_ = 0, n_vocab_pad_ = 0;
  float eps_ = 1e-6f, qk_clip_ = 0, rel_clip_ = 0, log_floor_ = 8192, log_alpha_ = 0, route_scale_ = 1;
  float situ_b_ = 4, situ_lb_ = 25, softcap_ = 0;
  std::vector<PLayer> L_;
  std::vector<Slot> slots_;
  W output_;
  float *output_norm_ = nullptr, *output_rq_ = nullptr, *tok_norm_ = nullptr;
  const GGUFTensor* tok_embd_ = nullptr;

  cudaStream_t s_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  std::vector<void*> allocs_, host_allocs_;
  size_t vram_used_ = 0;
  int U_ = 0;  // max tokens per chunk
  int seq_ = 0, next_seq_ = 1;
  const std::atomic<bool>* cancel_ = nullptr;

  // workspace
  std::vector<float*> pool_;  // hidden-state buffers [U][E]
  float *mixed_[3] = {}, *ins_[3] = {}, *xn_[3] = {}, *msgs_[4] = {};
  float *q_ = nullptr, *g_ = nullptr, *att_ = nullptr, *kp_ = nullptr, *vp_ = nullptr, *k_ = nullptr, *v_ = nullptr;
  float *r_ = nullptr, *sink_ = nullptr, *kd_ = nullptr, *vd_ = nullptr, *o_ = nullptr, *o2_ = nullptr;
  float *gcat_ = nullptr, *gd_ = nullptr, *gg_ = nullptr, *gf_ = nullptr;
  void* attn_work_ = nullptr;
  size_t attn_work_bytes_ = 0;
  // MoE
  float *logit_e_ = nullptr, *w_e_ = nullptr, *coef_ = nullptr, *ygu_ = nullptr, *yd_ = nullptr, *h_ = nullptr;
  int* goff_d_ = nullptr;
  int *ids_ = nullptr, *tok_d_ = nullptr, *eid_d_ = nullptr, *inv_d_ = nullptr;
  int *ids_h_ = nullptr, *perm_h_ = nullptr;
  uint8_t *stg_gu_ = nullptr, *stg_dn_ = nullptr;
  int stg_slots_ = 0;
  int8_t* xqs_ = nullptr;  // small-batch activations (decode: the prefill buffers below are lent to the cache)
  float* xds_ = nullptr;
  float *ygus_ = nullptr, *yds_ = nullptr, *hs_ = nullptr;
  uint8_t* big_ = nullptr;  // prefill-only buffers (ygu_, yd_, h_, xq_, xd_, xq2_, xd2_, w16_, a16_)
  size_t big_bytes_ = 0;
  int8_t* xq_ = nullptr;
  int8_t* xq2_ = nullptr;
  float* xd2_ = nullptr;
  cudaStream_t cs_ = nullptr;  // copy stream (host-layer expert streaming in prefill)
  cudaEvent_t stg_ready_[2] = {}, stg_free_[2] = {};
  float* xd_ = nullptr;
  __half *w16_ = nullptr, *a16_ = nullptr;
  size_t w16_elems_ = 0, a16_elems_ = 0;
  float* logits_d_ = nullptr;
  int* argmax_d_ = nullptr;
  float* embd_h_ = nullptr;
  int ctx_ = 0;
  int kv_bits_ = 16;
  size_t kv_rowb_ = 0;
  int ndev_layers_ = 0;
};
