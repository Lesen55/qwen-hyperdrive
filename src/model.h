// qwen35 hybrid (Gated DeltaNet + gated full attention) model runtime.
#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "kernels.h"
#include "cpu_ffn.h"
#include "imodel.h"
#include <cuda.h>

// A weight matrix [N rows][K] in GGUF quantized form.
struct Mat {
  std::string name;
  uint32_t type = 0;
  int K = 0, N = 0;
  size_t row_bytes = 0, bytes = 0;
  const uint8_t* host = nullptr;  // mmap / pinned host copy
  void* dev = nullptr;            // resident device copy (nullptr => streamed from host)
  int stream_slot_off = -1;       // offset inside the layer's streaming slot
  const uint8_t* cpu_w = nullptr; // weights used by the CPU kernels (native or R4)
  const int8_t* cpu_tab = nullptr;  // R4 value table (null = native format)
  bool valid() const { return K > 0; }
};

struct Layer {
  bool attn = false;
  float *attn_norm = nullptr, *post_norm = nullptr;
  // full attention
  Mat wq, wk, wv, wo;
  float *q_norm = nullptr, *k_norm = nullptr;
  // gated delta net
  Mat wqkv, wgate, walpha, wbeta, wssm_out;
  float *conv_w = nullptr, *dt_bias = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
  // ffn
  Mat ffn_gate, ffn_up, ffn_down;
  // MTP (nextn) block extras
  Mat eh_proj;
  float *enorm = nullptr, *hnorm = nullptr, *shared_head_norm = nullptr;
  // runtime state
  void *kcache = nullptr, *vcache = nullptr;
  float *conv_state = nullptr, *ssm_state = nullptr;
  float* ssm_alt = nullptr;  // ping-pong state for speculative verification
  float *sv_qkv = nullptr, *sv_g = nullptr, *sv_beta = nullptr;  // saved recurrence inputs of a verify pass
  float* conv_hist = nullptr;
  bool ffn_streamed = false;
  int cpu_rows = 0;                      // FFN rows [0, cpu_rows) computed on the CPU during decode
  uint32_t job_seq = 0;
  bool cpu_r4 = false;
  Mat gate_g, up_g, down_g;              // GPU (streamed) part
  Mat gate_c, up_c, down_c;              // CPU part (also streamed to the GPU for prefill)
  size_t stream_bytes_gpu = 0, stream_bytes_full = 0;
  const uint8_t* stream_host = nullptr;  // pinned host block
  uint8_t* borrowed_dev = nullptr;       // full copy of the block living in the KV pool tail (elastic KV)
  int gdn_idx = -1;                      // index among GDN layers (checkpoint layout)
  bool ffn_skip = false;                 // FFN dropped (--skip-ffn)
  bool hyb_on = true;                    // decode: CPU computes its row slice (else whole layer streamed)
  // sensenova_loop: residual-stream routers of the attention / FFN sublayers (fp32 [8][2E] + [8])
  float *mhc_attn_w = nullptr, *mhc_attn_b = nullptr, *mhc_ffn_w = nullptr, *mhc_ffn_b = nullptr;
};

class Model : public IModel {
 public:
  explicit Model(const EngineOptions& opt);
  ~Model();

  const Config& cfg() const override { return cfg_; }
  const GGUFFile& gguf() const override { return *gguf_; }

  // Run the trunk on `n` tokens at positions pos0..pos0+n-1.  Writes logits for the last
  // `n_logits` tokens into host buffer `logits` ([n_logits][n_vocab]) if non-null.
  // Run the trunk on `n` tokens at positions pos0..pos0+n-1.  For the last `n_logits` rows writes
  // logits (host, [n_logits][n_vocab]) if `logits` != null and/or the argmax token ids if
  // `argmax` != null.  `snapshot`: keep per-token recurrent snapshots so rollback() can undo all
  // but the first k tokens (speculative verification, n <= draft_max+1).
  // `mtp_prefill`: also feed the (h_{p-1}, x_p) pairs of these tokens through the MTP block so its
  // KV cache covers the prompt.
  void forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax = nullptr,
               bool snapshot = false, bool mtp_prefill = false) override;
  // Must follow every forward(..., snapshot=true): keep the recurrent effect of the first `n_keep`
  // tokens of that pass (n_keep == n commits everything).  KV entries beyond are overwritten later.
  void rollback(int n_keep) override;
  // MTP draft: pairs (h_row[i], tokens[i]) at MTP positions pos0+i, where h rows are the trunk's final
  // normed hidden states of the previous forward (rows hrow0..hrow0+n-1) or the carried row (hrow0 = -1).
  // Returns the greedy draft token predicted after the last pair.
  // logits_host (optional): receives the MTP logits of the last row (for sampled drafts).
  // head=false: only run the MTP block (fills its KV for these positions), no logits / draft
  int mtp_draft(int hrow0, const int* tokens, int n, int pos0, float* conf = nullptr, float* logits_host = nullptr,
                bool head = true) override;
  // Chained draft: pair (MTP output hidden of the previous mtp_* call, token) at MTP position pos.
  int mtp_chain(int token, int pos, float* conf = nullptr, float* logits_host = nullptr) override;
  int draft_max() const override { return spec_ ? opt_.draft_max : 0; }
  bool spec_ = true;  // verify passes with rollback (MTP or n-gram drafts)
  bool rope_table_ = false;  // RoPE from the (YaRN) frequency table
  bool state_save() override;
  void state_restore() override;
  uint8_t* state_bk_dev_ = nullptr;
  void set_carry_row(int row) override;  // carried hidden := row of the last forward (for prefix continuation)
  // ---- paged KV sequences + recurrent-state checkpoints (prompt caching across conversations) ----
  int seq_create() override;
  void seq_select(int id) override;   // KV positions of the selected sequence are used by forward()
  bool seq_alive(int id) const override { return seqs_.count(id) != 0; }
  void seq_drop(int id) override;
  int seq_current() const { return cur_seq_; }
  int ck_slots() const override { return (int)ck_host_.size(); }
  void ck_save_live(int slot) override;   // current GDN/conv state + MTP carry -> host slot
  void ck_restore(int slot) override;     // host slot -> current state
  // capture a checkpoint of the state after the first `pos` tokens during the next forward()
  void set_capture(int pos, int slot) override { cap_pos_ = pos; cap_slot_ = slot; }
  void draft_vocab_add(const int* ids, int n) override;
  // sampling top-K on the GPU: when K > 0, forward()/mtp_* with a host logits pointer only fill
  // topk rows (forward: row j of the requested logits; MTP: row kMaxB) instead of the full rows
  void set_topk(int K) override { topk_K_ = K; }
  void sync() override { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
  // sensenova_loop residual streams
  float *mhc_a_ = nullptr, *mhc_b_ = nullptr, *mhc_branch_ = nullptr, *mhc_pk_ = nullptr;
  bool mhc_on_ = false;
  const float* mhc_out_ = nullptr;
  int top2_[4] = {-1, -1, -1, -1};  // QE_MEASURE_TOP2: top-4 ids of the last MTP head output
  int topk_K() const override { return topk_K_; }
  const int* topk_idx(int row) const override { return topk_i_h_ + (size_t)row * kTopKMax; }
  const float* topk_val(int row) const override { return topk_v_h_ + (size_t)row * kTopKMax; }
  void trim_cache();  // drop old cached sequences beyond the budget and compact the KV pool  // tokens the MTP draft head may propose
  // Elastic KV: reserve KV space for `npos` positions and lend the rest of the KV pool to
  // streamed FFN layers (evicted automatically if the context later grows).
  void prepare_context(int npos, bool small_prefill = false) override;
  int borrowed_layers() const;
  KvInfo kv_info() const override;
  PlanStats plan_stats() const override { return {plan_res_mb_, plan_stream_mb_, plan_cpu_mb_, plan_res_ffn_}; }
  const char* backend() const override { return "cuda"; }
  const int* mtp_top4() const override { return top2_; }
  double plan_res_mb_ = 0, plan_stream_mb_ = 0, plan_cpu_mb_ = 0;
  int plan_res_ffn_ = 0;
  const EngineOptions& options() const override { return opt_; }
  int kv_limit() const;  // carried hidden := row of the last forward (for prefix continuation)
  bool has_mtp() const override { return cfg_.n_mtp > 0 && opt_.use_mtp; }
  void reset() override;  // clear recurrent state (KV positions are simply overwritten)

  size_t vram_used() const { return vram_used_; }
  double weight_bytes_streamed() const { return stream_total_; }

 private:
  void load_config();
  void plan_and_load();
  Mat make_mat(const std::string& name, bool required = true);
  float* upload_f32(const std::string& name);
  void* dalloc(size_t bytes);
  double cpu_bw(uint32_t type, bool r4 = false);
  std::vector<std::vector<uint8_t>> r4_store_;
  std::vector<std::vector<uint8_t>> requant_store_;  // load-time requantized weights (host)
  void requant_mtp();
  std::map<uint32_t, double> cpu_bw_;

  // compute helpers
  void linear(Layer* L, Mat& m, const float* x, int T, float* y, int ldy, bool acc,
              cudaStream_t s);
  const void* mat_ptr(Layer* L, Mat& m, cudaStream_t s);
  const float* gqp_ = nullptr;  // activation cached in int8-per-256 form for the GQ GEMM
  int gqp_K_ = 0, gqp_T_ = 0;
  bool gqp_xs_ = false, use_gq_ = true;
  void prepare_input(const float* x, int K, int T, cudaStream_t s, int fmt = 0);
  void layer_attn(int il, int T, int pos0);
  void layer_gdn(int il, int T);
  void layer_ffn(int il, int T);
  void embed(const int* tokens, int n, float* dst);

  EngineOptions opt_;
  Config cfg_{};
  std::unique_ptr<GGUFFile> gguf_;
  std::vector<Layer> layers_;  // n_layer + n_mtp
  Mat tok_embd_, output_;
  float* output_norm_ = nullptr;

  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  size_t vram_used_ = 0;
  uint8_t* arena_ = nullptr;
  // KV pool (elastic)
  uint8_t* kv_pool_ = nullptr;
  size_t kv_pool_bytes_ = 0, kv_block_bytes_ = 0, borrow_floor_ = 0;
  int n_kv_blocks_ = 0;
  std::vector<int> borrowed_;
  struct SeqInfo {
    std::vector<int> blocks;  // logical -> physical KV block
    std::vector<std::vector<uint8_t>> host;  // offloaded KV blocks (inactive conversation)
    uint64_t used = 0;
  };
  size_t host_kv_bytes_ = 0;
  void seq_offload(int id);
  void seq_restore(int id);
  std::map<int, SeqInfo> seqs_;
  int cur_seq_ = -1, next_seq_ = 1;
  uint64_t clock_ = 0;
  std::vector<int> blk_owner_;  // physical block -> sequence id (-1 free)
  int* btab_dev_ = nullptr;
  qk::KVMap kvmap() const { return {(long long)kv_block_bytes_, btab_dev_}; }
  void upload_btab();
  int alloc_block();
  int max_used_block() const;
  // checkpoints (pinned + mapped host memory, written by kernels directly)
  std::vector<uint8_t*> ck_host_, ck_dev_;
  size_t ck_state_off(int gi) const;
  size_t ck_conv_off(int gi) const;
  size_t ck_carry_off() const;
  int n_gdn_ = 0;
  int cap_pos_ = -1, cap_slot_ = -1, snap_t_ = -1;
  bool legacy16_ = false;
  bool zc_ = true;
  double measured_pcie_ = 0;
  void calibrate_host();
  bool state16_ = false;  // GDN recurrent state stored as fp16
  size_t state_bytes() const { return (size_t)cfg_.nv * cfg_.dv * cfg_.dk * (state16_ ? 2 : 4); }
  qk::GdnBatch* gdn_tab_ = nullptr;  // per-GDN-layer pointers for the batched rollback
  int n_gdn_tab_ = 0;
  // online choice of the decode CPU mode (0 = GPU-only, 1 = hybrid)
  static constexpr int kDmLevels = 5;
  double dm_ema_[kDmLevels] = {};
  int dm_cnt_[kDmLevels] = {}, dm_pass_ = 0, dm_explore_ = 0, dm_last_ = -2, dm_dir_ = 1;
  bool hyb_dirty_ = false;
  bool fuse_ = true;              // decode: fused residual + norm + quantize kernels
  const float* pend_ = nullptr;   // deferred residual (x_ += pend_ in the next norm_in)
  void norm_in(const float* w, float* y, int T, bool quant);
  void residual(const float* src, int T);  // small H2D transfers by kernels (zero-copy) instead of the copy engine
  static constexpr int kTopKMax = 128;
  int topk_K_ = 0;
  int* topk_i_h_ = nullptr;
  float* topk_v_h_ = nullptr;
  int* topk_i_d_ = nullptr;
  float* topk_v_d_ = nullptr;
  void* topk_work_ = nullptr;
  void topk_to_host(const float* dev_logits, int V, int rows, int host_row0);
  static constexpr int kDraftVocabCap = 131072;
  std::vector<int> dv_host_;
  std::vector<uint8_t> dv_has_, dv_cls_;
  uint32_t dv_cls_on_ = 0;
  std::vector<float> dv_tmp_;
  int* dv_dev_ = nullptr;
  bool dv_dirty_ = false;
  float* state_bk_host_ = nullptr;  // pinned backup of the GDN states for the fp16-GEMM guard
  void ensure_kv(int npos);
  void refresh_streamed();
  size_t arena_bytes_ = 0;
  double stream_total_ = 0;
  std::vector<void*> allocs_;
  std::vector<void*> host_allocs_;

  // streaming
  uint8_t* stream_buf_ = nullptr;
  size_t stream_buf_bytes_ = 0;
  cudaStream_t copy_stream_ = nullptr;
  static constexpr int kMaxSlots = 6;
  cudaEvent_t slot_ready_[kMaxSlots] = {}, slot_free_[kMaxSlots] = {};
  int slot_layer_[kMaxSlots] = {-1, -1, -1, -1, -1, -1};
  bool slot_waited_[kMaxSlots] = {};
  int nslots_ = 0, last_pf_ = -1;
  size_t slot_bytes_ = 0;
  int find_slot(int li) const;
  void ring_reset();
  std::vector<int> streamed_layers_;
  bool stream_full_ = false;  // current streaming mode: full layers (prefill) or GPU part only (decode)
  void set_stream_mode(bool full);
  // hybrid CPU FFN
  static constexpr int kCpuMaxB = 8;
  std::unique_ptr<cpu::ThreadPool> pool_;
  float *hx_in_ = nullptr, *hy_out_ = nullptr, *t5_ = nullptr;
  volatile uint32_t* hflags_ = nullptr;  // [0] = input ready seq, [16] = output ready seq
  CUdeviceptr dflags_ = 0;
  uint32_t seq_ = 0;
  double cpu_prof_wait_ = 0, cpu_prof_comp_ = 0, cpu_prof_total_ = 0;
  int cpu_prof_n_ = 0;
  double cpu_phase_[4] = {0, 0, 0, 0}, cpu_phase_bytes_ = 0;
  bool gprof_on_ = false;
  std::vector<cudaEvent_t> gp_evs_, cp_evs_;
  int gp_n_ = 0, cp_n_ = 0, gp_cnt_ = 0;
  double cp_bytes_ = 0, gp_acc_[4] = {0, 0, 0, 0};
  cudaEvent_t gp_start_ = nullptr, gp_end_ = nullptr;
  cudaEvent_t gp_ev();
  cudaEvent_t cp_ev();
  std::vector<cudaEvent_t> sec_evs_;
  double sec_acc_[5] = {0, 0, 0, 0, 0};
  cudaEvent_t sec_ev(int il, int k);
  cpu::Q8Act cx_, ch_;
  std::vector<float> cg_, cu_;
  void cpu_ffn_job(const std::vector<std::pair<int, uint32_t>>& jobs, int T, int tid);
  void issue_copy(int slot, int li);
  int next_streamed(int li) const;
  void stream_release(int li);

  // workspace
  struct WSet {
    uint8_t* base = nullptr;
    size_t bytes = 0;
    float *x = nullptr, *xn = nullptr, *t0 = nullptr, *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr;
    float *g = nullptr, *beta = nullptr, *xd = nullptr, *hid = nullptr;
    int8_t* xq = nullptr;
    __half *a16 = nullptr, *c16 = nullptr, *w16 = nullptr;
    void* fa = nullptr;
  };
  WSet ws_big_, ws_small_;
  bool ws_is_big_ = true, ws_lent_ = false;
  std::vector<int> ws_borrowed_;
  float* hsafe_ = nullptr;
  void select_ws(bool big);
  void ws_lend();
  void ws_reclaim();
  int Tmax_ = 0;
  float *x_ = nullptr, *xn_ = nullptr, *t0_ = nullptr, *t1_ = nullptr, *t2_ = nullptr,
        *t3_ = nullptr, *t4_ = nullptr;
  float *g_ = nullptr, *beta_ = nullptr;
  int8_t* xq_ = nullptr;
  float* xd_ = nullptr;
  __half* a16_ = nullptr;
  __half* w16_ = nullptr;
  __half* c16_ = nullptr;
  bool gemm16_ = false, use16_ = false;
  int* ovf_dev_ = nullptr;
  float* hcarry_bk_ = nullptr;
  size_t w16_elems_ = 0;
  const float* prepared_ = nullptr;  // input currently quantized in xq_/xd_
  const float* prep16_ = nullptr;    // input currently converted in a16_
  int prep16_K_ = 0, prep16_T_ = 0;
  bool use_mmq_ = true;
  int mmq_gs_ = 32, mmq_max_T_ = 384;
  int prepared_K_ = 0, prepared_T_ = 0;
  float* attn_work_ = nullptr;
  void* fa_work_ = nullptr;
  size_t attn_work_bytes_ = 0;
  int* pos_dev_ = nullptr;
  float* logits_dev_ = nullptr;
  float* hid_ = nullptr;        // [Tmax][n_embd] final normed hidden rows of the last forward
  float* hcarry_ = nullptr;     // [n_embd] normed hidden of the last token before the current chunk
  int* argmax_dev_ = nullptr;
  float* mtp_h_ = nullptr;      // [n_embd] normed output of the last MTP block call (for chaining)
  float* conf_dev_ = nullptr;
  int mtp_finish(int n, float* conf, float* logits_host);
  bool snap_valid_ = false;
  bool snapshot_ = false;
  int snap_T_ = 0;
  void mtp_block(const float* hrows, const int* tokens, int T, int pos0);
  float* embd_host_ = nullptr;
};
