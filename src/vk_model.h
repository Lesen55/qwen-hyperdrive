// qwen35 runtime on Vulkan compute (AMD Radeon, Intel Arc, NVIDIA, ...).  Weights stay in their GGUF
// formats and are dequantized inside the shaders; what does not fit in VRAM is read by the GPU from
// system RAM.  No MTP drafting (plain decoding); single KV sequence with recurrent-state checkpoints.
#pragma once
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cpu_ffn.h"
#include "imodel.h"
#include "vk_ctx.h"

// push constants of the matvec / GEMM / dequant shaders (64-bit addresses are uvec2 in GLSL)
struct MvPC { uint64_t w, x, y, tab; uint32_t K, N, T, rb, ldx, ldy, acc, wide, kg; };

class VkModel : public IModel {
 public:
  explicit VkModel(const EngineOptions& opt);
  ~VkModel() override;

  const Config& cfg() const override { return cfg_; }
  const GGUFFile& gguf() const override { return *gguf_; }
  const EngineOptions& options() const override { return opt_; }
  bool has_mtp() const override { return mtp_on_; }
  int draft_max() const override { return mtp_on_ ? std::clamp(opt_.draft_max, 1, kMaxTok - 1) : 0; }
  int mtp_draft(int hrow0, const int* tokens, int n, int pos0, float* conf = nullptr, float* logits_host = nullptr,
                bool head = true) override;
  int mtp_chain(int token, int pos, float* conf = nullptr, float* logits_host = nullptr) override;
  void rollback(int n_keep) override;
  void set_carry_row(int row) override;
  void draft_vocab_add(const int* ids, int n) override;
  const char* backend() const override { return "vulkan"; }
  void forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax = nullptr,
               bool snapshot = false, bool mtp_prefill = false) override;
  void reset() override;
  void prepare_context(int npos, bool small_prefill = false) override;
  int seq_create() override { cur_seq_ = ++next_seq_; return cur_seq_; }
  void seq_select(int id) override { QE_ASSERT_SEQ(id); }
  bool seq_alive(int id) const override { return id == cur_seq_; }
  void seq_drop(int id) override { if (id == cur_seq_) cur_seq_ = -1; }
  int ck_slots() const override { return kCkSlots; }
  void ck_save_live(int slot) override;
  void ck_restore(int slot) override;
  void set_capture(int pos, int slot) override { cap_pos_ = pos; cap_slot_ = slot; }
  KvInfo kv_info() const override;
  PlanStats plan_stats() const override { return plan_; }

 private:
  static constexpr int kCkSlots = 4;
  static constexpr int kLogitRows = 16;  // logits rows computed per submission
  struct VMat {
    uint32_t type = 0;
    int K = 0, N = 0;
    size_t rb = 0;
    VBuf buf;
    bool valid() const { return N > 0; }
  };
  struct VLayer {
    bool attn = false;
    VMat wq, wk, wv, wo, wqkv, wgate, walpha, wbeta, wssm_out, ffn_gate, ffn_up, ffn_down;
    VBuf attn_norm, post_norm, q_norm, k_norm, conv_w, dt_bias, ssm_a, ssm_norm;
    VBuf kc, vc;               // attention: KV cache
    size_t ssm_off = 0, conv_off = 0;  // GDN: offsets into the state buffers
    VBuf sv;                   // GDN: inputs of a speculative verify pass [raw | qkv | g | beta] for rollback
    VMat eh_proj;              // MTP block
    VBuf enorm, hnorm, shnorm;
  };
  void QE_ASSERT_SEQ(int id) const;
  VMat load_mat(const std::string& name, bool prefer_device);
  VBuf load_f32(const std::string& name);
  void upload(const VBuf& dst, const void* src, size_t n);
  void linear(const VMat& m, VkDeviceAddress x, int T, VkDeviceAddress y, bool acc = false, bool bar = true);  // bar: barrier after
  void matvec(const VMat& m, const MvPC& pc, bool bar);
  void gemm(const VMat& m, MvPC pc, const std::string& method, bool bar);
  double bench(const std::function<void()>& rec, int reps);
  void autotune();
  bool cm_selftest(const std::string& pipe, uint32_t M, uint32_t N);
  void release_pages(const void* p, size_t n);
  static std::string arch_profile(const VkCtx& v);
  struct Tune {
    int mv_teams = 4, mv_r = 4;  // decode matvec: 32-lane teams per workgroup x rows per team
    int mm_teams = 4, mm_r = 2;  // T = 2..8 (speculative verify) matvec
    bool coop = false;           // cooperative super-block kernel (mvc_*) for K-quants / Q8_0
    int c_teams = 16, c_r = 1, cm_teams = 16, cm_r = 1;  // its shapes for T = 1 / T > 1
    bool i8 = false;             // ... with int8 activations and packed integer dot products
    uint32_t mv_sg = 0;          // required subgroup size (0: driver default)
    std::string gemm = "gemm16"; // prefill: fused | gemm16 | cm_gemm_MxNxK
  };
  Tune tune_;
  std::string arch_;
  uint32_t wg_target_ = 512;
  size_t reserve_ = 0;
  bool drop_pages_ = false;
  int model_fd_ = -1;
  // unified memory (iGPU): weight matrices are GPU views of the mmapped model file itself (no copy,
  // no second copy in RAM); the file is split into page-aligned chunks that each hold whole tensors
  // CPU / GPU co-execution of the FFN gate / up projections on unified memory (decode and verify
  // passes): the CPU's big cores compute rows [0, r) straight from the mapped model file while the GPU
  // computes rows [r, n_ff); synchronized inside the submission (host flag, Vulkan event).  The share
  // r is chosen online from measured pass times.
  static constexpr int kHybLevels = 5;   // r = level/8 of n_ff
  bool hyb_ = false, pass_hyb_ = false;
  int hyb_level_ = 0, hyb_rows_ = 0, hyb_force_ = -1;
  std::unique_ptr<cpu::ThreadPool> pool_;
  std::vector<VkEvent> hyb_ev_;
  VBuf hyb_flags_;
  uint32_t hyb_serial_ = 0;
  cpu::Q8Act hyb_q_;
  int hyb_T_ = 0;
  double hyb_ema_[kHybLevels] = {};
  int hyb_cnt_[kHybLevels] = {}, hyb_pass_ = 0, hyb_explore_ = 0, hyb_dir_ = 1;
  void hyb_host_loop();
  void submit_pass();
  bool zero_copy_ = false;
  struct ZcChunk { size_t begin = 0, end = 0; VBuf buf; bool tried = false; };
  std::vector<ZcChunk> zc_chunks_;
  std::vector<std::pair<const uint8_t*, int>> zc_index_;  // tensor data -> chunk
  size_t zc_bytes_ = 0;
  void zc_plan();
  bool zc_view(const uint8_t* data, size_t bytes, VBuf& out);
  void norm_in(const VBuf& w, int T);
  void layer_attn(VLayer& L, int T, int pos0);
  void layer_gdn(VLayer& L, int T);
  void layer_ffn(VLayer& L, int T);
  void forward_chunk(const int* tokens, int T, int pos0);
  void forward_impl(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax, bool mtp_prefill);
  void fill_inputs(const int* tokens, int T, int pos0, const VBuf& emb, const VBuf& rope);
  void mtp_block(VkDeviceAddress hrows, const int* tokens, int T, int pos0);
  int mtp_finish(int n, float* conf, float* logits_host);
  void load_attn(VLayer& L, int il);
  static constexpr int kDraftVocabCap = 98304;
  VLayer mtp_;
  bool mtp_on_ = false;
  VBuf hid_, carry_, mtp_h_, hsafe_, hm_, emb2_h_, rope2_h_;
  VMat dvhead_;  // output head restricted to the draft vocabulary (rows copied from output_)
  std::vector<int> dv_host_;
  std::vector<uint8_t> dv_has_, dv_cls_;
  uint32_t dv_cls_on_ = 0;
  size_t dv_uploaded_ = 0;
  bool snapshot_ = false, snap_valid_ = false;
  int snap_T_ = 0;
  VkDeviceAddress rope_addr_ = 0;

  EngineOptions opt_;
  Config cfg_{};
  std::unique_ptr<GGUFFile> gguf_;
  std::unique_ptr<VkCtx> vk_;
  std::vector<VLayer> layers_;
  VMat output_, tok_embd_;
  VBuf output_norm_, tab_;
  int ub_ = 512, ctx_len_ = 0, submit_layers_ = 8;
  bool kv_q8_ = true;
  size_t kv_row_bytes_ = 0;  // bytes per (position, kv head)
  // activations
  VBuf x_, xn_, t0_, t1_, t2_, t3_, t4_, g_, beta_, logits_, part_, argmax_, ksplit_, w16_;
  VBuf x16_;
  VBuf xq8_, xd8_;                 // int8 activations of the current matvec input (DP4A kernels)
  VkDeviceAddress xq_src_ = 0;     // ... quantized from this address, still valid while ops() == xq_mark_
  uint32_t xq_T_ = 0, xq_K_ = 0;
  uint64_t xq_mark_ = ~0ull;
  VBuf emb_h_, rope_h_, logits_h_, argmax_h_, staging_;
  VBuf ssm_, conv_;
  size_t ssm_bytes_ = 0, conv_bytes_ = 0;
  std::vector<VBuf> ck_;
  bool pend_ = false;  // t4_ holds a residual still to be added to x_
  int cur_seq_ = -1, next_seq_ = 0;
  int cap_pos_ = -1, cap_slot_ = -1;
  int max_pos_ = 0;
  PlanStats plan_;
};
