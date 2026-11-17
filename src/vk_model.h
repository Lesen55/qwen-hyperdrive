// qwen35 runtime on Vulkan compute (AMD Radeon, Intel Arc, NVIDIA, ...).  Weights stay in their GGUF
// formats and are dequantized inside the shaders; what does not fit in VRAM is read by the GPU from
// system RAM.  No MTP drafting (plain decoding); single KV sequence with recurrent-state checkpoints.
#pragma once
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
  bool has_mtp() const override { return false; }
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
    uint32_t mv_sg = 0;          // required subgroup size (0: driver default)
    std::string gemm = "gemm16"; // prefill: fused | gemm16 | cm_gemm_MxNxK
  };
  Tune tune_;
  std::string arch_;
  uint32_t wg_target_ = 512;
  size_t reserve_ = 0;
  bool drop_pages_ = false;
  int model_fd_ = -1;
  void norm_in(const VBuf& w, int T);
  void layer_attn(VLayer& L, int T, int pos0);
  void layer_gdn(VLayer& L, int T);
  void layer_ffn(VLayer& L, int T);
  void forward_chunk(const int* tokens, int T, int pos0);
  void forward_impl(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax);

  EngineOptions opt_;
  Config cfg_{};
  std::unique_ptr<GGUFFile> gguf_;
  std::unique_ptr<VkCtx> vk_;
  std::vector<VLayer> layers_;
  VMat output_, tok_embd_;
  VBuf output_norm_, tab_;
  int ub_ = 512, ctx_len_ = 0;
  bool kv_q8_ = true;
  size_t kv_row_bytes_ = 0;  // bytes per (position, kv head)
  // activations
  VBuf x_, xn_, t0_, t1_, t2_, t3_, t4_, g_, beta_, logits_, part_, argmax_, ksplit_, w16_;
  VBuf x16_;
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
