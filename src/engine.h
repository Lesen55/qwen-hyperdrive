// Generation driver: sampling + MTP self-speculative decoding on top of Model.
#pragma once
#include <map>
#include <random>
#include <vector>

#include "model.h"

struct SamplingParams {
  float temperature = 1.0f;  // <= 0: greedy
  int top_k = 20;
  float top_p = 0.95f;
  float min_p = 0.0f;
  float presence_penalty = 0.0f;
  uint64_t seed = 0;  // 0 = random
};

// sparse probability distribution over the candidate tokens left after top-k/top-p/min-p
using Dist = std::vector<std::pair<int, float>>;

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p);
  bool greedy() const { return p_.temperature <= 0.f; }
  void dist(const float* logits, int n_vocab, const std::vector<int>& history, Dist& out);
  int draw(const Dist& d);
  double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
  int sample(const float* logits, int n_vocab, const std::vector<int>& history);

 private:
  SamplingParams p_;
  std::mt19937_64 rng_;
};

struct GenStats {
  int steps = 0, drafted = 0, accepted = 0, generated = 0;
  int acc_hist[4] = {0, 0, 0, 0};  // steps by number of accepted drafts
  double t_prefill = 0, t_decode = 0;
  double t_verify = 0, t_mtp = 0, t_sample = 0;
};

class Generator {
 public:
  Generator(Model& m, const SamplingParams& sp, bool use_mtp);
  void set_sampling(const SamplingParams& sp) { sampler_ = Sampler(sp); }
  void set_draft_pmin(float p) { draft_pmin_ = p; }
  void set_boundary_token(int id) { boundary_tok_ = id; }
  // Process the prompt; returns the first generated token.  If the model state currently holds a
  // strict prefix of `prompt` (previous conversation turn), only the new suffix is processed.
  int start(const std::vector<int>& prompt, bool allow_reuse = true, int reserve = 4096);
  int reused() const { return reused_; }
  // Generate the next tokens (1 .. draft_max+1, appended to `out`).
  void step(std::vector<int>& out);
  int pos() const { return pos_; }
  const GenStats& stats() const { return st_; }

 private:
  void make_drafts(int first_argmax, bool have_logits);
  Model& m_;
  Sampler sampler_;
  bool mtp_;
  int pos_ = 0;    // tokens committed into the KV/state
  int last_ = -1;  // sampled, not yet fed
  std::vector<int> drafts_;
  std::vector<Dist> dq_;     // draft distributions (sampling mode)
  float draft_pmin_ = 0.0f;  // greedy mode: stop chaining when the MTP's confidence drops below this
  std::vector<float> logits_, mtp_logits_;
  std::vector<int> history_;
  // ---- prompt cache: conversations = paged KV sequence + committed tokens + state checkpoints ----
  struct Conv {
    std::vector<int> toks;                    // tokens whose effect is in KV (and in the live state)
    std::vector<std::pair<int, int>> cks;     // (position, checkpoint slot)
    uint64_t used = 0;
  };
  std::map<int, Conv> convs_;                 // keyed by model sequence id
  int live_ = -1;                             // conversation whose state is live on the GPU
  std::vector<int> slot_seq_, slot_pos_;
  std::vector<uint64_t> slot_used_;
  uint64_t clk_ = 0;
  int boundary_tok_ = -1;                     // <|im_start|>: checkpoint before the generation prompt
  int take_slot(int seq, int pos);
  void drop_slot(int slot);
  std::vector<int>& committed() { return convs_[live_].toks; }
  // adaptive draft length: EMA of acceptance at each depth and of step time per verify size
  double alpha_[4] = {0.7, 0.6, 0.5, 0.5};
  double tstep_[5] = {0, 0, 0, 0, 0};
  int nstep_[5] = {0, 0, 0, 0, 0};
  int want_drafts_ = 3;
  void update_policy(int nd, int a, double dt);
  int reused_ = 0;
  GenStats st_;
};
