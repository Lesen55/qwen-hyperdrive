// Generation driver: sampling + MTP self-speculative decoding on top of Model.
#pragma once
#include <map>
#include <unordered_map>
#include <random>
#include <vector>

#include "common.h"
#include "imodel.h"

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
  // from GPU top-K candidates (sorted descending); exact when K >= top_k and no presence penalty
  void dist_sparse(const int* idx, const float* val, int K, Dist& out);
  const SamplingParams& params() const { return p_; }
  int draw(const Dist& d);
  double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
  int sample(const float* logits, int n_vocab, const std::vector<int>& history);

 private:
  SamplingParams p_;
  std::mt19937_64 rng_;
  void finish(std::vector<std::pair<float, int>>& c, Dist& out);
};

struct GenStats {
  int steps = 0, drafted = 0, accepted = 0, generated = 0;
  int acc_hist[4] = {0, 0, 0, 0};  // steps by number of accepted drafts
  int ng_steps = 0, ng_drafted = 0, ng_accepted = 0;  // n-gram drafted steps
  // MTP / draft-model steps: chain lengths, and per depth k how often it was reached / accepted
  int mtp_len[9] = {}, mtp_reach[8] = {}, mtp_acc[8] = {};
  double t_prefill = 0, t_decode = 0;
  double t_verify = 0, t_mtp = 0, t_sample = 0;
};

class Generator {
 public:
  Generator(IModel& m, const SamplingParams& sp, bool use_mtp);
  void set_sampling(const SamplingParams& sp) { sampler_ = Sampler(sp); }
  void set_draft_pmin(float p) { draft_pmin_ = p; }
  void set_boundary_token(int id) { boundary_tok_ = id; }
  void set_spec_relax(float l) { spec_relax_ = l < 1.f ? 1.f : l; }
  void set_mtp_max(int n) { mtp_max_ = n < 1 ? 1 : n; }
  void set_ngram(bool on, int n = 3) { ngram_on_ = on; ngram_n_ = n < 2 ? 2 : n; }
  // separate draft model (same tokenizer, e.g. Qwen3.5-0.8B) for targets without an MTP head
  void set_draft_model(IModel* d) { dm_ = d; }
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
  IModel& m_;
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
  int boundary_tok_ = -1;
  float spec_relax_ = 1.f;                     // <|im_start|>: checkpoint before the generation prompt
  int take_slot(int seq, int pos);
  void drop_slot(int slot);
  std::vector<int>& committed() { return convs_[live_].toks; }
  // adaptive draft length: EMA of acceptance at each depth and of step time per verify size
  double alpha_[8] = {0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  double tstep_[9] = {};
  int nstep_[9] = {};
  int want_drafts_ = 3;
  // draft model: it has processed dm_hist_ (a prefix of the committed tokens); dm_todo_ are committed
  // tokens it still has to read before drafting from last_
  IModel* dm_ = nullptr;
  int dm_seq_ = -1;
  std::vector<int> dm_hist_, dm_todo_;
  std::vector<float> dm_logits_;
  bool dm_drafted_ = false;
  void dm_sync(const std::vector<int>& toks);
  void dm_make_drafts(bool sampled);
  int mtp_max_ = 5;  // MTP chain length cap (the adaptive policy picks the length below it)
  int mtp_long_ = 7, long_ctx_ = 16384;  // longer chains at long context (passes are PCIe-bound, drafts cheap)
  int mtp_cap() const {
    static const int lc = getenv("QE_MTP_LONG") ? atoi(getenv("QE_MTP_LONG")) : -1;
    const int lng = lc >= 1 ? lc : mtp_long_;
    return pos_ >= long_ctx_ ? std::max(mtp_max_, lng) : mtp_max_;
  }
  // n-gram ("prompt lookup") drafts: continuation of the latest earlier occurrence of the last n tokens
  bool ngram_on_ = true, last_ng_ = false;
  double ng_run_ = 3.0;
  int first_top_[4] = {-1, -1, -1, -1};
  int ngram_n_ = 3;
  std::unordered_map<uint64_t, int> ng_idx_;
  size_t ng_built_ = 0;
  bool ngram_lookup(std::vector<int>& out);
  bool topk_ = false;
  void row_dist(int row, Dist& p);                                // verify row (logits_ or GPU top-K)
  void mtp_dist(const std::vector<int>& hist, Dist& q);          // last MTP head output
  void update_policy(int nd, int a, double dt);
  int reused_ = 0;
  GenStats st_;
};
