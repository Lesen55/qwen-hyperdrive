#include "engine.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <queue>

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}

void Sampler::dist(const float* l, int n, const std::vector<int>& history, Dist& out) {
  out.clear();
  const int k = std::min(n, p_.top_k > 0 ? p_.top_k : 256);
  // presence penalty applies to tokens seen so far
  std::vector<int> pen_ids;
  if (p_.presence_penalty != 0.f) {
    pen_ids = history;
    std::sort(pen_ids.begin(), pen_ids.end());
    pen_ids.erase(std::unique(pen_ids.begin(), pen_ids.end()), pen_ids.end());
  }
  auto logit = [&](int i) {
    float v = l[i];
    if (!pen_ids.empty() && std::binary_search(pen_ids.begin(), pen_ids.end(), i)) v -= p_.presence_penalty;
    return v;
  };
  // top-k by a min-heap scan (k is small)
  using E = std::pair<float, int>;
  std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
  if (pen_ids.empty()) {
    for (int i = 0; i < n; i++) {
      const float v = l[i];
      if ((int)heap.size() < k) heap.emplace(v, i);
      else if (v > heap.top().first) { heap.pop(); heap.emplace(v, i); }
    }
  } else {
    for (int i = 0; i < n; i++) {
      const float v = logit(i);
      if ((int)heap.size() < k) heap.emplace(v, i);
      else if (v > heap.top().first) { heap.pop(); heap.emplace(v, i); }
    }
  }
  std::vector<E> c;
  c.reserve(k);
  while (!heap.empty()) { c.push_back(heap.top()); heap.pop(); }
  std::reverse(c.begin(), c.end());  // descending
  finish(c, out);
}

void Sampler::dist_sparse(const int* idx, const float* val, int K, Dist& out) {
  out.clear();
  const int k = std::min(K, p_.top_k > 0 ? p_.top_k : K);
  std::vector<std::pair<float, int>> c(k);
  for (int i = 0; i < k; i++) c[i] = {val[i], idx[i]};
  finish(c, out);
}

void Sampler::finish(std::vector<std::pair<float, int>>& c, Dist& out) {
  std::vector<double> pr(c.size());
  const double mx = c[0].first;
  double sum = 0;
  for (size_t i = 0; i < c.size(); i++) {
    pr[i] = std::exp((c[i].first - mx) / p_.temperature);
    sum += pr[i];
  }
  size_t kk = c.size();
  if (p_.min_p > 0.f) {
    const double thr = p_.min_p * pr[0];
    while (kk > 1 && pr[kk - 1] < thr) sum -= pr[--kk];
  }
  if (p_.top_p < 1.f) {
    double acc = 0;
    for (size_t i = 0; i < kk; i++) {
      acc += pr[i] / sum;
      if (acc >= p_.top_p) { kk = i + 1; break; }
    }
  }
  double tot = 0;
  for (size_t i = 0; i < kk; i++) tot += pr[i];
  out.reserve(kk);
  for (size_t i = 0; i < kk; i++) out.emplace_back(c[i].second, (float)(pr[i] / tot));
}

int Sampler::draw(const Dist& d) {
  double r = uniform();
  for (auto& [id, p] : d) {
    r -= p;
    if (r <= 0) return id;
  }
  return d.back().first;
}

int Sampler::sample(const float* l, int n, const std::vector<int>& history) {
  if (greedy()) return (int)(std::max_element(l, l + n) - l);
  Dist d;
  dist(l, n, history, d);
  return draw(d);
}

static float prob_of(const Dist& d, int id) {
  for (auto& [i, p] : d)
    if (i == id) return p;
  return 0.f;
}

Generator::Generator(IModel& m, const SamplingParams& sp, bool use_mtp)
    : m_(m), sampler_(sp), mtp_(use_mtp && m.has_mtp()) {
  logits_.resize((size_t)kMaxTok * m.cfg().n_vocab);  // verify rows (up to draft_max + 1)
  mtp_logits_.resize(m.cfg().n_vocab);
}

int Generator::take_slot(int seq, int pos) {
  const int ns = m_.ck_slots();
  if ((int)slot_seq_.size() != ns) { slot_seq_.assign(ns, -1); slot_pos_.assign(ns, -1); slot_used_.assign(ns, 0); }
  int best = -1;
  for (int i = 0; i < ns; i++) {
    if (slot_seq_[i] < 0) { best = i; break; }
    if (best < 0 || slot_used_[i] < slot_used_[best]) best = i;
  }
  if (best < 0) return -1;
  drop_slot(best);
  slot_seq_[best] = seq;
  slot_pos_[best] = pos;
  slot_used_[best] = ++clk_;
  return best;
}

void Generator::drop_slot(int slot) {
  if (slot < 0 || slot >= (int)slot_seq_.size() || slot_seq_[slot] < 0) return;
  auto it = convs_.find(slot_seq_[slot]);
  if (it != convs_.end()) {
    auto& v = it->second.cks;
    v.erase(std::remove_if(v.begin(), v.end(), [&](auto& e) { return e.second == slot; }), v.end());
  }
  slot_seq_[slot] = -1;
}

void Generator::abandon() {
  if (live_ >= 0) {
    m_.set_capture(-1, -1);
    m_.seq_drop(live_);
    for (auto& [pos, slot] : convs_[live_].cks) slot_seq_[slot] = -1;
    convs_.erase(live_);
    live_ = -1;
  }
  history_.clear();
  pos_ = 0;
}

// Prompt cache lookup.  The recurrent (GDN) state cannot be rewound, so a request can resume from
//  (a) the live state of a conversation whose committed tokens are a prefix of the prompt, or
//  (b) a checkpoint (state after the first p tokens) whose prefix matches.
// Checkpoints are taken at the start of each generation prompt (so the next turn hits even when the
// client drops the model's reasoning from the history) and when switching to another conversation.
int Generator::start(const std::vector<int>& prompt, bool allow_reuse, int reserve, const std::atomic<bool>* cancel) {
  const double t0 = qe_now();
  const int n = (int)prompt.size();
  for (auto it = convs_.begin(); it != convs_.end();) {  // conversations whose KV was evicted
    if (!m_.seq_alive(it->first)) {
      for (auto& [p, sl] : std::vector<std::pair<int, int>>(it->second.cks)) slot_seq_[sl] = -1;
      if (live_ == it->first) live_ = -1;
      it = convs_.erase(it);
    } else {
      ++it;
    }
  }
  int best_seq = -1, best_pos = 0, best_slot = -1;
  if (allow_reuse) {
    for (auto& [sid, cv] : convs_) {
      size_t l = 0;
      while (l < cv.toks.size() && l < prompt.size() && cv.toks[l] == prompt[l]) l++;
      l = std::min<size_t>(l, n - 1);  // at least one token must be processed for fresh logits
      if (sid == live_ && cv.toks.size() <= l && (int)cv.toks.size() > best_pos) {
        best_seq = sid; best_pos = (int)cv.toks.size(); best_slot = -1;
      }
      for (auto& [p, sl] : cv.cks)
        if (p <= (int)l && p > best_pos) { best_seq = sid; best_pos = p; best_slot = sl; }
    }
  }
  // leaving the live conversation: keep its exact end state as a checkpoint
  if (live_ >= 0 && convs_.count(live_) && !(best_seq == live_ && best_slot < 0) && m_.ck_slots() > 0) {
    Conv& lc = convs_[live_];
    const int lt = (int)lc.toks.size();
    bool have = false;
    for (auto& [p, sl] : lc.cks) have |= (p == lt);
    if (!have && lt > 0) {
      const int sl = take_slot(live_, lt);
      if (sl >= 0 && sl != best_slot) {
        m_.ck_save_live(sl);
        lc.cks.emplace_back(lt, sl);
      } else if (sl >= 0) {
        slot_seq_[sl] = best_seq;  // never overwrite the slot we are about to restore
      }
    }
  }
  if (best_seq < 0 || best_pos == 0) {
    best_seq = m_.seq_create();
    m_.seq_select(best_seq);
    m_.reset();
    convs_[best_seq] = Conv{};
    best_pos = 0;
  } else {
    m_.seq_select(best_seq);
    if (best_slot >= 0) {
      m_.ck_restore(best_slot);
      slot_used_[best_slot] = ++clk_;
    }
    Conv& cv = convs_[best_seq];
    cv.toks.resize(best_pos);
    std::vector<int> stale;
    for (auto& [p, sl] : cv.cks)
      if (p > best_pos) stale.push_back(sl);  // their KV is about to be overwritten
    for (int sl : stale) drop_slot(sl);
  }
  live_ = best_seq;
  convs_[live_].used = ++clk_;
  const int p0 = best_pos;
  const double t_lookup = qe_now();
  // checkpoint at the last <|im_start|> of the prompt (start of the generation prompt)
  static const bool no_ck = getenv("QE_NO_CK") != nullptr;
  if (boundary_tok_ >= 0 && m_.ck_slots() > 0 && !no_ck) {
    for (int b = n - 1; b > p0; b--)
      if (prompt[b] == boundary_tok_) {
        const int sl = take_slot(live_, b);
        if (sl >= 0) {
          m_.set_capture(b, sl);
          convs_[live_].cks.emplace_back(b, sl);
        }
        break;
      }
  }
  reused_ = p0;
  m_.draft_vocab_add(prompt.data(), n);
  m_.prepare_context(n + reserve, n - p0 <= kMaxTok);
  const double t_prep = qe_now();
  history_ = prompt;
  {
    const SamplingParams& sp = sampler_.params();
    topk_ = !sampler_.greedy() && sp.top_k > 0 && sp.top_k <= 128 && sp.presence_penalty == 0.f && !getenv("QE_NO_GPU_TOPK");
    m_.set_topk(topk_ ? sp.top_k : 0);
    topk_ = topk_ && m_.topk_K() > 0;  // backends without GPU top-K return full logits
  }
  ng_idx_.clear();
  ng_built_ = 0;
  last_ng_ = false;
  const int m = n - p0;
  const bool greedy = sampler_.greedy();
  m_.forward(prompt.data() + p0, m, p0, greedy ? nullptr : logits_.data(), 1, greedy ? &last_ : nullptr, false, mtp_);
  if (cancel && cancel->load(std::memory_order_relaxed)) return kCancelled;
  if (!greedy) {
    if (topk_) { Dist p; row_dist(0, p); last_ = sampler_.draw(p); }
    else last_ = sampler_.sample(logits_.data(), m_.cfg().n_vocab, history_);
  }
  committed().assign(prompt.begin(), prompt.end());
  if (dm_) dm_sync(prompt);
  QE_DBG("[gen] start: lookup/restore %.1f ms, prepare_context %.1f ms, prefill %.1f ms (%d tokens)\n",
         (t_lookup - t0) * 1e3, (t_prep - t_lookup) * 1e3, (qe_now() - t_prep) * 1e3, n - p0);
  pos_ = n;
  history_.push_back(last_);
  st_ = GenStats{};
  if (mtp_) {
    const bool g = sampler_.greedy();
    float conf = 1.f;
    const int d = m_.mtp_draft(-1, &last_, 1, pos_, g && draft_pmin_ > 0 ? &conf : nullptr, g ? nullptr : mtp_logits_.data());
    drafts_.clear();
    dq_.clear();
    if (conf >= draft_pmin_) make_drafts(d, !g);
  }
  st_.t_prefill = qe_now() - t0;
  st_.generated = 1;
  return last_;
}

// Draft model: bring it to the committed prompt (prefix reuse when it continues what it has read,
// else from scratch: a recurrent state cannot be rewound).
void Generator::dm_sync(const std::vector<int>& toks) {
  if (dm_seq_ < 0 || !dm_->seq_alive(dm_seq_)) {
    dm_seq_ = dm_->seq_create();
    dm_hist_.clear();
  }
  dm_->seq_select(dm_seq_);
  dm_todo_.clear();
  size_t l = 0;
  while (l < dm_hist_.size() && l < toks.size() && dm_hist_[l] == toks[l]) l++;
  if (l < dm_hist_.size()) {
    dm_->reset();
    dm_hist_.clear();
    l = 0;
  }
  if (l < toks.size()) {
    dm_->prepare_context((int)toks.size() + 16);
    dm_->forward(toks.data() + l, (int)(toks.size() - l), (int)l, nullptr, 0);
    dm_hist_.assign(toks.begin(), toks.end());
  }
}

// Draft model proposals: read the pending committed tokens and last_ in one pass (its argmax / sample
// is d1), save the state, then chain single-token passes for d2..dk.  Sampled mode keeps each draft's
// distribution q so verification is exact rejection sampling.
void Generator::dm_make_drafts(bool sampled) {
  const int V = m_.cfg().n_vocab;
  const int k = std::max(1, std::min({want_drafts_, m_.draft_max(), mtp_cap()}));
  if (sampled && (int)dm_logits_.size() < V) dm_logits_.resize(V);
  std::vector<int> in = dm_todo_;
  in.push_back(last_);
  std::vector<int> hist = history_;
  auto next = [&](const int* toks, int n, int pos) {
    int d = -1;
    if (!sampled) {
      dm_->forward(toks, n, pos, nullptr, 1, &d);
    } else {
      dm_->forward(toks, n, pos, dm_logits_.data(), 1);
      dq_.emplace_back();
      sampler_.dist(dm_logits_.data(), V, hist, dq_.back());
      d = sampler_.draw(dq_.back());
    }
    return d;
  };
  int d = next(in.data(), (int)in.size(), (int)dm_hist_.size());
  dm_hist_.insert(dm_hist_.end(), in.begin(), in.end());
  dm_todo_.clear();
  if (!dm_->state_save()) { dm_ = nullptr; return; }  // cannot rewind: stop using the draft model
  drafts_.push_back(d);
  hist.push_back(d);
  for (int i = 1; i < k; i++) {
    d = next(&drafts_.back(), 1, (int)dm_hist_.size() + i - 1);
    drafts_.push_back(d);
    hist.push_back(d);
  }
  dm_drafted_ = true;
}

// Build the draft chain [d1 .. dk] (k <= draft_max).  Greedy: MTP argmax tokens (stop when the
// MTP's own confidence < draft_pmin).  Sampling: each draft is *sampled* from the MTP distribution
// q (kept in dq_) so verification can use exact speculative rejection sampling.
void Generator::make_drafts(int first_argmax, bool have_logits) {
  const int V = m_.cfg().n_vocab;
  const bool g = sampler_.greedy();
  std::vector<int> hist = history_;
  int d;
  if (g) {
    d = first_argmax;
  } else {
    (void)have_logits;
    dq_.emplace_back();
    mtp_dist(hist, dq_.back());
    d = sampler_.draw(dq_.back());
  }
  drafts_.push_back(d);
  hist.push_back(d);
  const int maxd = std::min({m_.draft_max(), mtp_cap(), want_drafts_});
  for (int k = 1; k < maxd; k++) {
    float conf = 1.f;
    const int a = m_.mtp_chain(d, pos_ + k, g && draft_pmin_ > 0 ? &conf : nullptr, g ? nullptr : mtp_logits_.data());
    if (g) {
      if (conf < draft_pmin_) break;
      d = a;
    } else {
      dq_.emplace_back();
      mtp_dist(hist, dq_.back());
      d = sampler_.draw(dq_.back());
    }
    drafts_.push_back(d);
    hist.push_back(d);
  }
}

void Generator::row_dist(int row, Dist& p) {
  if (topk_) sampler_.dist_sparse(m_.topk_idx(row), m_.topk_val(row), m_.topk_K(), p);
  else sampler_.dist(logits_.data() + (size_t)row * m_.cfg().n_vocab, m_.cfg().n_vocab, history_, p);
}
void Generator::mtp_dist(const std::vector<int>& hist, Dist& q) {
  if (topk_) sampler_.dist_sparse(m_.topk_idx(kMaxTok), m_.topk_val(kMaxTok), m_.topk_K(), q);
  else sampler_.dist(mtp_logits_.data(), m_.cfg().n_vocab, hist, q);
}

bool Generator::ngram_lookup(std::vector<int>& out) {
  const int N = ngram_n_, H = (int)history_.size();
  auto key = [&](int end) {  // hash of history_[end-N+1 .. end]
    uint64_t h = 1469598103934665603ull;
    for (int i = end - N + 1; i <= end; i++) h = (h ^ (uint32_t)history_[i]) * 1099511628211ull;
    return h;
  };
  // index every n-gram that has a continuation (ends before the last token), latest occurrence wins
  for (; (int)ng_built_ + 1 < H; ng_built_++)
    if ((int)ng_built_ >= N - 1) ng_idx_[key((int)ng_built_)] = (int)ng_built_;
  if (H < N) return false;
  auto it = ng_idx_.find(key(H - 1));
  if (it == ng_idx_.end()) return false;
  const int p = it->second;
  for (int i = 0; i < N; i++)  // guard against hash collisions
    if (history_[p - i] != history_[H - 1 - i]) return false;
  const int maxd = m_.draft_max();
  for (int i = p + 1; i < H && (int)out.size() < maxd; i++) out.push_back(history_[i]);
  return !out.empty();
}

// Choose the draft length maximizing expected tokens per second:
//   E[tokens | n] = 1 + sum_{k<=n} prod_{j<=k} alpha_j,   time(n) = EMA step time with n drafts.
// Unmeasured lengths are explored occasionally so their time estimate stays current.
void Generator::update_policy(int nd, int a, double dt) {
  const double ew = 0.05;
  for (int k = 0; k < nd && k <= a; k++) {
    const double hit = k < a ? 1.0 : 0.0;  // depth k was reached; accepted iff k < a
    alpha_[k] = (1 - ew) * alpha_[k] + ew * hit;
  }
  tstep_[nd] = nstep_[nd] ? (1 - 0.1) * tstep_[nd] + 0.1 * dt : dt;
  nstep_[nd]++;
  const int maxd = std::min(m_.draft_max(), mtp_cap());
  int best = maxd;
  double best_rate = 0;
  bool explore = false;
  for (int n = 1; n <= maxd; n++) {
    if (nstep_[n] < 3) { explore = true; best = n; break; }
    double e = 1, pr = 1;
    for (int k = 0; k < n; k++) { pr *= alpha_[k]; e += pr; }
    const double rate = e / tstep_[n];
    if (rate > best_rate) { best_rate = rate; best = n; }
  }
  // periodic re-measurement of the neighbouring lengths (cycling through all lengths costs too much
  // once long chains are allowed)
  if (!explore && st_.steps % 32 == 0) best = std::clamp(best + ((st_.steps / 32) % 2 ? 1 : -1), 1, maxd);
  static const bool fixed = getenv("QE_FIXED_DRAFT") != nullptr;  // tests: no timing-dependent choice
  if (fixed) best = maxd;
  want_drafts_ = best;
}

void Generator::step(std::vector<int>& out) {
  const double t0 = qe_now();
  const int V = m_.cfg().n_vocab;
  const bool g = sampler_.greedy();
  st_.steps++;
  if (!mtp_) {
    // no MTP head: n-gram (prompt / history lookup) drafts when the continuation repeats earlier text
    drafts_.clear();
    dq_.clear();
    last_ng_ = false;
    std::vector<int> ng;
    if (ngram_on_ && m_.draft_max() > 0 && ngram_lookup(ng)) {
      const int cap = std::clamp((int)std::lround(ng_run_ + 2.0), 2, m_.draft_max());
      for (int i = 0; i < (int)ng.size() && i < cap; i++) {
        drafts_.push_back(ng[i]);
        if (!g) dq_.push_back(Dist{{ng[i], 1.f}});  // deterministic proposal: q = point mass
      }
      last_ng_ = !drafts_.empty();
    }
    dm_drafted_ = false;
    if (drafts_.empty() && dm_) dm_make_drafts(!g);
  }
  if (!mtp_ && drafts_.empty()) {
    int next;
    if (g) m_.forward(&last_, 1, pos_, nullptr, 1, &next);
    else {
      m_.forward(&last_, 1, pos_, logits_.data(), 1);
      if (topk_) { Dist p; row_dist(0, p); next = sampler_.draw(p); }
      else next = sampler_.sample(logits_.data(), V, history_);
    }
    committed().push_back(last_);
    if (dm_) dm_todo_.push_back(last_);
    pos_ += 1;
    last_ = next;
    history_.push_back(next);
    out.push_back(next);
    st_.generated++;
    st_.t_decode += qe_now() - t0;
    return;
  }
  // verify [last, d1..dn] in one pass
  const int nd = (int)drafts_.size();
  const int T = 1 + nd;
  int in[8] = {last_};
  for (int i = 0; i < nd; i++) in[1 + i] = drafts_[i];
  int am[8];
  double tv = qe_now();
  if (g) m_.forward(in, T, pos_, nullptr, T, am, nd > 0);
  else m_.forward(in, T, pos_, logits_.data(), T, nullptr, nd > 0);
  st_.t_verify += qe_now() - tv;
  tv = qe_now();
  int a = 0, s;
  if (g) {
    s = am[0];
    while (a < nd && s == drafts_[a]) {
      history_.push_back(s);
      out.push_back(s);
      a++;
      s = am[a];
    }
  } else {
    // speculative rejection sampling: accept d_i with prob min(1, p_i(d_i)/q_i(d_i));
    // on rejection sample from norm(max(0, p_i - q_i)); if all accepted sample from p_nd.
    Dist p;
    for (;;) {
      row_dist(a, p);
      for (auto& [id, pv] : p) m_.draft_vocab_add(&id, 1);  // learn the vocabulary in use
      if (a == nd) { s = sampler_.draw(p); break; }
      const int x = drafts_[a];
      const float px = prob_of(p, x), qx = prob_of(dq_[a], x);
      // spec_relax_ > 1 accepts drafts more leniently (min(1, lambda*p/q)): faster, slightly biased
      // n-gram drafts (copies of earlier text) are verified exactly: relaxing them would feed loops
      const double lam = last_ng_ ? 1.0 : spec_relax_;
      if (qx > 0 && sampler_.uniform() * qx < lam * px) {
        history_.push_back(x);
        out.push_back(x);
        a++;
        continue;
      }
      Dist r;
      double tot = 0;
      for (auto& [id, pv] : p) {
        const double v = pv - prob_of(dq_[a], id);
        if (v > 0) { r.emplace_back(id, (float)v); tot += v; }
      }
      if (tot <= 0) s = sampler_.draw(p);
      else {
        for (auto& e : r) e.second = (float)(e.second / tot);
        s = sampler_.draw(r);
      }
      break;
    }
  }
  if (last_ng_) {
    st_.ng_steps++;
    st_.ng_drafted += nd;
    st_.ng_accepted += a;
    ng_run_ = 0.7 * ng_run_ + 0.3 * a;  // recent accepted run length -> n-gram draft cap
  } else {
    update_policy(nd, a, qe_now() - t0);
    st_.mtp_len[std::min(nd, 8)]++;
    for (int k = 0; k < nd && k < 8 && k <= a; k++) { st_.mtp_reach[k]++; st_.mtp_acc[k] += k < a; }
  }
  static const bool m2 = getenv("QE_MEASURE_TOP2") != nullptr;
  static long m2_rej = 0, m2_hit[4] = {0, 0, 0, 0};
  if (m2 && nd > 0 && !last_ng_) {
    if (a == 0) {
      m2_rej++;
      for (int k = 1; k < 4; k++) if (first_top_[k] == s) m2_hit[k]++;
    }
    if (st_.steps % 100 == 0 && m2_rej)
      fprintf(stderr, "[top2] first draft rejected %ld times: 2nd candidate right %.1f%%, 3rd %.1f%%, 4th %.1f%% (steps %d)\n", m2_rej,
              100.0 * m2_hit[1] / m2_rej, 100.0 * m2_hit[2] / m2_rej, 100.0 * m2_hit[3] / m2_rej, st_.steps);
  }
  st_.drafted += nd;
  st_.accepted += a;
  st_.acc_hist[std::min(a, 3)]++;
  history_.push_back(s);
  out.push_back(s);
  committed().push_back(last_);
  for (int i = 0; i < a; i++) committed().push_back(drafts_[i]);
  st_.t_sample += qe_now() - tv;
  tv = qe_now();
  static const bool hp = getenv("QE_HPROF") != nullptr;
  static double hp_rb = 0, hp_d1 = 0, hp_ch = 0, hp_n = 0;
  if (nd > 0) m_.rollback(a + 1);  // commit the accepted prefix of the verify pass
  if (!mtp_) {  // n-gram / draft model: commit and stop (no MTP head)
    if (dm_) {
      if (dm_drafted_ && a == nd) {
        // the draft read last_ and d1..d_{k-1} while drafting: only d_k is still to read
        dm_hist_.insert(dm_hist_.end(), drafts_.begin(), drafts_.end() - 1);
        dm_todo_.assign(1, drafts_.back());
      } else if (dm_drafted_) {
        dm_->state_restore();  // back to "after last_", then read the accepted drafts
        dm_todo_.assign(drafts_.begin(), drafts_.begin() + a);
      } else {  // n-gram step: everything committed is still unread by the draft
        dm_todo_.push_back(in[0]);
        dm_todo_.insert(dm_todo_.end(), drafts_.begin(), drafts_.begin() + a);
      }
    }
    pos_ += a + 1;
    last_ = s;
    st_.generated += a + 1;
    st_.t_decode += qe_now() - t0;
    return;
  }
  m_.set_carry_row(a);
  if (hp) { m_.sync(); hp_rb += qe_now() - tv; }
  const double hp1 = qe_now();
  // MTP: pairs (h_{pos+i}, x_{pos+i+1}) for i = 0..a  ->  next drafts
  int toks[8];
  for (int i = 0; i < a; i++) toks[i] = drafts_[i];
  toks[a] = s;
  pos_ += a + 1;
  last_ = s;
  m_.draft_vocab_add(&s, 1);
  st_.generated += a + 1;
  float conf = 1.f;
  const int d = m_.mtp_draft(0, toks, a + 1, pos_ - a, g && draft_pmin_ > 0 ? &conf : nullptr, g ? nullptr : mtp_logits_.data());
  for (int k = 0; k < 4; k++) first_top_[k] = m_.mtp_top4()[k];
  const double hp2 = qe_now();
  drafts_.clear();
  dq_.clear();
  last_ng_ = false;
  std::vector<int> ng;
  if (ngram_on_ && ngram_lookup(ng)) {
    // first draft from the MTP head; when it agrees with the n-gram continuation, extend with the
    // rest of the n-gram (free) instead of running the MTP chain
    int d0 = d;
    if (!g) {
      dq_.emplace_back();
      mtp_dist(history_, dq_.back());
      d0 = sampler_.draw(dq_.back());
    }
    if (d0 == ng[0]) {
      last_ng_ = true;
      const int cap = std::clamp((int)std::lround(ng_run_ + 2.0), 2, m_.draft_max());
      drafts_.push_back(d0);
      for (int i = 1; i < (int)ng.size() && i < cap; i++) {
        drafts_.push_back(ng[i]);
        if (!g) dq_.push_back(Dist{{ng[i], 1.f}});  // deterministic proposal: q = point mass
      }
    } else if (!g) {
      // keep the sampled first draft and continue with the MTP chain
      drafts_.push_back(d0);
      std::vector<int> hist = history_;
      hist.push_back(d0);
      const int maxd = std::min({m_.draft_max(), mtp_cap(), want_drafts_});
      int dd = d0;
      for (int k = 1; k < maxd; k++) {
        m_.mtp_chain(dd, pos_ + k, nullptr, mtp_logits_.data());
        dq_.emplace_back();
        mtp_dist(hist, dq_.back());
        dd = sampler_.draw(dq_.back());
        drafts_.push_back(dd);
        hist.push_back(dd);
      }
    }
  }
  if (drafts_.empty() && conf >= draft_pmin_) make_drafts(d, !g);
  if (hp) {
    hp_d1 += hp2 - hp1; hp_ch += qe_now() - hp2; hp_n += 1;
    if ((int)hp_n % 100 == 0)
      fprintf(stderr, "[hprof] step: rollback %.3f ms, mtp first draft %.3f ms, chain %.3f ms (%.2f drafts/step)\n",
              hp_rb / hp_n * 1e3, hp_d1 / hp_n * 1e3, hp_ch / hp_n * 1e3, (double)st_.drafted / st_.steps);
  }
  st_.t_mtp += qe_now() - tv;
  st_.t_decode += qe_now() - t0;
}
