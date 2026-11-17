#include "engine.h"

#include <algorithm>
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

Generator::Generator(Model& m, const SamplingParams& sp, bool use_mtp)
    : m_(m), sampler_(sp), mtp_(use_mtp && m.has_mtp()) {
  logits_.resize((size_t)qk::kMaxB * m.cfg().n_vocab);  // verify rows (up to draft_max + 1)
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

// Prompt cache lookup.  The recurrent (GDN) state cannot be rewound, so a request can resume from
//  (a) the live state of a conversation whose committed tokens are a prefix of the prompt, or
//  (b) a checkpoint (state after the first p tokens) whose prefix matches.
// Checkpoints are taken at the start of each generation prompt (so the next turn hits even when the
// client drops the model's reasoning from the history) and when switching to another conversation.
int Generator::start(const std::vector<int>& prompt, bool allow_reuse, int reserve) {
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
  m_.prepare_context(n + reserve);
  const double t_prep = qe_now();
  history_ = prompt;
  const int m = n - p0;
  if (sampler_.greedy()) m_.forward(prompt.data() + p0, m, p0, nullptr, 1, &last_, false, mtp_);
  else {
    m_.forward(prompt.data() + p0, m, p0, logits_.data(), 1, nullptr, false, mtp_);
    last_ = sampler_.sample(logits_.data(), m_.cfg().n_vocab, history_);
  }
  committed().assign(prompt.begin(), prompt.end());
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
    sampler_.dist(mtp_logits_.data(), V, hist, dq_.back());
    d = sampler_.draw(dq_.back());
  }
  drafts_.push_back(d);
  hist.push_back(d);
  const int maxd = std::min(m_.draft_max(), want_drafts_);
  for (int k = 1; k < maxd; k++) {
    float conf = 1.f;
    const int a = m_.mtp_chain(d, pos_ + k, g && draft_pmin_ > 0 ? &conf : nullptr, g ? nullptr : mtp_logits_.data());
    if (g) {
      if (conf < draft_pmin_) break;
      d = a;
    } else {
      dq_.emplace_back();
      sampler_.dist(mtp_logits_.data(), V, hist, dq_.back());
      d = sampler_.draw(dq_.back());
    }
    drafts_.push_back(d);
    hist.push_back(d);
  }
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
  const int maxd = m_.draft_max();
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
  if (!explore && st_.steps % 32 == 0) best = 1 + (st_.steps / 32) % maxd;  // periodic re-measurement
  want_drafts_ = best;
}

void Generator::step(std::vector<int>& out) {
  const double t0 = qe_now();
  const int V = m_.cfg().n_vocab;
  const bool g = sampler_.greedy();
  st_.steps++;
  if (!mtp_) {
    int next;
    if (g) m_.forward(&last_, 1, pos_, nullptr, 1, &next);
    else {
      m_.forward(&last_, 1, pos_, logits_.data(), 1);
      next = sampler_.sample(logits_.data(), V, history_);
    }
    committed().push_back(last_);
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
      sampler_.dist(logits_.data() + (size_t)a * V, V, history_, p);
      if (a == nd) { s = sampler_.draw(p); break; }
      const int x = drafts_[a];
      const float px = prob_of(p, x), qx = prob_of(dq_[a], x);
      // spec_relax_ > 1 accepts drafts more leniently (min(1, lambda*p/q)): faster, slightly biased
      if (qx > 0 && sampler_.uniform() * qx < spec_relax_ * px) {
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
  update_policy(nd, a, qe_now() - t0);
  st_.drafted += nd;
  st_.accepted += a;
  st_.acc_hist[std::min(a, 3)]++;
  history_.push_back(s);
  out.push_back(s);
  committed().push_back(last_);
  for (int i = 0; i < a; i++) committed().push_back(drafts_[i]);
  st_.t_sample += qe_now() - tv;
  tv = qe_now();
  if (nd > 0) m_.rollback(a + 1);  // commit the accepted prefix of the verify pass
  m_.set_carry_row(a);
  // MTP: pairs (h_{pos+i}, x_{pos+i+1}) for i = 0..a  ->  next drafts
  int toks[8];
  for (int i = 0; i < a; i++) toks[i] = drafts_[i];
  toks[a] = s;
  pos_ += a + 1;
  last_ = s;
  st_.generated += a + 1;
  float conf = 1.f;
  const int d = m_.mtp_draft(0, toks, a + 1, pos_ - a, g && draft_pmin_ > 0 ? &conf : nullptr, g ? nullptr : mtp_logits_.data());
  drafts_.clear();
  dq_.clear();
  if (conf >= draft_pmin_) make_drafts(d, !g);
  st_.t_mtp += qe_now() - tv;
  st_.t_decode += qe_now() - t0;
}
