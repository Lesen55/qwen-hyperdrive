#include "tokenizer.h"

#include <algorithm>
#include <climits>

#include "unicode_tables.inc"

namespace {

template <size_t N>
bool in_table(const uint32_t (&t)[N][2], uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < t[mid][0]) hi = mid;
    else if (cp > t[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}
inline bool isL(uint32_t c) { return c < 128 ? ((c | 32) >= 'a' && (c | 32) <= 'z') : in_table(kUni_L, c); }
inline bool isM(uint32_t c) { return c >= 128 && in_table(kUni_M, c); }
inline bool isN(uint32_t c) { return c < 128 ? (c >= '0' && c <= '9') : in_table(kUni_N, c); }
inline bool isWS(uint32_t c) { return in_table(kUni_WS, c); }
inline bool isLM(uint32_t c) { return isL(c) || isM(c); }
inline bool isNL(uint32_t c) { return c == '\r' || c == '\n'; }

void utf8_append(std::string& s, uint32_t cp) {
  if (cp < 0x80) s += (char)cp;
  else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
  else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
  else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
}

// decode UTF-8 to codepoints, recording byte offsets; invalid bytes map to U+FFFD-like raw values
void utf8_decode(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& offs) {
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = s[i];
    uint32_t cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c >> 5) == 6 && i + 1 < s.size()) { cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); len = 2; }
    else if ((c >> 4) == 14 && i + 2 < s.size()) { cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); len = 3; }
    else if ((c >> 3) == 30 && i + 3 < s.size()) { cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F); len = 4; }
    else { cp = 0xFFFD; len = 1; }
    cps.push_back(cp);
    offs.push_back(i);
    i += len;
  }
  offs.push_back(s.size());
}

// qwen35 pre-tokenizer:
// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
//  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
size_t match_at(const std::vector<uint32_t>& c, size_t i) {
  const size_t n = c.size();
  auto lower = [](uint32_t x) { return x < 128 ? (x | 32) : x; };
  // contractions
  if (c[i] == '\'' && i + 1 < n) {
    const uint32_t a = lower(c[i + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (i + 2 < n) {
      const uint32_t b = lower(c[i + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
  }
  // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
  {
    size_t j = i;
    if (!isLM(c[j]) && !isNL(c[j]) && !isL(c[j]) && !isN(c[j]) && j + 1 < n && isLM(c[j + 1])) j++;
    if (isLM(c[j])) {
      while (j < n && isLM(c[j])) j++;
      return j - i;
    }
  }
  if (isN(c[i])) return 1;
  // ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
  {
    size_t j = i;
    if (c[j] == ' ') j++;
    if (j < n && !isWS(c[j]) && !isLM(c[j]) && !isN(c[j])) {
      while (j < n && !isWS(c[j]) && !isLM(c[j]) && !isN(c[j])) j++;
      while (j < n && isNL(c[j])) j++;
      return j - i;
    }
  }
  // whitespace forms
  size_t e = i;
  while (e < n && isWS(c[e])) e++;
  if (e > i) {
    // \s*[\r\n]+ : up to the last newline in the run
    for (size_t k = e; k > i; k--)
      if (isNL(c[k - 1])) return k - i;
    // \s+(?!\S)
    if (e == n) return e - i;
    if (e - i >= 2) return e - i - 1;
    return 1;  // \s+
  }
  return 1;
}

// GPT-4o pre-tokenizer (inkling2, gpt-4o, llama4):
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
// \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
inline bool isUp(uint32_t c) { return c < 128 ? (c >= 'A' && c <= 'Z') : in_table(kUni_LUT, c); }
inline bool isLo(uint32_t c) { return c < 128 ? (c >= 'a' && c <= 'z') : in_table(kUni_LL, c); }
// U = upper-or-caseless letter / mark, Lw = lower-or-caseless letter / mark
inline bool isU(uint32_t c) { return isM(c) || (isL(c) && !isLo(c)); }
inline bool isLw(uint32_t c) { return isM(c) || (isL(c) && !isUp(c)); }

size_t contraction(const std::vector<uint32_t>& c, size_t j) {
  const size_t n = c.size();
  auto lower = [](uint32_t x) { return x < 128 ? (x | 32) : x; };
  if (j < n && c[j] == '\'' && j + 1 < n) {
    const uint32_t a = lower(c[j + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (j + 2 < n) {
      const uint32_t b = lower(c[j + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
  }
  return 0;
}

// letter run U*Lw+ (alt 1) or U+Lw* (alt 2) starting at j; returns its end or 0 when neither matches
size_t letters_at(const std::vector<uint32_t>& c, size_t j) {
  const size_t n = c.size();
  size_t e = j;
  while (e < n && isU(c[e])) e++;
  // alt 1: U* greedy, then Lw+ (backtracking into U* to the last caseless char if needed)
  if (e < n && isLw(c[e])) {
    while (e < n && isLw(c[e])) e++;
    return e;
  }
  for (size_t k = e; k > j; k--)
    if (isLw(c[k - 1])) return k;
  // alt 2: U+ Lw* (Lw* is empty here: c[e] is not Lw)
  return e > j ? e : 0;
}

size_t match_gpt4o(const std::vector<uint32_t>& c, size_t i) {
  const size_t n = c.size();
  // optional [^\r\n\p{L}\p{N}] prefix, then the letter run
  for (int pre = 1; pre >= 0; pre--) {
    const size_t j = i + pre;
    if (pre && (isL(c[i]) || isN(c[i]) || isNL(c[i]) || j >= n)) continue;
    if (const size_t e = letters_at(c, j)) return e + contraction(c, e) - i;
  }
  if (isN(c[i])) {
    size_t j = i;
    while (j < n && j < i + 3 && isN(c[j])) j++;
    return j - i;
  }
  // ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
  {
    size_t j = i;
    if (c[j] == ' ') j++;
    if (j < n && !isWS(c[j]) && !isL(c[j]) && !isN(c[j])) {
      while (j < n && !isWS(c[j]) && !isL(c[j]) && !isN(c[j])) j++;
      while (j < n && (isNL(c[j]) || c[j] == '/')) j++;
      return j - i;
    }
  }
  size_t e = i;
  while (e < n && isWS(c[e])) e++;
  if (e > i) {
    for (size_t k = e; k > i; k--)
      if (isNL(c[k - 1])) return k - i;
    if (e == n) return e - i;
    if (e - i >= 2) return e - i - 1;
    return 1;
  }
  return 1;
}

}  // namespace

Tokenizer::Tokenizer(const GGUFFile& f) {
  const GGUFValue* tk = f.get("tokenizer.ggml.tokens");
  const GGUFValue* mg = f.get("tokenizer.ggml.merges");
  if (!tk || !mg) QE_FATAL("GGUF has no BPE tokenizer");
  if (f.get_str("tokenizer.ggml.model", "") != "gpt2") QE_FATAL("unsupported tokenizer model");
  tokens_.reserve(tk->arr.size());
  for (auto& v : tk->arr) tokens_.push_back(v.as_str());
  if (const GGUFValue* tt = f.get("tokenizer.ggml.token_type"))
    for (auto& v : tt->arr) type_.push_back((int)v.as_int());
  type_.resize(tokens_.size(), 1);
  vocab_.reserve(tokens_.size() * 2);
  for (size_t i = 0; i < tokens_.size(); i++) vocab_.emplace(tokens_[i], (int)i);
  merges_.reserve(mg->arr.size() * 2);
  for (size_t i = 0; i < mg->arr.size(); i++) merges_.emplace(mg->arr[i].as_str(), (int)i);
  for (size_t i = 0; i < tokens_.size(); i++)
    if (type_[i] == 3 || type_[i] == 4) specials_.emplace_back(tokens_[i], (int)i);
  std::sort(specials_.begin(), specials_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
  eos_ = (int)f.get_int("tokenizer.ggml.eos_token_id", -1);
  const std::string pre = f.get_str("tokenizer.ggml.pre", "");
  gpt4o_ = pre == "inkling2" || pre == "gpt-4o" || pre == "llama4";
  ignore_merges_ = pre == "inkling2";
  for (auto& [t, id] : specials_) {
    special_ids_.emplace(t, id);
    if (std::find(special_lens_.begin(), special_lens_.end(), t.size()) == special_lens_.end()) special_lens_.push_back(t.size());
    special_first_[(unsigned char)t[0]] = true;
  }
  std::sort(special_lens_.begin(), special_lens_.end(), std::greater<size_t>());
  // GPT-2 byte <-> unicode mapping
  int n = 0;
  for (int b = 0; b < 256; b++) {
    const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    const uint32_t cp = direct ? (uint32_t)b : 256u + n++;
    utf8_append(byte_enc_[b], cp);
    byte_dec_[cp] = (uint8_t)b;
  }
}

int Tokenizer::find(const std::string& s) const {
  auto it = vocab_.find(s);
  return it == vocab_.end() ? -1 : it->second;
}

void Tokenizer::bpe(const std::string& word, std::vector<int>& out) const {
  auto cit = cache_.find(word);
  if (cit != cache_.end()) {
    out.insert(out.end(), cit->second.begin(), cit->second.end());
    return;
  }
  // symbols = mapped chars of each byte
  std::vector<std::string> sym;
  if (ignore_merges_) {  // a pre-token that is itself a vocabulary entry is kept whole
    std::string m;
    for (unsigned char b : word) m += byte_enc_[b];
    auto it = vocab_.find(m);
    if (it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  for (unsigned char b : word) sym.push_back(byte_enc_[b]);
  while (sym.size() > 1) {
    int best = INT_MAX;
    size_t bi = 0;
    for (size_t i = 0; i + 1 < sym.size(); i++) {
      auto it = merges_.find(sym[i] + " " + sym[i + 1]);
      if (it != merges_.end() && it->second < best) { best = it->second; bi = i; }
    }
    if (best == INT_MAX) break;
    sym[bi] += sym[bi + 1];
    sym.erase(sym.begin() + bi + 1);
  }
  std::vector<int> ids;
  for (auto& s : sym) {
    auto it = vocab_.find(s);
    if (it != vocab_.end()) ids.push_back(it->second);
    else  // should not happen with a complete byte-level vocab: fall back to single bytes
      for (unsigned char ch : s) {
        auto jt = vocab_.find(std::string(1, (char)ch));
        if (jt != vocab_.end()) ids.push_back(jt->second);
      }
  }
  if (cache_.size() < 200000) cache_.emplace(word, ids);
  out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
  std::vector<int> out;
  auto encode_plain = [&](const std::string& seg) {
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
    utf8_decode(seg, cps, offs);
    size_t i = 0;
    while (i < cps.size()) {
      const size_t len = gpt4o_ ? match_gpt4o(cps, i) : match_at(cps, i);
      bpe(seg.substr(offs[i], offs[i + len] - offs[i]), out);
      i += len;
    }
  };
  // control tokens: scan for positions where one starts (first-byte filter, then exact lookups by length)
  auto special_at = [&](size_t p, size_t& len) -> int {
    if (!special_first_[(unsigned char)text[p]]) return -1;
    for (size_t l : special_lens_) {
      if (p + l > text.size()) continue;
      auto it = special_ids_.find(text.substr(p, l));
      if (it != special_ids_.end()) { len = l; return it->second; }
    }
    return -1;
  };
  size_t seg = 0;
  if (parse_special && !specials_.empty()) {
    for (size_t p = 0; p < text.size(); p++) {
      size_t len = 0;
      const int sid = special_at(p, len);
      if (sid < 0) continue;
      if (p > seg) encode_plain(text.substr(seg, p - seg));
      out.push_back(sid);
      seg = p + len;
      p = seg - 1;
    }
  }
  if (seg < text.size()) encode_plain(text.substr(seg));

  return out;
}

std::string Tokenizer::token_text(int id) const {
  if (id < 0 || id >= (int)tokens_.size()) return "";
  const std::string& t = tokens_[id];
  if (is_special(id)) return t;
  std::string out;
  std::vector<uint32_t> cps;
  std::vector<size_t> offs;
  utf8_decode(t, cps, offs);
  for (uint32_t cp : cps) {
    auto it = byte_dec_.find(cp);
    if (it != byte_dec_.end()) out += (char)it->second;
    else utf8_append(out, cp);
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string s;
  for (int id : ids) s += token_text(id);
  return s;
}
