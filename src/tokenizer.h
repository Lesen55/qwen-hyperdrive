// Byte-level BPE tokenizer (GPT-2 style) with the qwen35 pre-tokenizer, loaded from GGUF metadata.
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

class Tokenizer {
 public:
  explicit Tokenizer(const GGUFFile& f);
  // parse_special: recognize control tokens like <|im_start|> inside the text
  std::vector<int> encode(const std::string& text, bool parse_special = true) const;
  std::string decode(const std::vector<int>& ids) const;
  std::string token_text(int id) const;  // decoded bytes of one token (may be partial UTF-8)
  int find(const std::string& token_str) const;  // exact token string -> id, -1 if absent
  int eos() const { return eos_; }
  int vocab_size() const { return (int)tokens_.size(); }
  bool is_special(int id) const { return id >= 0 && id < (int)type_.size() && (type_[id] == 3 || type_[id] == 4); }

 private:
  void bpe(const std::string& word, std::vector<int>& out) const;
  std::vector<std::string> tokens_;   // GPT-2 unicode-mapped strings
  std::vector<int> type_;
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> merges_;  // "a b" -> rank
  std::vector<std::pair<std::string, int>> specials_;  // sorted by length desc
  std::string byte_enc_[256];            // byte -> mapped UTF-8 string
  std::unordered_map<uint32_t, uint8_t> byte_dec_;  // mapped codepoint -> byte
  int eos_ = -1;
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
};
