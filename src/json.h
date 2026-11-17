// Minimal JSON value / parser / serializer (enough for the OpenAI-compatible API).
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class Json {
 public:
  enum Type { Null, Bool, Number, String, Array, Object };
  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : t_(Bool), b_(b) {}
  Json(int v) : t_(Number), n_(v) {}
  Json(int64_t v) : t_(Number), n_((double)v) {}
  Json(double v) : t_(Number), n_(v) {}
  Json(const char* s) : t_(String), s_(s) {}
  Json(std::string s) : t_(String), s_(std::move(s)) {}
  static Json array() { Json j; j.t_ = Array; return j; }
  static Json object() { Json j; j.t_ = Object; return j; }

  Type type() const { return t_; }
  bool is_null() const { return t_ == Null; }
  bool is_string() const { return t_ == String; }
  bool is_array() const { return t_ == Array; }
  bool is_object() const { return t_ == Object; }
  bool is_number() const { return t_ == Number; }
  bool is_bool() const { return t_ == Bool; }

  const std::string& str() const { return s_; }
  double num() const { return n_; }
  bool boolean() const { return b_; }
  const std::vector<Json>& arr() const { return a_; }
  std::vector<Json>& arr() { return a_; }
  const std::vector<std::pair<std::string, Json>>& items() const { return o_; }

  bool has(const std::string& k) const {
    for (auto& [kk, v] : o_) if (kk == k) return true;
    return false;
  }
  const Json& operator[](const std::string& k) const {
    static const Json null;
    for (auto& [kk, v] : o_) if (kk == k) return v;
    return null;
  }
  Json& operator[](const std::string& k) {
    if (t_ == Null) t_ = Object;
    for (auto& [kk, v] : o_) if (kk == k) return v;
    o_.emplace_back(k, Json());
    return o_.back().second;
  }
  void push_back(Json v) {
    if (t_ == Null) t_ = Array;
    a_.push_back(std::move(v));
  }
  // typed getters with defaults
  std::string get_str(const std::string& k, const std::string& def = "") const {
    const Json& v = (*this)[k];
    return v.is_string() ? v.s_ : def;
  }
  double get_num(const std::string& k, double def) const {
    const Json& v = (*this)[k];
    return v.is_number() ? v.n_ : def;
  }
  bool get_bool(const std::string& k, bool def) const {
    const Json& v = (*this)[k];
    return v.is_bool() ? v.b_ : def;
  }

  static Json parse(const std::string& s);
  std::string dump() const;
  std::string dump_py() const;  // Python json.dumps style separators (", " and ": "), as Jinja tojson

 private:
  Type t_ = Null;
  bool b_ = false;
  double n_ = 0;
  std::string s_;
  std::vector<Json> a_;
  std::vector<std::pair<std::string, Json>> o_;
  friend struct JsonParser;
};

std::string json_escape(const std::string& s);
