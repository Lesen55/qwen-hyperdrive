#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

struct JsonParser {
  const std::string& s;
  size_t i = 0;
  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m + " at " + std::to_string(i)); }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
  }
  static void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
  }
  uint32_t hex4() {
    if (i + 4 > s.size()) fail("bad \\u escape");
    uint32_t v = (uint32_t)std::strtoul(s.substr(i, 4).c_str(), nullptr, 16);
    i += 4;
    return v;
  }
  std::string str() {
    if (s[i] != '"') fail("expected string");
    i++;
    std::string o;
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c != '\\') { o += c; continue; }
      if (i >= s.size()) fail("bad escape");
      char e = s[i++];
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            i += 2;
            const uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          put_utf8(o, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    if (i >= s.size()) fail("unterminated string");
    i++;
    return o;
  }
  Json value() {
    ws();
    if (i >= s.size()) fail("unexpected end");
    const char c = s[i];
    if (c == '{') {
      i++;
      Json o = Json::object();
      ws();
      if (s[i] == '}') { i++; return o; }
      for (;;) {
        ws();
        std::string k = str();
        ws();
        if (s[i] != ':') fail("expected :");
        i++;
        o.o_.emplace_back(std::move(k), value());
        ws();
        if (s[i] == ',') { i++; continue; }
        if (s[i] == '}') { i++; return o; }
        fail("expected , or }");
      }
    }
    if (c == '[') {
      i++;
      Json a = Json::array();
      ws();
      if (s[i] == ']') { i++; return a; }
      for (;;) {
        a.a_.push_back(value());
        ws();
        if (s[i] == ',') { i++; continue; }
        if (s[i] == ']') { i++; return a; }
        fail("expected , or ]");
      }
    }
    if (c == '"') return Json(str());
    if (s.compare(i, 4, "true") == 0) { i += 4; return Json(true); }
    if (s.compare(i, 5, "false") == 0) { i += 5; return Json(false); }
    if (s.compare(i, 4, "null") == 0) { i += 4; return Json(); }
    char* end = nullptr;
    const double v = std::strtod(s.c_str() + i, &end);
    if (end == s.c_str() + i) fail("bad value");
    i = end - s.c_str();
    return Json(v);
  }
};

Json Json::parse(const std::string& s) {
  JsonParser p{s};
  Json v = p.value();
  return v;
}

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o += (char)c;
        }
    }
  }
  return o;
}

std::string Json::dump() const {
  switch (t_) {
    case Null: return "null";
    case Bool: return b_ ? "true" : "false";
    case Number: {
      if (std::isfinite(n_) && n_ == std::floor(n_) && std::fabs(n_) < 1e15) return std::to_string((int64_t)n_);
      char b[32];
      snprintf(b, sizeof b, "%.6g", n_);
      return b;
    }
    case String: return "\"" + json_escape(s_) + "\"";
    case Array: {
      std::string o = "[";
      for (size_t k = 0; k < a_.size(); k++) o += (k ? "," : "") + a_[k].dump();
      return o + "]";
    }
    case Object: {
      std::string o = "{";
      for (size_t k = 0; k < o_.size(); k++) o += (k ? ",\"" : "\"") + json_escape(o_[k].first) + "\":" + o_[k].second.dump();
      return o + "}";
    }
  }
  return "null";
}
