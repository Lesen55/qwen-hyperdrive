#include "websearch.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <thread>

namespace {

const char* kUA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0";

// run curl (no shell), capture at most max_bytes of stdout
std::string curl_get(const std::string& url, size_t max_bytes, int timeout_s) {
  // everything the child needs is built before fork (no allocation between fork and exec)
  const std::string to = std::to_string(timeout_s), mx = std::to_string(max_bytes * 4);
  int pfd[2];
  if (pipe(pfd) != 0) return "";
  const pid_t pid = fork();
  if (pid < 0) { close(pfd[0]); close(pfd[1]); return ""; }
  if (pid == 0) {
    dup2(pfd[1], 1);
    close(pfd[0]);
    close(pfd[1]);
    execlp("curl", "curl", "-sL", "--compressed", "--max-time", to.c_str(), "--max-filesize", mx.c_str(),
           "--max-redirs", "3", "--proto", "=http,https", "--proto-redir", "=http,https", "-A", kUA,
           "-H", "Accept-Language: ja,en;q=0.8", "--", url.c_str(), (char*)nullptr);
    _exit(127);
  }
  close(pfd[1]);
  std::string out;
  char buf[65536];
  for (;;) {
    const ssize_t n = read(pfd[0], buf, sizeof buf);
    if (n <= 0) break;
    out.append(buf, n);
    if (out.size() >= max_bytes) { kill(pid, SIGKILL); break; }
  }
  close(pfd[0]);
  int st = 0;
  waitpid(pid, &st, 0);
  if (out.size() > max_bytes) out.resize(max_bytes);
  return out;
}

std::string url_encode(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : s) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char)c;
    else if (c == ' ') o += '+';
    else { o += '%'; o += hex[c >> 4]; o += hex[c & 15]; }
  }
  return o;
}

void append_utf8(std::string& o, unsigned cp) {
  if (cp < 0x80) o += (char)cp;
  else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
  else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
  else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
}

std::string decode_entities(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] != '&') { o += s[i]; continue; }
    const size_t e = s.find(';', i);
    if (e == std::string::npos || e - i > 10) { o += s[i]; continue; }
    const std::string ent = s.substr(i + 1, e - i - 1);
    if (ent == "amp") o += '&';
    else if (ent == "lt") o += '<';
    else if (ent == "gt") o += '>';
    else if (ent == "quot") o += '"';
    else if (ent == "apos" || ent == "#39") o += '\'';
    else if (ent == "nbsp") o += ' ';
    else if (!ent.empty() && ent[0] == '#') {
      const unsigned cp = (ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X')) ? strtoul(ent.c_str() + 2, nullptr, 16)
                                                                              : strtoul(ent.c_str() + 1, nullptr, 10);
      if (cp) append_utf8(o, cp);
    } else { o += s.substr(i, e - i + 1); }
    i = e;
  }
  return o;
}

std::string lower(std::string s) {
  for (auto& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

std::string strip_tags(const std::string& s) {
  std::string o;
  bool in = false;
  for (char c : s) {
    if (c == '<') in = true;
    else if (c == '>') in = false;
    else if (!in) o += c;
  }
  return o;
}

std::string collapse_ws(const std::string& s) {
  std::string o;
  bool sp = false;
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') sp = true;
    else { if (sp && !o.empty()) o += ' '; sp = false; o += c; }
  }
  return o;
}

// readable text of an HTML page: drop non-content blocks, keep line structure, keep only lines that
// look like prose (drops most navigation / menus)
std::string html_to_text(const std::string& html, size_t max_chars) {
  std::string h = html;
  const std::string lh = lower(h);
  // prefer <article> or <main> when present
  for (const char* tag : {"<article", "<main"}) {
    const size_t a = lh.find(tag);
    if (a != std::string::npos) {
      const std::string close = std::string("</") + (tag + 1) + ">";
      const size_t b = lh.find(close, a);
      if (b != std::string::npos && b - a > 2000) { h = h.substr(a, b - a); break; }
    }
  }
  std::string l2 = lower(h), o;
  o.reserve(h.size() / 2);
  size_t i = 0;
  while (i < h.size()) {
    if (h[i] == '<') {
      bool skipped = false;
      for (const char* blk : {"script", "style", "noscript", "svg", "head", "nav", "footer", "header", "form"}) {
        const std::string open = std::string("<") + blk;
        if (l2.compare(i, open.size(), open) == 0 && i + open.size() < h.size() &&
            (h[i + open.size()] == '>' || h[i + open.size()] == ' ')) {
          const size_t e = l2.find(std::string("</") + blk, i);
          const size_t g = e == std::string::npos ? std::string::npos : l2.find('>', e);
          i = g == std::string::npos ? h.size() : g + 1;
          skipped = true;
          break;
        }
      }
      if (skipped) continue;
      const size_t e = h.find('>', i);
      if (e == std::string::npos) break;
      const std::string tag = l2.substr(i, std::min<size_t>(e - i, 6));
      if (tag.rfind("<br", 0) == 0 || tag.rfind("</p", 0) == 0 || tag.rfind("</div", 0) == 0 || tag.rfind("</li", 0) == 0 ||
          tag.rfind("</h", 0) == 0 || tag.rfind("<li", 0) == 0 || tag.rfind("</tr", 0) == 0)
        o += '\n';
      else
        o += ' ';
      i = e + 1;
      continue;
    }
    o += h[i++];
  }
  o = decode_entities(o);
  std::string res;
  size_t p = 0;
  while (p < o.size() && res.size() < max_chars) {
    size_t q = o.find('\n', p);
    if (q == std::string::npos) q = o.size();
    std::string line = collapse_ws(o.substr(p, q - p));
    p = q + 1;
    // prose filter: long enough (bytes; CJK chars are 3 bytes) or ends with a sentence mark
    if (line.size() >= 60 || (line.size() >= 24 && (line.find("。") != std::string::npos || line.back() == '.')))
      res += line + "\n";
  }
  if (res.size() > max_chars) {
    size_t cut = max_chars;
    while (cut > 0 && ((unsigned char)res[cut] & 0xC0) == 0x80) cut--;  // UTF-8 boundary
    res.resize(cut);
    res += "…";
  }
  return res;
}

bool host_allowed(const std::string& url) {
  const size_t s = url.find("://");
  if (s == std::string::npos) return false;
  const std::string scheme = lower(url.substr(0, s));
  if (scheme != "http" && scheme != "https") return false;
  size_t a = s + 3, b = url.find_first_of("/?#", a);
  std::string host = lower(url.substr(a, b == std::string::npos ? std::string::npos : b - a));
  const size_t at = host.rfind('@');
  if (at != std::string::npos) return false;  // credentials in URL: refuse
  if (!host.empty() && host[0] == '[') return false;  // IPv6 literal: refuse
  const size_t colon = host.find(':');
  if (colon != std::string::npos) host = host.substr(0, colon);
  if (host.empty() || host == "localhost" || host.find('.') == std::string::npos) return false;
  const size_t L = host.size();
  for (const char* suf : {".local", ".localhost", ".internal", ".lan", ".home"})
    if (L > strlen(suf) && host.compare(L - strlen(suf), strlen(suf), suf) == 0) return false;
  unsigned o[4];
  char tail;
  if (sscanf(host.c_str(), "%u.%u.%u.%u%c", &o[0], &o[1], &o[2], &o[3], &tail) == 4) {
    if (o[0] == 10 || o[0] == 127 || o[0] == 0 || (o[0] == 169 && o[1] == 254) || (o[0] == 172 && o[1] >= 16 && o[1] < 32) ||
        (o[0] == 192 && o[1] == 168) || (o[0] == 100 && o[1] >= 64 && o[1] < 128) || o[0] >= 224)
      return false;
  }
  return true;
}

}  // namespace

std::string url_decode(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1]) && isxdigit((unsigned char)s[i + 2])) {
      o += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
      i += 2;
    } else if (s[i] == '+') {
      o += ' ';
    } else {
      o += s[i];
    }
  }
  return o;
}

namespace {
// DuckDuckGo HTML results
std::vector<WebResult> parse_ddg(const std::string& page, int n_results) {
  std::vector<WebResult> res;
  size_t p = 0;
  while ((int)res.size() < n_results) {
    p = page.find("class=\"result__a\"", p);
    if (p == std::string::npos) break;
    const size_t tag_end = page.find('>', p);
    const size_t close = tag_end == std::string::npos ? std::string::npos : page.find("</a>", tag_end);
    if (close == std::string::npos) break;
    const std::string tag = page.substr(p, tag_end - p);
    WebResult r;
    r.title = collapse_ws(decode_entities(strip_tags(page.substr(tag_end + 1, close - tag_end - 1))));
    const size_t u = tag.find("uddg=");
    if (u != std::string::npos) {
      const size_t ue = tag.find_first_of("&\"", u + 5);
      r.url = url_decode(tag.substr(u + 5, ue == std::string::npos ? std::string::npos : ue - u - 5));
    }
    p = close;
    const size_t sn = page.find("class=\"result__snippet\"", p);
    const size_t next = page.find("class=\"result__a\"", p);
    if (sn != std::string::npos && (next == std::string::npos || sn < next)) {
      const size_t se = page.find('>', sn), sc = se == std::string::npos ? se : page.find("</a>", se);
      if (sc != std::string::npos) r.snippet = collapse_ws(decode_entities(strip_tags(page.substr(se + 1, sc - se - 1))));
    }
    if (r.url.empty() || r.url.find("duckduckgo.com/y.js") != std::string::npos || !host_allowed(r.url)) continue;
    res.push_back(std::move(r));
  }
  return res;
}

std::string between_div(const std::string& seg, const char* cls) {
  const size_t c = seg.find(cls);
  if (c == std::string::npos) return "";
  const size_t s = seg.find('>', c), e = s == std::string::npos ? s : seg.find("</div>", s);
  if (e == std::string::npos) return "";
  return collapse_ws(decode_entities(strip_tags(seg.substr(s + 1, e - s - 1))));
}

// Brave Search HTML results (fallback when DuckDuckGo answers with a bot challenge)
std::vector<WebResult> parse_brave(const std::string& page, int n_results) {
  std::vector<WebResult> res;
  const std::string key = "data-type=\"web\"";
  size_t p = page.find(key);
  while (p != std::string::npos && (int)res.size() < n_results) {
    const size_t q = page.find(key, p + key.size());
    const std::string seg = page.substr(p, (q == std::string::npos ? page.size() : q) - p);
    p = q;
    WebResult r;
    const size_t h = seg.find("href=\"http");
    if (h == std::string::npos) continue;
    r.url = decode_entities(seg.substr(h + 6, seg.find('"', h + 6) - h - 6));
    r.title = between_div(seg, "class=\"title");
    r.snippet = between_div(seg, "class=\"content");
    if (r.snippet.empty()) r.snippet = between_div(seg, "class=\"generic-snippet");
    if (r.title.empty() || !host_allowed(r.url)) continue;
    res.push_back(std::move(r));
  }
  return res;
}
}  // namespace

std::vector<WebResult> web_search(const std::string& query, int n_results, int n_fetch, size_t max_chars) {
  std::vector<WebResult> res = parse_ddg(curl_get("https://html.duckduckgo.com/html/?q=" + url_encode(query), 2 << 20, 10), n_results);
  if (res.empty()) res = parse_brave(curl_get("https://search.brave.com/search?q=" + url_encode(query), 3 << 20, 10), n_results);
  // fetch the top pages in parallel
  std::vector<std::thread> th;
  for (int i = 0; i < (int)res.size() && i < n_fetch; i++)
    th.emplace_back([&res, i, max_chars] { res[i].text = html_to_text(curl_get(res[i].url, 3 << 20, 8), max_chars); });
  for (auto& t : th) t.join();
  return res;
}
