#include "websearch.h"

#include "common.h"

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

// base64url / base64 decode (Bing wraps result URLs as u=a1<base64url>)
std::string b64_decode(const std::string& in) {
  std::string out;
  int val = 0, bits = -8;
  for (unsigned char ch : in) {
    int v;
    if (ch >= 'A' && ch <= 'Z') v = ch - 'A';
    else if (ch >= 'a' && ch <= 'z') v = ch - 'a' + 26;
    else if (ch >= '0' && ch <= '9') v = ch - '0' + 52;
    else if (ch == '+' || ch == '-') v = 62;
    else if (ch == '/' || ch == '_') v = 63;
    else break;
    val = (val << 6) | v;
    bits += 6;
    if (bits >= 0) {
      out += (char)((val >> bits) & 0xff);
      bits -= 8;
    }
  }
  return out;
}

std::string inner_text(const std::string& seg, size_t from, const char* close) {
  const size_t s = seg.find('>', from);
  if (s == std::string::npos) return "";
  const size_t e = seg.find(close, s);
  if (e == std::string::npos) return "";
  return collapse_ws(decode_entities(strip_tags(seg.substr(s + 1, e - s - 1))));
}

// Bing HTML results: <li class="b_algo"> ... <h2><a href="https://www.bing.com/ck/a?...&u=a1<b64>">title</a></h2>
// ... <p class="b_lineclamp...">snippet</p>
std::vector<WebResult> parse_bing(const std::string& page, int n_results) {
  std::vector<WebResult> res;
  const std::string key = "class=\"b_algo\"";
  size_t p = page.find(key);
  while (p != std::string::npos && (int)res.size() < n_results) {
    const size_t q = page.find(key, p + key.size());
    const std::string seg = page.substr(p, (q == std::string::npos ? page.size() : q) - p);
    p = q;
    const size_t h2 = seg.find("<h2");
    if (h2 == std::string::npos) continue;
    const size_t a = seg.find("href=\"", h2);
    if (a == std::string::npos) continue;
    WebResult r;
    std::string href = decode_entities(seg.substr(a + 6, seg.find('"', a + 6) - a - 6));
    if (href.find("bing.com/ck/a") != std::string::npos) {
      const size_t u = href.find("&u=");
      if (u == std::string::npos) continue;
      std::string enc = href.substr(u + 3, href.find('&', u + 3) - u - 3);
      if (enc.rfind("a1", 0) == 0) enc = enc.substr(2);
      href = b64_decode(enc);
    }
    r.url = href;
    r.title = inner_text(seg, a, "</a>");
    const size_t sn = seg.find("class=\"b_lineclamp");
    if (sn != std::string::npos) r.snippet = inner_text(seg, sn, "</p>");
    else {
      const size_t cp = seg.find("class=\"b_caption");
      const size_t pp = cp == std::string::npos ? cp : seg.find("<p", cp);
      if (pp != std::string::npos) r.snippet = inner_text(seg, pp, "</p>");
    }
    if (r.title.empty() || !host_allowed(r.url)) continue;
    res.push_back(std::move(r));
  }
  return res;
}

// Yahoo! JAPAN results: <div class="sw-Card__title"><a href="URL" ...><h3 ...>title</h3> ...
// <div class="sw-Card__summary">snippet</div>
std::vector<WebResult> parse_yahoo_jp(const std::string& page, int n_results) {
  std::vector<WebResult> res;
  const std::string key = "class=\"sw-Card__title\"";
  size_t p = page.find(key);
  while (p != std::string::npos && (int)res.size() < n_results) {
    const size_t q = page.find(key, p + key.size());
    const std::string seg = page.substr(p, (q == std::string::npos ? page.size() : q) - p);
    p = q;
    const size_t a = seg.find("href=\"");
    if (a == std::string::npos) continue;
    WebResult r;
    r.url = decode_entities(seg.substr(a + 6, seg.find('"', a + 6) - a - 6));
    const size_t t = seg.find("sw-Card__titleMain");
    if (t != std::string::npos) r.title = inner_text(seg, t, "</h3>");
    const size_t sm = seg.find("class=\"sw-Card__summary");
    if (sm != std::string::npos) r.snippet = inner_text(seg, sm, "</div>");
    if (r.title.empty() || !host_allowed(r.url) || r.url.find(".yahoo.co.jp/") != std::string::npos) continue;
    bool dup = false;
    for (auto& o : res) dup |= o.url == r.url;
    if (!dup) res.push_back(std::move(r));
  }
  return res;
}

// does a result mention the query?  Every whitespace-separated query term (lower-cased) is looked up in
// title + snippet + URL; a term without spaces in a CJK script also matches by any of its 2-character
// pieces (Japanese queries are rarely split into words).  Relevant when at least half the terms match.
std::vector<std::string> utf8_chars(const std::string& s) {
  std::vector<std::string> v;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = (unsigned char)s[i];
    const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    v.push_back(s.substr(i, n));
    i += n;
  }
  return v;
}
bool relevant(const std::string& query, const WebResult& r) {
  const std::string hay = lower(r.title + " " + r.snippet + " " + url_decode(r.url));
  std::vector<std::string> terms;
  std::string cur;
  for (char ch : lower(query) + " ") {
    if (ch == ' ' || ch == '\t' || ch == '"') {
      if (!cur.empty()) terms.push_back(cur);
      cur.clear();
    } else {
      cur += ch;
    }
  }
  if (terms.empty()) return true;
  int hit = 0;
  for (const std::string& t : terms) {
    bool ok = hay.find(t) != std::string::npos;
    const std::vector<std::string> cs = utf8_chars(t);
    for (size_t i = 0; !ok && cs.size() >= 2 && i + 1 < cs.size(); i++)
      if (cs[i].size() > 1 && cs[i + 1].size() > 1) ok = hay.find(cs[i] + cs[i + 1]) != std::string::npos;
    hit += ok;
  }
  return 2 * hit >= (int)terms.size();
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
  // engines in order until one returns relevant results.  Scrapers get bot challenges now and then
  // (DuckDuckGo HTTP 202 "anomaly", Brave HTTP 429 captcha), and Bing may answer them with plausible
  // but unrelated pages, so every engine's results go through relevant() first.
  const std::string q = url_encode(query);
  std::vector<WebResult> res;
  const struct { const char* name; std::string url; std::vector<WebResult> (*parse)(const std::string&, int); size_t cap; } engines[] = {
      {"Yahoo! JAPAN", "https://search.yahoo.co.jp/search?p=" + q, parse_yahoo_jp, 3u << 20},
      {"DuckDuckGo", "https://html.duckduckgo.com/html/?q=" + q, parse_ddg, 2u << 20},
      {"Brave", "https://search.brave.com/search?q=" + q, parse_brave, 3u << 20},
      {"Bing", "https://www.bing.com/search?q=" + q + "&setlang=ja", parse_bing, 3u << 20},
  };
  for (const auto& e : engines) {
    std::vector<WebResult> got = e.parse(curl_get(e.url, e.cap, 10), n_results + 4);
    const size_t n_all = got.size();
    got.erase(std::remove_if(got.begin(), got.end(), [&](const WebResult& r) { return !relevant(query, r); }), got.end());
    if (n_all && got.size() * 3 < n_all) {  // mostly unrelated: treat the engine as failed
      QE_LOG("[search] %s: %zu of %zu results unrelated to \"%s\", trying the next engine\n", e.name, n_all - got.size(), n_all,
             query.c_str());
      continue;
    }
    if (got.empty()) continue;
    if ((int)got.size() > n_results) got.resize(n_results);
    res = std::move(got);
    break;
  }
  if (res.empty()) QE_LOG("[search] no results from Bing / DuckDuckGo / Yahoo! JAPAN / Brave for \"%s\"\n", query.c_str());
  // fetch the top pages in parallel
  std::vector<std::thread> th;
  for (int i = 0; i < (int)res.size() && i < n_fetch; i++)
    th.emplace_back([&res, i, max_chars] { res[i].text = html_to_text(curl_get(res[i].url, 3 << 20, 8), max_chars); });
  for (auto& t : th) t.join();
  return res;
}
