#include "websearch.h"

#include "common.h"

#ifndef _WIN32
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <string_view>
#include <thread>

namespace {

const char* kUA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0";
using FetchClock = std::chrono::steady_clock;
constexpr size_t kMaxUrlBytes = 8192;

struct CurlOutput {
  std::string text;
  std::string last_line;
  size_t received = 0, last_break = std::string::npos;
  int exit_code = -1;

  void append(std::string_view data, size_t body_cap) {
    text.append(data.substr(0, body_cap - text.size()));
    const auto pos = data.rfind('\n');
    if (pos != std::string::npos) {
      last_break = received + pos;
      last_line.clear();
    }
    received += data.size();
    // Keep the final write-out line even after the captured body fills.
    constexpr size_t cap = kMaxUrlBytes + 5;
    last_line.append(data.substr(pos == std::string::npos ? 0 : pos + 1, cap - last_line.size()));
  }
};

#ifdef _WIN32
// Windows: spawn the system curl.exe (ships with Windows 10 1803+) without a console
// window and read its stdout through an anonymous pipe.
std::wstring widen(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
  return w;
}

// Launch only the copy supplied by Windows.
CurlOutput run_curl(const std::vector<std::string>& args, size_t max_bytes, int timeout_s) {
  wchar_t directory[MAX_PATH];
  const auto len = GetSystemDirectoryW(directory, MAX_PATH);
  if (!len || len >= MAX_PATH) return {};
  const std::wstring application = std::wstring(directory, len) + L"\\curl.exe";
  // Arguments are fixed strings, validated URLs, or addresses formatted by inet_ntop.
  std::string cmd = "curl.exe";
  for (const auto& arg : args) cmd += " \"" + arg + "\"";
  // curl accepts the executable name as argv[0] and disables curlrc with the first option.
  std::wstring wcmd = widen(cmd);

  SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 0)) return {};
  if (!SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0)) { CloseHandle(rd); CloseHandle(wr); return {}; }
  HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
  if (nul == INVALID_HANDLE_VALUE) { CloseHandle(rd); CloseHandle(wr); return {}; }

  SIZE_T attr_size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
  std::vector<unsigned char> storage(attr_size);
  STARTUPINFOEXW si{};
  si.StartupInfo.cb = sizeof si;
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdOutput = wr;
  si.StartupInfo.hStdError = nul;
  si.StartupInfo.hStdInput = nul;
  si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
  const BOOL initialized = attr_size && InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &attr_size);
  HANDLE inherited[] = {wr, nul};
  PROCESS_INFORMATION pi{};
  const BOOL ok = initialized &&
      UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof inherited,
                                nullptr, nullptr) &&
      CreateProcessW(application.c_str(), wcmd.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &si.StartupInfo, &pi);
  if (initialized) DeleteProcThreadAttributeList(si.lpAttributeList);
  CloseHandle(wr);
  CloseHandle(nul);
  if (!ok) { CloseHandle(rd); return {}; }

  CurlOutput out;
  char buf[65536];
  for (;;) {
    DWORD n = 0;
    if (!ReadFile(rd, buf, sizeof buf, &n, nullptr) || n == 0) break;
    out.append({buf, n}, max_bytes);
  }
  CloseHandle(rd);
  const DWORD waited = WaitForSingleObject(pi.hProcess, (DWORD)(timeout_s + 5) * 1000);
  if (waited == WAIT_TIMEOUT) TerminateProcess(pi.hProcess, 1);
  DWORD status = 0;
  if (waited == WAIT_OBJECT_0 && GetExitCodeProcess(pi.hProcess, &status)) out.exit_code = (int)status;
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return out;
}
#else
// Build the child's argv before forking from a multithreaded server.
CurlOutput run_curl(const std::vector<std::string>& args, size_t max_bytes, int) {
  // everything the child needs is built before fork (no allocation between fork and exec)
  char program[] = "curl";
  std::vector<char*> argv{program};
  for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
  argv.push_back(nullptr);
  int pfd[2];
  if (pipe2(pfd, O_CLOEXEC) != 0) return {};
  const pid_t pid = fork();
  if (pid < 0) { close(pfd[0]); close(pfd[1]); return {}; }
  if (pid == 0) {
    dup2(pfd[1], 1);
    close(pfd[0]);
    close(pfd[1]);
    execvp(program, argv.data());
    _exit(127);
  }
  close(pfd[1]);
  CurlOutput out;
  char buf[65536];
  for (;;) {
    const ssize_t n = read(pfd[0], buf, sizeof buf);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    out.append({buf, (size_t)n}, max_bytes);
  }
  close(pfd[0]);
  int st = 0;
  pid_t waited;
  do { waited = waitpid(pid, &st, 0); } while (waited < 0 && errno == EINTR);
  if (waited == pid && WIFEXITED(st)) out.exit_code = WEXITSTATUS(st);
  return out;
}
#endif  // _WIN32

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

struct HttpTarget {
  std::string host, port, url;
};

bool public_ipv4(const in_addr& address) {
  const uint32_t ip = ntohl(address.s_addr);
  const unsigned a = ip >> 24, b = (ip >> 16) & 255, c = (ip >> 8) & 255;
  return a != 0 && a != 10 && a != 127 && a < 224 &&
         !(a == 100 && b >= 64 && b < 128) && !(a == 169 && b == 254) &&
         !(a == 172 && b >= 16 && b < 32) &&
         !(a == 192 && (b == 168 || (b == 0 && (c == 0 || c == 2)) || (b == 88 && c == 99))) &&
         !(a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100))) &&
         !(a == 203 && b == 0 && c == 113);
}

bool public_ipv6(const in6_addr& address) {
  const auto* p = reinterpret_cast<const unsigned char*>(&address);
  // Global unicast only; exclude special-use protocol, transition, and documentation prefixes.
  return (p[0] & 0xe0) == 0x20 &&
         !(p[0] == 0x20 && (p[1] == 2 || (p[1] == 1 && (p[2] < 2 || (p[2] == 0x0d && p[3] == 0xb8))))) &&
         !(p[0] == 0x3f && p[1] == 0xff && (p[2] & 0xf0) == 0);
}

bool parse_http_target(const std::string& url, HttpTarget& target) {
  if (url.size() > kMaxUrlBytes) return false;
  for (unsigned char c : url)
    if (c <= ' ' || c == '\x7f' || c == '"' || c == '\\') return false;
  const size_t s = url.find("://");
  if (s == std::string::npos) return false;
  const std::string scheme = lower(url.substr(0, s));
  if (scheme != "http" && scheme != "https") return false;
  const size_t a = s + 3, b = url.find_first_of("/?#", a);
  std::string host = lower(url.substr(a, b == std::string::npos ? b : b - a));
  std::string port = scheme == "https" ? "443" : "80";
  const size_t colon = host.find(':');
  if (colon != std::string::npos) {
    const auto value = host.substr(colon + 1);
    if (value.empty() || value.size() > 5 || value.find_first_not_of("0123456789") != std::string::npos) return false;
    const int n = std::stoi(value);
    if (!n || n > 65535) return false;
    port = std::to_string(n);
    host.resize(colon);
  }
  const size_t dot = host.rfind('.');
  if (host.empty() || host.size() > 253 || host.front() == '.' || dot == std::string::npos || dot + 1 == host.size() ||
      host.find("..") != std::string::npos || host.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789.-") != std::string::npos)
    return false;
  const size_t L = host.size();
  for (const char* suf : {".local", ".localhost", ".internal", ".lan", ".home"})
    if (L > strlen(suf) && host.compare(L - strlen(suf), strlen(suf), suf) == 0) return false;
  in_addr ipv4{};
  if (inet_pton(AF_INET, host.c_str(), &ipv4) == 1) {
    if (!public_ipv4(ipv4)) return false;
  } else if (host[dot + 1] < 'a' || host[dot + 1] > 'z') {
    // Refuse non-canonical numeric names so curl and getaddrinfo cannot interpret them differently.
    return false;
  }
  target = {host, port, scheme + "://" + host + (colon == std::string::npos ? "" : ":" + port) +
                          (b == std::string::npos ? "" : url.substr(b))};
  return true;
}

bool host_allowed(const std::string& url) {
  HttpTarget target;
  return parse_http_target(url, target);
}

std::string lookup_public_addresses(const HttpTarget& target) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  if (getaddrinfo(target.host.c_str(), target.port.c_str(), &hints, &addresses) != 0) return {};
  std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> entries(addresses, freeaddrinfo);
  std::vector<std::string> approved;
  for (const addrinfo* i = entries.get(); i; i = i->ai_next) {
    char numeric[INET6_ADDRSTRLEN];
    std::string address;
    if (i->ai_family == AF_INET) {
      const auto& a = reinterpret_cast<const sockaddr_in*>(i->ai_addr)->sin_addr;
      if (public_ipv4(a) && inet_ntop(AF_INET, &a, numeric, sizeof numeric)) address = numeric;
    } else if (i->ai_family == AF_INET6) {
      const auto& a = reinterpret_cast<const sockaddr_in6*>(i->ai_addr)->sin6_addr;
      if (public_ipv6(a) && inet_ntop(AF_INET6, &a, numeric, sizeof numeric)) address = "[" + std::string(numeric) + "]";
    }
    if (!address.empty() && std::find(approved.begin(), approved.end(), address) == approved.end())
      approved.push_back(std::move(address));
  }
  std::string result;
  for (const auto& address : approved) {
    if (!result.empty()) result += ',';
    result += address;
  }
  return result;
}

std::string resolve_public(const HttpTarget& target, FetchClock::time_point deadline) {
  // The platform resolver is synchronous. Workers own their inputs, and the
  // bound also counts workers whose DNS lookup outlives the request deadline.
  static std::atomic<int> outstanding{0};
  constexpr int max_outstanding = 16;
  if (FetchClock::now() >= deadline) return {};
  std::promise<std::string> result;
  auto future = result.get_future();
  if (outstanding.fetch_add(1) >= max_outstanding) {
    outstanding.fetch_sub(1);
    return {};
  }
  try {
    std::thread([target, deadline, result = std::move(result)]() mutable {
      try {
        result.set_value(FetchClock::now() < deadline ? lookup_public_addresses(target) : std::string{});
      } catch (...) {
        result.set_exception(std::current_exception());
      }
      outstanding.fetch_sub(1);
    }).detach();
  } catch (const std::system_error&) {
    outstanding.fetch_sub(1);
    return {};
  } catch (...) {
    outstanding.fetch_sub(1);
    throw;
  }
  if (future.wait_until(deadline) != std::future_status::ready) return {};
  return future.get();
}

std::string curl_get(std::string url, size_t max_bytes, int timeout_s) {
  const auto deadline = FetchClock::now() + std::chrono::seconds(timeout_s);
  for (int redirects = 0; redirects <= 3; ++redirects) {
    HttpTarget target;
    if (!parse_http_target(url, target)) return {};
    const std::string addresses = resolve_public(target, deadline);
    const auto remaining = std::chrono::floor<std::chrono::milliseconds>(deadline - FetchClock::now());
    if (addresses.empty() || remaining.count() <= 0) return {};
    // Disable curlrc/proxies and pin the checked binding for this single HTTP hop.
    const std::vector<std::string> args = {
        "-q", "-s", "--globoff", "--compressed", "--max-time", std::to_string(remaining.count() / 1000.0), "--max-filesize",
        std::to_string(max_bytes * 4), "--proto", "=http,https", "--noproxy", "*", "--resolve",
        target.host + ":" + target.port + ":" + addresses, "-A", kUA, "-H", "Accept-Language: ja,en;q=0.8",
        "--write-out", "\\n%{http_code} %{redirect_url}", "--", target.url};
    const auto out = run_curl(args, max_bytes, (int)std::chrono::ceil<std::chrono::seconds>(remaining).count());
    // curl still writes metadata on a maximum-file-size exit (63).
    if (out.exit_code != 0 && out.exit_code != 63) return {};
    if (out.last_break == std::string::npos || out.last_line.size() < 4 || out.last_line[3] != ' ') return {};
    const auto code = out.last_line.substr(0, 3);
    if (code.find_first_not_of("0123456789") != std::string::npos) return {};
    url = out.last_line.substr(4);
    if ((code == "301" || code == "302" || code == "303" || code == "307" || code == "308") && !url.empty()) continue;
    return out.text.substr(0, out.last_break);
  }
  return {};
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
