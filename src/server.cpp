// qwen35-server: OpenAI-compatible HTTP server on top of the engine.
//   POST /v1/chat/completions   (stream / non-stream; reasoning_content for <think> output)
//   POST /v1/completions        (raw prompt)
//   GET  /v1/models, /health
//   GET  /            web UI (chat + live hardware / engine dashboard)
//   GET  /api/stats   telemetry for the web UI
//   GET  /api/search?q=...&n=5&fetch=3   web search (DuckDuckGo) + page text for the UI
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <map>
#include <atomic>
#include <deque>
#include <cmath>
#include <ctime>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>

#include "chat.h"
#include "hwmon.h"
#include "websearch.h"
#include "webui_html.h"
#include "backend.h"
#include "engine.h"
#include "gguf.h"
#include "json.h"
#include "tokenizer.h"

// true when the peer has closed the connection (EOF or reset) - checked without blocking
static bool client_gone(int fd) {
  pollfd p{fd, (short)(POLLIN | POLLRDHUP), 0};
  if (poll(&p, 1, 0) <= 0) return false;
  if (p.revents & (POLLHUP | POLLERR | POLLRDHUP | POLLNVAL)) return true;
  if (p.revents & POLLIN) {
    char c;
    const ssize_t r = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    return r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
  }
  return false;
}


namespace {

struct Request {
  std::string target;
  std::string method, path;
  std::string body;
};

bool send_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    const ssize_t n = send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
    if (n <= 0) return false;
    off += n;
  }
  return true;
}

bool read_request(int fd, Request& rq) {
  std::string buf;
  char tmp[65536];
  size_t hdr_end = std::string::npos;
  while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
    const ssize_t n = recv(fd, tmp, sizeof tmp, 0);
    if (n <= 0) return false;
    buf.append(tmp, n);
    if (buf.size() > (64u << 20)) return false;
  }
  const std::string head = buf.substr(0, hdr_end);
  const size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
  rq.method = head.substr(0, sp1);
  rq.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
  rq.target = rq.path;  // with the query string
  const size_t q = rq.path.find('?');
  if (q != std::string::npos) rq.path.resize(q);
  size_t clen = 0;
  {
    std::string lower = head;
    for (auto& ch : lower) ch = (char)tolower((unsigned char)ch);
    const size_t p = lower.find("\r\ncontent-length:");
    if (p != std::string::npos) clen = std::strtoull(lower.c_str() + p + 17, nullptr, 10);
  }
  rq.body = buf.substr(hdr_end + 4);
  while (rq.body.size() < clen) {
    const ssize_t n = recv(fd, tmp, sizeof tmp, 0);
    if (n <= 0) return false;
    rq.body.append(tmp, n);
  }
  return true;
}

const char* kCors =
    "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

void respond(int fd, int code, const std::string& body, const char* ctype = "application/json") {
  const char* msg = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Error";
  send_all(fd, "HTTP/1.1 " + std::to_string(code) + " " + msg + "\r\nContent-Type: " + ctype +
                   "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n" + kCors + "Connection: close\r\n\r\n" + body);
}

// keep only complete UTF-8 sequences in `out`, carry the rest
std::string take_utf8(std::string& buf) {
  size_t n = buf.size(), cut = n;
  // find start of a possibly incomplete trailing sequence
  for (size_t k = 1; k <= 4 && k <= n; k++) {
    const unsigned char c = buf[n - k];
    if ((c & 0xC0) == 0x80) continue;  // continuation byte
    const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    if (need > k) cut = n - k;
    break;
  }
  std::string out = buf.substr(0, cut);
  buf.erase(0, cut);
  return out;
}

struct Server {
  EngineOptions opt;
  bool ngram = true;
  std::string draft_model;
  std::unique_ptr<IModel> draft;
  int mtp_max = 5;
  float spec_relax = 1.f;  // >1: lenient speculative acceptance (faster, slightly biased sampling)
  std::unique_ptr<IModel> model;
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Generator> gen;
  std::mutex mu;  // one generation at a time
  std::string model_name;
  std::atomic<int> id_counter{0};
  int stop_ids[2] = {-1, -1};
  // compaction summaries: hash of the dropped message prefix -> summary of it
  std::map<size_t, std::string> summaries;
  // Summarize `part` (with an earlier summary `prev` of what came before it) using the model itself:
  // greedy, no thinking, at most 512 tokens; the transcript is cut to its most recent ~24K tokens so
  // the call stays well inside proxy timeouts.  Call with `mu` held.
  std::string summarize(const std::vector<ChatMessage>& part, const std::string& prev) {
    std::string tr;
    for (const auto& m : part) {
      if (m.content.empty() && !m.tool_calls.is_array()) continue;
      tr += (m.role == "user" ? "ユーザー" : m.role == "assistant" ? "アシスタント" : m.role) + std::string(": ") + m.content;
      if (m.tool_calls.is_array()) tr += " [ツール呼び出し " + m.tool_calls.dump() + "]";
      tr += "\n\n";
    }
    std::vector<int> t = tok->encode(tr, false);
    const int cap = std::min(24576, opt.ctx / 2);
    if ((int)t.size() > cap) tr = "（…前略…）\n" + tok->decode(std::vector<int>(t.end() - cap, t.end()));
    std::vector<ChatMessage> req;
    req.push_back({"system", "あなたは会話の要約係です。後で会話を続けるために必要な事実・決定事項・ユーザーの要望・未解決の質問・固有名詞や数値を落とさず、日本語の箇条書きで簡潔にまとめてください。", "", Json()});
    std::string u;
    if (!prev.empty()) u += "これまでの要約:\n" + prev + "\n\n";
    u += "続きの会話:\n" + tr + "\n上の内容（これまでの要約があればそれも含めて）を 1 つの要約にまとめてください。";
    req.push_back({"user", u, "", Json()});
    std::vector<int> ids = tok->encode(chat_render(req, true, false), true);
    SamplingParams g;
    g.temperature = 0.f;
    gen->set_sampling(g);
    std::vector<int> out;
    out.push_back(gen->start(ids));
    while ((int)out.size() < 512 && out.back() != stop_ids[0] && out.back() != stop_ids[1]) gen->step(out);
    while (!out.empty() && (out.back() == stop_ids[0] || out.back() == stop_ids[1])) out.pop_back();
    for (size_t i = 0; i < out.size(); i++)
      if (out[i] == stop_ids[0] || out[i] == stop_ids[1]) { out.resize(i); break; }
    ThinkSplitter sp(false);
    std::string r, c;
    sp.feed(tok->decode(out), r, c, true);
    return c.empty() ? r : c;
  }
  std::string skip_ffn_str;
  std::mutex search_mu;
  static std::string query_param(const std::string& path, const std::string& key, const std::string& def = "") {
    const size_t q = path.find('?');
    if (q == std::string::npos) return def;
    size_t p = q + 1;
    while (p < path.size()) {
      size_t e = path.find('&', p);
      if (e == std::string::npos) e = path.size();
      const size_t eq = path.find('=', p);
      if (eq != std::string::npos && eq < e && path.compare(p, eq - p, key) == 0 && eq - p == key.size())
        return url_decode(path.substr(eq + 1, e - eq - 1));
      p = e + 1;
    }
    return def;
  }

  // ---- live engine telemetry (written by the generating thread, read by /api/stats) ----
  struct ReqStat {
    std::string id, finish;
    double t_end = 0, ttft_ms = 0, prefill_s = 0, prefill_tps = 0, decode_s = 0, decode_tps = 0;
    int prompt_n = 0, cache_n = 0, gen_n = 0, drafted = 0, accepted = 0, steps = 0;
  };
  struct Live {
    std::mutex m;
    std::string phase = "idle", id;
    double t_start = 0;
    ReqStat cur;
    std::deque<ReqStat> hist;
    KvInfo kv;
    int pos = 0;
    uint64_t total_gen = 0, total_prompt = 0, requests = 0;
    // records and token-weighted averages (prefill: prompts >= 128 new tokens, decode: >= 32 tokens)
    ReqStat best_prefill, best_decode;
    double sum_pf_tok = 0, sum_pf_s = 0, sum_dec_tok = 0, sum_dec_s = 0;
  } live;
  const double t_boot = qe_now();

  void live_update(const char* phase, const ReqStat* r, bool kv) {
    KvInfo k;
    if (kv) k = model->kv_info();
    std::lock_guard<std::mutex> g(live.m);
    if (phase) live.phase = phase;
    if (r) live.cur = *r;
    if (kv) { live.kv = k; live.pos = gen->pos(); }
  }

  std::string stats_json() {
    const GpuStats gs = read_gpu_stats();
    const CpuStats cs = read_cpu_stats();
    Json j = Json::object();
    Json g = Json::object();
    g["ok"] = gs.ok;
    g["name"] = gs.name;
    g["temp_c"] = gs.temp_c;
    g["fan_pct"] = gs.fan_pct;
    g["power_w"] = gs.power_w;
    g["power_limit_w"] = gs.power_limit_w;
    g["util_gpu"] = gs.util_gpu;
    g["util_mem"] = gs.util_mem;
    g["mem_used_mb"] = gs.mem_used_mb;
    g["mem_total_mb"] = gs.mem_total_mb;
    g["clock_gpu_mhz"] = gs.clock_gpu_mhz;
    g["clock_mem_mhz"] = gs.clock_mem_mhz;
    g["clock_sm_max_mhz"] = gs.clock_sm_max_mhz;
    g["pcie_rx_mbs"] = gs.pcie_rx_mbs;
    g["pcie_tx_mbs"] = gs.pcie_tx_mbs;
    g["pcie_gen"] = gs.pcie_gen;
    g["pcie_width"] = gs.pcie_width;
    j["gpu"] = g;
    Json c = Json::object();
    c["name"] = cs.name;
    c["temp_c"] = cs.temp_c;
    c["util_pct"] = cs.util_pct;
    c["freq_avg_mhz"] = cs.freq_avg_mhz;
    c["freq_max_mhz"] = cs.freq_max_mhz;
    c["threads"] = cs.threads;
    c["ram_used_mb"] = cs.ram_used_mb;
    c["ram_total_mb"] = cs.ram_total_mb;
    j["cpu"] = c;
    auto rs = [](const ReqStat& r) {
      Json o = Json::object();
      o["id"] = r.id;
      o["finish"] = r.finish;
      o["t_end"] = r.t_end;
      o["ttft_ms"] = r.ttft_ms;
      o["prompt_n"] = r.prompt_n;
      o["cache_n"] = r.cache_n;
      o["prefill_s"] = r.prefill_s;
      o["prefill_tps"] = r.prefill_tps;
      o["gen_n"] = r.gen_n;
      o["decode_s"] = r.decode_s;
      o["decode_tps"] = r.decode_tps;
      o["drafted"] = r.drafted;
      o["accepted"] = r.accepted;
      o["steps"] = r.steps;
      return o;
    };
    Json e = Json::object();
    {
      std::lock_guard<std::mutex> lk(live.m);
      e["phase"] = live.phase;
      e["elapsed_s"] = live.phase == "idle" ? 0.0 : qe_now() - live.t_start;
      e["current"] = rs(live.cur);
      e["history"] = Json::array();
      for (auto& h : live.hist) e["history"].push_back(rs(h));
      e["pos"] = live.pos;
      Json k = Json::object();
      k["blocks_used"] = live.kv.blocks_used;
      k["blocks_total"] = live.kv.blocks_total;
      k["blocks_lendable"] = live.kv.blocks_lendable;
      k["block_mb"] = live.kv.block_mb;
      k["seqs"] = live.kv.seqs;
      k["seqs_offloaded"] = live.kv.seqs_offloaded;
      k["host_mb"] = live.kv.host_mb;
      k["borrowed_kv"] = live.kv.borrowed_kv;
      k["borrowed_ws"] = live.kv.borrowed_ws;
      k["streamed"] = live.kv.streamed;
      e["kv"] = k;
      e["requests"] = (double)live.requests;
      e["total_gen"] = (double)live.total_gen;
      e["total_prompt"] = (double)live.total_prompt;
      e["best_prefill"] = rs(live.best_prefill);
      e["best_decode"] = rs(live.best_decode);
      e["avg_prefill_tps"] = live.sum_pf_s > 0 ? live.sum_pf_tok / live.sum_pf_s : -1.0;
      e["avg_decode_tps"] = live.sum_dec_s > 0 ? live.sum_dec_tok / live.sum_dec_s : -1.0;
    }
    e["uptime_s"] = qe_now() - t_boot;
    j["engine"] = e;
    Json cf = Json::object();
    cf["model"] = model_name;
    cf["ctx"] = opt.ctx;
    const char* kvn[] = {"f16", "q8", "q4", "q3"};
    cf["kv_type"] = kvn[std::clamp((int)opt.kv_type, 0, 3)];
    cf["mtp"] = opt.use_mtp;
    cf["draft_max"] = model->options().draft_max;
    cf["spec_relax"] = spec_relax;
    cf["skip_ffn"] = skip_ffn_str;
    cf["resident_mb"] = model->plan_stats().res_mb;
    cf["streamed_mb"] = model->plan_stats().stream_mb;
    cf["cpu_mb"] = model->plan_stats().cpu_mb;
    cf["resident_ffn"] = model->plan_stats().res_ffn;
    cf["n_layer"] = model->cfg().n_layer;
    j["config"] = cf;
    return j.dump();
  }

  SamplingParams sampling_from(const Json& b) {
    SamplingParams sp;
    sp.temperature = (float)b.get_num("temperature", 1.0);
    sp.top_p = (float)b.get_num("top_p", 0.95);
    sp.top_k = (int)b.get_num("top_k", 20);
    sp.min_p = (float)b.get_num("min_p", 0.0);
    sp.presence_penalty = (float)b.get_num("presence_penalty", 0.0);
    sp.seed = (uint64_t)b.get_num("seed", 0);
    return sp;
  }

  void handle(int fd) {
    Request rq;
    if (!read_request(fd, rq)) { close(fd); return; }
    if (rq.method == "OPTIONS") {
      send_all(fd, std::string("HTTP/1.1 204 No Content\r\n") + kCors + "Content-Length: 0\r\nConnection: close\r\n\r\n");
    } else if (rq.method == "GET" && rq.path == "/health") {
      respond(fd, 200, "{\"status\":\"ok\"}");
    } else if (rq.method == "GET" && (rq.path == "/" || rq.path == "/ui" || rq.path == "/index.html")) {
      respond(fd, 200, kWebUi, "text/html; charset=utf-8");
    } else if (rq.method == "GET" && rq.path.rfind("/api/stats", 0) == 0) {
      respond(fd, 200, stats_json());
    } else if (rq.method == "GET" && rq.path.rfind("/api/search", 0) == 0) {
      const std::string q = query_param(rq.target, "q");
      if (q.empty() || q.size() > 400) { respond(fd, 400, "{\"error\":{\"message\":\"q required\"}}"); close(fd); return; }
      const int n = std::clamp(atoi(query_param(rq.target, "n", "6").c_str()), 1, 10);
      const int nf = std::clamp(atoi(query_param(rq.target, "fetch", "3").c_str()), 0, 5);
      const int mc = std::clamp(atoi(query_param(rq.target, "chars", "2500").c_str()), 200, 8000);
      {
        std::lock_guard<std::mutex> g(search_mu);  // one search at a time (public endpoint)
        const auto rs = web_search(q, n, nf, (size_t)mc);
        Json j = Json::object();
        j["query"] = q;
        j["results"] = Json::array();
        for (auto& r : rs) {
          Json o = Json::object();
          o["title"] = r.title;
          o["url"] = r.url;
          o["snippet"] = r.snippet;
          o["text"] = r.text;
          j["results"].push_back(o);
        }
        respond(fd, 200, j.dump());
      }
    } else if (rq.method == "GET" && rq.path == "/v1/models") {
      Json list = Json::object();
      list["object"] = "list";
      list["data"] = Json::array();
      for (const std::string id : {model_name, std::string("qwen-hyperdrive"), std::string("qwen-hyperdrive-low"), std::string("qwen-hyperdrive-fast")}) {
        Json m = Json::object();
        m["id"] = id;
        m["object"] = "model";
        m["owned_by"] = "local";
        m["meta"]["n_ctx"] = opt.ctx;
        list["data"].push_back(m);
      }
      respond(fd, 200, list.dump());
    } else if (rq.method == "POST" && (rq.path == "/v1/chat/completions" || rq.path == "/v1/completions")) {
      try {
        const Json body = Json::parse(rq.body);
        completion(fd, body, rq.path == "/v1/chat/completions");
      } catch (const std::exception& e) {
        Json err = Json::object();
        err["error"]["message"] = e.what();
        err["error"]["type"] = "invalid_request_error";
        respond(fd, 400, err.dump());
      }
    } else {
      respond(fd, 404, "{\"error\":{\"message\":\"not found\"}}");
    }
    close(fd);
  }

  void completion(int fd, const Json& b, bool chat) {
    std::string prompt;
    bool thinking = true;
    Json tools;
    int alias_budget = -1;
    std::string effort, roles;
    const int req_max = (int)b.get_num("max_completion_tokens", b.get_num("max_tokens", -1));
    if (chat) {
      if (!b["messages"].is_array()) throw std::runtime_error("messages must be an array");
      const Json& kw = b["chat_template_kwargs"];
      // model aliases: <name>-fast = no thinking, <name>-low = short thinking (agents)
      const std::string mname = b.get_str("model");
      auto ends = [&](const char* suf) { const size_t n = strlen(suf); return mname.size() >= n && mname.compare(mname.size() - n, n, suf) == 0; };
      bool def_think = true;
      if (ends("-fast")) def_think = false;
      if (ends("-low")) { effort = "low"; alias_budget = 2048; }
      thinking = kw.get_bool("enable_thinking", b.get_bool("enable_thinking", def_think));
      // reasoning_effort as sent by OpenAI-style clients (none/minimal/low/medium/high/xhigh)
      const std::string re = b.get_str("reasoning_effort");
      if (chat_format() == ChatFormat::Inkling && !re.empty() && re != "none") effort = re;  // its own effort levels
      else if (re == "none") thinking = false;
      else if (re == "minimal" || re == "low") effort = "low";
      else if (re == "medium") effort = "medium";
      else if (re == "high" || re == "xhigh") effort = "xhigh";
      if (req_max > 0 && req_max < 256) thinking = false;  // short calls (titles etc.): no room to think
      if (b["tools"].is_array() && !b["tools"].arr().empty() && b.get_str("tool_choice") != "none") tools = b["tools"];
      std::map<std::string, int> rc;
      for (auto& m : b["messages"].arr()) rc[m.get_str("role")]++;
      for (auto& [r, n] : rc) roles += r + ":" + std::to_string(n) + " ";
      QE_LOG("[server] request: model=%s think=%d effort=%s max_tokens=%d tools=%zu roles=%s\n", mname.c_str(), (int)thinking,
             effort.empty() ? (thinking ? "xhigh" : "-") : effort.c_str(), req_max, tools.is_array() ? tools.arr().size() : (size_t)0,
             roles.c_str());
      std::vector<ChatMessage> msgs = chat_messages_from_json(b["messages"]);
      // Compaction: a conversation that no longer fits the context (prompt + room for the answer)
      // loses its oldest messages after the leading system prompt; the last user turn always stays,
      // and the kept part starts at a user message.  A note in the system prompt says how many were
      // dropped.  (Crude by design: no summarization.)
      const int reserve = std::clamp(req_max > 0 ? req_max : 8192, 1024, 16384);
      const int limit = opt.ctx - reserve;
      std::string rendered = chat_render(msgs, true, thinking, tools, effort);
      int ntok = (int)tok->encode(rendered, true).size();
      if (ntok > limit) {
        size_t first = 0;
        while (first < msgs.size() && (msgs[first].role == "system" || msgs[first].role == "developer")) first++;
        size_t last_user = msgs.size();
        for (size_t i = msgs.size(); i-- > first;) if (msgs[i].role == "user") { last_user = i; break; }
        std::vector<int> cost(msgs.size(), 0);
        for (size_t i = first; i < msgs.size(); i++)
          cost[i] = (int)tok->encode(msgs[i].content + msgs[i].reasoning + (msgs[i].tool_calls.is_array() ? msgs[i].tool_calls.dump() : ""), false).size() + 8;
        size_t cut = first;  // messages [first, cut) are dropped
        int over = ntok - limit;
        auto advance = [&] {
          if (cut < last_user) { over -= cost[cut]; cut++; }
          while (cut < last_user && msgs[cut].role != "user") { over -= cost[cut]; cut++; }  // keep starting at a user turn
        };
        while (over > 0 && cut < last_user) advance();
        // summary of the dropped prefix [first, cut): reuse the longest cached prefix summary and fold
        // the newly dropped messages into it
        auto prefix_hash = [&](size_t end) {
          size_t h = 1469598103934665603ull;
          for (size_t i = first; i < end; i++) h = (h ^ std::hash<std::string>()(msgs[i].role + "\x1f" + msgs[i].content)) * 1099511628211ull;
          return h;
        };
        std::string summary;
        size_t sum_cut = first;
        auto make_summary = [&] {
          if (cut == sum_cut) return;
          const size_t hc = prefix_hash(cut);
          std::lock_guard<std::mutex> g(mu);
          if (summaries.count(hc)) { summary = summaries[hc]; sum_cut = cut; return; }
          size_t j = cut;
          std::string prev;
          while (j > first) {  // longest cached prefix
            j--;
            auto it = summaries.find(prefix_hash(j));
            if (it != summaries.end()) { prev = it->second; break; }
          }
          if (j == first && prev.empty()) j = first;
          const double ts = qe_now();
          summary = summarize(std::vector<ChatMessage>(msgs.begin() + j, msgs.begin() + cut), prev);
          summaries[hc] = summary;
          sum_cut = cut;
          QE_LOG("[server] compaction: summarized messages %zu..%zu (%zu chars) in %.1f s\n", j - first, cut - first,
                 summary.size(), qe_now() - ts);
        };
        for (;;) {
          std::vector<ChatMessage> kept(msgs.begin(), msgs.begin() + first);
          kept.insert(kept.end(), msgs.begin() + cut, msgs.end());
          make_summary();
          const std::string note = "（注: 会話が文脈長の上限を超えたため、古いメッセージ " + std::to_string(cut - first) +
                                   " 件を次の要約に置き換えています。）\n【これまでの会話の要約】\n" + summary;
          if (cut > first) {
            if (first > 0) kept[0].content += "\n\n" + note;
            else kept.insert(kept.begin(), ChatMessage{"system", note, "", Json()});
          }
          rendered = chat_render(kept, true, thinking, tools, effort);
          ntok = (int)tok->encode(rendered, true).size();
          if (ntok <= limit || cut >= last_user) break;
          advance();
        }
        QE_LOG("[server] compaction: dropped %zu of %zu messages, prompt now %d tokens (limit %d)\n", cut - first, msgs.size(),
               ntok, limit);
      }
      prompt = rendered;
    } else {
      prompt = b.get_str("prompt");
    }
    const bool stream = b.get_bool("stream", false);
    int max_tokens = req_max;
    std::vector<std::string> stops;
    if (b["stop"].is_string()) stops.push_back(b["stop"].str());
    if (b["stop"].is_array())
      for (auto& s : b["stop"].arr()) if (s.is_string()) stops.push_back(s.str());
    const SamplingParams sp = sampling_from(b);

    const double t_req = qe_now();
    std::unique_lock<std::mutex> lk(mu);
    std::vector<int> ids = tok->encode(prompt, true);
    if ((int)ids.size() >= opt.ctx - 1) throw std::runtime_error("prompt too long: " + std::to_string(ids.size()) + " tokens");
    const int budget = opt.ctx - (int)ids.size() - 2;
    if (max_tokens < 0 || max_tokens > budget) max_tokens = budget;
    if (max_tokens <= 0) max_tokens = 1;
    // thinking budget: when the reasoning reaches it, the thinking is closed (Qwen's recommended
    // early-exit phrase + </think>) so an answer is always produced within max_tokens
    int think_budget = -1;
    if (chat && thinking) {
      const Json& kw2 = b["chat_template_kwargs"];
      const int reserve = std::min(2048, std::max(128, max_tokens / 4));
      think_budget = (int)b.get_num("thinking_budget", kw2.get_num("thinking_budget", alias_budget > 0 ? alias_budget : max_tokens - reserve));
      think_budget = std::clamp(think_budget, 1, std::max(1, max_tokens - 64));
    }

    const std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(++id_counter);
    const int64_t created = (int64_t)time(nullptr);
    if (stream)
      // no-transform / X-Accel-Buffering: keep proxies (Cloudflare, nginx) from compressing or buffering the
      // stream; the padded comment pushes past proxies that hold back the first few KB
      send_all(fd, std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                               "Cache-Control: no-cache, no-transform\r\nX-Accel-Buffering: no\r\n") + kCors +
                       "Connection: close\r\n\r\n:" + std::string(2048, ' ') + "\n\n");
    auto chunk = [&](const std::string& reasoning, const std::string& content, const char* finish) -> bool {
      Json c = Json::object();
      c["id"] = id;
      c["object"] = chat ? "chat.completion.chunk" : "text_completion";
      c["created"] = created;
      c["model"] = model_name;
      Json ch = Json::object();
      ch["index"] = 0;
      if (chat) {
        Json d = Json::object();
        if (!reasoning.empty()) d["reasoning_content"] = reasoning;
        if (!content.empty()) d["content"] = content;
        ch["delta"] = d;
      } else {
        ch["text"] = content;
      }
      ch["finish_reason"] = finish ? Json(finish) : Json();
      c["choices"] = Json::array();
      c["choices"].push_back(ch);
      return send_all(fd, "data: " + c.dump() + "\n\n");
    };

    // Prompt processing can run for minutes (long contexts) without a byte on the wire, and proxies
    // (the Cloudflare tunnel: ~100 s) drop idle streams, which the browser reports as "network error".
    // While a prefill runs, a helper thread sends SSE comment lines every 10 s; it is joined before the
    // request thread writes again, so the two never write at the same time.
    // The same thread watches the connection every second (streaming or not) and raises `cancel`
    // when the client has gone, which stops the prefill between chunks.
    std::atomic<bool> cancel{false};
    auto keepalive = [&](auto&& fn) {
      std::mutex km;
      std::condition_variable kcv;
      bool kdone = false;
      model->set_cancel(&cancel);
      std::thread th([&] {
        std::unique_lock<std::mutex> lk(km);
        for (int tick = 1; !kcv.wait_for(lk, std::chrono::seconds(1), [&] { return kdone; }); tick++) {
          if (client_gone(fd)) { cancel = true; break; }
          if (stream && tick % 10 == 0 && !send_all(fd, ": keep-alive\n\n")) { cancel = true; break; }
        }
      });
      struct Join {
        std::mutex& m; std::condition_variable& cv; bool& done; std::thread& t;
        ~Join() { { std::lock_guard<std::mutex> g(m); done = true; } cv.notify_all(); t.join(); }
      } join{km, kcv, kdone, th};
      const int r = fn();
      model->set_cancel(nullptr);
      return r;
    };

    gen->set_sampling(sp);
    std::vector<int> out;
    std::string bytes_pending, reasoning, content, text_all;
    ThinkSplitter split(chat && thinking);
    ToolCallFilter tfilter;
    const bool use_tools = chat && tools.is_array();
    // a tool call written although no tools were offered (e.g. the client's last round, when the
    // history still holds earlier calls) must not leak into the answer as raw <tool_call> markup
    const bool filter_tools = chat;
    const char* finish = "length";
    bool alive = true;
    ReqStat rsn;
    int reused0 = 0;
    rsn.id = id;
    rsn.prompt_n = (int)ids.size();
    {
      std::lock_guard<std::mutex> g(live.m);
      live.t_start = t_req;
      live.cur = rsn;
    }
    live_update("prefill", nullptr, false);
    out.push_back(keepalive([&] { return gen->start(ids); }));
    if (cancel) {  // the client left during the prompt: drop the partial state, send nothing
      gen->abandon();
      QE_LOG("[server] %s: client disconnected during prompt processing, cancelled\n", id.c_str());
      live_update("idle", nullptr, true);
      return;
    }
    {
      const auto& st0 = gen->stats();
      reused0 = gen->reused();
      rsn.cache_n = gen->reused();
      rsn.prompt_n = (int)ids.size() - gen->reused();
      rsn.prefill_s = st0.t_prefill;
      rsn.prefill_tps = rsn.prompt_n / std::max(st0.t_prefill, 1e-9);
      rsn.ttft_ms = (qe_now() - t_req) * 1e3;
      rsn.gen_n = 1;
    }
    live_update("decode", &rsn, true);
    size_t emitted = 0;
    auto flush = [&](bool final) {
      std::string piece_r, piece_c;
      for (; emitted < out.size(); emitted++) bytes_pending += tok->token_text(out[emitted]);
      std::string text = final ? bytes_pending : take_utf8(bytes_pending);
      if (final) bytes_pending.clear();
      text_all += text;
      if (chat) split.feed(text, piece_r, piece_c, final);
      else piece_c = text;
      if (filter_tools) piece_c = tfilter.feed(piece_c, final);
      reasoning += piece_r;
      content += piece_c;
      if (stream && alive && (!piece_r.empty() || !piece_c.empty())) alive = chunk(piece_r, piece_c, nullptr);
    };
    GenStats pre{};  // stats of the generation segments before a forced end of thinking
    bool forced = false;
    for (;;) {
      // stop on EOS tokens (do not emit them)
      bool stop = false;
      for (size_t k = emitted; k < out.size(); k++)
        if (out[k] == stop_ids[0] || out[k] == stop_ids[1]) {
          out.resize(k);
          finish = "stop";
          stop = true;
          break;
        }
      flush(stop);
      if (!stops.empty()) {
        for (auto& st : stops) {
          const size_t p = content.find(st);
          if (p != std::string::npos) {
            content.resize(p);
            finish = "stop";
            stop = true;
          }
        }
      }
      if (alive && client_gone(fd)) {  // no reader any more (non-streaming requests never write before the end)
        alive = false;
        QE_LOG("[server] %s: client disconnected, generation stopped after %zu tokens\n", id.c_str(), out.size());
      }
      if (stop || (int)out.size() >= max_tokens || !alive) break;
      // a reasoning loop (the same passage repeated >= 4 times at the end) is closed like the budget
      bool looping = false;
      if (chat && thinking && !forced && split.in_think && (out.size() & 7) == 0 && reasoning.size() >= 800) {
        const size_t L = reasoning.size();
        for (size_t per = 12; per <= 400 && !looping && 4 * per <= L; per++) {
          if (4 * per < 200) continue;
          bool rep = true;
          for (size_t k = 1; k < 4 && rep; k++)
            rep = reasoning.compare(L - per, per, reasoning, L - (k + 1) * per, per) == 0;
          looping = rep;
        }
      }
      if (!forced && split.in_think && ((think_budget > 0 && (int)out.size() >= think_budget) || looping)) {
        if (looping) QE_LOG("[server] %s: reasoning loop detected, closing the reasoning\n", id.c_str());
        // close the reasoning: feed the closing text as a prompt continuation (state is reused)
        forced = true;
        const std::vector<int> fx =
            tok->encode(chat_close_reasoning());
        const GenStats sx = gen->stats();
        pre.steps += sx.steps; pre.drafted += sx.drafted; pre.accepted += sx.accepted; pre.generated += sx.generated;
        pre.t_prefill += sx.t_prefill; pre.t_decode += sx.t_decode;
        std::vector<int> cont = ids;
        cont.insert(cont.end(), out.begin(), out.end());
        cont.insert(cont.end(), fx.begin(), fx.end());
        out.insert(out.end(), fx.begin(), fx.end());
        if (!looping) QE_LOG("[server] %s: thinking budget %d reached, closing the reasoning\n", id.c_str(), think_budget);
        out.push_back(keepalive([&] { return gen->start(cont); }));
        if (cancel) { gen->abandon(); alive = false; break; }
        pre.t_decode += gen->stats().t_prefill;  // the forced tokens count as decode time
        continue;
      }
      gen->step(out);
      {
        const auto& sx = gen->stats();
        rsn.gen_n = (int)out.size();
        rsn.decode_s = sx.t_decode;
        rsn.decode_tps = std::max(0, sx.generated - 1) / std::max(sx.t_decode, 1e-9);
        rsn.drafted = sx.drafted;
        rsn.accepted = sx.accepted;
        rsn.steps = sx.steps;
        live_update(nullptr, &rsn, (sx.steps & 15) == 0);
      }
    }
    if (std::string(finish) != "stop") flush(true);
    GenStats st = gen->stats();
    if (forced) {
      st.steps += pre.steps; st.drafted += pre.drafted; st.accepted += pre.accepted;
      st.generated += pre.generated;
      st.t_prefill = pre.t_prefill;
      st.t_decode += pre.t_decode;
    }
    Json usage = Json::object();
    usage["prompt_tokens"] = (int)ids.size();
    usage["completion_tokens"] = (int)out.size();
    usage["total_tokens"] = (int)(ids.size() + out.size());
    Json timings = Json::object();
    timings["prompt_n"] = (int)ids.size() - reused0;
    timings["cache_n"] = reused0;
    timings["prompt_ms"] = st.t_prefill * 1e3;
    timings["prompt_per_second"] = (ids.size() - reused0) / std::max(st.t_prefill, 1e-9);
    timings["predicted_n"] = (int)out.size();
    timings["predicted_ms"] = st.t_decode * 1e3;
    timings["predicted_per_second"] = std::max(0, st.generated - 1) / std::max(st.t_decode, 1e-9);
    timings["draft_n"] = st.drafted;
    timings["draft_n_accepted"] = st.accepted;
    rsn.gen_n = (int)out.size();
    rsn.decode_s = st.t_decode;
    rsn.decode_tps = std::max(0, st.generated - 1) / std::max(st.t_decode, 1e-9);
    rsn.drafted = st.drafted;
    rsn.accepted = st.accepted;
    rsn.steps = st.steps;
    rsn.finish = finish;
    rsn.t_end = qe_now() - t_boot;
    live_update("idle", &rsn, true);
    {
      std::lock_guard<std::mutex> g(live.m);
      live.hist.push_back(rsn);
      if (live.hist.size() > 100) live.hist.pop_front();
      live.requests++;
      live.total_gen += out.size();
      live.total_prompt += rsn.prompt_n;
      if (rsn.prompt_n >= 128) {
        live.sum_pf_tok += rsn.prompt_n;
        live.sum_pf_s += rsn.prefill_s;
        if (rsn.prefill_tps > live.best_prefill.prefill_tps) live.best_prefill = rsn;
      }
      if (rsn.gen_n >= 32 && rsn.decode_s > 0) {
        live.sum_dec_tok += rsn.gen_n - 1;
        live.sum_dec_s += rsn.decode_s;
        if (rsn.decode_tps > live.best_decode.decode_tps) live.best_decode = rsn;
      }
    }
    if (!use_tools && tfilter.in_call && content.find_first_not_of(" \t\r\n") == std::string::npos) {
      // the model wanted to call a tool in a round without tools: say so instead of an empty answer
      const std::string note = "（検索の回数上限に達したため、ここで回答を終えます。続けて質問すると、これまでの結果を踏まえて答えます）";
      content += note;
      if (stream && alive) alive = chunk("", note, nullptr);
      QE_LOG("[server] %s: tool call without tools offered: dropped\n", id.c_str());
    }
    Json tool_calls;
    if (use_tools && tfilter.in_call) {
      tool_calls = parse_tool_calls(tfilter.raw, tools, id_counter.load());
      if (!tool_calls.arr().empty()) finish = "tool_calls";
    }
    QE_LOG("[server] %s: prompt %zu (+%d cached) in %.2fs, gen %zu tok %.2f tok/s, mtp %d/%d, finish=%s\n", id.c_str(),
           ids.size() - reused0, reused0, st.t_prefill, out.size(),
           std::max(0, st.generated - 1) / std::max(st.t_decode, 1e-9), st.accepted, st.drafted, finish);
    if (stream) {
      if (alive && tool_calls.is_array() && !tool_calls.arr().empty()) {
        Json c = Json::object();
        c["id"] = id;
        c["object"] = "chat.completion.chunk";
        c["created"] = created;
        c["model"] = model_name;
        Json ch = Json::object();
        ch["index"] = 0;
        Json d = Json::object();
        d["tool_calls"] = Json::array();
        int k = 0;
        for (const Json& tc : tool_calls.arr()) {
          Json t = tc;
          t["index"] = k++;
          d["tool_calls"].push_back(t);
        }
        ch["delta"] = d;
        ch["finish_reason"] = Json();
        c["choices"] = Json::array();
        c["choices"].push_back(ch);
        alive = send_all(fd, "data: " + c.dump() + "\n\n");
      }
      if (alive) {
        Json c = Json::object();
        c["id"] = id;
        c["object"] = chat ? "chat.completion.chunk" : "text_completion";
        c["created"] = created;
        c["model"] = model_name;
        Json ch = Json::object();
        ch["index"] = 0;
        if (chat) ch["delta"] = Json::object();
        else ch["text"] = "";
        ch["finish_reason"] = finish;
        c["choices"] = Json::array();
        c["choices"].push_back(ch);
        c["usage"] = usage;
        c["timings"] = timings;
        send_all(fd, "data: " + c.dump() + "\n\ndata: [DONE]\n\n");
      }
      return;
    }
    Json r = Json::object();
    r["id"] = id;
    r["object"] = chat ? "chat.completion" : "text_completion";
    r["created"] = created;
    r["model"] = model_name;
    Json ch = Json::object();
    ch["index"] = 0;
    if (chat) {
      Json msg = Json::object();
      msg["role"] = "assistant";
      const bool tc = tool_calls.is_array() && !tool_calls.arr().empty();
      msg["content"] = tc && content.empty() ? Json() : Json(content);
      if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
      if (tc) msg["tool_calls"] = tool_calls;
      ch["message"] = msg;
    } else {
      ch["text"] = content;
    }
    ch["finish_reason"] = finish;
    r["choices"] = Json::array();
    r["choices"].push_back(ch);
    r["usage"] = usage;
    r["timings"] = timings;
    lk.unlock();
    respond(fd, 200, r.dump());
  }
};

}  // namespace

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  Server srv;
  std::string host = "127.0.0.1";
  int port = 8080;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) QE_FATAL("missing value for %s", a.c_str()); return argv[++i]; };
    if (a == "-m" || a == "--model") srv.opt.model_path = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "-c" || a == "--ctx") srv.opt.ctx = std::stoi(next());
    else if (a == "--kv") {
      std::string k = next();
      srv.opt.kv_type = k == "f16" ? 0 : k == "q8" ? 1 : k == "q3" ? 3 : 2;
    } else if (a == "--cpu-frac") {
      const std::string v = next();
      srv.opt.cpu_frac = v == "auto" ? -1.0 : std::stod(v);
    }
    else if (a == "-t" || a == "--threads") srv.opt.cpu_threads = std::stoi(next());
    else if (a == "--pcie-gbps") srv.opt.pcie_gbps = std::stod(next());
    else if (a == "--ubatch") srv.opt.ubatch = std::stoi(next());
    else if (a == "--vram-reserve") srv.opt.vram_reserve_mb = std::stoul(next());
    else if (a == "--no-mtp") srv.opt.use_mtp = false;
    else if (a == "-v" || a == "--verbose") srv.opt.verbose = true;
    else if (a == "--spec-relax") srv.spec_relax = std::stof(next());
    else if (a == "--no-ngram") srv.ngram = false;
    else if (a == "--draft-model") srv.draft_model = next();
    else if (a == "--mtp-vocab") srv.opt.mtp_vocab = std::stoi(next());
    else if (a == "--draft") srv.mtp_max = std::stoi(next());
    else if (a == "--cache-keep") srv.opt.cache_keep = std::stoi(next());
    else if (a == "--cache-blocks") srv.opt.cache_blocks = std::stoi(next());
    else if (a == "--cache-host-mb") srv.opt.cache_host_mb = std::stoi(next());
    else if (a == "--skip-ffn") {
      const std::string v = next();
      srv.skip_ffn_str = v;
      for (size_t i = 0; i < v.size();) { srv.opt.skip_ffn.push_back(std::stoi(v.substr(i))); i = v.find(',', i); if (i == std::string::npos) break; i++; }
    }
    else if (a == "--no-elastic") srv.opt.elastic_kv = false;
    else if (a == "-h" || a == "--help") {
      printf("usage: qwen35-server -m model.gguf [--host 127.0.0.1] [--port 8080] [-c 131072] [--kv q4|q8|f16]\n"
             "                    [--cpu-frac auto|0..1] [--pcie-gbps 20] [-t 6] [--ubatch 512] [--vram-reserve 64] [--no-mtp] [--no-elastic] [--spec-relax 1]\n");
      return 0;
    } else QE_FATAL("unknown argument %s (see --help)", a.c_str());
  }
  if (srv.opt.model_path.empty()) QE_FATAL("-m model.gguf is required");
  if (!srv.draft_model.empty()) {
    // a DSpark block drafter (dflash) runs inside the target; other drafts are separate models
    if (GGUFFile(srv.draft_model).get_str("general.architecture", "") == "dflash") srv.opt.dspark_path = srv.draft_model;
    else srv.draft = create_draft_model(srv.draft_model, srv.opt);  // before the target
  }
  srv.model = create_model(srv.opt);
  srv.tok = std::make_unique<Tokenizer>(srv.model->gguf());
  srv.gen = std::make_unique<Generator>(*srv.model, SamplingParams{}, srv.opt.use_mtp);
  srv.stop_ids[0] = srv.tok->eos();
  srv.stop_ids[1] = srv.tok->find("<|endoftext|>");
  srv.gen->set_boundary_token(srv.tok->find("<|im_start|>"));
  if (srv.tok->find("<|message_model|>") >= 0 && srv.tok->find("<|content_model_end_sampling|>") >= 0) {
    chat_set_format(ChatFormat::Inkling);
    srv.gen->set_boundary_token(srv.tok->find("<|end_message|>"));  // checkpoint at the end of the last message
    QE_LOG("[server] chat format: Inkling 2\n");
  }
  srv.gen->set_spec_relax(srv.spec_relax);
  srv.gen->set_mtp_max(srv.mtp_max);
  srv.gen->set_ngram(srv.ngram);
  if (srv.draft) {
    if (srv.draft->cfg().n_vocab != srv.model->cfg().n_vocab) QE_FATAL("draft model vocabulary differs from the model's");
    srv.gen->set_draft_model(srv.draft.get());
  }
  {
    std::string p = srv.opt.model_path;
    const size_t sl = p.find_last_of('/');
    srv.model_name = sl == std::string::npos ? p : p.substr(sl + 1);
  }
  const int ls = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) QE_FATAL("bad host %s", host.c_str());
  if (bind(ls, (sockaddr*)&addr, sizeof addr) != 0) QE_FATAL("bind %s:%d failed", host.c_str(), port);
  listen(ls, 64);
  QE_LOG("[server] listening on http://%s:%d  (model %s, ctx %d, mtp %s)\n", host.c_str(), port, srv.model_name.c_str(),
         srv.opt.ctx, srv.model->has_mtp() ? "on" : "off");
  for (;;) {
    const int fd = accept(ls, nullptr, nullptr);
    if (fd < 0) continue;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    std::thread([&srv, fd] { srv.handle(fd); }).detach();
  }
}
