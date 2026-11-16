// qwen35-server: OpenAI-compatible HTTP server on top of the engine.
//   POST /v1/chat/completions   (stream / non-stream; reasoning_content for <think> output)
//   POST /v1/completions        (raw prompt)
//   GET  /v1/models, /health
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <ctime>
#include <mutex>
#include <thread>

#include "chat.h"
#include "engine.h"
#include "json.h"
#include "tokenizer.h"

namespace {

struct Request {
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
  std::unique_ptr<Model> model;
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Generator> gen;
  std::mutex mu;  // one generation at a time
  std::string model_name;
  std::atomic<int> id_counter{0};
  int stop_ids[2] = {-1, -1};

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
    } else if (rq.method == "GET" && (rq.path == "/health" || rq.path == "/")) {
      respond(fd, 200, "{\"status\":\"ok\"}");
    } else if (rq.method == "GET" && rq.path == "/v1/models") {
      Json m = Json::object();
      m["id"] = model_name;
      m["object"] = "model";
      m["owned_by"] = "local";
      m["meta"]["n_ctx"] = opt.ctx;
      Json list = Json::object();
      list["object"] = "list";
      list["data"] = Json::array();
      list["data"].push_back(m);
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
    if (chat) {
      if (!b["messages"].is_array()) throw std::runtime_error("messages must be an array");
      const Json& kw = b["chat_template_kwargs"];
      thinking = kw.get_bool("enable_thinking", b.get_bool("enable_thinking", true));
      if (b.get_str("reasoning_effort") == "none") thinking = false;
      prompt = chat_render(chat_messages_from_json(b["messages"]), true, thinking);
    } else {
      prompt = b.get_str("prompt");
    }
    const bool stream = b.get_bool("stream", false);
    int max_tokens = (int)b.get_num("max_completion_tokens", b.get_num("max_tokens", -1));
    std::vector<std::string> stops;
    if (b["stop"].is_string()) stops.push_back(b["stop"].str());
    if (b["stop"].is_array())
      for (auto& s : b["stop"].arr()) if (s.is_string()) stops.push_back(s.str());
    const SamplingParams sp = sampling_from(b);

    std::unique_lock<std::mutex> lk(mu);
    std::vector<int> ids = tok->encode(prompt, true);
    if ((int)ids.size() >= opt.ctx - 1) throw std::runtime_error("prompt too long: " + std::to_string(ids.size()) + " tokens");
    const int budget = opt.ctx - (int)ids.size() - 2;
    if (max_tokens < 0 || max_tokens > budget) max_tokens = budget;
    if (max_tokens <= 0) max_tokens = 1;

    const std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(++id_counter);
    const int64_t created = (int64_t)time(nullptr);
    if (stream)
      send_all(fd, std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n") + kCors +
                       "Connection: close\r\n\r\n");
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

    gen->set_sampling(sp);
    std::vector<int> out;
    std::string bytes_pending, reasoning, content, text_all;
    ThinkSplitter split(chat && thinking);
    const char* finish = "length";
    bool alive = true;
    out.push_back(gen->start(ids));
    size_t emitted = 0;
    auto flush = [&](bool final) {
      std::string piece_r, piece_c;
      for (; emitted < out.size(); emitted++) bytes_pending += tok->token_text(out[emitted]);
      std::string text = final ? bytes_pending : take_utf8(bytes_pending);
      if (final) bytes_pending.clear();
      text_all += text;
      if (chat) split.feed(text, piece_r, piece_c, final);
      else piece_c = text;
      reasoning += piece_r;
      content += piece_c;
      if (stream && alive && (!piece_r.empty() || !piece_c.empty())) alive = chunk(piece_r, piece_c, nullptr);
    };
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
      if (stop || (int)out.size() >= max_tokens || !alive) break;
      gen->step(out);
    }
    if (std::string(finish) != "stop") flush(true);
    const auto& st = gen->stats();
    Json usage = Json::object();
    usage["prompt_tokens"] = (int)ids.size();
    usage["completion_tokens"] = (int)out.size();
    usage["total_tokens"] = (int)(ids.size() + out.size());
    Json timings = Json::object();
    timings["prompt_n"] = (int)ids.size() - gen->reused();
    timings["cache_n"] = gen->reused();
    timings["prompt_ms"] = st.t_prefill * 1e3;
    timings["prompt_per_second"] = (ids.size() - gen->reused()) / std::max(st.t_prefill, 1e-9);
    timings["predicted_n"] = (int)out.size();
    timings["predicted_ms"] = st.t_decode * 1e3;
    timings["predicted_per_second"] = std::max(0, st.generated - 1) / std::max(st.t_decode, 1e-9);
    timings["draft_n"] = st.drafted;
    timings["draft_n_accepted"] = st.accepted;
    QE_LOG("[server] %s: prompt %zu (+%d cached) in %.2fs, gen %zu tok %.2f tok/s, mtp %d/%d, finish=%s\n", id.c_str(),
           ids.size() - gen->reused(), gen->reused(), st.t_prefill, out.size(),
           std::max(0, st.generated - 1) / std::max(st.t_decode, 1e-9), st.accepted, st.drafted, finish);
    if (stream) {
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
      msg["content"] = content;
      if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
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
      srv.opt.kv_type = k == "f16" ? qk::KV_F16 : k == "q8" ? qk::KV_Q8 : k == "q3" ? qk::KV_Q3 : qk::KV_Q4;
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
    else if (a == "--no-elastic") srv.opt.elastic_kv = false;
    else if (a == "-h" || a == "--help") {
      printf("usage: qwen35-server -m model.gguf [--host 127.0.0.1] [--port 8080] [-c 131072] [--kv q4|q8|f16]\n"
             "                    [--cpu-frac auto|0..1] [--pcie-gbps 20] [-t 6] [--ubatch 512] [--vram-reserve 300] [--no-mtp] [--no-elastic]\n");
      return 0;
    } else QE_FATAL("unknown argument %s (see --help)", a.c_str());
  }
  if (srv.opt.model_path.empty()) QE_FATAL("-m model.gguf is required");
  srv.model = std::make_unique<Model>(srv.opt);
  srv.tok = std::make_unique<Tokenizer>(srv.model->gguf());
  srv.gen = std::make_unique<Generator>(*srv.model, SamplingParams{}, srv.opt.use_mtp);
  srv.stop_ids[0] = srv.tok->eos();
  srv.stop_ids[1] = srv.tok->find("<|endoftext|>");
  srv.gen->set_boundary_token(srv.tok->find("<|im_start|>"));
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
