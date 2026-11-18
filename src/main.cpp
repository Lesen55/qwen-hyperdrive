// qwen35-engine CLI.
//   qwen35-engine ppl  -m model.gguf --ids file   : perplexity over token ids (whitespace separated)
//   qwen35-engine gen  -m model.gguf --ids file -n N : greedy continuation, prints token ids
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "engine.h"
#include "backend.h"
#include "gguf.h"

static std::vector<int> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) QE_FATAL("cannot read %s", path.c_str());
  std::string s((std::istreambuf_iterator<char>(f)), {});
  for (char& ch : s)
    if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
  std::istringstream is(s);
  std::vector<int> v;
  int x;
  while (is >> x) v.push_back(x);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: qwen35-engine <ppl|gen> -m model.gguf --ids file [-n N] [-c ctx] [--kv f16|q8|q4] [--chunk N]\n");
    return 1;
  }
  std::string cmd = argv[1];
  EngineOptions opt;
  std::string ids_path;
  int n_gen = 32, chunk = 512, reserve = 4096;
  float spec_relax = 1.f;
  int mtp_max = 5;
  bool ngram = true;
  float draft_pmin = 0.f;
  SamplingParams sp;
  sp.temperature = 0.f;
  std::string draft_model;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) QE_FATAL("missing value for %s", a.c_str()); return argv[++i]; };
    if (a == "-m") opt.model_path = next();
    else if (a == "--ids") ids_path = next();
    else if (a == "-n") n_gen = std::stoi(next());
    else if (a == "-c") opt.ctx = std::stoi(next());
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--ubatch") opt.ubatch = std::stoi(next());
    else if (a == "--vram-budget") opt.vram_budget_mb = std::stoll(next());
    else if (a == "--kv") {
      std::string k = next();
      opt.kv_type = k == "f16" ? 0 : k == "q8" ? 1 : k == "q3" ? 3 : 2;
    } else if (a == "--cpu-frac") opt.cpu_frac = std::stod(next());
    else if (a == "-t") opt.cpu_threads = std::stoi(next());
    else if (a == "--pcie-gbps") opt.pcie_gbps = std::stod(next());
    else if (a == "--dram-gbps") opt.dram_gbps = std::stod(next());
    else if (a == "--no-r4") opt.cpu_r4 = false;
    else if (a == "--temp") sp.temperature = std::stof(next());
    else if (a == "--seed") sp.seed = std::stoull(next());
    else if (a == "--no-mtp") opt.use_mtp = false;
    else if (a == "--no-elastic") opt.elastic_kv = false;
    else if (a == "--reserve") reserve = std::stoi(next());
    else if (a == "--draft") mtp_max = std::stoi(next());
    else if (a == "--no-ngram") ngram = false;
    else if (a == "--draft-model") draft_model = next();
    else if (a == "--cache-keep") opt.cache_keep = std::stoi(next());
    else if (a == "--cache-blocks") opt.cache_blocks = std::stoi(next());
    else if (a == "--cache-host-mb") opt.cache_host_mb = std::stoi(next());
    else if (a == "--skip-ffn") {
      const std::string v = next();
      for (size_t i = 0; i < v.size();) { opt.skip_ffn.push_back(std::stoi(v.substr(i))); i = v.find(',', i); if (i == std::string::npos) break; i++; }
    }
    else if (a == "--vram-reserve") opt.vram_reserve_mb = std::stoul(next());
    else if (a == "--spec-relax") spec_relax = std::stof(next());
    else if (a == "--mtp-vocab") opt.mtp_vocab = std::stoi(next());
    else if (a == "--draft-pmin") draft_pmin = std::stof(next());
    else if (a == "-v") opt.verbose = true;
    else QE_FATAL("unknown argument %s", a.c_str());
  }
  std::unique_ptr<IModel> dmp;
  if (!draft_model.empty()) dmp = create_draft_model(draft_model, opt);  // first: it takes its share of VRAM
  std::unique_ptr<IModel> mp = create_model(opt);
  IModel& model = *mp;
  const int V = model.cfg().n_vocab;
  std::vector<int> ids = read_ids(ids_path);
  if (cmd == "ppl") {
    // non-overlapping chunks of `chunk` tokens, score second half of each chunk (like llama-perplexity)
    double nll = 0;
    int cnt = 0;
    std::vector<float> logits((size_t)chunk * V);
    const int nchunks = (int)ids.size() / chunk;
    for (int ci = 0; ci < nchunks; ci++) {
      model.reset();
      const double t0 = qe_now();
      model.forward(ids.data() + (size_t)ci * chunk, chunk, 0, logits.data(), chunk);
      const double dt = qe_now() - t0;
      for (int i = chunk / 2; i < chunk - 1; i++) {
        const float* l = &logits[(size_t)i * V];
        float mx = -INFINITY;
        for (int j = 0; j < V; j++) mx = std::max(mx, l[j]);
        double se = 0;
        for (int j = 0; j < V; j++) se += std::exp((double)l[j] - mx);
        nll += -((double)l[ids[(size_t)ci * chunk + i + 1]] - mx - std::log(se));
        cnt++;
      }
      printf("[%d] ppl=%.4f  (%.1f tok/s)\n", ci + 1, std::exp(nll / cnt), chunk / dt);
      fflush(stdout);
    }
    printf("final ppl = %.4f over %d tokens\n", std::exp(nll / cnt), cnt);
  } else if (cmd == "gen") {
    Generator gen(model, sp, opt.use_mtp);
    gen.set_draft_pmin(draft_pmin);
    gen.set_spec_relax(spec_relax);
    gen.set_mtp_max(mtp_max);
    gen.set_ngram(ngram);
    if (dmp) gen.set_draft_model(dmp.get());
    std::vector<int> out;
    // QE_REPEAT=n: run the same generation n times through the prompt cache (server-like reuse)
    for (int rep = getenv("QE_REPEAT") ? atoi(getenv("QE_REPEAT")) : 1; rep > 1; rep--) {
      out.clear();
      out.push_back(gen.start(ids, getenv("QE_REPEAT_NOREUSE") == nullptr, reserve));
      while ((int)out.size() < n_gen) gen.step(out);
      const auto& st = gen.stats();
      fprintf(stderr, "[repeat] gen %d tok, mtp accepted %d/%d:", st.generated - 1, st.accepted, st.drafted);
      for (int i = 0; i < 40 && i < (int)out.size(); i++) fprintf(stderr, " %d", out[i]);
      fprintf(stderr, "\n");
    }
    out.clear();
    out.push_back(gen.start(ids, true, reserve));
    while ((int)out.size() < n_gen) gen.step(out);
    out.resize(n_gen);
    for (int x : out) printf("%d ", x);
    const auto& st = gen.stats();
    printf("\nprompt %zu tok in %.2f s (%.1f tok/s), gen %d tok in %.2f s (%.2f tok/s), mtp accepted %d/%d\n",
           ids.size(), st.t_prefill, ids.size() / st.t_prefill, st.generated - 1, st.t_decode,
           (st.generated - 1) / st.t_decode, st.accepted, st.drafted);
    printf("time: verify %.2f s, sample %.2f s, mtp+rollback %.2f s\n", st.t_verify, st.t_sample, st.t_mtp);
    printf("n-gram drafted steps %d: accepted %d/%d\n", st.ng_steps, st.ng_accepted, st.ng_drafted);
    printf("steps %d, accepted-per-step histogram: 0:%d 1:%d 2:%d 3:%d  (%.2f tok/step)\n", st.steps, st.acc_hist[0],
           st.acc_hist[1], st.acc_hist[2], st.acc_hist[3], (st.generated - 1) / (double)std::max(1, st.steps));
  }
  return 0;
}
