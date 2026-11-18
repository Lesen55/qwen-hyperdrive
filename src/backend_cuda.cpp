#include "backend.h"

#include "common.h"
#include "gguf.h"
#include "model.h"

std::unique_ptr<IModel> create_model(const EngineOptions& opt) { return std::make_unique<Model>(opt); }
const char* backend_name() { return "cuda"; }

std::unique_ptr<IModel> create_draft_model(const std::string& path, const EngineOptions& target) {
  EngineOptions o;
  o.model_path = path;
  o.ctx = target.ctx;
  o.kv_type = 2;
  o.ubatch = 512;
  o.use_mtp = false;
  o.elastic_kv = false;
  o.ck_slots = 0;
  o.cache_keep = 0;
  o.verbose = target.verbose;
  o.cpu_threads = target.cpu_threads;
  {
    GGUFFile g(path);
    const Config c = load_qwen35_config(g);
    if (c.mhc_streams) QE_FATAL("draft model %s: looped models cannot be drafts", path.c_str());
    int n_attn = 0;
    for (int il = 0; il < c.n_layer; il++) n_attn += c.is_attn(il);
    const double kv = (double)o.ctx * n_attn * 2 * c.n_kv * c.hd * 0.6;  // ~q4 with scales
    o.vram_budget_mb = (int64_t)((g.size() * 1.08 + kv) / 1048576.0) + 640;  // + workspace / state
  }
  QE_LOG("[plan] draft model %s: VRAM budget %lld MiB\n", path.c_str(), (long long)o.vram_budget_mb);
  return create_model(o);
}
