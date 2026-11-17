// Backend-independent model interface used by the generator and the server (CUDA: Model, Vulkan: VkModel).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class GGUFFile;

constexpr int kMaxTok = 8;  // max tokens per small-batch (decode / speculative verify) forward

struct EngineOptions {
  std::string model_path;
  int ctx = 131072;            // max context length (tokens)
  int kv_type = 2;              // KV cache format (0 f16, 1 q8, 2 q4, 3 q3)     // KV cache format
  int ubatch = 2048;           // prefill chunk (its workspace is lent to FFN weights while decoding)
  size_t vram_reserve_mb = 64;   // keep this much VRAM free (all big buffers are allocated up front)
  int64_t vram_budget_mb = -1;  // override total VRAM the engine may use (-1 = auto)
  int cpu_threads = -1;        // -1: physical cores
  bool use_mtp = true;
  bool elastic_kv = true;      // lend unused KV pool space to FFN weights
  int ck_slots = 8;
  std::vector<int> skip_ffn;
  int cache_keep = 1;          // inactive conversations kept in VRAM (if within cache_blocks)
  int cache_blocks = 0;        // ... resident in VRAM with at most this many KV blocks (others: host RAM)
  int cache_host_mb = 6144;    // host RAM for offloaded KV of inactive conversations (LRU beyond)   // lossy: layers whose FFN block is dropped (saves their PCIe/VRAM bytes)
  int mtp_vocab = 32768;       // MTP draft head vocabulary: first ids + scripts in use + seen tokens (0 = full)            // prompt-cache state checkpoints kept in pinned host memory
  int draft_max = 7;           // max drafts per verify pass (n-gram drafts; MTP chains are capped separately)
  double cpu_frac = -1;         // fraction of streamed FFN rows computed on the CPU (<0: auto per layer)
  double pcie_gbps = -1;        // host->GPU bandwidth assumed by the auto balancer (-1: measured at start-up)
  double dram_gbps = -1;        // host DRAM bandwidth shared by CPU kernels and DMA (-1: measured)
  bool cpu_r4 = true;           // allow lossless R4 repacking of the CPU part
  bool verbose = false;
};

struct Config {
  int n_embd, n_layer, n_ff, n_head, n_kv, hd, rope_dims, n_vocab;
  float rope_base, eps;
  int conv_k, dk, dv, nk, nv, d_inner, attn_interval;
  int n_mtp;
  int conv_dim() const { return 2 * nk * dk + nv * dv; }
  bool is_attn(int il) const { return il >= n_layer || (il + 1) % attn_interval == 0; }
};

struct KvInfo {
  int blocks_used = 0, blocks_total = 0, blocks_lendable = 0, seqs = 0, seqs_offloaded = 0, cur_positions = 0;
  int borrowed_kv = 0, borrowed_ws = 0, streamed = 0;
  double host_mb = 0, block_mb = 0;
};
struct PlanStats {
  double res_mb = 0, stream_mb = 0, cpu_mb = 0;
  int res_ffn = 0;
};

Config load_qwen35_config(const GGUFFile& g);  // fatal on unsupported architecture / dims

class IModel {
 public:
  virtual ~IModel() = default;
  virtual const Config& cfg() const = 0;
  virtual const GGUFFile& gguf() const = 0;
  virtual const EngineOptions& options() const = 0;
  virtual bool has_mtp() const = 0;
  virtual const char* backend() const = 0;
  // Process tokens[0..n) at positions pos0..; logits (host, n_logits rows) and/or argmax of the last
  // n_logits rows.  snapshot: keep the recurrence inputs for rollback() (speculative verify).
  virtual void forward(const int* tokens, int n, int pos0, float* logits, int n_logits, int* argmax = nullptr,
                       bool snapshot = false, bool mtp_prefill = false) = 0;
  virtual void reset() = 0;
  virtual void prepare_context(int npos, bool small_prefill = false) = 0;
  // speculative decoding (MTP); defaults for backends without it
  virtual int draft_max() const { return 0; }
  virtual int mtp_draft(int, const int*, int, int, float* = nullptr, float* = nullptr, bool = true) { return -1; }
  virtual int mtp_chain(int, int, float* = nullptr, float* = nullptr) { return -1; }
  virtual void rollback(int) {}
  virtual void set_carry_row(int) {}
  virtual void draft_vocab_add(const int*, int) {}
  virtual const int* mtp_top4() const { static const int none[4] = {-1, -1, -1, -1}; return none; }
  // prompt cache: KV sequences and recurrent-state checkpoints
  virtual int seq_create() = 0;
  virtual void seq_select(int id) = 0;
  virtual bool seq_alive(int id) const = 0;
  virtual void seq_drop(int id) = 0;
  virtual int ck_slots() const { return 0; }
  virtual void ck_save_live(int) {}
  virtual void ck_restore(int) {}
  virtual void set_capture(int, int) {}
  // GPU-side top-K for sampling (0 = unsupported: full logits are returned)
  virtual void set_topk(int) {}
  virtual int topk_K() const { return 0; }
  virtual const int* topk_idx(int) const { return nullptr; }
  virtual const float* topk_val(int) const { return nullptr; }
  virtual void sync() {}
  // telemetry
  virtual KvInfo kv_info() const { return {}; }
  virtual PlanStats plan_stats() const { return {}; }
};
