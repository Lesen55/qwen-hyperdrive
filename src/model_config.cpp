// qwen35 hyper-parameters from GGUF metadata (shared by the CUDA and Vulkan backends)
#include "common.h"
#include "gguf.h"
#include "imodel.h"

#include <algorithm>
#include <cmath>
#include <string>

Config load_qwen35_config(const GGUFFile& g) {
  const std::string arch = g.get_str("general.architecture", "");
  if (arch != "qwen35" && arch != "sensenova_loop")
    QE_FATAL("unsupported architecture '%s' (need qwen35 or sensenova_loop)", arch.c_str());
  Config c{};
  c.arch = arch;
  const std::string a = arch + ".";
  c.n_embd = (int)g.get_int(a + "embedding_length", 0);
  const int blocks = (int)g.get_int(a + "block_count", 0);
  c.n_mtp = (int)g.get_int(a + "nextn_predict_layers", 0);
  c.n_layer = blocks - c.n_mtp;
  c.n_ff = (int)g.get_int(a + "feed_forward_length", 0);
  c.n_head = (int)g.get_int(a + "attention.head_count", 0);
  c.n_kv = (int)g.get_int(a + "attention.head_count_kv", 0);
  c.hd = (int)g.get_int(a + "attention.key_length", 256);
  c.rope_dims = (int)g.get_int(a + "rope.dimension_count", 64);
  c.rope_base = (float)g.get_float(a + "rope.freq_base", 1e7);
  c.eps = (float)g.get_float(a + "attention.layer_norm_rms_epsilon", 1e-6);
  c.conv_k = (int)g.get_int(a + "ssm.conv_kernel", 4);
  c.dk = c.dv = (int)g.get_int(a + "ssm.state_size", 128);
  c.nk = (int)g.get_int(a + "ssm.group_count", 16);
  c.nv = (int)g.get_int(a + "ssm.time_step_rank", 48);
  c.d_inner = (int)g.get_int(a + "ssm.inner_size", 6144);
  c.attn_interval = (int)g.get_int(a + "full_attention_interval", 4);
  c.n_vocab = (int)g.need("token_embd.weight").ne[1];
  if (c.hd != 256 || c.dk != 128 || c.conv_k != 4 || c.d_inner != c.nv * c.dv || c.n_head % c.n_kv || c.nv % c.nk)
    QE_FATAL("unsupported qwen35 dimensions: attention head %d (need 256), GDN state %d (need 128), conv %d (need 4), "
             "heads %d/%d, GDN heads %d/%d", c.hd, c.dk, c.conv_k, c.n_head, c.n_kv, c.nk, c.nv);
  auto int_array = [&](const std::string& key) {
    std::vector<int64_t> v;
    if (const GGUFValue* f = g.get(key))
      for (const auto& e : f->arr) v.push_back(e.as_int());
    return v;
  };
  // per-block layer types (recurrent = Gated DeltaNet), when the file lists them
  const std::vector<int64_t> rec = int_array(a + "attention.recurrent_layers");
  if ((int)rec.size() >= c.n_layer)
    for (int il = 0; il < c.n_layer; il++) c.attn_at.push_back(rec[il] ? 0 : 1);
  // looped model: residual-stream kinds, and the second loop visit re-using the first visit's weights
  c.mhc_streams = (int)g.get_int(a + "mhc.streams", 0);
  if (c.mhc_streams) {
    if (c.mhc_streams != 2) QE_FATAL("mHC with %d residual streams is not supported (2 are)", c.mhc_streams);
    const std::vector<int64_t> k = int_array(a + "mhc.layer_kinds");
    if ((int)k.size() < c.n_layer) QE_FATAL("mhc.layer_kinds has %zu entries for %d blocks", k.size(), c.n_layer);
    for (int il = 0; il < c.n_layer; il++) c.mhc_kind.push_back((uint8_t)k[il]);
    int first = -1, start2 = -1;  // as the reference loader: the second "enter" marks the second visit
    for (int il = 0; il < c.n_layer && start2 < 0; il++)
      if (c.mhc_kind[il] == 1) (first < 0 ? first : start2) = il;
    c.wlayer.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; il++) c.wlayer[il] = il;
    if (start2 > 0) {
      const int shift = start2 - first;
      for (int il = start2; il < start2 + shift && il < c.n_layer; il++) c.wlayer[il] = il - shift;
    }
  }
  // RoPE frequencies, YaRN folded in (ggml's rope_yarn: ramp between the correction dims, ext_factor 1,
  // attention factor 1 + 0.1 ln(scale))
  {
    const int half = c.rope_dims / 2;
    const std::string st = g.get_str(a + "rope.scaling.type", "");
    const double factor = g.get_float(a + "rope.scaling.factor", 1.0);
    const bool yarn = st == "yarn" && factor > 1.0;
    const double fs = yarn ? 1.0 / factor : 1.0;
    double low = 0, high = 0;
    if (yarn) {
      const double n_orig = (double)g.get_int(a + "rope.scaling.original_context_length", 262144);
      const double beta_fast = g.get_float(a + "rope.scaling.yarn_beta_fast", 32.0);
      const double beta_slow = g.get_float(a + "rope.scaling.yarn_beta_slow", 1.0);
      auto corr = [&](double nr) { return c.rope_dims * std::log(n_orig / (nr * 2 * M_PI)) / (2 * std::log((double)c.rope_base)); };
      low = std::max(0.0, std::floor(corr(beta_fast)));
      high = std::min((double)c.rope_dims - 1, std::ceil(corr(beta_slow)));
      c.rope_mscale = (float)(g.get_float(a + "rope.scaling.attn_factor", 1.0) * (1.0 + 0.1 * std::log(1.0 / fs)));
    }
    for (int i = 0; i < half; i++) {
      const double f = std::pow((double)c.rope_base, -2.0 * i / c.rope_dims);
      double mix = 0;  // 1: extrapolate (original frequency), 0: interpolate (frequency * fs)
      if (yarn) mix = 1.0 - std::min(1.0, std::max(0.0, (i - low) / std::max(0.001, high - low)));
      c.rope_freq.push_back(yarn ? f * (fs * (1 - mix) + mix) : f);
    }
    if (yarn) QE_LOG("[model] RoPE: YaRN x%.1f (original context %lld), attention factor %.4f\n", factor,
                     (long long)g.get_int(a + "rope.scaling.original_context_length", 0), c.rope_mscale);
  }
  int n_alias = 0, n_mhc = 0;
  for (int il = 0; il < c.n_layer; il++) { n_alias += c.aliased(il); n_mhc += c.mhc(il) != 0; }
  QE_LOG("[model] %s: %d layers (+%d MTP), embd %d, ff %d, heads %d/%d x %d, GDN %d/%d x %d, vocab %d\n", arch.c_str(),
         c.n_layer, c.n_mtp, c.n_embd, c.n_ff, c.n_head, c.n_kv, c.hd, c.nk, c.nv, c.dk, c.n_vocab);
  if (c.mhc_streams)
    QE_LOG("[model] looped: %d executed blocks, %d of them re-use earlier weights, %d in the %d-stream mHC residual\n",
           c.n_layer, n_alias, n_mhc, c.mhc_streams);
  return c;
}

// Script class of a vocabulary entry (GPT-2 byte-level token string): 1 kana, 2 CJK, 3 hangul,
// 4 cyrillic, 5 arabic, 6 thai, 0 other.  Used to seed the draft vocabulary per script in use.
int token_script(const std::string& tok) {
  static int inv[324];
  static bool init = false;
  if (!init) {
    for (int& v : inv) v = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
      const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      inv[printable ? b : 256 + n++] = b;
    }
    init = true;
  }
  std::string bytes;
  for (size_t i = 0; i < tok.size();) {  // UTF-8 decode of the mapped characters
    const unsigned char c0 = tok[i];
    int cp, len;
    if (c0 < 0x80) { cp = c0; len = 1; }
    else if ((c0 >> 5) == 6 && i + 1 < tok.size()) { cp = ((c0 & 31) << 6) | (tok[i + 1] & 63); len = 2; }
    else return 0;
    if (cp >= 324 || inv[cp] < 0) return 0;
    bytes += (char)inv[cp];
    i += len;
  }
  for (size_t i = 0; i < bytes.size();) {
    const unsigned char c0 = bytes[i];
    int cp, len;
    if (c0 < 0x80) { cp = c0; len = 1; }
    else if ((c0 >> 5) == 6) { cp = c0 & 31; len = 2; }
    else if ((c0 >> 4) == 14) { cp = c0 & 15; len = 3; }
    else if ((c0 >> 3) == 30) { cp = c0 & 7; len = 4; }
    else return 0;
    if (i + len > bytes.size()) return 0;  // partial character
    for (int k = 1; k < len; k++) cp = (cp << 6) | (bytes[i + k] & 63);
    i += len;
    if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0xFF66 && cp <= 0xFF9F)) return 1;
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0xF900 && cp <= 0xFAFF)) return 2;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 3;
    if (cp >= 0x0400 && cp <= 0x04FF) return 4;
    if (cp >= 0x0600 && cp <= 0x06FF) return 5;
    if (cp >= 0x0E00 && cp <= 0x0E7F) return 6;
  }
  return 0;
}
