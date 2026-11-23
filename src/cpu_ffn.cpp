#include "cpu_ffn.h"

#include <immintrin.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>

#include <bit>
#else
#include <pthread.h>
#include <sched.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "quant.h"

namespace cpu {

bool type_supported(uint32_t t) {
  switch (t) {
    case QT_IQ2_XXS: case QT_IQ2_XS: case QT_IQ2_S: case QT_IQ3_XXS: case QT_IQ3_S:
    case QT_IQ4_XS: case QT_Q2_K: case QT_Q4_K:
      return true;
    default:
      return false;
  }
}

void Q8Act::resize(int K_, int B_) {
  K = K_;
  B = B_;
  q.resize((size_t)K * B);
  d.resize((size_t)K / 256 * B);
  bs16.resize((size_t)K / 16 * B);
}

void quantize(const float* x, int K, int b, Q8Act& a, int blk0, int blk1) {
  const float* xr = x + (size_t)b * K;
  int8_t* q = a.q.data() + (size_t)b * K;
  const __m256 sign = _mm256_set1_ps(-0.f);
  for (int blk = blk0; blk < blk1; blk++) {
    const float* p = xr + 256 * blk;
    __m256 m = _mm256_setzero_ps();
    for (int i = 0; i < 256; i += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(p + i)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    const float amax = _mm_cvtss_f32(m4);
    const float dd = amax / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
    a.d[(size_t)b * (K / 256) + blk] = dd;
    const __m256 vid = _mm256_set1_ps(id);
    int8_t* qb = q + 256 * blk;
    for (int i = 0; i < 256; i += 32) {
      __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i), vid));
      __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i + 8), vid));
      __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i + 16), vid));
      __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + i + 24), vid));
      __m256i s01 = _mm256_packs_epi32(i0, i1), s23 = _mm256_packs_epi32(i2, i3);
      __m256i v = _mm256_packs_epi16(s01, s23);
      v = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
      _mm256_storeu_si256((__m256i*)(qb + i), v);
    }
    int16_t* bs = a.bs16.data() + (size_t)b * (K / 16) + 16 * blk;
    for (int j = 0; j < 16; j++) {
      int s = 0;
      for (int k = 0; k < 16; k++) s += qb[16 * j + k];
      bs[j] = (int16_t)s;
    }
  }
}

// ------------------------------------------------------------------------------------------------
namespace {

uint64_t g_smask[256];  // byte j = 0xFF if bit j of the index is set
struct InitMask {
  InitMask() {
    for (int i = 0; i < 256; i++) {
      uint64_t m = 0;
      for (int j = 0; j < 8; j++)
        if (i & (1 << j)) m |= 0xFFull << (8 * j);
      g_smask[i] = m;
    }
  }
} g_init_mask;

#define QE_ISA_AVX2
namespace isa_avx2 {
#include "cpu_kernels.inc"
}
#undef QE_ISA_AVX2
#ifdef __clang__
// clang (notably with the MSVC ABI) ignores #pragma GCC target; use the
// equivalent clang attribute pragma to stamp the target features on every
// function defined in the region instead.
#pragma clang attribute push(__attribute__((target("avxvnni"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avxvnni")
#endif
#define QE_ISA_VNNI
namespace isa_vnni {
#include "cpu_kernels.inc"
}
#undef QE_ISA_VNNI
#ifdef __clang__
#pragma clang attribute pop
#else
#pragma GCC pop_options
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512vnni")
#endif
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni"))), apply_to = function)
#endif
#define QE_ISA_AVX512
namespace isa_avx512 {
#include "cpu_kernels.inc"
}
#undef QE_ISA_AVX512
#ifdef __clang__
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
using namespace isa_avx2;  // R4 conversion (load time only)

struct R4Tables {
  int8_t tab[32][16];  // indexed by ggml type id
  int8_t idx[32][256]; // signed value (as uint8) -> nibble, -1 if not representable
  R4Tables() {
    memset(tab, 0, sizeof tab);
    memset(idx, -1, sizeof idx);
    auto build = [&](int t, std::vector<int> mags) {
      for (size_t i = 0; i < mags.size(); i++) {
        tab[t][i] = (int8_t)mags[i];
        tab[t][8 + i] = (int8_t)-mags[i];
        idx[t][(uint8_t)(int8_t)mags[i]] = (int8_t)i;
        idx[t][(uint8_t)(int8_t)-mags[i]] = (int8_t)(8 + i);
      }
    };
    for (int t : {QT_IQ2_XXS, QT_IQ2_XS, QT_IQ2_S}) build(t, {8, 25, 43});
    build(QT_IQ3_XXS, {4, 12, 20, 28, 36, 44, 52, 62});
    build(QT_IQ3_S, {1, 3, 5, 7, 9, 11, 13, 15});
  }
} g_r4;

template <int T>
void r4_convert_t(const uint8_t* src, size_t src_rb, int K, int N, uint8_t* dst) {
  using D = Dec<T>;
  const int nb = K / 256;
  for (int r = 0; r < N; r++) {
    for (int j = 0; j < nb; j++) {
      const uint8_t* blk = src + (size_t)r * src_rb + (size_t)j * D::BB;
      uint8_t* o = dst + ((size_t)r * nb + j) * R4_BLOCK_BYTES;
      const float f = D::factor(blk);
      memcpy(o, &f, 4);
      uint8_t* sc = o + 4;
      uint8_t* q = o + 20;
      for (int ib = 0; ib < 8; ib++) {
        D32 w;
        D::dec(blk, ib, w);
        alignas(32) int8_t v[32];
        alignas(32) int16_t s[16];
        _mm256_store_si256((__m256i*)v, w.w);
        _mm256_store_si256((__m256i*)s, w.s16);
        sc[2 * ib] = (uint8_t)s[0];
        sc[2 * ib + 1] = (uint8_t)s[8];
        for (int e = 0; e < 16; e++) {
          const int8_t a = g_r4.idx[T][(uint8_t)v[e]], b = g_r4.idx[T][(uint8_t)v[e + 16]];
          if (a < 0 || b < 0) QE_FATAL("r4: value not in codebook");
          q[16 * ib + e] = (uint8_t)(a | (b << 4));
        }
      }
    }
  }
}

}  // namespace

bool r4_supported(uint32_t t) {
  return t == QT_IQ2_XXS || t == QT_IQ2_XS || t == QT_IQ2_S || t == QT_IQ3_XXS || t == QT_IQ3_S;
}
const int8_t* r4_table(uint32_t t) { return g_r4.tab[t]; }
void r4_convert(uint32_t type, const uint8_t* src, size_t src_rb, int K, int N, uint8_t* dst) {
  switch (type) {
    case QT_IQ2_XXS: r4_convert_t<QT_IQ2_XXS>(src, src_rb, K, N, dst); break;
    case QT_IQ2_XS: r4_convert_t<QT_IQ2_XS>(src, src_rb, K, N, dst); break;
    case QT_IQ2_S: r4_convert_t<QT_IQ2_S>(src, src_rb, K, N, dst); break;
    case QT_IQ3_XXS: r4_convert_t<QT_IQ3_XXS>(src, src_rb, K, N, dst); break;
    case QT_IQ3_S: r4_convert_t<QT_IQ3_S>(src, src_rb, K, N, dst); break;
    default: QE_FATAL("r4_convert: unsupported type %u", type);
  }
}
// ---- instruction-set selection (once, at first use): AVX-512 VNNI > AVX-VNNI > AVX2; QE_CPU_ISA
// = avx2 | vnni | avx512 overrides (only to something the CPU has)
#ifdef _WIN32
// __builtin_cpu_supports() lowers to libgcc's __cpu_model / __cpu_indicator_init,
// which do not exist in the MSVC CRT, so the link fails. Do the CPUID leaf 7
// checks directly (with the OS XCR0 state checks the builtin also performs).
static bool qe_cpu_has(const char* name) {
  struct Feats {
    bool avx2, fma, f16c, bmi2, avxvnni, avx512f, avx512bw, avx512vl, avx512vnni, avx512bf16, avxvnniint8, amx_int8;
  };
  static const Feats ft = [] {
    Feats f{};
    int regs[4] = {};
    __cpuidex(regs, 0, 0);
    const int max_leaf = regs[0];
    int l1[4] = {}, l7[4] = {}, l7s1[4] = {};
    if (max_leaf >= 1) __cpuidex(l1, 1, 0);
    if (max_leaf >= 7) {
      __cpuidex(l7, 7, 0);
      if (l7[0] >= 1) __cpuidex(l7s1, 7, 1);
    }
    const bool osxsave = (l1[2] >> 27) & 1;
    const unsigned long long xcr0 = osxsave ? (unsigned long long)_xgetbv(0) : 0;
    const bool os_avx = ((xcr0 & 0x6) == 0x6) && ((l1[2] >> 28) & 1);
    const bool os_512 = os_avx && ((xcr0 & 0xE6) == 0xE6);
    f.fma = ((l1[2] >> 12) & 1) && os_avx;
    f.f16c = ((l1[2] >> 29) & 1) && os_avx;
    f.avx2 = ((l7[1] >> 5) & 1) && os_avx;
    f.bmi2 = (l7[1] >> 8) & 1;
    f.avxvnni = ((l7s1[0] >> 4) & 1) && os_avx;
    f.avx512f = ((l7[1] >> 16) & 1) && os_512;
    f.avx512bw = ((l7[1] >> 30) & 1) && os_512;
    f.avx512vl = ((l7[1] >> 31) & 1) && os_512;
    f.avx512vnni = ((l7[2] >> 11) & 1) && os_512;
    f.avx512bf16 = ((l7s1[0] >> 5) & 1) && os_512;
    f.avxvnniint8 = ((l7s1[3] >> 4) & 1) && os_avx;
    f.amx_int8 = (l7[3] >> 25) & 1;
    return f;
  }();
  if (!strcmp(name, "avx2")) return ft.avx2;
  if (!strcmp(name, "fma")) return ft.fma;
  if (!strcmp(name, "f16c")) return ft.f16c;
  if (!strcmp(name, "bmi2")) return ft.bmi2;
  if (!strcmp(name, "avxvnni")) return ft.avxvnni;
  if (!strcmp(name, "avx512f")) return ft.avx512f;
  if (!strcmp(name, "avx512bw")) return ft.avx512bw;
  if (!strcmp(name, "avx512vl")) return ft.avx512vl;
  if (!strcmp(name, "avx512vnni")) return ft.avx512vnni;
  if (!strcmp(name, "avx512bf16")) return ft.avx512bf16;
  if (!strcmp(name, "avxvnniint8")) return ft.avxvnniint8;
  if (!strcmp(name, "amx-int8")) return ft.amx_int8;
  return false;
}
#define QE_CPU_INIT() ((void)0)
#define QE_CPU_HAS(name) qe_cpu_has(name)
#else
#define QE_CPU_INIT() __builtin_cpu_init()
#define QE_CPU_HAS(name) __builtin_cpu_supports(name)
#endif
bool baseline_supported() {
  QE_CPU_INIT();
  return QE_CPU_HAS("avx2") && QE_CPU_HAS("fma") && QE_CPU_HAS("f16c") && QE_CPU_HAS("bmi2");
}

static Isa detect_isa() {
  QE_CPU_INIT();
  const bool a512 = QE_CPU_HAS("avx512f") && QE_CPU_HAS("avx512bw") &&
                    QE_CPU_HAS("avx512vl") && QE_CPU_HAS("avx512vnni");
  const bool vnni = QE_CPU_HAS("avxvnni");
  Isa best = a512 ? Isa::AVX512 : vnni ? Isa::VNNI : Isa::AVX2;
  if (const char* e = getenv("QE_CPU_ISA")) {
    const std::string v = e;
    if (v == "avx2") best = Isa::AVX2;
    else if (v == "vnni" && (vnni || a512)) best = vnni ? Isa::VNNI : Isa::AVX512;
    else if (v == "avx512" && a512) best = Isa::AVX512;
  }
  return best;
}
Isa isa() {
  static const Isa i = detect_isa();
  return i;
}
const char* isa_name(Isa i) {
  switch (i) {
    case Isa::AVX512: return "AVX-512 VNNI";
    case Isa::VNNI: return "AVX-VNNI";
    default: return "AVX2";
  }
}
std::string cpu_features() {
  QE_CPU_INIT();
  std::string s;
  for (const char* f : {"avx2", "fma", "f16c", "avxvnni", "avx512f", "avx512bw", "avx512vl", "avx512vnni", "avx512bf16",
                        "avxvnniint8", "amx-int8"}) {
    bool has = false;
#define F(name) if (!strcmp(f, name)) has = QE_CPU_HAS(name);
    F("avx2") F("fma") F("f16c") F("avxvnni") F("avx512f") F("avx512bw") F("avx512vl") F("avx512vnni") F("avx512bf16")
    F("avxvnniint8") F("amx-int8")
#undef F
    if (has) s += std::string(s.empty() ? "" : " ") + f;
  }
  return s;
}

template <class F2, class F1, class F0>
static void by_isa(F2 f512, F1 fvnni, F0 favx2) {
  switch (isa()) {
    case Isa::AVX512: f512(); break;
    case Isa::VNNI: fvnni(); break;
    default: favx2(); break;
  }
}

void matvec_r4(const int8_t* tab, const uint8_t* W, int K, int r0, int r1, const Q8Act& x, int B, float* y,
               int ldy) {
#define R4(ns)                                                         \
  switch (B) {                                                         \
    case 1: ns::mv_r4<1>(tab, W, K, r0, r1, x, y, ldy); break;         \
    case 2: ns::mv_r4<2>(tab, W, K, r0, r1, x, y, ldy); break;         \
    case 3: ns::mv_r4<3>(tab, W, K, r0, r1, x, y, ldy); break;         \
    case 4: ns::mv_r4<4>(tab, W, K, r0, r1, x, y, ldy); break;         \
    case 5: ns::mv_r4<5>(tab, W, K, r0, r1, x, y, ldy); break;         \
    case 6: ns::mv_r4<6>(tab, W, K, r0, r1, x, y, ldy); break;         \
    case 7: ns::mv_r4<7>(tab, W, K, r0, r1, x, y, ldy); break;         \
    case 8: ns::mv_r4<8>(tab, W, K, r0, r1, x, y, ldy); break;         \
    default: QE_FATAL("cpu matvec_r4: batch %d > 8", B);               \
  }
  by_isa([&] { R4(isa_avx512) }, [&] { R4(isa_vnni) }, [&] { R4(isa_avx2) });
#undef R4
}

void matvec(uint32_t type, const uint8_t* W, size_t rb, int K, int r0, int r1, const Q8Act& x, int B,
            float* y, int ldy) {
#define MV(ns)                                                                                              \
  switch (type) {                                                                                           \
    case QT_IQ2_XXS: ns::mv_b<QT_IQ2_XXS>(W, rb, K, r0, r1, x, B, y, ldy); break;                           \
    case QT_IQ2_XS: ns::mv_b<QT_IQ2_XS>(W, rb, K, r0, r1, x, B, y, ldy); break;                             \
    case QT_IQ2_S: ns::mv_b<QT_IQ2_S>(W, rb, K, r0, r1, x, B, y, ldy); break;                               \
    case QT_IQ3_XXS: ns::mv_b<QT_IQ3_XXS>(W, rb, K, r0, r1, x, B, y, ldy); break;                           \
    case QT_IQ3_S: ns::mv_b<QT_IQ3_S>(W, rb, K, r0, r1, x, B, y, ldy); break;                               \
    case QT_IQ4_XS: ns::mv_b<QT_IQ4_XS>(W, rb, K, r0, r1, x, B, y, ldy); break;                             \
    case QT_Q2_K: ns::mv_b<QT_Q2_K>(W, rb, K, r0, r1, x, B, y, ldy); break;                                 \
    case QT_Q4_K: ns::mv_b<QT_Q4_K>(W, rb, K, r0, r1, x, B, y, ldy); break;                                 \
    default: QE_FATAL("cpu matvec: unsupported type %u", type);                                             \
  }
  by_isa([&] { MV(isa_avx512) }, [&] { MV(isa_vnni) }, [&] { MV(isa_avx2) });
#undef MV
}

// ------------------------------------------------------------------------------------------------
// One logical CPU per physical core (SMT siblings share the core's load ports and caches), fastest
// cores first: Intel hybrid parts list their P-cores in /sys/devices/cpu_core/cpus, other systems
// may report cpu_capacity; ties keep the CPU numbering.
#ifdef _WIN32
struct WindowsCore {
  int cpu;
  unsigned efficiency;
};

static const std::vector<WindowsCore>& windows_cores() {
  static const auto cores = [] {
    std::vector<WindowsCore> result;
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::vector<uint64_t> storage((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    auto* data = reinterpret_cast<unsigned char*>(storage.data());
    if (!bytes || !GetLogicalProcessorInformationEx(RelationProcessorCore,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data), &bytes)) return result;
    DWORD_PTR process_mask = ~DWORD_PTR{0};
    if (GetActiveProcessorGroupCount() == 1) {
      DWORD_PTR process = 0, system = 0;
      if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) process_mask = process;
    }
    for (DWORD offset = 0; offset < bytes;) {
      const auto& entry = *reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(data + offset);
      const auto& core = entry.Processor;
      const auto& group = core.GroupMask[0];  // a physical core belongs to one processor group
      const KAFFINITY mask = group.Mask & process_mask;
      if (mask) result.push_back({int(group.Group) * 64 + std::countr_zero(mask), core.EfficiencyClass});
      offset += entry.Size;
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
      return a.efficiency == b.efficiency ? a.cpu < b.cpu : a.efficiency > b.efficiency;
    });
    return result;
  }();
  return cores;
}
#else
static std::vector<int> read_cpu_list(const char* path) {
  std::vector<int> v;
  FILE* f = fopen(path, "r");
  if (!f) return v;
  char buf[4096] = "";
  if (!fgets(buf, sizeof buf, f)) buf[0] = 0;
  fclose(f);
  for (char* p = buf; *p;) {
    char* e;
    const long a = strtol(p, &e, 10);
    if (e == p) break;
    long b = a;
    if (*e == '-') b = strtol(e + 1, &e, 10);
    for (long i = a; i <= b; i++) v.push_back((int)i);
    p = *e == ',' ? e + 1 : e;
    if (*p == '\n') break;
  }
  return v;
}
#endif
const std::vector<int>& core_order() {
  static const std::vector<int> order = [] {
#ifdef _WIN32
    std::vector<int> o;
    for (const auto& core : windows_cores()) o.push_back(core.cpu);
    return o;
#else
    std::vector<int> online = read_cpu_list("/sys/devices/system/cpu/online");
    const std::vector<int> pcores = read_cpu_list("/sys/devices/cpu_core/cpus");
    std::vector<std::pair<std::pair<int, int>, int>> cand;  // ((-rank, cpu), cpu)
    std::vector<std::string> seen;
    for (int cpu : online) {
      const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/";
      std::string sib;
      if (FILE* f = fopen((base + "topology/thread_siblings_list").c_str(), "r")) {
        char b[256] = "";
        if (fgets(b, sizeof b, f)) sib = b;
        fclose(f);
      }
      if (!sib.empty() && std::find(seen.begin(), seen.end(), sib) != seen.end()) continue;  // SMT sibling
      seen.push_back(sib);
      int rank = 0;
      if (!pcores.empty()) rank = std::find(pcores.begin(), pcores.end(), cpu) != pcores.end() ? 2048 : 0;
      if (FILE* f = fopen((base + "cpu_capacity").c_str(), "r")) {
        int cap = 0;
        if (fscanf(f, "%d", &cap) == 1) rank += cap;
        fclose(f);
      }
      cand.push_back({{-rank, cpu}, cpu});
    }
    std::sort(cand.begin(), cand.end());
    std::vector<int> o;
    for (auto& c : cand) o.push_back(c.second);
    return o;
#endif
  }();
  return order;
}

// Memory-bound int8 matvecs on a hybrid CPU are best left to the big cores: the slow (low-power)
// cores finish their row share last and the whole job waits for them (Core Ultra 5 228V: 4 P-cores
// 57.6 GB/s, all 8 cores 19.3 GB/s).
int fast_cores() {
#ifdef _WIN32
  const auto& cores = windows_cores();
  if (cores.empty()) return std::max(1u, std::thread::hardware_concurrency());
  static const int n = (int)std::count_if(cores.begin(), cores.end(), [&](const auto& core) {
    return core.efficiency == cores.front().efficiency;
  });
  return n;
#else
  static const int n = [] {
    const std::vector<int> pcores = read_cpu_list("/sys/devices/cpu_core/cpus");
    const std::vector<int>& order = core_order();
    if (!pcores.empty()) {
      int k = 0;
      for (int c : order) k += std::find(pcores.begin(), pcores.end(), c) != pcores.end();
      if (k > 0) return k;
    }
    int best = -1, k = 0;
    for (int c : order) {
      int cap = 1024;
      if (FILE* f = fopen(("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpu_capacity").c_str(), "r")) {
        if (fscanf(f, "%d", &cap) != 1) cap = 1024;
        fclose(f);
      }
      if (cap > best) { best = cap; k = 1; }
      else if (cap == best) k++;
    }
    return std::max(1, k ? k : (int)order.size());
  }();
  return n;
#endif
}

ThreadPool::ThreadPool(int n) : n_(n) {
  for (int i = 0; i < n; i++) threads_.emplace_back([this, i] { worker(i); });
}
ThreadPool::~ThreadPool() {
  stop_.store(true);
  gen_.fetch_add(1);
  gen_.notify_all();
  for (auto& t : threads_) t.join();
}
void ThreadPool::start(std::function<void(int)> fn) {
  job_ = std::move(fn);
  done_.store(0, std::memory_order_relaxed);
  gen_.fetch_add(1, std::memory_order_release);
  gen_.notify_all();
}
void ThreadPool::wait() {
  while (done_.load(std::memory_order_acquire) < n_) _mm_pause();
}
void ThreadPool::barrier() {
  const uint64_t g = bar_gen_.load(std::memory_order_acquire);
  if (bar_count_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
    bar_count_.store(0, std::memory_order_relaxed);
    bar_gen_.fetch_add(1, std::memory_order_release);
  } else {
    while (bar_gen_.load(std::memory_order_acquire) == g) _mm_pause();
  }
}
void ThreadPool::worker(int idx) {
  {
    const std::vector<int>& order = core_order();
    if (!order.empty()) {
      const int cpu = order[idx % order.size()];
#ifdef _WIN32
      GROUP_AFFINITY affinity{};
      affinity.Group = (WORD)(cpu / 64);
      affinity.Mask = KAFFINITY{1} << (cpu % 64);
      SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr);
#else
      cpu_set_t cs;
      CPU_ZERO(&cs);
      CPU_SET(cpu, &cs);
      pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
    }
  }
  uint64_t seen = 0;
  for (;;) {
    uint64_t g;
    int spins = 0;
    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
      if (++spins < 200000) _mm_pause();
      else gen_.wait(seen, std::memory_order_acquire);
    }
    seen = g;
    if (stop_.load()) return;
    job_(idx);
    done_.fetch_add(1, std::memory_order_acq_rel);
  }
}

}  // namespace cpu
