// Common definitions for qwen35-engine.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define QE_FATAL(...)                                                          \
  do {                                                                         \
    fprintf(stderr, "[fatal] %s:%d: ", __FILE__, __LINE__);                    \
    fprintf(stderr, __VA_ARGS__);                                              \
    fprintf(stderr, "\n");                                                     \
    std::abort();                                                              \
  } while (0)

#define QE_ASSERT(c)                                                           \
  do {                                                                         \
    if (!(c)) QE_FATAL("assertion failed: %s", #c);                            \
  } while (0)

extern int g_verbose;
#define QE_LOG(...)                                                            \
  do {                                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0)
#define QE_DBG(...)                                                            \
  do {                                                                         \
    if (g_verbose) fprintf(stderr, __VA_ARGS__);                               \
  } while (0)

double qe_now();  // seconds, monotonic

// GGML tensor type ids (as stored in GGUF).
enum QType : uint32_t {
  QT_F32 = 0,
  QT_F16 = 1,
  QT_Q8_0 = 8,
  QT_Q2_K = 10,
  QT_Q3_K = 11,
  QT_Q4_K = 12,
  QT_Q5_K = 13,
  QT_Q6_K = 14,
  QT_IQ2_XXS = 16,
  QT_IQ2_XS = 17,
  QT_IQ3_XXS = 18,
  QT_IQ1_S = 19,
  QT_IQ4_NL = 20,
  QT_IQ3_S = 21,
  QT_IQ2_S = 22,
  QT_IQ4_XS = 23,
  QT_IQ1_M = 29,
  QT_BF16 = 30,
  QT_NVFP4 = 40,  // 64-element blocks: 4 UE4M3 scales (per 16) + 32 bytes of E2M1 nibbles
};

struct QTypeInfo {
  const char* name;
  int block_elems;  // elements per block
  int block_bytes;  // bytes per block
};
const QTypeInfo& qtype_info(uint32_t t);
bool qtype_supported(uint32_t t);
inline size_t qtype_row_bytes(uint32_t t, int64_t n) {
  const auto& i = qtype_info(t);
  return (size_t)(n / i.block_elems) * i.block_bytes;
}
