// CPU (AVX2) side of the hybrid FFN: int8-activation dot products for GGUF quant formats and a
// spinning thread pool that computes a row-slice of a SwiGLU FFN concurrently with the GPU.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "common.h"

namespace cpu {

bool type_supported(uint32_t t);
bool baseline_supported();

// Instruction set of the int8 matvec kernels, detected at first use (QE_CPU_ISA overrides)
enum class Isa { AVX2, VNNI, AVX512 };
Isa isa();
const char* isa_name(Isa i);
std::string cpu_features();              // inference-relevant x86 extensions the CPU reports
const std::vector<int>& core_order();    // one logical CPU per physical core, fastest cores first
int fast_cores();                        // cores of the fastest class (P-cores of a hybrid CPU), >= 1

// int8 activations, one scale per 256-block (q8_K style) + int16 sums per 16 (for min formats).
// Weight sub-block scales are applied in the integer domain; one float multiply per 256 block.
struct Q8Act {
  int K = 0, B = 0;
  std::vector<int8_t> q;      // [B][K]
  std::vector<float> d;       // [B][K/256]
  std::vector<int16_t> bs16;  // [B][K/16]
  void resize(int K_, int B_);
};
void quantize(const float* x, int K, int b, Q8Act& a, int blk0, int blk1);  // 256-blocks [blk0,blk1) of row b

// "R4": lossless CPU-friendly repacking of IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S.  Their codebooks
// only contain 3 (IQ2) or 8 (IQ3) distinct magnitudes, so every weight is exactly a 4-bit
// (sign, magnitude-index) nibble -> decoded with one pshufb like IQ4_XS, instead of per-8 grid
// lookups.  Super-block (256 weights, 148 B): float factor | uint8 scale per 16 | 128 B nibbles.
constexpr int R4_BLOCK_BYTES = 148;
bool r4_supported(uint32_t type);
const int8_t* r4_table(uint32_t type);  // 16 signed values
// convert rows [0,N) of a GGUF matrix; dst must hold N * (K/256) * R4_BLOCK_BYTES bytes
void r4_convert(uint32_t type, const uint8_t* src, size_t src_rb, int K, int N, uint8_t* dst);
void matvec_r4(const int8_t* tab, const uint8_t* W, int K, int r0, int r1, const Q8Act& x, int B,
               float* y, int ldy);

// y[b*ldy + r] = dot(W[r,:], x[b]) for r in [r0, r1)
void matvec(uint32_t type, const uint8_t* W, size_t row_bytes, int K, int r0, int r1,
            const Q8Act& x, int B, float* y, int ldy);

// Persistent worker threads (pinned to physical cores).  start() hands a job to all workers and
// returns immediately; wait() blocks until every worker finished it.  Workers spin briefly between
// jobs (low latency during decode) and then sleep.
class ThreadPool {
 public:
  explicit ThreadPool(int n);
  ~ThreadPool();
  int size() const { return n_; }
  void start(std::function<void(int)> fn);
  void wait();
  void run(std::function<void(int)> fn) { start(std::move(fn)); wait(); }
  void barrier();  // only valid inside a job; all n workers must call it

 private:
  void worker(int idx);
  int n_;
  std::vector<std::thread> threads_;
  std::function<void(int)> job_;
  std::atomic<uint64_t> gen_{0};
  std::atomic<int> done_{0};
  std::atomic<int> bar_count_{0};
  std::atomic<uint64_t> bar_gen_{0};
  std::atomic<bool> stop_{false};
};

}  // namespace cpu
