#pragma once
#include "argmax.h"
#include "attn_f16.h"
#include "attn_q8.h"
#include "attn_combine.h"
#include "cm_gemm.h"
#include "conv.h"
#include "dq_f32.h"
#include "dq_f16.h"
#include "dq_bf16.h"
#include "dq_q8_0.h"
#include "dq_q2_k.h"
#include "dq_q3_k.h"
#include "dq_q4_k.h"
#include "dq_q5_k.h"
#include "dq_q6_k.h"
#include "dq_iq2_xxs.h"
#include "dq_iq2_xs.h"
#include "dq_iq2_s.h"
#include "dq_iq3_xxs.h"
#include "dq_iq3_s.h"
#include "dq_iq1_s.h"
#include "dq_iq1_m.h"
#include "dq_iq4_xs.h"
#include "dq16_f32.h"
#include "dq16_f16.h"
#include "dq16_bf16.h"
#include "dq16_q8_0.h"
#include "dq16_q2_k.h"
#include "dq16_q3_k.h"
#include "dq16_q4_k.h"
#include "dq16_q5_k.h"
#include "dq16_q6_k.h"
#include "dq16_iq2_xxs.h"
#include "dq16_iq2_xs.h"
#include "dq16_iq2_s.h"
#include "dq16_iq3_xxs.h"
#include "dq16_iq3_s.h"
#include "dq16_iq1_s.h"
#include "dq16_iq1_m.h"
#include "dq16_iq4_xs.h"
#include "silu_mul.h"
#include "gate.h"
#include "gdn_gb.h"
#include "ksum.h"
#include "to_f16.h"
#include "gated_norm.h"
#include "gdn_rec.h"
#include "mm_f32.h"
#include "mm_f16.h"
#include "mm_bf16.h"
#include "mm_q8_0.h"
#include "mm_q2_k.h"
#include "mm_q3_k.h"
#include "mm_q4_k.h"
#include "mm_q5_k.h"
#include "mm_q6_k.h"
#include "mm_iq2_xxs.h"
#include "mm_iq2_xs.h"
#include "mm_iq2_s.h"
#include "mm_iq3_xxs.h"
#include "mm_iq3_s.h"
#include "mm_iq1_s.h"
#include "mm_iq1_m.h"
#include "mm_iq4_xs.h"
#include "gemm16.h"
#include "headnorm.h"
#include "kv_store.h"
#include "mv_f32.h"
#include "mv_f16.h"
#include "mv_bf16.h"
#include "mv_q8_0.h"
#include "mv_q2_k.h"
#include "mv_q3_k.h"
#include "mv_q4_k.h"
#include "mv_q5_k.h"
#include "mv_q6_k.h"
#include "mv_iq2_xxs.h"
#include "mv_iq2_xs.h"
#include "mv_iq2_s.h"
#include "mv_iq3_xxs.h"
#include "mv_iq3_s.h"
#include "mv_iq1_s.h"
#include "mv_iq1_m.h"
#include "mv_iq4_xs.h"
#include "rmsnorm.h"
#include "rope.h"
struct SpvEntry { const char* name; const unsigned* code; unsigned size; };
static const SpvEntry kSpv[] = {
  {"argmax", spv_argmax, sizeof(spv_argmax)},
  {"attn_f16", spv_attn_f16, sizeof(spv_attn_f16)},
  {"attn_q8", spv_attn_q8, sizeof(spv_attn_q8)},
  {"attn_combine", spv_attn_combine, sizeof(spv_attn_combine)},
  {"cm_gemm", spv_cm_gemm, sizeof(spv_cm_gemm)},
  {"conv", spv_conv, sizeof(spv_conv)},
  {"dq_f32", spv_dq_f32, sizeof(spv_dq_f32)},
  {"dq_f16", spv_dq_f16, sizeof(spv_dq_f16)},
  {"dq_bf16", spv_dq_bf16, sizeof(spv_dq_bf16)},
  {"dq_q8_0", spv_dq_q8_0, sizeof(spv_dq_q8_0)},
  {"dq_q2_k", spv_dq_q2_k, sizeof(spv_dq_q2_k)},
  {"dq_q3_k", spv_dq_q3_k, sizeof(spv_dq_q3_k)},
  {"dq_q4_k", spv_dq_q4_k, sizeof(spv_dq_q4_k)},
  {"dq_q5_k", spv_dq_q5_k, sizeof(spv_dq_q5_k)},
  {"dq_q6_k", spv_dq_q6_k, sizeof(spv_dq_q6_k)},
  {"dq_iq2_xxs", spv_dq_iq2_xxs, sizeof(spv_dq_iq2_xxs)},
  {"dq_iq2_xs", spv_dq_iq2_xs, sizeof(spv_dq_iq2_xs)},
  {"dq_iq2_s", spv_dq_iq2_s, sizeof(spv_dq_iq2_s)},
  {"dq_iq3_xxs", spv_dq_iq3_xxs, sizeof(spv_dq_iq3_xxs)},
  {"dq_iq3_s", spv_dq_iq3_s, sizeof(spv_dq_iq3_s)},
  {"dq_iq1_s", spv_dq_iq1_s, sizeof(spv_dq_iq1_s)},
  {"dq_iq1_m", spv_dq_iq1_m, sizeof(spv_dq_iq1_m)},
  {"dq_iq4_xs", spv_dq_iq4_xs, sizeof(spv_dq_iq4_xs)},
  {"dq16_f32", spv_dq16_f32, sizeof(spv_dq16_f32)},
  {"dq16_f16", spv_dq16_f16, sizeof(spv_dq16_f16)},
  {"dq16_bf16", spv_dq16_bf16, sizeof(spv_dq16_bf16)},
  {"dq16_q8_0", spv_dq16_q8_0, sizeof(spv_dq16_q8_0)},
  {"dq16_q2_k", spv_dq16_q2_k, sizeof(spv_dq16_q2_k)},
  {"dq16_q3_k", spv_dq16_q3_k, sizeof(spv_dq16_q3_k)},
  {"dq16_q4_k", spv_dq16_q4_k, sizeof(spv_dq16_q4_k)},
  {"dq16_q5_k", spv_dq16_q5_k, sizeof(spv_dq16_q5_k)},
  {"dq16_q6_k", spv_dq16_q6_k, sizeof(spv_dq16_q6_k)},
  {"dq16_iq2_xxs", spv_dq16_iq2_xxs, sizeof(spv_dq16_iq2_xxs)},
  {"dq16_iq2_xs", spv_dq16_iq2_xs, sizeof(spv_dq16_iq2_xs)},
  {"dq16_iq2_s", spv_dq16_iq2_s, sizeof(spv_dq16_iq2_s)},
  {"dq16_iq3_xxs", spv_dq16_iq3_xxs, sizeof(spv_dq16_iq3_xxs)},
  {"dq16_iq3_s", spv_dq16_iq3_s, sizeof(spv_dq16_iq3_s)},
  {"dq16_iq1_s", spv_dq16_iq1_s, sizeof(spv_dq16_iq1_s)},
  {"dq16_iq1_m", spv_dq16_iq1_m, sizeof(spv_dq16_iq1_m)},
  {"dq16_iq4_xs", spv_dq16_iq4_xs, sizeof(spv_dq16_iq4_xs)},
  {"silu_mul", spv_silu_mul, sizeof(spv_silu_mul)},
  {"gate", spv_gate, sizeof(spv_gate)},
  {"gdn_gb", spv_gdn_gb, sizeof(spv_gdn_gb)},
  {"ksum", spv_ksum, sizeof(spv_ksum)},
  {"to_f16", spv_to_f16, sizeof(spv_to_f16)},
  {"gated_norm", spv_gated_norm, sizeof(spv_gated_norm)},
  {"gdn_rec", spv_gdn_rec, sizeof(spv_gdn_rec)},
  {"mm_f32", spv_mm_f32, sizeof(spv_mm_f32)},
  {"mm_f16", spv_mm_f16, sizeof(spv_mm_f16)},
  {"mm_bf16", spv_mm_bf16, sizeof(spv_mm_bf16)},
  {"mm_q8_0", spv_mm_q8_0, sizeof(spv_mm_q8_0)},
  {"mm_q2_k", spv_mm_q2_k, sizeof(spv_mm_q2_k)},
  {"mm_q3_k", spv_mm_q3_k, sizeof(spv_mm_q3_k)},
  {"mm_q4_k", spv_mm_q4_k, sizeof(spv_mm_q4_k)},
  {"mm_q5_k", spv_mm_q5_k, sizeof(spv_mm_q5_k)},
  {"mm_q6_k", spv_mm_q6_k, sizeof(spv_mm_q6_k)},
  {"mm_iq2_xxs", spv_mm_iq2_xxs, sizeof(spv_mm_iq2_xxs)},
  {"mm_iq2_xs", spv_mm_iq2_xs, sizeof(spv_mm_iq2_xs)},
  {"mm_iq2_s", spv_mm_iq2_s, sizeof(spv_mm_iq2_s)},
  {"mm_iq3_xxs", spv_mm_iq3_xxs, sizeof(spv_mm_iq3_xxs)},
  {"mm_iq3_s", spv_mm_iq3_s, sizeof(spv_mm_iq3_s)},
  {"mm_iq1_s", spv_mm_iq1_s, sizeof(spv_mm_iq1_s)},
  {"mm_iq1_m", spv_mm_iq1_m, sizeof(spv_mm_iq1_m)},
  {"mm_iq4_xs", spv_mm_iq4_xs, sizeof(spv_mm_iq4_xs)},
  {"gemm16", spv_gemm16, sizeof(spv_gemm16)},
  {"headnorm", spv_headnorm, sizeof(spv_headnorm)},
  {"kv_store", spv_kv_store, sizeof(spv_kv_store)},
  {"mv_f32", spv_mv_f32, sizeof(spv_mv_f32)},
  {"mv_f16", spv_mv_f16, sizeof(spv_mv_f16)},
  {"mv_bf16", spv_mv_bf16, sizeof(spv_mv_bf16)},
  {"mv_q8_0", spv_mv_q8_0, sizeof(spv_mv_q8_0)},
  {"mv_q2_k", spv_mv_q2_k, sizeof(spv_mv_q2_k)},
  {"mv_q3_k", spv_mv_q3_k, sizeof(spv_mv_q3_k)},
  {"mv_q4_k", spv_mv_q4_k, sizeof(spv_mv_q4_k)},
  {"mv_q5_k", spv_mv_q5_k, sizeof(spv_mv_q5_k)},
  {"mv_q6_k", spv_mv_q6_k, sizeof(spv_mv_q6_k)},
  {"mv_iq2_xxs", spv_mv_iq2_xxs, sizeof(spv_mv_iq2_xxs)},
  {"mv_iq2_xs", spv_mv_iq2_xs, sizeof(spv_mv_iq2_xs)},
  {"mv_iq2_s", spv_mv_iq2_s, sizeof(spv_mv_iq2_s)},
  {"mv_iq3_xxs", spv_mv_iq3_xxs, sizeof(spv_mv_iq3_xxs)},
  {"mv_iq3_s", spv_mv_iq3_s, sizeof(spv_mv_iq3_s)},
  {"mv_iq1_s", spv_mv_iq1_s, sizeof(spv_mv_iq1_s)},
  {"mv_iq1_m", spv_mv_iq1_m, sizeof(spv_mv_iq1_m)},
  {"mv_iq4_xs", spv_mv_iq4_xs, sizeof(spv_mv_iq4_xs)},
  {"rmsnorm", spv_rmsnorm, sizeof(spv_rmsnorm)},
  {"rope", spv_rope, sizeof(spv_rope)},
};
