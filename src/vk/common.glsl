// Shared definitions for the Vulkan compute shaders.
// All buffers are reached through buffer device addresses passed in push constants (no descriptor
// sets), so every shader is a plain function of its push block.  Weights are GGUF blocks read as
// 32-bit words with byte-granular helpers (block sizes like 98 or 210 bytes are only 2-byte aligned).
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32 { int d[]; };

bool is_null(uvec2 a) { return a.x == 0u && a.y == 0u; }

uint ld8(U32 w, uint b) { return (w.d[b >> 2] >> ((b & 3u) << 3)) & 0xffu; }
uint ld16(U32 w, uint b) { return (w.d[b >> 2] >> ((b & 2u) << 3)) & 0xffffu; }  // b even
uint ld32(U32 w, uint b) {  // b even (all GGUF block fields are 2-byte aligned)
  const uint i = b >> 2;
  if ((b & 2u) == 0u) return w.d[i];
  return (w.d[i] >> 16) | (w.d[i + 1u] << 16);
}
float ldh(U32 w, uint b) { return unpackHalf2x16(ld16(w, b)).x; }
uint byte_of(uint x, uint j) { return (x >> (j << 3)) & 0xffu; }
int sbyte_of(uint x, uint j) { return bitfieldExtract(int(x), int(j << 3), 8); }

// ---- codebook table (word offsets; filled by the host from quant_tables.inc)
#define TAB_IQ2XXS 0u     // 256  x u64
#define TAB_IQ2XS 512u    // 512  x u64
#define TAB_IQ2S 1536u    // 1024 x u64
#define TAB_IQ3XXS 3584u  // 256  x u32
#define TAB_IQ3S 3840u    // 512  x u32
#define TAB_IQ1S 4352u    // 2048 x u64 (signed bytes)
#define TAB_IQ4NL 8448u   // 16 x i8
#define TAB_WORDS 8452u

uint signs8(uint s7) { return s7 | ((bitCount(s7) & 1u) << 7); }

// ---- weight types (GGML type ids)
#define QT_F32 0
#define QT_F16 1
#define QT_Q8_0 8
#define QT_Q2_K 10
#define QT_Q3_K 11
#define QT_Q4_K 12
#define QT_Q5_K 13
#define QT_Q6_K 14
#define QT_IQ2_XXS 16
#define QT_IQ2_XS 17
#define QT_IQ3_XXS 18
#define QT_IQ1_S 19
#define QT_IQ3_S 21
#define QT_IQ2_S 22
#define QT_IQ4_XS 23
#define QT_IQ1_M 29
#define QT_BF16 30

// bytes per 256-element super block (Q8_0: 8 blocks of 32; float types: 256 elements)
#ifdef QT
#if QT == QT_F32
#define BB 1024u
#elif QT == QT_F16 || QT == QT_BF16
#define BB 512u
#elif QT == QT_Q8_0
#define BB 272u
#elif QT == QT_Q2_K
#define BB 84u
#elif QT == QT_Q3_K
#define BB 110u
#elif QT == QT_Q4_K
#define BB 144u
#elif QT == QT_Q5_K
#define BB 176u
#elif QT == QT_Q6_K
#define BB 210u
#elif QT == QT_IQ2_XXS
#define BB 66u
#elif QT == QT_IQ2_XS
#define BB 74u
#elif QT == QT_IQ2_S
#define BB 82u
#elif QT == QT_IQ3_XXS
#define BB 98u
#elif QT == QT_IQ3_S
#define BB 110u
#elif QT == QT_IQ1_S
#define BB 50u
#elif QT == QT_IQ1_M
#define BB 56u
#elif QT == QT_IQ4_XS
#define BB 136u
#endif

void q4k_scale_min(U32 w, uint sc, uint j, out float d, out float m) {
  if (j < 4u) {
    d = float(ld8(w, sc + j) & 63u);
    m = float(ld8(w, sc + j + 4u) & 63u);
  } else {
    const uint a = ld8(w, sc + j + 4u), lo = ld8(w, sc + j - 4u), hi = ld8(w, sc + j);
    d = float((a & 0xFu) | ((lo >> 6) << 4));
    m = float((a >> 4) | ((hi >> 6) << 4));
  }
}

// dequantize the 32 elements of group g (0..7) of the super block at byte offset sb
void deq32(U32 w, U32 tab, uint sb, uint g, out float v[32]) {
#if QT == QT_F32
  for (uint i = 0u; i < 32u; i++) v[i] = uintBitsToFloat(w.d[(sb >> 2) + g * 32u + i]);
#elif QT == QT_F16
  for (uint i = 0u; i < 16u; i++) {
    const vec2 f = unpackHalf2x16(w.d[(sb >> 2) + g * 16u + i]);
    v[2u * i] = f.x; v[2u * i + 1u] = f.y;
  }
#elif QT == QT_BF16
  for (uint i = 0u; i < 16u; i++) {
    const uint u = w.d[(sb >> 2) + g * 16u + i];
    v[2u * i] = uintBitsToFloat(u << 16); v[2u * i + 1u] = uintBitsToFloat(u & 0xffff0000u);
  }
#elif QT == QT_Q8_0
  const uint b = sb + 34u * g;
  const float d = ldh(w, b);
  for (uint i = 0u; i < 8u; i++) {
    const uint q = ld32(w, b + 2u + 4u * i);
    for (uint j = 0u; j < 4u; j++) v[4u * i + j] = d * float(sbyte_of(q, j));
  }
#elif QT == QT_Q2_K
  const float d = ldh(w, sb + 80u), dmin = ldh(w, sb + 82u);
  const uint q = sb + 16u + (g / 4u) * 32u, shift = 2u * (g % 4u);
  for (uint h = 0u; h < 2u; h++) {
    const uint sc = ld8(w, sb + 2u * g + h);
    const float dl = d * float(sc & 0xFu), ml = dmin * float(sc >> 4);
    for (uint i = 0u; i < 4u; i++) {
      const uint x = ld32(w, q + 16u * h + 4u * i);
      for (uint j = 0u; j < 4u; j++) v[16u * h + 4u * i + j] = dl * float((byte_of(x, j) >> shift) & 3u) - ml;
    }
  }
#elif QT == QT_Q3_K
  const uint a0 = ld32(w, sb + 96u), a1 = ld32(w, sb + 100u), tmp = ld32(w, sb + 104u);
  const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
  const uint wi = g / 2u;
  uint aux;
  if (wi == 0u) aux = (a0 & k2) | (((tmp >> 0) & k1) << 4);
  else if (wi == 1u) aux = (a1 & k2) | (((tmp >> 2) & k1) << 4);
  else if (wi == 2u) aux = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
  else aux = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
  const uint sh = 16u * (g & 1u);
  const float d = ldh(w, sb + 108u);
  const float dl0 = d * float(int((aux >> sh) & 0xffu) - 32), dl1 = d * float(int((aux >> (sh + 8u)) & 0xffu) - 32);
  const uint q = sb + 32u + 32u * (g / 4u), shift = 2u * (g % 4u);
  for (uint i = 0u; i < 8u; i++) {
    const uint x = ld32(w, q + 4u * i), hm = ld32(w, sb + 4u * i);
    const float dl = i < 4u ? dl0 : dl1;
    for (uint j = 0u; j < 4u; j++) {
      const int val = int((byte_of(x, j) >> shift) & 3u) - (((byte_of(hm, j) >> g) & 1u) != 0u ? 0 : 4);
      v[4u * i + j] = dl * float(val);
    }
  }
#elif QT == QT_Q4_K || QT == QT_Q5_K
  float sc, m;
  q4k_scale_min(w, sb + 4u, g, sc, m);
  const float d = ldh(w, sb) * sc, mn = ldh(w, sb + 2u) * m;
#if QT == QT_Q4_K
  const uint q = sb + 16u + (g / 2u) * 32u;
#else
  const uint q = sb + 48u + (g / 2u) * 32u;
#endif
  const uint sh = 4u * (g & 1u);
  for (uint i = 0u; i < 8u; i++) {
    const uint x = (ld32(w, q + 4u * i) >> sh) & 0x0F0F0F0Fu;
#if QT == QT_Q5_K
    const uint hb = (ld32(w, sb + 16u + 4u * i) >> g) & 0x01010101u;
    const uint xx = x | (hb << 4);
#else
    const uint xx = x;
#endif
    for (uint j = 0u; j < 4u; j++) v[4u * i + j] = d * float(byte_of(xx, j)) - mn;
  }
#elif QT == QT_Q6_K
  const float d = ldh(w, sb + 208u);
  const uint h = g / 4u, r = g % 4u;
  const uint ql = sb + 64u * h + (r & 1u) * 32u, qh = sb + 128u + 32u * h;
  const float s0 = d * float(sbyte_of(ld8(w, sb + 192u + 8u * h + 2u * r), 0u));
  const float s1 = d * float(sbyte_of(ld8(w, sb + 192u + 8u * h + 2u * r + 1u), 0u));
  for (uint i = 0u; i < 8u; i++) {
    const uint lo = (ld32(w, ql + 4u * i) >> (r >= 2u ? 4u : 0u)) & 0x0F0F0F0Fu;
    const uint hi = (ld32(w, qh + 4u * i) >> (2u * r)) & 0x03030303u;
    const uint x = lo | (hi << 4);
    for (uint j = 0u; j < 4u; j++) v[4u * i + j] = (i < 4u ? s0 : s1) * float(int(byte_of(x, j)) - 32);
  }
#elif QT == QT_IQ2_XXS
  const uint a0 = ld32(w, sb + 2u + 8u * g), a1 = ld32(w, sb + 6u + 8u * g);
  const float db = ldh(w, sb) * (0.5 + float(a1 >> 28)) * 0.25;
  for (uint l = 0u; l < 4u; l++) {
    const uint e = (a0 >> (8u * l)) & 255u;
    const uvec2 gr = uvec2(tab.d[TAB_IQ2XXS + 2u * e], tab.d[TAB_IQ2XXS + 2u * e + 1u]);
    const uint s = signs8((a1 >> (7u * l)) & 127u);
    for (uint j = 0u; j < 8u; j++)
      v[8u * l + j] = db * float(byte_of(j < 4u ? gr.x : gr.y, j & 3u)) * (((s >> j) & 1u) != 0u ? -1.0 : 1.0);
  }
#elif QT == QT_IQ2_XS || QT == QT_IQ2_S
  const float d = ldh(w, sb);
#if QT == QT_IQ2_XS
  const uint scb = ld8(w, sb + 66u + g);
#else
  const uint scb = ld8(w, sb + 74u + g);
  const uint qh = ld8(w, sb + 66u + g);
#endif
  for (uint l = 0u; l < 4u; l++) {
    const float db = d * (0.5 + float((scb >> (4u * (l / 2u))) & 0xfu)) * 0.25;
#if QT == QT_IQ2_XS
    const uint q = ld16(w, sb + 2u + 2u * (4u * g + l));
    const uint e = q & 511u, s = signs8(q >> 9);
    const uvec2 gr = uvec2(tab.d[TAB_IQ2XS + 2u * e], tab.d[TAB_IQ2XS + 2u * e + 1u]);
#else
    const uint e = ld8(w, sb + 2u + 4u * g + l) | ((qh << (8u - 2u * l)) & 0x300u);
    const uint s = ld8(w, sb + 34u + 4u * g + l);
    const uvec2 gr = uvec2(tab.d[TAB_IQ2S + 2u * e], tab.d[TAB_IQ2S + 2u * e + 1u]);
#endif
    for (uint j = 0u; j < 8u; j++)
      v[8u * l + j] = db * float(byte_of(j < 4u ? gr.x : gr.y, j & 3u)) * (((s >> j) & 1u) != 0u ? -1.0 : 1.0);
  }
#elif QT == QT_IQ3_XXS
  const uint a = ld32(w, sb + 66u + 4u * g);
  const float db = ldh(w, sb) * (0.5 + float(a >> 28)) * 0.5;
  for (uint l = 0u; l < 4u; l++) {
    const uint s = signs8((a >> (7u * l)) & 127u);
    const uint g1 = tab.d[TAB_IQ3XXS + ld8(w, sb + 2u + 8u * g + 2u * l)];
    const uint g2 = tab.d[TAB_IQ3XXS + ld8(w, sb + 3u + 8u * g + 2u * l)];
    for (uint j = 0u; j < 4u; j++) {
      v[8u * l + j] = db * float(byte_of(g1, j)) * (((s >> j) & 1u) != 0u ? -1.0 : 1.0);
      v[8u * l + j + 4u] = db * float(byte_of(g2, j)) * (((s >> (j + 4u)) & 1u) != 0u ? -1.0 : 1.0);
    }
  }
#elif QT == QT_IQ3_S
  const float db = ldh(w, sb) * float(1u + 2u * ((ld8(w, sb + 106u + g / 2u) >> (4u * (g & 1u))) & 0xfu));
  const uint qh = ld8(w, sb + 66u + g);
  for (uint l = 0u; l < 4u; l++) {
    const uint s = ld8(w, sb + 74u + 4u * g + l);
    const uint g1 = tab.d[TAB_IQ3S + (ld8(w, sb + 2u + 8u * g + 2u * l) | ((qh << (8u - 2u * l)) & 256u))];
    const uint g2 = tab.d[TAB_IQ3S + (ld8(w, sb + 3u + 8u * g + 2u * l) | ((qh << (7u - 2u * l)) & 256u))];
    for (uint j = 0u; j < 4u; j++) {
      v[8u * l + j] = db * float(byte_of(g1, j)) * (((s >> j) & 1u) != 0u ? -1.0 : 1.0);
      v[8u * l + j + 4u] = db * float(byte_of(g2, j)) * (((s >> (j + 4u)) & 1u) != 0u ? -1.0 : 1.0);
    }
  }
#elif QT == QT_IQ1_S
  const uint qh = ld16(w, sb + 34u + 2u * g);
  const float dl = ldh(w, sb) * float(2u * ((qh >> 12) & 7u) + 1u);
  const float delta = (qh & 0x8000u) != 0u ? -0.125 : 0.125;
  for (uint l = 0u; l < 4u; l++) {
    const uint e = ld8(w, sb + 2u + 4u * g + l) | (((qh >> (3u * l)) & 7u) << 8);
    const uvec2 gr = uvec2(tab.d[TAB_IQ1S + 2u * e], tab.d[TAB_IQ1S + 2u * e + 1u]);
    for (uint j = 0u; j < 8u; j++) v[8u * l + j] = dl * (float(sbyte_of(j < 4u ? gr.x : gr.y, j & 3u)) + delta);
  }
#elif QT == QT_IQ1_M
  const uint sc0 = ld16(w, sb + 48u), sc1 = ld16(w, sb + 50u), sc2 = ld16(w, sb + 52u), sc3 = ld16(w, sb + 54u);
  const uint u = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u);
  const float d = unpackHalf2x16(u).x;
  const uint scg = ld16(w, sb + 48u + 2u * (g / 2u));
  const float dl1 = d * float(2u * ((scg >> (6u * (g % 2u))) & 7u) + 1u);
  const float dl2 = d * float(2u * ((scg >> (6u * (g % 2u) + 3u)) & 7u) + 1u);
  const uint qs = ld32(w, sb + 4u * g);
  const uint qh0 = ld8(w, sb + 32u + 2u * g), qh1 = ld8(w, sb + 33u + 2u * g);
  for (uint l = 0u; l < 4u; l++) {
    const uint hb = l < 2u ? qh0 : qh1;
    const uint e = byte_of(qs, l) | ((hb << ((l & 1u) != 0u ? 4u : 8u)) & 0x700u);
    const float delta = (hb & ((l & 1u) != 0u ? 0x80u : 0x08u)) != 0u ? -0.125 : 0.125;
    const float dl = l < 2u ? dl1 : dl2;
    const uvec2 gr = uvec2(tab.d[TAB_IQ1S + 2u * e], tab.d[TAB_IQ1S + 2u * e + 1u]);
    for (uint j = 0u; j < 8u; j++) v[8u * l + j] = dl * (float(sbyte_of(j < 4u ? gr.x : gr.y, j & 3u)) + delta);
  }
#elif QT == QT_IQ4_XS
  const uint ls = ((ld8(w, sb + 4u + g / 2u) >> (4u * (g % 2u))) & 0xfu) | (((ld16(w, sb + 2u) >> (2u * g)) & 3u) << 4);
  const float dl = ldh(w, sb) * float(int(ls) - 32);
  for (uint i = 0u; i < 4u; i++) {
    const uint x = ld32(w, sb + 8u + 16u * g + 4u * i);
    for (uint j = 0u; j < 4u; j++) {
      const uint q = byte_of(x, j);
      v[4u * i + j] = dl * float(sbyte_of(tab.d[TAB_IQ4NL + ((q & 0xfu) >> 2)], q & 3u));
      v[4u * i + j + 16u] = dl * float(sbyte_of(tab.d[TAB_IQ4NL + ((q >> 4) >> 2)], (q >> 4) & 3u));
    }
  }
#endif
}
#endif  // QT
