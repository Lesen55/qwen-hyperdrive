#pragma once
#include <cstddef>
#include <cstdint>

size_t iq4_xs_row_bytes(int K);
// dequantize N rows of src_type and requantize them to IQ4_XS (multi-threaded)
void requant_rows_iq4_xs(uint32_t src_type, const uint8_t* src, size_t src_row_bytes, int K, int N, uint8_t* dst);
