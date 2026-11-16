// qe_dump <model.gguf> <tensor> <nrows> : write the first nrows dequantized rows (f32) to stdout.
// Used to validate our dequantization against an independent implementation (gguf-py).
#include "gguf.h"
#include "quant.h"

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: qe_dump model tensor nrows\n"); return 1; }
  GGUFFile f(argv[1]);
  const auto& t = f.need(argv[2]);
  int64_t nr = std::min<int64_t>(atoll(argv[3]), t.rows());
  size_t rb = qtype_row_bytes(t.type, t.ne[0]);
  std::vector<float> y(t.ne[0]);
  for (int64_t r = 0; r < nr; r++) {
    dequant_row_ref(t.type, t.data + r * rb, t.ne[0], y.data());
    fwrite(y.data(), 4, y.size(), stdout);
  }
  return 0;
}
