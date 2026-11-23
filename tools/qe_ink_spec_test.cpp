// Speculative verify / rollback check: logits of a verify pass and after rollback against one-token steps.
//   qe_ink_spec_test model.gguf ids_file
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "backend.h"
#include "common.h"

static int argmax(const float* x, int n) {
  int b = 0;
  for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
  return b;
}
static double maxdiff(const float* a, const float* b, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: qe_ink_spec_test model.gguf ids\n"); return 1; }
  EngineOptions o;
  o.model_path = argv[1];
  o.ctx = 4096;
  o.kv_type = 1;
  std::vector<int> ids;
  std::ifstream in(argv[2]);
  for (int x; in >> x;) ids.push_back(x);
  auto m = create_model(o);
  const int V = m->cfg().n_vocab;
  const int P = (int)ids.size() - 6;
  const int d[4] = {ids[P], ids[P + 1], ids[P + 2], ids[P + 3]};
  const int alt = ids[P + 5];
  std::vector<float> seq(4 * (size_t)V), ver(4 * (size_t)V), a1(V), b1(V);
  // A: one token at a time
  m->reset();
  m->forward(ids.data(), P, 0, nullptr, 1);
  for (int i = 0; i < 4; i++) m->forward(&d[i], 1, P + i, seq.data() + (size_t)i * V, 1);
  // C: two drafts kept, then a different third token
  m->reset();
  m->forward(ids.data(), P, 0, nullptr, 1);
  m->forward(&d[0], 1, P, nullptr, 1);
  m->forward(&d[1], 1, P + 1, nullptr, 1);
  m->forward(&alt, 1, P + 2, a1.data(), 1);
  // B: verify pass of 4, rollback to 2, then the different third token
  m->reset();
  m->forward(ids.data(), P, 0, nullptr, 1);
  m->forward(d, 4, P, ver.data(), 4, nullptr, true);
  m->rollback(2);
  m->forward(&alt, 1, P + 2, b1.data(), 1);
  for (int i = 0; i < 4; i++)
    printf("verify row %d: max |dlogit| %.4f, argmax %d vs %d\n", i,
           maxdiff(seq.data() + (size_t)i * V, ver.data() + (size_t)i * V, V), argmax(seq.data() + (size_t)i * V, V),
           argmax(ver.data() + (size_t)i * V, V));
  printf("after rollback(2): max |dlogit| %.4f, argmax %d vs %d\n", maxdiff(a1.data(), b1.data(), V), argmax(a1.data(), V),
         argmax(b1.data(), V));
  return 0;
}
