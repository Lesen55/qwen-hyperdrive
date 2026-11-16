// qe_tokenize model.gguf file  -> prints token ids (one line, space separated); --decode ids_file
#include <fstream>
#include <sstream>

#include "gguf.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: qe_tokenize model.gguf textfile [--decode]\n"); return 1; }
  GGUFFile f(argv[1]);
  Tokenizer tok(f);
  std::ifstream in(argv[2], std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(in)), {});
  if (argc > 3 && std::string(argv[3]) == "--decode") {
    std::istringstream is(s);
    std::vector<int> ids;
    int x;
    while (is >> x) ids.push_back(x);
    fwrite(tok.decode(ids).data(), 1, tok.decode(ids).size(), stdout);
    return 0;
  }
  const double t0 = qe_now();
  auto ids = tok.encode(s);
  fprintf(stderr, "%zu tokens in %.3f s\n", ids.size(), qe_now() - t0);
  for (size_t i = 0; i < ids.size(); i++) printf(i ? " %d" : "%d", ids[i]);
  printf("\n");
}
