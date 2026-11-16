#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>

int g_verbose = 0;

double qe_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const QTypeInfo kTypes[] = {
    /*0*/ {"F32", 1, 4},       /*1*/ {"F16", 1, 2},     {"?", 0, 0},          {"?", 0, 0},
    {"?", 0, 0},               {"?", 0, 0},             {"?", 0, 0},          {"?", 0, 0},
    /*8*/ {"Q8_0", 32, 34},    {"?", 0, 0},             /*10*/ {"Q2_K", 256, 84},
    /*11*/ {"Q3_K", 256, 110}, /*12*/ {"Q4_K", 256, 144}, /*13*/ {"Q5_K", 256, 176},
    /*14*/ {"Q6_K", 256, 210}, /*15*/ {"Q8_K", 256, 292}, /*16*/ {"IQ2_XXS", 256, 66},
    /*17*/ {"IQ2_XS", 256, 74}, /*18*/ {"IQ3_XXS", 256, 98}, /*19*/ {"IQ1_S", 256, 50},
    /*20*/ {"IQ4_NL", 32, 18}, /*21*/ {"IQ3_S", 256, 110}, /*22*/ {"IQ2_S", 256, 82},
    /*23*/ {"IQ4_XS", 256, 136}, {"?", 0, 0},           {"?", 0, 0},          {"?", 0, 0},
    {"?", 0, 0},               {"?", 0, 0},             /*29*/ {"IQ1_M", 256, 56},
    /*30*/ {"BF16", 1, 2},
};

const QTypeInfo& qtype_info(uint32_t t) {
  if (t >= sizeof(kTypes) / sizeof(kTypes[0]) || kTypes[t].block_elems == 0)
    QE_FATAL("unknown ggml type %u", t);
  return kTypes[t];
}

bool qtype_supported(uint32_t t) {
  switch (t) {
    case QT_F32: case QT_F16: case QT_BF16: case QT_Q2_K: case QT_Q4_K: case QT_Q6_K:
    case QT_IQ2_XXS: case QT_IQ2_XS: case QT_IQ2_S: case QT_IQ3_XXS: case QT_IQ3_S:
    case QT_IQ1_S: case QT_IQ1_M: case QT_IQ4_XS:
      return true;
    default:
      return false;
  }
}

int64_t GGUFValue::as_int() const {
  if (auto p = std::get_if<uint64_t>(&scalar)) return (int64_t)*p;
  if (auto p = std::get_if<int64_t>(&scalar)) return *p;
  if (auto p = std::get_if<bool>(&scalar)) return *p;
  if (auto p = std::get_if<double>(&scalar)) return (int64_t)*p;
  QE_FATAL("gguf value is not an int");
}
double GGUFValue::as_float() const {
  if (auto p = std::get_if<double>(&scalar)) return *p;
  return (double)as_int();
}
const std::string& GGUFValue::as_str() const {
  if (auto p = std::get_if<std::string>(&scalar)) return *p;
  QE_FATAL("gguf value is not a string");
}

namespace {
struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  template <class T>
  T rd() {
    if (p + sizeof(T) > end) QE_FATAL("gguf: truncated file");
    T v;
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string str() {
    uint64_t n = rd<uint64_t>();
    if (p + n > end) QE_FATAL("gguf: truncated string");
    std::string s((const char*)p, n);
    p += n;
    return s;
  }
  GGUFValue value(uint32_t t) {
    GGUFValue v;
    v.type = t;
    switch (t) {
      case 0: v.scalar = (uint64_t)rd<uint8_t>(); break;
      case 1: v.scalar = (int64_t)rd<int8_t>(); break;
      case 2: v.scalar = (uint64_t)rd<uint16_t>(); break;
      case 3: v.scalar = (int64_t)rd<int16_t>(); break;
      case 4: v.scalar = (uint64_t)rd<uint32_t>(); break;
      case 5: v.scalar = (int64_t)rd<int32_t>(); break;
      case 6: v.scalar = (double)rd<float>(); break;
      case 7: v.scalar = (bool)rd<uint8_t>(); break;
      case 8: v.scalar = str(); break;
      case 9: {
        v.arr_type = rd<uint32_t>();
        uint64_t n = rd<uint64_t>();
        v.arr.reserve(n);
        for (uint64_t i = 0; i < n; i++) v.arr.push_back(value(v.arr_type));
        break;
      }
      case 10: v.scalar = rd<uint64_t>(); break;
      case 11: v.scalar = rd<int64_t>(); break;
      case 12: v.scalar = rd<double>(); break;
      default: QE_FATAL("gguf: bad value type %u", t);
    }
    return v;
  }
};
}  // namespace

GGUFFile::GGUFFile(const std::string& path) : path_(path) {
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0) QE_FATAL("cannot open %s", path.c_str());
  struct stat st;
  fstat(fd_, &st);
  size_ = st.st_size;
  base_ = (uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (base_ == MAP_FAILED) QE_FATAL("mmap failed for %s", path.c_str());
  Reader r{base_, base_ + size_};
  if (r.rd<uint32_t>() != 0x46554747u) QE_FATAL("%s is not a GGUF file", path.c_str());
  uint32_t ver = r.rd<uint32_t>();
  if (ver != 3) QE_FATAL("unsupported GGUF version %u", ver);
  uint64_t nt = r.rd<uint64_t>(), nkv = r.rd<uint64_t>();
  for (uint64_t i = 0; i < nkv; i++) {
    std::string k = r.str();
    uint32_t t = r.rd<uint32_t>();
    kv_[k] = r.value(t);
  }
  tensors_.resize(nt);
  for (uint64_t i = 0; i < nt; i++) {
    auto& t = tensors_[i];
    t.name = r.str();
    uint32_t nd = r.rd<uint32_t>();
    t.ne.resize(nd);
    for (auto& x : t.ne) x = (int64_t)r.rd<uint64_t>();
    t.type = r.rd<uint32_t>();
    t.offset = r.rd<uint64_t>();
    tindex_[t.name] = i;
  }
  uint64_t align = (uint64_t)get_int("general.alignment", 32);
  uint64_t off = (uint64_t)(r.p - base_);
  off = (off + align - 1) / align * align;
  for (auto& t : tensors_) {
    const auto& ti = qtype_info(t.type);
    t.nbytes = (size_t)(t.nelems() / ti.block_elems) * ti.block_bytes;
    t.data = base_ + off + t.offset;
    if (t.data + t.nbytes > base_ + size_) QE_FATAL("tensor %s out of file bounds", t.name.c_str());
  }
}

GGUFFile::~GGUFFile() {
  if (base_ && base_ != MAP_FAILED) munmap(base_, size_);
  if (fd_ >= 0) close(fd_);
}

const GGUFValue* GGUFFile::get(const std::string& key) const {
  auto it = kv_.find(key);
  return it == kv_.end() ? nullptr : &it->second;
}
int64_t GGUFFile::get_int(const std::string& key, int64_t def) const {
  auto v = get(key);
  return v ? v->as_int() : def;
}
double GGUFFile::get_float(const std::string& key, double def) const {
  auto v = get(key);
  return v ? v->as_float() : def;
}
std::string GGUFFile::get_str(const std::string& key, const std::string& def) const {
  auto v = get(key);
  return v ? v->as_str() : def;
}
const GGUFTensor* GGUFFile::tensor(const std::string& name) const {
  auto it = tindex_.find(name);
  return it == tindex_.end() ? nullptr : &tensors_[it->second];
}
const GGUFTensor& GGUFFile::need(const std::string& name) const {
  auto t = tensor(name);
  if (!t) QE_FATAL("missing tensor %s", name.c_str());
  return *t;
}
