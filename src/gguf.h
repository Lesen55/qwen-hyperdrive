// Minimal zero-copy GGUF v3 reader (mmap).
#pragma once
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "common.h"

struct GGUFTensor {
  std::string name;
  std::vector<int64_t> ne;  // ne[0] = innermost (row length)
  uint32_t type = 0;
  uint64_t offset = 0;      // relative to data section
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  int64_t nelems() const {
    int64_t n = 1;
    for (auto x : ne) n *= x;
    return n;
  }
  int64_t rows() const { return nelems() / ne[0]; }
};

struct GGUFValue {
  uint32_t type = 0;
  std::variant<uint64_t, int64_t, double, bool, std::string> scalar;
  uint32_t arr_type = 0;
  std::vector<GGUFValue> arr;
  bool is_array() const { return type == 9; }
  int64_t as_int() const;
  double as_float() const;
  const std::string& as_str() const;
};

class GGUFFile {
 public:
  explicit GGUFFile(const std::string& path);
  ~GGUFFile();
  GGUFFile(const GGUFFile&) = delete;
  GGUFFile& operator=(const GGUFFile&) = delete;

  const GGUFValue* get(const std::string& key) const;
  int64_t get_int(const std::string& key, int64_t def) const;
  double get_float(const std::string& key, double def) const;
  std::string get_str(const std::string& key, const std::string& def) const;
  const GGUFTensor* tensor(const std::string& name) const;
  const GGUFTensor& need(const std::string& name) const;

  const std::vector<GGUFTensor>& tensors() const { return tensors_; }
  const uint8_t* base() const { return base_; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
#ifdef _WIN32
  void* win_map_ = nullptr;  // QeFileMapping* (src/win_compat.h), opaque to keep windows.h out of this header
#endif
  uint8_t* base_ = nullptr;
  size_t size_ = 0;
  std::map<std::string, GGUFValue> kv_;
  std::vector<GGUFTensor> tensors_;
  std::map<std::string, size_t> tindex_;
};
