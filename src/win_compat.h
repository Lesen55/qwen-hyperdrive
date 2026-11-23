// Windows compatibility shims for the POSIX facilities the engine uses.
// Currently: read-only whole-file mapping (the mmap(PROT_READ, MAP_SHARED) equivalent
// used for GGUF model files). Everything here is _WIN32-only; other platforms never see it.
#pragma once
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstddef>
#include <cstdint>

// A read-only whole-file mapping. Owns the file + mapping handles until qe_unmap_file.
struct QeFileMapping {
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
  uint8_t* base = nullptr;
  size_t size = 0;
  bool ok() const { return base != nullptr; }
};

// Maps `path` read-only; returns false on failure (handles cleaned up).
inline bool qe_map_file_readonly(const char* path, QeFileMapping& out) {
  out.file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (out.file == INVALID_HANDLE_VALUE) return false;
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(out.file, &sz) || sz.QuadPart <= 0) {
    CloseHandle(out.file);
    out.file = INVALID_HANDLE_VALUE;
    return false;
  }
  out.size = (size_t)sz.QuadPart;
  out.mapping = CreateFileMappingA(out.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!out.mapping) {
    CloseHandle(out.file);
    out.file = INVALID_HANDLE_VALUE;
    return false;
  }
  out.base = (uint8_t*)MapViewOfFile(out.mapping, FILE_MAP_READ, 0, 0, 0);
  if (!out.base) {
    CloseHandle(out.mapping);
    CloseHandle(out.file);
    out.mapping = nullptr;
    out.file = INVALID_HANDLE_VALUE;
    return false;
  }
  return true;
}

inline void qe_unmap_file(QeFileMapping& m) {
  if (m.base) UnmapViewOfFile(m.base);
  if (m.mapping) CloseHandle(m.mapping);
  if (m.file != INVALID_HANDLE_VALUE) CloseHandle(m.file);
  m = QeFileMapping{};
}

#endif  // _WIN32
