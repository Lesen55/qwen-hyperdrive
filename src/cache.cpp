#include "cache.h"

#include <atomic>
#include <cstdlib>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

std::filesystem::path cache_directory() {
#ifdef _WIN32
  if (const wchar_t* xdg = _wgetenv(L"XDG_CACHE_HOME"); xdg && *xdg)
    return std::filesystem::path(xdg) / "qwen-hyperdrive";
  const wchar_t* home = _wgetenv(L"HOME");
  if (!home || !*home) home = _wgetenv(L"USERPROFILE");
#else
  if (const char* xdg = getenv("XDG_CACHE_HOME"); xdg && *xdg)
    return std::filesystem::path(xdg) / "qwen-hyperdrive";
  const char* home = getenv("HOME");
#endif
  if (!home || !*home) return {};
  return std::filesystem::path(home) / ".cache" / "qwen-hyperdrive";
}

FILE* open_cache_file(const std::filesystem::path& path) {
  if (path.empty()) return nullptr;
#ifdef _WIN32
  return _wfopen(path.c_str(), L"rb");
#else
  return fopen(path.c_str(), "rb");
#endif
}

void write_cache_file(const std::filesystem::path& path, const void* data, size_t size) {
  if (path.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) return;
  static std::atomic<unsigned> serial{0};
  auto tmp = path;
#ifdef _WIN32
  const auto pid = _getpid();
#else
  const auto pid = getpid();
#endif
  tmp += ".tmp-" + std::to_string(pid) + "-" + std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
#ifdef _WIN32
  FILE* f = _wfopen(tmp.c_str(), L"wb");
#else
  FILE* f = fopen(tmp.c_str(), "wb");
#endif
  if (!f) return;
  const bool written = fwrite(data, 1, size, f) == size;
  const bool closed = fclose(f) == 0;
  if (written && closed) {
    std::filesystem::rename(tmp, path, ec);
    if (!ec) return;
  }
  std::filesystem::remove(tmp, ec);
}
