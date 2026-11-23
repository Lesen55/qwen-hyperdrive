#pragma once

#include <cstddef>
#include <cstdio>
#include <filesystem>

std::filesystem::path cache_directory();
FILE* open_cache_file(const std::filesystem::path& path);
void write_cache_file(const std::filesystem::path& path, const void* data, size_t size);
