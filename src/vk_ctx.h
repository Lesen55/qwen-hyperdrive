// Minimal Vulkan compute runtime: one device + queue, buffers addressed by device address (shaders
// get raw pointers through push constants, so there are no descriptor sets), pipelines built from
// the SPIR-V embedded at build time, and a single command buffer recorded and submitted per pass.
#pragma once
#include <vulkan/vulkan.h>

#include <map>
#include <string>
#include <vector>

enum class Mem { Device, Host, Readback };  // Host: system RAM the GPU reads (PCIe); Readback: cached

struct VBuf {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceAddress addr = 0;
  size_t size = 0;
  void* map = nullptr;  // host-visible buffers stay mapped
  bool device_local = false;
  VkDeviceAddress at(size_t off) const { return addr + off; }
};

class VkCtx {
 public:
  VkCtx();
  ~VkCtx();
  VkCtx(const VkCtx&) = delete;
  VkCtx& operator=(const VkCtx&) = delete;

  // Device allocation falls back to host memory when VRAM is exhausted (unless strict).
  VBuf alloc(size_t bytes, Mem kind, bool strict = false);
  void free(VBuf& b);
  size_t vram_free() const;   // VK_EXT_memory_budget if available, else heap size - our allocations
  size_t vram_total() const { return vram_total_; }
  size_t vram_used() const { return vram_used_; }

  void begin();
  // dispatch + a full compute/transfer memory barrier after it (barrier=false: the next dispatch is
  // independent, e.g. the q/k/v projections reading the same input)
  void dispatch(const char* pipe, const void* pc, uint32_t pc_size, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
                bool barrier = true);
  void barrier();
  void copy(const VBuf& src, size_t src_off, const VBuf& dst, size_t dst_off, size_t n);
  void fill(const VBuf& dst, size_t off, size_t n, uint32_t value = 0);
  void submit();  // end, submit, wait
  bool recording() const { return recording_; }
  bool has_pipeline(const char* name) const;
  void warm_pipelines();  // compile every embedded shader now (first request would pay for it)

  std::string device_name;
  std::string driver_name;
  uint32_t subgroup_size = 0;
  bool integrated = false;
  bool coopmat = false;  // VK_KHR_cooperative_matrix with 16x16x16 f16 x f16 -> f32 (subgroup scope)

 private:
  VkPipeline pipeline(const char* name);
  int mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) const;

  VkInstance inst_ = VK_NULL_HANDLE;
  VkPhysicalDevice pdev_ = VK_NULL_HANDLE;
  VkDevice dev_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t qfam_ = 0;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cb_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipelineCache cache_ = VK_NULL_HANDLE;
  std::string cache_path_;  // on-disk pipeline cache (per device + driver version)
  std::map<std::string, VkPipeline> pipes_;
  VkPhysicalDeviceMemoryProperties mprops_{};
  bool budget_ext_ = false;
  int vram_heap_ = -1;
  size_t vram_total_ = 0, vram_used_ = 0;
  bool recording_ = false;
  // QE_VK_PROF: GPU time per pipeline (timestamps around every dispatch)
  VkQueryPool qpool_ = VK_NULL_HANDLE;
  float ts_period_ = 1.f;
  std::vector<std::string> q_names_;
  std::map<std::string, std::pair<double, long>> prof_;
  int prof_submits_ = 0;
  void prof_collect();
 public:
  void prof_report();
};
