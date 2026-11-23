// Minimal Vulkan compute runtime: one device + queue, buffers addressed by device address (shaders
// get raw pointers through push constants, so there are no descriptor sets), pipelines built from
// the SPIR-V embedded at build time, and a single command buffer recorded and submitted per pass.
#pragma once
#include <vulkan/vulkan.h>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

// Host: system RAM the GPU reads (PCIe); Readback: cached; DeviceMapped: device-local and CPU-writable
// (resizable BAR on discrete GPUs, all memory on integrated ones) -> uploads are a plain memcpy
enum class Mem { Device, Host, Readback, DeviceMapped, SharedWeights };

// per-pipeline options: specialization constants 0 / 1 (-1 = shader default) and a required subgroup
// size (0 = driver's choice; ignored without VK_EXT_subgroup_size_control)
struct PipeOpt {
  int spec0 = -1, spec1 = -1, spec2 = -1, spec3 = -1;
  uint32_t subgroup = 0;
  bool operator<(const PipeOpt& o) const {
    if (spec0 != o.spec0) return spec0 < o.spec0;
    if (spec1 != o.spec1) return spec1 < o.spec1;
    if (spec2 != o.spec2) return spec2 < o.spec2;
    if (spec3 != o.spec3) return spec3 < o.spec3;
    return subgroup < o.subgroup;
  }
};
struct CmShape { uint32_t M, N, K; };

struct VBuf {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceAddress addr = 0;
  size_t size = 0;
  void* map = nullptr;  // host-visible buffers stay mapped
  bool device_local = false;
  bool imported = false;
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
                bool barrier = true, const PipeOpt& opt = PipeOpt{});
  void barrier();
  void copy(const VBuf& src, size_t src_off, const VBuf& dst, size_t dst_off, size_t n);
  void copy_regions(const VBuf& src, const VBuf& dst, const std::vector<VkBufferCopy>& r);  // one command
  void fill(const VBuf& dst, size_t off, size_t n, uint32_t value = 0);
  void submit();  // end, submit, wait
  // CPU / GPU co-execution inside one submission (unified memory): the GPU writes `value` to a
  // host-visible word when it reaches signal_host(); the CPU sets an event the GPU waits for
  void submit_nowait();           // end + submit; finish() waits
  void finish();
  VkEvent make_event();
  void wait_event(VkEvent e);     // recorded: the GPU waits until the host sets e
  void set_event(VkEvent e);      // host
  void reset_event(VkEvent e);    // host (event must not be waited on by pending work)
  void signal_host(const VBuf& flags, size_t off, uint32_t value);  // recorded: flags[off] = value
  bool pending() const { return pending_; }
  bool recording() const { return recording_; }
  bool has_pipeline(const char* name) const;
  void warm_pipelines();  // compile every embedded shader now (first request would pay for it)

  std::string device_name;
  std::string driver_name;
  uint32_t subgroup_size = 0;
  bool integrated = false;
  bool coopmat = false;           // VK_KHR_cooperative_matrix, f16 x f16 -> f32 at subgroup scope
  std::vector<CmShape> cm_shapes;  // ... in these M x N x K shapes
  bool sg_control = false;        // VK_EXT_subgroup_size_control for compute
  uint32_t sg_min = 0, sg_max = 0;
  uint32_t vendor_id = 0, device_id = 0, driver_version = 0;
  uint32_t compute_units = 0;     // CUs (AMD) / SMs (NVIDIA) when the driver tells, else 0
  uint32_t max_shared = 0, max_wg_invocations = 0;
  bool int_dot = false;           // VK_KHR_shader_integer_dot_product (packed 4 x 8-bit dot, DP4A)
  bool host_import = false;       // VK_EXT_external_memory_host: GPU buffers over existing host memory
  size_t import_align = 4096;
  size_t max_buffer = 0;          // largest single buffer (VkPhysicalDeviceMaintenance4: maxBufferSize)
  // wrap [p, p + n) (p and n multiples of import_align) as a GPU buffer; empty VBuf on failure
  VBuf import_host(const void* p, size_t n);
  uint64_t ops() const { return ops_; }  // dispatches / copies recorded so far
  bool rebar = false;             // a CPU-mappable device-local type spans the VRAM heap
  bool has_cm(uint32_t M, uint32_t N, uint32_t K) const;

 private:
  VkPipeline pipeline(const char* name, const PipeOpt& opt = PipeOpt{});
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
  std::filesystem::path cache_path_;  // on-disk pipeline cache (per device + driver version)
  std::map<std::pair<std::string, PipeOpt>, VkPipeline> pipes_;
  VkPhysicalDeviceMemoryProperties mprops_{};
  bool budget_ext_ = false;
  int vram_heap_ = -1;
  size_t vram_total_ = 0, vram_used_ = 0;
  bool recording_ = false;
  bool pending_ = false;
  uint64_t ops_ = 0;
  std::vector<VkEvent> events_;
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
