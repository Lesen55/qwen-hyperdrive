#include "vk_ctx.h"

#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <cstring>

#include "common.h"
#include "spv_all.h"

#define VK_CHECK(x)                                                                      \
  do {                                                                                   \
    VkResult r_ = (x);                                                                   \
    if (r_ != VK_SUCCESS) QE_FATAL("Vulkan error %d at %s", (int)r_, #x);                 \
  } while (0)

static constexpr uint32_t kPushBytes = 128;  // guaranteed minimum on every implementation

VkCtx::VkCtx() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "qwen-hyperdrive";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (vkCreateInstance(&ici, nullptr, &inst_) != VK_SUCCESS)
    QE_FATAL("no Vulkan 1.2 driver (vkCreateInstance failed; is a Vulkan ICD installed?)");

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst_, &n, nullptr);
  if (!n) QE_FATAL("no Vulkan devices");
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(inst_, &n, devs.data());
  auto rank = [](VkPhysicalDeviceType t) {
    switch (t) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
      case VK_PHYSICAL_DEVICE_TYPE_CPU: return 4;
      default: return 3;
    }
  };
  int pick = -1;
  for (uint32_t i = 0; i < n; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    QE_DBG("[vk] device %u: %s (type %d, api %u.%u)\n", i, p.deviceName, (int)p.deviceType,
           VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion));
  }
  if (const char* e = getenv("QE_VK_DEVICE")) {
    pick = atoi(e);
    if (pick < 0 || pick >= (int)n) QE_FATAL("QE_VK_DEVICE=%d out of range (%u devices)", pick, n);
  } else {
    int best = 99;
    for (uint32_t i = 0; i < n; i++) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(devs[i], &p);
      if (p.apiVersion < VK_API_VERSION_1_2) continue;
      if (rank(p.deviceType) < best) { best = rank(p.deviceType); pick = (int)i; }
    }
    if (pick < 0) QE_FATAL("no Vulkan 1.2 capable device");
  }
  pdev_ = devs[pick];

  VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  sg.pNext = &drv;
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &sg;
  vkGetPhysicalDeviceProperties2(pdev_, &p2);
  device_name = p2.properties.deviceName;
  driver_name = drv.driverName;
  integrated = p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  subgroup_size = sg.subgroupSize;
  const VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                      VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
  if ((sg.supportedOperations & need) != need || !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || sg.subgroupSize < 8)
    QE_FATAL("%s: needs compute subgroups of >= 8 lanes with arithmetic + clustered operations", device_name.c_str());
  if (p2.properties.limits.maxPushConstantsSize < kPushBytes) QE_FATAL("push constants < 128 bytes");
  if (p2.properties.limits.maxComputeSharedMemorySize < 32 * 1024)
    QE_FATAL("%s: needs 32 KiB of compute shared memory", device_name.c_str());

  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &f12;
  vkGetPhysicalDeviceFeatures2(pdev_, &f2);
  if (!f12.bufferDeviceAddress) QE_FATAL("%s: bufferDeviceAddress not supported", device_name.c_str());

  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pdev_, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq);
  vkGetPhysicalDeviceQueueFamilyProperties(pdev_, &nq, qf.data());
  int fam = -1;
  for (uint32_t i = 0; i < nq; i++)  // prefer a compute queue of the graphics family (always present)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { fam = (int)i; break; }
  if (fam < 0) QE_FATAL("no compute queue");
  qfam_ = (uint32_t)fam;

  uint32_t ne = 0;
  vkEnumerateDeviceExtensionProperties(pdev_, nullptr, &ne, nullptr);
  std::vector<VkExtensionProperties> exts(ne);
  vkEnumerateDeviceExtensionProperties(pdev_, nullptr, &ne, exts.data());
  std::vector<const char*> en;
  bool cm_ext = false;
  for (auto& e : exts) {
    if (!strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) { budget_ext_ = true; en.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME); }
    if (!strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) cm_ext = true;
  }
  // cooperative matrices: extension + features + the exact 16x16x16 f16/f32 configuration
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  {
    VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q.pNext = &f11;
    if (cm_ext) f11.pNext = &cmf;
    vkGetPhysicalDeviceFeatures2(pdev_, &q);
  }
  if (cm_ext && cmf.cooperativeMatrix && f12.shaderFloat16 && f12.vulkanMemoryModel && f11.storageBuffer16BitAccess &&
      !getenv("QE_VK_NO_COOPMAT")) {
    auto fn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
        inst_, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t nc = 0;
    if (fn && fn(pdev_, &nc, nullptr) == VK_SUCCESS && nc) {
      std::vector<VkCooperativeMatrixPropertiesKHR> cp(nc, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
      fn(pdev_, &nc, cp.data());
      for (auto& c : cp)
        if (c.MSize == 16 && c.NSize == 16 && c.KSize == 16 && c.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            c.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            c.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && c.scope == VK_SCOPE_SUBGROUP_KHR)
          coopmat = true;
    }
  }
  if (coopmat) en.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

  VkPhysicalDeviceVulkan12Features on12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  on12.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceVulkan11Features on11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR oncm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceFeatures2 on{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  on.pNext = &on12;
  if (coopmat) {
    on12.shaderFloat16 = VK_TRUE;
    on12.vulkanMemoryModel = VK_TRUE;
    on11.storageBuffer16BitAccess = VK_TRUE;
    oncm.cooperativeMatrix = VK_TRUE;
    on12.pNext = &on11;
    on11.pNext = &oncm;
  }
  const float prio = 1.f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfam_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &on;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)en.size();
  dci.ppEnabledExtensionNames = en.data();
  VK_CHECK(vkCreateDevice(pdev_, &dci, nullptr, &dev_));
  vkGetDeviceQueue(dev_, qfam_, 0, &queue_);

  vkGetPhysicalDeviceMemoryProperties(pdev_, &mprops_);
  for (uint32_t i = 0; i < mprops_.memoryHeapCount; i++)
    if ((mprops_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mprops_.memoryHeaps[i].size > vram_total_) {
      vram_total_ = mprops_.memoryHeaps[i].size;
      vram_heap_ = (int)i;
    }

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = qfam_;
  VK_CHECK(vkCreateCommandPool(dev_, &pci, nullptr, &pool_));
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool_;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(dev_, &cai, &cb_));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &fence_));
  VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
  VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  lci.pushConstantRangeCount = 1;
  lci.pPushConstantRanges = &pr;
  VK_CHECK(vkCreatePipelineLayout(dev_, &lci, nullptr, &layout_));
  if (getenv("QE_VK_PROF")) {
    ts_period_ = p2.properties.limits.timestampPeriod;
    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qp.queryCount = 8192;
    VK_CHECK(vkCreateQueryPool(dev_, &qp, nullptr, &qpool_));
  }
  // pipeline cache on disk: ~/.cache/qwen-hyperdrive/vk-<vendor>-<device>-<driver>.bin
  std::vector<char> blob;
  if (const char* home = getenv("HOME")) {
    char nm[128];
    snprintf(nm, sizeof nm, "vk-%x-%x-%x.bin", p2.properties.vendorID, p2.properties.deviceID, p2.properties.driverVersion);
    cache_path_ = std::string(home) + "/.cache/qwen-hyperdrive/" + nm;
    if (FILE* f = fopen(cache_path_.c_str(), "rb")) {
      fseek(f, 0, SEEK_END);
      blob.resize((size_t)std::max(0L, ftell(f)));
      fseek(f, 0, SEEK_SET);
      if (fread(blob.data(), 1, blob.size(), f) != blob.size()) blob.clear();
      fclose(f);
    }
  }
  VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  cci.initialDataSize = blob.size();
  cci.pInitialData = blob.empty() ? nullptr : blob.data();
  if (vkCreatePipelineCache(dev_, &cci, nullptr, &cache_) != VK_SUCCESS) {  // stale / foreign blob
    cci.initialDataSize = 0;
    cci.pInitialData = nullptr;
    VK_CHECK(vkCreatePipelineCache(dev_, &cci, nullptr, &cache_));
  }
}

VkCtx::~VkCtx() {
  if (!dev_) return;
  vkDeviceWaitIdle(dev_);
  if (qpool_) { prof_report(); vkDestroyQueryPool(dev_, qpool_, nullptr); }
  for (auto& [k, p] : pipes_) vkDestroyPipeline(dev_, p, nullptr);
  vkDestroyPipelineCache(dev_, cache_, nullptr);
  vkDestroyPipelineLayout(dev_, layout_, nullptr);
  vkDestroyFence(dev_, fence_, nullptr);
  vkDestroyCommandPool(dev_, pool_, nullptr);
  vkDestroyDevice(dev_, nullptr);
  vkDestroyInstance(inst_, nullptr);
}

int VkCtx::mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) const {
  for (uint32_t i = 0; i < mprops_.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mprops_.memoryTypes[i].propertyFlags & want) == want &&
        !(mprops_.memoryTypes[i].propertyFlags & avoid))
      return (int)i;
  return -1;
}

size_t VkCtx::vram_free() const {
  if (budget_ext_ && vram_heap_ >= 0) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 m2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    m2.pNext = &b;
    vkGetPhysicalDeviceMemoryProperties2(pdev_, &m2);
    const size_t bud = b.heapBudget[vram_heap_], use = b.heapUsage[vram_heap_];
    return bud > use ? bud - use : 0;
  }
  const size_t cap = vram_total_ / 10 * 9;
  return cap > vram_used_ ? cap - vram_used_ : 0;
}

VBuf VkCtx::alloc(size_t bytes, Mem kind, bool strict) {
  VBuf b;
  b.size = bytes;
  bytes = (bytes + 255) / 256 * 256;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = bytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &b.buf));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev_, b.buf, &req);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &fi;
  ai.allocationSize = req.size;
  const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  auto try_alloc = [&](int t) {
    if (t < 0) return false;
    ai.memoryTypeIndex = (uint32_t)t;
    return vkAllocateMemory(dev_, &ai, nullptr, &b.mem) == VK_SUCCESS;
  };
  bool ok = false;
  if (kind == Mem::Device) {
    const int t = mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    ok = (integrated || req.size <= vram_free() + (64u << 20)) && try_alloc(t);
    if (ok) b.device_local = true;
    else if (strict) QE_FATAL("out of device memory allocating %.1f MiB", req.size / 1048576.0);
    else kind = Mem::Host;  // weights that do not fit: read by the GPU from system RAM
  }
  if (!ok && kind == Mem::Readback) {
    ok = try_alloc(mem_type(req.memoryTypeBits, hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)) ||
         try_alloc(mem_type(req.memoryTypeBits, hv, 0));
  }
  if (!ok && kind == Mem::Host) {
    ok = try_alloc(mem_type(req.memoryTypeBits, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ||
         try_alloc(mem_type(req.memoryTypeBits, hv, 0));
  }
  if (!ok) QE_FATAL("Vulkan allocation of %.1f MiB failed", req.size / 1048576.0);
  VK_CHECK(vkBindBufferMemory(dev_, b.buf, b.mem, 0));
  VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  dai.buffer = b.buf;
  b.addr = vkGetBufferDeviceAddress(dev_, &dai);
  const uint32_t flags = mprops_.memoryTypes[ai.memoryTypeIndex].propertyFlags;
  if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK_CHECK(vkMapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
  if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) { b.device_local = true; vram_used_ += req.size; }
  return b;
}

void VkCtx::free(VBuf& b) {
  if (!b.buf) return;
  if (b.device_local) vram_used_ -= std::min(vram_used_, (b.size + 255) / 256 * 256);
  vkDestroyBuffer(dev_, b.buf, nullptr);
  vkFreeMemory(dev_, b.mem, nullptr);
  b = VBuf{};
}

void VkCtx::warm_pipelines() {
  const double t0 = qe_now();
  for (const auto& e : kSpv)
    if (coopmat || strncmp(e.name, "cm_", 3)) pipeline(e.name);
  if (!cache_path_.empty()) {
    size_t n = 0;
    vkGetPipelineCacheData(dev_, cache_, &n, nullptr);
    std::vector<char> blob(n);
    if (n && vkGetPipelineCacheData(dev_, cache_, &n, blob.data()) == VK_SUCCESS) {
      const std::string dir = cache_path_.substr(0, cache_path_.rfind('/'));
      (void)!system(("mkdir -p '" + dir + "'").c_str());
      const std::string tmp = cache_path_ + ".tmp";
      if (FILE* f = fopen(tmp.c_str(), "wb")) {
        const bool ok = fwrite(blob.data(), 1, n, f) == n;
        fclose(f);
        if (ok) rename(tmp.c_str(), cache_path_.c_str());
      }
    }
  }
  QE_DBG("[vk] %zu pipelines ready in %.0f ms\n", sizeof(kSpv) / sizeof(kSpv[0]), (qe_now() - t0) * 1e3);
}

bool VkCtx::has_pipeline(const char* name) const {
  for (const auto& e : kSpv)
    if (!strcmp(e.name, name)) return true;
  return false;
}

VkPipeline VkCtx::pipeline(const char* name) {
  auto it = pipes_.find(name);
  if (it != pipes_.end()) return it->second;
  const SpvEntry* sp = nullptr;
  for (const auto& e : kSpv)
    if (!strcmp(e.name, name)) sp = &e;
  if (!sp) QE_FATAL("no shader '%s'", name);
  VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  mci.codeSize = sp->size;
  mci.pCode = sp->code;
  VkShaderModule mod;
  VK_CHECK(vkCreateShaderModule(dev_, &mci, nullptr, &mod));
  VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pci.stage.module = mod;
  pci.stage.pName = "main";
  pci.layout = layout_;
  VkPipeline p;
  VK_CHECK(vkCreateComputePipelines(dev_, cache_, 1, &pci, nullptr, &p));
  vkDestroyShaderModule(dev_, mod, nullptr);
  pipes_[name] = p;
  return p;
}

void VkCtx::begin() {
  if (recording_) return;
  VK_CHECK(vkResetCommandBuffer(cb_, 0));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cb_, &bi));
  recording_ = true;
  if (qpool_) { vkCmdResetQueryPool(cb_, qpool_, 0, 8192); q_names_.clear(); }
}

void VkCtx::barrier() {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                     VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                       0, 1, &mb, 0, nullptr, 0, nullptr);
}

void VkCtx::dispatch(const char* name, const void* pc, uint32_t pc_size, uint32_t gx, uint32_t gy, uint32_t gz,
                     bool bar) {
  QE_ASSERT(recording_ && pc_size <= kPushBytes);
  if (!gx || !gy || !gz) return;
  vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(name));
  vkCmdPushConstants(cb_, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
  const bool prof = qpool_ && q_names_.size() < 4095;
  if (prof) vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool_, 2 * (uint32_t)q_names_.size());
  vkCmdDispatch(cb_, gx, gy, gz);
  if (prof) {
    vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, 2 * (uint32_t)q_names_.size() + 1);
    q_names_.push_back(name);
  }
  if (bar) barrier();
}

void VkCtx::copy(const VBuf& src, size_t so, const VBuf& dst, size_t dof, size_t n) {
  QE_ASSERT(recording_);
  if (!n) return;
  VkBufferCopy r{so, dof, n};
  vkCmdCopyBuffer(cb_, src.buf, dst.buf, 1, &r);
  barrier();
}

void VkCtx::fill(const VBuf& dst, size_t off, size_t n, uint32_t v) {
  QE_ASSERT(recording_);
  vkCmdFillBuffer(cb_, dst.buf, off, (n + 3) / 4 * 4, v);
  barrier();
}

void VkCtx::submit() {
  if (!recording_) return;
  VK_CHECK(vkEndCommandBuffer(cb_));
  recording_ = false;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb_;
  VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
  VK_CHECK(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
  VK_CHECK(vkResetFences(dev_, 1, &fence_));
  if (qpool_) prof_collect();
}

void VkCtx::prof_collect() {
  if (q_names_.empty()) return;
  std::vector<uint64_t> ts(2 * q_names_.size());
  vkGetQueryPoolResults(dev_, qpool_, 0, (uint32_t)ts.size(), ts.size() * 8, ts.data(), 8,
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  for (size_t i = 0; i < q_names_.size(); i++) {
    auto& e = prof_[q_names_[i]];
    e.first += (ts[2 * i + 1] - ts[2 * i]) * (double)ts_period_ * 1e-6;  // ms
    e.second++;
  }
  q_names_.clear();
}

void VkCtx::prof_report() {
  std::vector<std::pair<double, std::string>> v;
  double tot = 0;
  for (auto& [k, e] : prof_) { v.emplace_back(e.first, k); tot += e.first; }
  std::sort(v.rbegin(), v.rend());
  fprintf(stderr, "[vkprof] GPU time per pipeline (total %.1f ms)\n", tot);
  for (auto& [t, k] : v)
    fprintf(stderr, "  %-14s %9.2f ms %5.1f%%  %7ld calls  %8.1f us/call\n", k.c_str(), t, 100 * t / tot, prof_[k].second,
            1e3 * t / prof_[k].second);
}
