#include "hwmon.h"

#ifdef _WIN32
// Windows port: no telemetry yet (NVML .so / /proc / /sys do not exist); the web UI
// shows the "unknown" defaults from hwmon.h.
GpuStats read_gpu_stats() { return GpuStats{}; }
CpuStats read_cpu_stats() { return CpuStats{}; }
#else

#include <dirent.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <type_traits>
#include <mutex>
#include <sstream>

namespace {

// ---- minimal NVML ABI (stable C API) ----
typedef struct nvmlDevice_st* nvmlDevice_t;
struct nvmlUtilization_t { unsigned gpu, memory; };
struct nvmlMemory_t { unsigned long long total, free, used; };

struct Nvml {
  bool ok = false;
  nvmlDevice_t dev = nullptr;
  int (*Init)() = nullptr;
  int (*GetHandleByIndex)(unsigned, nvmlDevice_t*) = nullptr;
  int (*GetName)(nvmlDevice_t, char*, unsigned) = nullptr;
  int (*GetTemperature)(nvmlDevice_t, int, unsigned*) = nullptr;
  int (*GetPowerUsage)(nvmlDevice_t, unsigned*) = nullptr;
  int (*GetPowerLimit)(nvmlDevice_t, unsigned*) = nullptr;
  int (*GetUtil)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
  int (*GetMem)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
  int (*GetClock)(nvmlDevice_t, int, unsigned*) = nullptr;
  int (*GetMaxClock)(nvmlDevice_t, int, unsigned*) = nullptr;
  int (*GetFan)(nvmlDevice_t, unsigned*) = nullptr;
  int (*GetPcieTp)(nvmlDevice_t, int, unsigned*) = nullptr;
  int (*GetPcieGen)(nvmlDevice_t, unsigned*) = nullptr;
  int (*GetPcieWidth)(nvmlDevice_t, unsigned*) = nullptr;
  std::string name;

  Nvml() {
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("/run/opengl-driver/lib/libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    auto sym = [&](auto& fn, const char* n) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(h, n)); };
    sym(Init, "nvmlInit_v2");
    sym(GetHandleByIndex, "nvmlDeviceGetHandleByIndex_v2");
    sym(GetName, "nvmlDeviceGetName");
    sym(GetTemperature, "nvmlDeviceGetTemperature");
    sym(GetPowerUsage, "nvmlDeviceGetPowerUsage");
    sym(GetPowerLimit, "nvmlDeviceGetEnforcedPowerLimit");
    sym(GetUtil, "nvmlDeviceGetUtilizationRates");
    sym(GetMem, "nvmlDeviceGetMemoryInfo");
    sym(GetClock, "nvmlDeviceGetClockInfo");
    sym(GetMaxClock, "nvmlDeviceGetMaxClockInfo");
    sym(GetFan, "nvmlDeviceGetFanSpeed");
    sym(GetPcieTp, "nvmlDeviceGetPcieThroughput");
    sym(GetPcieGen, "nvmlDeviceGetCurrPcieLinkGeneration");
    sym(GetPcieWidth, "nvmlDeviceGetCurrPcieLinkWidth");
    if (!Init || !GetHandleByIndex || Init() != 0 || GetHandleByIndex(0, &dev) != 0) return;
    char buf[96] = {0};
    if (GetName && GetName(dev, buf, sizeof buf) == 0) name = buf;
    ok = true;
  }
};

Nvml& nvml() {
  static Nvml n;
  return n;
}

std::string read_file(const std::string& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

double cpu_temp() {
  // AMD k10temp (Tctl) or Intel coretemp (package), first match wins
  DIR* d = opendir("/sys/class/hwmon");
  if (!d) return -1;
  double t = -1;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    const std::string base = std::string("/sys/class/hwmon/") + e->d_name;
    std::string name = read_file(base + "/name");
    while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
    if (name == "k10temp" || name == "coretemp" || name == "zenpower") {
      const std::string v = read_file(base + "/temp1_input");
      if (!v.empty()) { t = atof(v.c_str()) / 1000.0; break; }
    }
  }
  closedir(d);
  return t;
}

}  // namespace

GpuStats read_gpu_stats() {
  GpuStats g;
  Nvml& n = nvml();
  if (!n.ok) return g;
  g.ok = true;
  g.name = n.name;
  unsigned v = 0;
  if (n.GetTemperature && n.GetTemperature(n.dev, 0, &v) == 0) g.temp_c = (int)v;
  if (n.GetFan && n.GetFan(n.dev, &v) == 0) g.fan_pct = (int)v;
  if (n.GetPowerUsage && n.GetPowerUsage(n.dev, &v) == 0) g.power_w = v / 1000.0;
  if (n.GetPowerLimit && n.GetPowerLimit(n.dev, &v) == 0) g.power_limit_w = v / 1000.0;
  nvmlUtilization_t u{};
  if (n.GetUtil && n.GetUtil(n.dev, &u) == 0) { g.util_gpu = (int)u.gpu; g.util_mem = (int)u.memory; }
  nvmlMemory_t m{};
  if (n.GetMem && n.GetMem(n.dev, &m) == 0) { g.mem_used_mb = m.used / 1048576.0; g.mem_total_mb = m.total / 1048576.0; }
  if (n.GetClock && n.GetClock(n.dev, 0, &v) == 0) g.clock_gpu_mhz = (int)v;
  if (n.GetClock && n.GetClock(n.dev, 2, &v) == 0) g.clock_mem_mhz = (int)v;
  if (n.GetMaxClock && n.GetMaxClock(n.dev, 1, &v) == 0) g.clock_sm_max_mhz = (int)v;
  if (n.GetPcieTp && n.GetPcieTp(n.dev, 1, &v) == 0) g.pcie_rx_mbs = v / 1024.0;  // KB/s, host -> GPU
  if (n.GetPcieTp && n.GetPcieTp(n.dev, 0, &v) == 0) g.pcie_tx_mbs = v / 1024.0;
  if (n.GetPcieGen && n.GetPcieGen(n.dev, &v) == 0) g.pcie_gen = (int)v;
  if (n.GetPcieWidth && n.GetPcieWidth(n.dev, &v) == 0) g.pcie_width = (int)v;
  return g;
}

CpuStats read_cpu_stats() {
  CpuStats c;
  c.temp_c = cpu_temp();
  // utilization from /proc/stat deltas between calls
  static std::mutex mu;
  static unsigned long long prev_idle = 0, prev_total = 0;
  {
    std::ifstream f("/proc/stat");
    std::string cpu;
    unsigned long long user = 0, nice = 0, sys = 0, idle = 0, iow = 0, irq = 0, sirq = 0, steal = 0;
    f >> cpu >> user >> nice >> sys >> idle >> iow >> irq >> sirq >> steal;
    const unsigned long long idle_all = idle + iow, total = user + nice + sys + idle + iow + irq + sirq + steal;
    std::lock_guard<std::mutex> lk(mu);
    if (prev_total && total > prev_total)
      c.util_pct = 100.0 * (1.0 - (double)(idle_all - prev_idle) / (double)(total - prev_total));
    prev_idle = idle_all;
    prev_total = total;
  }
  {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    double sum = 0, mx = 0;
    int n = 0;
    while (std::getline(f, line)) {
      if (line.rfind("cpu MHz", 0) == 0) {
        const double v = atof(line.substr(line.find(':') + 1).c_str());
        sum += v;
        mx = std::max(mx, v);
        n++;
      } else if (c.name.empty() && line.rfind("model name", 0) == 0) {
        c.name = line.substr(line.find(':') + 2);
      }
    }
    if (n) { c.freq_avg_mhz = sum / n; c.freq_max_mhz = mx; c.threads = n; }
  }
  {
    std::ifstream f("/proc/meminfo");
    std::string k;
    double v;
    std::string unit;
    double total = -1, avail = -1;
    while (f >> k >> v >> unit) {
      if (k == "MemTotal:") total = v / 1024.0;
      else if (k == "MemAvailable:") avail = v / 1024.0;
    }
    if (total > 0 && avail >= 0) { c.ram_total_mb = total; c.ram_used_mb = total - avail; }
  }
  return c;
}

#endif  // _WIN32
