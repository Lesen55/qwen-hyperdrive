#pragma once
// Hardware telemetry for the web UI: GPU via NVML (dlopen'ed at runtime, no build dependency),
// CPU / RAM via /proc and /sys.
#include <string>

struct GpuStats {
  bool ok = false;
  std::string name;
  int temp_c = -1, fan_pct = -1;
  double power_w = -1, power_limit_w = -1;
  int util_gpu = -1, util_mem = -1;
  double mem_used_mb = -1, mem_total_mb = -1;
  int clock_gpu_mhz = -1, clock_mem_mhz = -1, clock_sm_max_mhz = -1;
  double pcie_rx_mbs = -1, pcie_tx_mbs = -1;
  int pcie_gen = -1, pcie_width = -1;
};

struct CpuStats {
  std::string name;
  double temp_c = -1;
  double util_pct = -1;
  double freq_avg_mhz = -1, freq_max_mhz = -1;
  int threads = 0;
  double ram_used_mb = -1, ram_total_mb = -1;
};

GpuStats read_gpu_stats();
CpuStats read_cpu_stats();
