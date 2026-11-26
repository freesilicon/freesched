/*
 * freesched topology probing and static SoC hardware cost profiles.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "fs_internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#ifdef __linux__
#include <unistd.h>
#if defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP2_SME2
#define HWCAP2_SME2 (1UL << 37)
#endif
#endif
#endif

uint64_t fs_now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000ULL +
         static_cast<uint64_t>(ts.tv_nsec) / 1000ULL;
}

void fs_set_error(fs_context_t* ctx, const char* fmt, ...) {
  if (!ctx || !fmt) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(ctx->last_error, sizeof(ctx->last_error), fmt, args);
  va_end(args);
}

int fs_dev_is_valid(fs_dev_t dev) {
  return dev > FS_DEV_NONE && dev < FS_DEV_COUNT;
}

int fs_dev_is_cpu(fs_dev_t dev) {
  return dev == FS_DEV_CPU_NEON || dev == FS_DEV_CPU_SME2;
}

size_t fs_align_up(size_t value, size_t alignment) {
  if (alignment == 0u) {
    return value;
  }
  return (value + alignment - 1u) & ~(alignment - 1u);
}

static void fs_copy_str(char* dst, size_t dst_size, const char* src) {
  if (!dst || dst_size == 0u) {
    return;
  }
  if (!src) {
    dst[0] = '\0';
    return;
  }
  std::snprintf(dst, dst_size, "%s", src);
}

static void fs_fill_profile_laguna(fs_topology_t* topo, int is_mock) {
  const uint32_t mock_bit = is_mock ? static_cast<uint32_t>(FS_CAP_MOCK) : 0u;
  const uint32_t base_mem = FS_CAP_UNIFIED_MEMORY | FS_CAP_DMABUF | FS_CAP_SYNC_FILE | mock_bit;

  topo->profile             = FS_PROFILE_PIXEL10_LAGUNA;
  fs_copy_str(topo->soc_name, sizeof(topo->soc_name), "laguna");
  topo->unified_ram_bytes   = 16ULL * 1024ULL * 1024ULL * 1024ULL;
  topo->global_caps         = base_mem | FS_CAP_VK_EXT_MEMORY_FD | FS_CAP_EDGETPU_MAP_DMABUF;
  topo->prefill_m_threshold = FS_DEFAULT_PREFILL_M;

  /* EdgeTPU: Tensor G5 BUENOS_A0 */
  fs_dev_info_t& tpu        = topo->devices[FS_DEV_TPU];
  tpu.dev                   = FS_DEV_TPU;
  tpu.available             = 1;
  tpu.caps                  = base_mem | FS_CAP_EDGETPU_MAP_DMABUF | FS_CAP_TPU_SPLINE_0X640 |
                              FS_CAP_TPU_SWIGLU_CHAIN | FS_CAP_FP16 | FS_CAP_INT8 | FS_CAP_Q8_0 |
                              FS_CAP_Q4_0;
  fs_copy_str(tpu.name, sizeof(tpu.name), "BUENOS_A0 EdgeTPU");
  tpu.peak_gflops_fp16      = 6000.0f;
  tpu.peak_tops_int8        = 12.0f;
  tpu.mem_bw_gbps           = 48.0f;
  tpu.launch_overhead_us    = 18.0f;
  tpu.handoff_overhead_us   = 8.0f;
  tpu.typical_watts         = 1.8f;
  tpu.sram_kb               = 2048u;
  tpu.max_inflight          = 16u;

  /* Vulkan GPU: PowerVR Volcanic DXT-48-1536 */
  fs_dev_info_t& gpu        = topo->devices[FS_DEV_GPU_VK];
  gpu.dev                   = FS_DEV_GPU_VK;
  gpu.available             = 1;
  gpu.caps                  = base_mem | FS_CAP_VK_EXT_MEMORY_FD | FS_CAP_FP16 | FS_CAP_INT8 |
                              FS_CAP_Q8_0 | FS_CAP_Q4_0;
  fs_copy_str(gpu.name, sizeof(gpu.name), "PowerVR DXT-48-1536 (Vulkan)");
  gpu.peak_gflops_fp16      = 2400.0f;
  gpu.peak_tops_int8        = 4.8f;
  gpu.mem_bw_gbps           = 54.0f;
  gpu.launch_overhead_us    = 65.0f;
  gpu.handoff_overhead_us   = 10.0f;
  gpu.typical_watts         = 4.5f;
  gpu.sram_kb               = 1024u;
  gpu.max_inflight          = 8u;

  /* ARMv9 NEON CPU */
  fs_dev_info_t& neon       = topo->devices[FS_DEV_CPU_NEON];
  neon.dev                  = FS_DEV_CPU_NEON;
  neon.available            = 1;
  neon.caps                 = base_mem | FS_CAP_FP16 | FS_CAP_INT8 | FS_CAP_Q8_0 | FS_CAP_Q4_0;
  fs_copy_str(neon.name, sizeof(neon.name), "ARMv9 NEON CPU");
  neon.peak_gflops_fp16     = 240.0f;
  neon.peak_tops_int8       = 0.96f;
  neon.mem_bw_gbps          = 42.0f;
  neon.launch_overhead_us   = 3.0f;
  neon.handoff_overhead_us  = 4.0f;
  neon.typical_watts        = 2.6f;
  neon.sram_kb              = 4096u;
  neon.max_inflight         = 4u;

  /* SME2 unavailable on laguna */
  fs_dev_info_t& sme2       = topo->devices[FS_DEV_CPU_SME2];
  sme2.dev                  = FS_DEV_CPU_SME2;
  sme2.available            = 0;
  sme2.caps                 = mock_bit;
  fs_copy_str(sme2.name, sizeof(sme2.name), "ARMv9 SME2 CPU (unavailable)");
  sme2.peak_gflops_fp16     = 0.0f;
  sme2.peak_tops_int8       = 0.0f;
  sme2.mem_bw_gbps          = 0.0f;
  sme2.launch_overhead_us   = 0.0f;
  sme2.handoff_overhead_us  = 0.0f;
  sme2.typical_watts        = 0.0f;
  sme2.sram_kb              = 0u;
  sme2.max_inflight         = 0u;
}

static void fs_fill_profile_malibu(fs_topology_t* topo, int is_mock) {
  const uint32_t mock_bit = is_mock ? static_cast<uint32_t>(FS_CAP_MOCK) : 0u;
  const uint32_t base_mem = FS_CAP_UNIFIED_MEMORY | FS_CAP_DMABUF | FS_CAP_SYNC_FILE | mock_bit;

  topo->profile             = FS_PROFILE_PIXEL11_MALIBU;
  fs_copy_str(topo->soc_name, sizeof(topo->soc_name), "malibu");
  topo->unified_ram_bytes   = 16ULL * 1024ULL * 1024ULL * 1024ULL;
  topo->global_caps         = base_mem | FS_CAP_VK_EXT_MEMORY_FD | FS_CAP_EDGETPU_MAP_DMABUF |
                              FS_CAP_SME2_MATMUL;
  topo->prefill_m_threshold = FS_DEFAULT_PREFILL_M;

  /* EdgeTPU: Tensor G6 malibu TPU */
  fs_dev_info_t& tpu        = topo->devices[FS_DEV_TPU];
  tpu.dev                   = FS_DEV_TPU;
  tpu.available             = 1;
  tpu.caps                  = base_mem | FS_CAP_EDGETPU_MAP_DMABUF | FS_CAP_TPU_SPLINE_0X640 |
                              FS_CAP_TPU_SWIGLU_CHAIN | FS_CAP_FP16 | FS_CAP_INT8 | FS_CAP_Q8_0 |
                              FS_CAP_Q4_0;
  fs_copy_str(tpu.name, sizeof(tpu.name), "malibu EdgeTPU");
  tpu.peak_gflops_fp16      = 12000.0f;
  tpu.peak_tops_int8        = 24.0f;
  tpu.mem_bw_gbps           = 68.0f;
  tpu.launch_overhead_us    = 14.0f;
  tpu.handoff_overhead_us   = 6.0f;
  tpu.typical_watts         = 1.6f;
  tpu.sram_kb               = 4096u;
  tpu.max_inflight          = 16u;

  /* Vulkan GPU: PowerVR CXTP-48-1536 */
  fs_dev_info_t& gpu        = topo->devices[FS_DEV_GPU_VK];
  gpu.dev                   = FS_DEV_GPU_VK;
  gpu.available             = 1;
  gpu.caps                  = base_mem | FS_CAP_VK_EXT_MEMORY_FD | FS_CAP_FP16 | FS_CAP_INT8 |
                              FS_CAP_Q8_0 | FS_CAP_Q4_0;
  fs_copy_str(gpu.name, sizeof(gpu.name), "PowerVR CXTP-48-1536 (Vulkan)");
  gpu.peak_gflops_fp16      = 3600.0f;
  gpu.peak_tops_int8        = 7.2f;
  gpu.mem_bw_gbps           = 68.0f;
  gpu.launch_overhead_us    = 55.0f;
  gpu.handoff_overhead_us   = 8.0f;
  gpu.typical_watts         = 4.2f;
  gpu.sram_kb               = 2048u;
  gpu.max_inflight          = 12u;

  /* ARMv9 NEON CPU */
  fs_dev_info_t& neon       = topo->devices[FS_DEV_CPU_NEON];
  neon.dev                  = FS_DEV_CPU_NEON;
  neon.available            = 1;
  neon.caps                 = base_mem | FS_CAP_FP16 | FS_CAP_INT8 | FS_CAP_Q8_0 | FS_CAP_Q4_0;
  fs_copy_str(neon.name, sizeof(neon.name), "ARMv9 NEON CPU");
  neon.peak_gflops_fp16     = 320.0f;
  neon.peak_tops_int8       = 1.28f;
  neon.mem_bw_gbps          = 56.0f;
  neon.launch_overhead_us   = 2.5f;
  neon.handoff_overhead_us  = 3.5f;
  neon.typical_watts        = 2.4f;
  neon.sram_kb              = 6144u;
  neon.max_inflight         = 4u;

  /* ARMv9 SME2 CPU */
  fs_dev_info_t& sme2       = topo->devices[FS_DEV_CPU_SME2];
  sme2.dev                  = FS_DEV_CPU_SME2;
  sme2.available            = 1;
  sme2.caps                 = base_mem | FS_CAP_SME2_MATMUL | FS_CAP_FP16 | FS_CAP_INT8 |
                              FS_CAP_Q8_0 | FS_CAP_Q4_0;
  fs_copy_str(sme2.name, sizeof(sme2.name), "ARMv9 SME2 CPU");
  sme2.peak_gflops_fp16     = 1200.0f;
  sme2.peak_tops_int8       = 4.8f;
  sme2.mem_bw_gbps          = 68.0f;
  sme2.launch_overhead_us   = 3.5f;
  sme2.handoff_overhead_us  = 3.5f;
  sme2.typical_watts        = 2.2f;
  sme2.sram_kb              = 6144u;
  sme2.max_inflight         = 4u;
}

static void fs_fill_profile_host_mock(fs_topology_t* topo) {
  fs_fill_profile_malibu(topo, 1);
  topo->profile = FS_PROFILE_HOST_MOCK;
  fs_copy_str(topo->soc_name, sizeof(topo->soc_name), "host_mock");
  fs_copy_str(topo->devices[FS_DEV_TPU].name, sizeof(topo->devices[FS_DEV_TPU].name),
              "Mock EdgeTPU (BUENOS/malibu)");
  fs_copy_str(topo->devices[FS_DEV_GPU_VK].name, sizeof(topo->devices[FS_DEV_GPU_VK].name),
              "Mock Vulkan GPU (PowerVR)");
  fs_copy_str(topo->devices[FS_DEV_CPU_NEON].name, sizeof(topo->devices[FS_DEV_CPU_NEON].name),
              "Host Reference CPU (NEON)");
  fs_copy_str(topo->devices[FS_DEV_CPU_SME2].name, sizeof(topo->devices[FS_DEV_CPU_SME2].name),
              "Host Reference CPU (SME2)");
}

static fs_hw_profile_t fs_detect_profile_from_env_or_os(int* out_is_mock) {
  const char* mock_env = std::getenv("FREESCHED_MOCK");
  const int   env_mock = (mock_env && std::strcmp(mock_env, "0") != 0) ? 1 : 0;
  const char* prof_env = std::getenv("FREESCHED_PROFILE");

  if (prof_env) {
    if (std::strcmp(prof_env, "laguna") == 0) {
      *out_is_mock = env_mock;
      return FS_PROFILE_PIXEL10_LAGUNA;
    }
    if (std::strcmp(prof_env, "malibu") == 0) {
      *out_is_mock = env_mock;
      return FS_PROFILE_PIXEL11_MALIBU;
    }
    if (std::strcmp(prof_env, "host_mock") == 0) {
      *out_is_mock = 1;
      return FS_PROFILE_HOST_MOCK;
    }
  }

  if (env_mock) {
    *out_is_mock = 1;
    return FS_PROFILE_HOST_MOCK;
  }

#ifdef __linux__
  {
    FILE* f = std::fopen("/proc/device-tree/compatible", "rb");
    if (f) {
      char   buf[256];
      size_t n = std::fread(buf, 1, sizeof(buf) - 1u, f);
      std::fclose(f);
      for (size_t i = 0; i < n; ++i) {
        if (buf[i] == '\0') {
          buf[i] = ' ';
        }
      }
      buf[n] = '\0';
      if (std::strstr(buf, "malibu") != nullptr) {
        *out_is_mock = 0;
        return FS_PROFILE_PIXEL11_MALIBU;
      }
      if (std::strstr(buf, "laguna") != nullptr || std::strstr(buf, "rango") != nullptr) {
        *out_is_mock = 0;
        return FS_PROFILE_PIXEL10_LAGUNA;
      }
    }
  }
#endif

  *out_is_mock = 1;
  return FS_PROFILE_HOST_MOCK;
}

extern "C" {

fs_status_t fs_topology_probe(fs_hw_profile_t profile_hint, fs_topology_t* out_topo) {
  if (!out_topo) {
    return FS_ERR_INVALID_ARG;
  }

  std::memset(out_topo, 0, sizeof(*out_topo));
  out_topo->struct_size = static_cast<uint32_t>(sizeof(fs_topology_t));

  int             is_mock = 0;
  fs_hw_profile_t profile = profile_hint;

  if (profile == FS_PROFILE_AUTO) {
    profile = fs_detect_profile_from_env_or_os(&is_mock);
  } else if (profile == FS_PROFILE_HOST_MOCK) {
    is_mock = 1;
  } else {
    const char* mock_env = std::getenv("FREESCHED_MOCK");
#ifndef __linux__
    is_mock = 1;
    (void)mock_env;
#else
    is_mock = (mock_env && std::strcmp(mock_env, "0") != 0) ? 1 : 0;
#endif
  }

  switch (profile) {
    case FS_PROFILE_PIXEL10_LAGUNA:
      fs_fill_profile_laguna(out_topo, is_mock);
      break;
    case FS_PROFILE_PIXEL11_MALIBU:
      fs_fill_profile_malibu(out_topo, is_mock);
      break;
    case FS_PROFILE_HOST_MOCK:
    default:
      fs_fill_profile_host_mock(out_topo);
      break;
  }

#ifdef __linux__
  if (!is_mock) {
    if (access("/dev/edgetpu-soc", F_OK) != 0) {
      out_topo->devices[FS_DEV_TPU].available = 0;
    }
    if (access("/dev/dri/renderD128", F_OK) != 0 && access("/dev/kgsl-3d0", F_OK) != 0) {
      out_topo->devices[FS_DEV_GPU_VK].available = 0;
    }
    if (access("/dev/dma_heap/system", F_OK) != 0) {
      out_topo->global_caps &= ~FS_CAP_DMABUF;
    }
#if defined(__aarch64__)
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if ((hwcap2 & HWCAP2_SME2) != 0UL) {
      out_topo->devices[FS_DEV_CPU_SME2].available = 1;
      out_topo->devices[FS_DEV_CPU_SME2].caps |= FS_CAP_SME2_MATMUL;
    }
#endif
  }
#endif

  return FS_OK;
}

fs_status_t fs_topology_set_device_enabled(fs_topology_t* topo, fs_dev_t dev, int enabled) {
  if (!topo || !fs_dev_is_valid(dev)) {
    return FS_ERR_INVALID_ARG;
  }
  topo->devices[dev].available = enabled ? 1 : 0;
  return FS_OK;
}

int fs_topology_has_device(const fs_topology_t* topo, fs_dev_t dev) {
  if (!topo || !fs_dev_is_valid(dev)) {
    return 0;
  }
  return topo->devices[dev].available ? 1 : 0;
}

const char* fs_status_string(fs_status_t status) {
  switch (status) {
    case FS_OK:              return "FS_OK";
    case FS_ERR_INVALID_ARG: return "FS_ERR_INVALID_ARG";
    case FS_ERR_NO_MEMORY:   return "FS_ERR_NO_MEMORY";
    case FS_ERR_DEVICE:      return "FS_ERR_DEVICE";
    case FS_ERR_IOCTL:       return "FS_ERR_IOCTL";
    case FS_ERR_TIMEOUT:     return "FS_ERR_TIMEOUT";
    case FS_ERR_OWNERSHIP:   return "FS_ERR_OWNERSHIP";
    case FS_ERR_BUSY:        return "FS_ERR_BUSY";
    case FS_ERR_NOT_FOUND:   return "FS_ERR_NOT_FOUND";
    case FS_ERR_UNSUPPORTED: return "FS_ERR_UNSUPPORTED";
    case FS_ERR_OVERFLOW:    return "FS_ERR_OVERFLOW";
    case FS_ERR_IO:          return "FS_ERR_IO";
    case FS_ERR_FENCE:       return "FS_ERR_FENCE";
    case FS_ERR_NO_BACKEND:  return "FS_ERR_NO_BACKEND";
    default:                 return "FS_ERR_UNKNOWN";
  }
}

const char* fs_dev_name(fs_dev_t dev) {
  switch (dev) {
    case FS_DEV_NONE:     return "NONE";
    case FS_DEV_TPU:      return "TPU";
    case FS_DEV_GPU_VK:   return "GPU_VK";
    case FS_DEV_CPU_NEON: return "CPU_NEON";
    case FS_DEV_CPU_SME2: return "CPU_SME2";
    default:              return "UNKNOWN";
  }
}

const char* fs_profile_name(fs_hw_profile_t profile) {
  switch (profile) {
    case FS_PROFILE_AUTO:           return "auto";
    case FS_PROFILE_HOST_MOCK:      return "host_mock";
    case FS_PROFILE_PIXEL10_LAGUNA: return "laguna";
    case FS_PROFILE_PIXEL11_MALIBU: return "malibu";
    default:                        return "unknown";
  }
}

const char* fs_op_type_name(fs_op_type_t type) {
  switch (type) {
    case FS_OP_GEMM:         return "GEMM";
    case FS_OP_PREFILL_GEMM: return "PREFILL_GEMM";
    case FS_OP_DECODE_GEMV:  return "DECODE_GEMV";
    case FS_OP_SWIGLU:       return "SWIGLU";
    case FS_OP_ATTN_SOFTMAX: return "ATTN_SOFTMAX";
    case FS_OP_ROPE:         return "ROPE";
    case FS_OP_RMS_NORM:     return "RMS_NORM";
    case FS_OP_VISION_CONV:  return "VISION_CONV";
    case FS_OP_ELEMENTWISE:  return "ELEMENTWISE";
    default:                 return "UNKNOWN_OP";
  }
}

} /* extern "C" */
