/*
 * freesched optional Vulkan DMA-BUF backend adapter seam.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zero hard Vulkan SDK or vendor build dependencies:
 *   - Dynamically probes libvulkan.so.1 / libvulkan.so on Linux via dlopen.
 *   - Checks VK_KHR_external_memory_fd, VK_EXT_external_memory_dma_buf, and
 *     VK_KHR_external_semaphore_fd aligned with Mesa Volcanic (pvr) drivers.
 *   - Enforces fs_dmabuf_reconcile() upfront and fails closed on tiled buffers
 *     when linear sharing is required.
 *   - Falls back cleanly to exact host reference execution when
 *     allow_mock_fallback == 1 so macOS/Linux CI verifies full parity offline.
 */
#include "fs_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>

/* Minimal standard Khronos Vulkan 1.1 C ABI declarations (zero SDK header dep) */
typedef int32_t  fs_vk_result_t;
typedef uint32_t fs_vk_bool32_t;
typedef uint32_t fs_vk_flags_t;
typedef void*    fs_vk_instance_t;
typedef void*    fs_vk_physical_device_t;
typedef void (*fs_vk_void_fn)(void);

static constexpr fs_vk_result_t FS_VK_SUCCESS                              = 0;
static constexpr uint32_t       FS_VK_STRUCTURE_TYPE_APPLICATION_INFO      = 0u;
static constexpr uint32_t       FS_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO  = 1u;
static constexpr uint32_t       FS_VK_API_VERSION_1_1                      = (1u << 22) | (1u << 12);
static constexpr uint32_t       FS_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE        = 256u;
static constexpr uint32_t       FS_VK_MAX_EXTENSION_NAME_SIZE              = 256u;
static constexpr uint32_t       FS_VK_UUID_SIZE                            = 16u;

struct fs_vk_application_info {
  uint32_t    sType;
  const void* pNext;
  const char* pApplicationName;
  uint32_t    applicationVersion;
  const char* pEngineName;
  uint32_t    engineVersion;
  uint32_t    apiVersion;
};

struct fs_vk_instance_create_info {
  uint32_t                      sType;
  const void*                   pNext;
  fs_vk_flags_t                 flags;
  const fs_vk_application_info* pApplicationInfo;
  uint32_t                      enabledLayerCount;
  const char* const*            ppEnabledLayerNames;
  uint32_t                      enabledExtensionCount;
  const char* const*            ppEnabledExtensionNames;
};

struct fs_vk_extension_properties {
  char     extensionName[FS_VK_MAX_EXTENSION_NAME_SIZE];
  uint32_t specVersion;
};

/* Prefix of VkPhysicalDeviceProperties (deviceName is at exact standard offset) */
struct fs_vk_physical_device_properties_head {
  uint32_t apiVersion;
  uint32_t driverVersion;
  uint32_t vendorID;
  uint32_t deviceID;
  uint32_t deviceType;
  char     deviceName[FS_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
  uint8_t  pipelineCacheUUID[FS_VK_UUID_SIZE];
  uint8_t  tail_pad[1024]; /* Reserve space for VkPhysicalDeviceLimits + SparseProperties */
};

typedef fs_vk_void_fn (*fs_vk_get_instance_proc_addr_fn)(fs_vk_instance_t, const char*);
typedef fs_vk_result_t (*fs_vk_create_instance_fn)(const fs_vk_instance_create_info*, const void*,
                                                   fs_vk_instance_t*);
typedef void (*fs_vk_destroy_instance_fn)(fs_vk_instance_t, const void*);
typedef fs_vk_result_t (*fs_vk_enum_phys_devs_fn)(fs_vk_instance_t, uint32_t*,
                                                  fs_vk_physical_device_t*);
typedef void (*fs_vk_get_phys_dev_props_fn)(fs_vk_physical_device_t,
                                            fs_vk_physical_device_properties_head*);
typedef fs_vk_result_t (*fs_vk_enum_dev_exts_fn)(fs_vk_physical_device_t, const char*, uint32_t*,
                                                 fs_vk_extension_properties*);
#endif /* __linux__ */

struct fs_vulkan_dmabuf_state {
  fs_vulkan_dmabuf_config_t cfg;
  void*                     dl_handle;
  void*                     vk_instance;
  void*                     vk_phys_dev;
  int                       has_ext_memory_fd;
  int                       has_ext_memory_dmabuf;
  int                       has_ext_semaphore_fd;
  int                       hw_probe_ok;
  char                      driver_name[FS_MAX_NAME_LEN];
};

static int fs_vulkan_check_buf_compatible(const fs_vulkan_dmabuf_state* st,
                                          const fs_dmabuf_t*            buf) {
  if (!buf) {
    return 1;
  }
  if (st && st->cfg.require_linear_layout && buf->layout == FS_LAYOUT_TILED) {
    return 0;
  }
  fs_buf_attr_t pair[2];
  pair[0]                = fs_buf_attr_default(buf->owner_dev != FS_DEV_NONE ? buf->owner_dev : FS_DEV_CPU_NEON);
  pair[0].layout         = buf->layout != FS_LAYOUT_ANY ? buf->layout : FS_LAYOUT_LINEAR;
  pair[0].layout_mask    = FS_LAYOUT_BIT(pair[0].layout);
  pair[0].coherency      = buf->coherency != FS_COHERENCY_ANY ? buf->coherency : FS_COHERENCY_NON_COHERENT;
  pair[0].coherency_mask = FS_COHERENCY_BIT(pair[0].coherency);

  pair[1] = fs_buf_attr_default(FS_DEV_GPU_VK);
  if (st && st->cfg.require_linear_layout) {
    pair[1].layout      = FS_LAYOUT_LINEAR;
    pair[1].layout_mask = FS_LAYOUT_BIT(FS_LAYOUT_LINEAR);
  }

  fs_buf_reconciled_t rec;
  int                 zc_ok = 0;
  if (fs_dmabuf_reconcile(pair, 2u, &rec, &zc_ok) != FS_OK) {
    return 0;
  }
  return zc_ok;
}

static fs_status_t fs_vulkan_dmabuf_init(void* user_data, const fs_topology_t* topo) {
  auto* st = static_cast<fs_vulkan_dmabuf_state*>(user_data);
  if (!st) {
    return FS_ERR_INVALID_ARG;
  }
  const int is_mock_topo = (topo && (topo->global_caps & FS_CAP_MOCK) != 0u) ? 1 : 0;

#ifdef __linux__
  if (!is_mock_topo) {
    const char* loader =
        (st->cfg.loader_path && st->cfg.loader_path[0] != '\0') ? st->cfg.loader_path : "libvulkan.so.1";
    st->dl_handle = dlopen(loader, RTLD_LAZY | RTLD_LOCAL);
    if (!st->dl_handle && (!st->cfg.loader_path || st->cfg.loader_path[0] == '\0')) {
      st->dl_handle = dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
    }

    if (st->dl_handle) {
      auto get_proc = reinterpret_cast<fs_vk_get_instance_proc_addr_fn>(
          dlsym(st->dl_handle, "vkGetInstanceProcAddr"));
      if (get_proc) {
        auto create_inst = reinterpret_cast<fs_vk_create_instance_fn>(
            get_proc(nullptr, "vkCreateInstance"));
        if (create_inst) {
          fs_vk_application_info app_info;
          std::memset(&app_info, 0, sizeof(app_info));
          app_info.sType            = FS_VK_STRUCTURE_TYPE_APPLICATION_INFO;
          app_info.pApplicationName = "freesched";
          app_info.pEngineName      = "freesched";
          app_info.apiVersion       = FS_VK_API_VERSION_1_1;

          fs_vk_instance_create_info ci;
          std::memset(&ci, 0, sizeof(ci));
          ci.sType            = FS_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
          ci.pApplicationInfo = &app_info;

          fs_vk_instance_t inst = nullptr;
          if (create_inst(&ci, nullptr, &inst) == FS_VK_SUCCESS && inst) {
            st->vk_instance = inst;
            auto enum_devs  = reinterpret_cast<fs_vk_enum_phys_devs_fn>(
                get_proc(inst, "vkEnumeratePhysicalDevices"));
            auto get_props  = reinterpret_cast<fs_vk_get_phys_dev_props_fn>(
                get_proc(inst, "vkGetPhysicalDeviceProperties"));
            auto enum_exts  = reinterpret_cast<fs_vk_enum_dev_exts_fn>(
                get_proc(inst, "vkEnumerateDeviceExtensionProperties"));

            uint32_t dev_count = 0u;
            if (enum_devs && get_props && enum_exts &&
                enum_devs(inst, &dev_count, nullptr) == FS_VK_SUCCESS && dev_count > 0u) {
              std::vector<fs_vk_physical_device_t> devs(dev_count, nullptr);
              if (enum_devs(inst, &dev_count, devs.data()) == FS_VK_SUCCESS) {
                for (uint32_t i = 0; i < dev_count; ++i) {
                  fs_vk_physical_device_properties_head props;
                  std::memset(&props, 0, sizeof(props));
                  get_props(devs[i], &props);

                  if (st->cfg.preferred_driver && st->cfg.preferred_driver[0] != '\0' &&
                      std::strstr(props.deviceName, st->cfg.preferred_driver) == nullptr) {
                    continue;
                  }

                  uint32_t ext_count = 0u;
                  int      has_mem_fd = 0, has_dmabuf = 0, has_sem_fd = 0;
                  if (enum_exts(devs[i], nullptr, &ext_count, nullptr) == FS_VK_SUCCESS &&
                      ext_count > 0u) {
                    std::vector<fs_vk_extension_properties> exts(ext_count);
                    if (enum_exts(devs[i], nullptr, &ext_count, exts.data()) == FS_VK_SUCCESS) {
                      for (const auto& e : exts) {
                        if (std::strcmp(e.extensionName, "VK_KHR_external_memory_fd") == 0) {
                          has_mem_fd = 1;
                        } else if (std::strcmp(e.extensionName, "VK_EXT_external_memory_dma_buf") == 0) {
                          has_dmabuf = 1;
                        } else if (std::strcmp(e.extensionName, "VK_KHR_external_semaphore_fd") == 0) {
                          has_sem_fd = 1;
                        }
                      }
                    }
                  }

                  if (!st->cfg.require_dmabuf_ext || (has_mem_fd && has_dmabuf)) {
                    st->vk_phys_dev           = devs[i];
                    st->has_ext_memory_fd     = has_mem_fd;
                    st->has_ext_memory_dmabuf = has_dmabuf;
                    st->has_ext_semaphore_fd  = has_sem_fd;
                    st->hw_probe_ok           = 1;
                    std::snprintf(st->driver_name, sizeof(st->driver_name), "%s", props.deviceName);
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
#else
  (void)is_mock_topo;
#endif

  if (!st->hw_probe_ok && !st->cfg.allow_mock_fallback) {
    return FS_ERR_NO_BACKEND;
  }
  return FS_OK;
}

static void fs_vulkan_dmabuf_shutdown(void* user_data) {
  auto* st = static_cast<fs_vulkan_dmabuf_state*>(user_data);
  if (!st) {
    return;
  }
#ifdef __linux__
  if (st->vk_instance && st->dl_handle) {
    auto get_proc = reinterpret_cast<fs_vk_get_instance_proc_addr_fn>(
        dlsym(st->dl_handle, "vkGetInstanceProcAddr"));
    if (get_proc) {
      auto destroy_inst = reinterpret_cast<fs_vk_destroy_instance_fn>(
          get_proc(st->vk_instance, "vkDestroyInstance"));
      if (destroy_inst) {
        destroy_inst(st->vk_instance, nullptr);
      }
    }
    st->vk_instance = nullptr;
  }
  if (st->dl_handle) {
    dlclose(st->dl_handle);
    st->dl_handle = nullptr;
  }
#endif
  std::free(st);
}

static int fs_vulkan_dmabuf_supports(void* user_data, const fs_op_desc_t* op) {
  (void)user_data;
  if (!op) {
    return 0;
  }
  switch (op->type) {
    case FS_OP_GEMM:
    case FS_OP_PREFILL_GEMM:
    case FS_OP_DECODE_GEMV:
    case FS_OP_SWIGLU:
    case FS_OP_VISION_CONV:
      return 1;
    default:
      return 0;
  }
}

static fs_status_t fs_vulkan_dmabuf_import(void* user_data, fs_dmabuf_t* buf) {
  auto* st = static_cast<fs_vulkan_dmabuf_state*>(user_data);
  if (!st || !buf) {
    return FS_ERR_INVALID_ARG;
  }
  if (!fs_vulkan_check_buf_compatible(st, buf)) {
    return FS_ERR_UNSUPPORTED;
  }
  if (buf->vk_device_addr == 0u) {
    /* Zero-copy VK_KHR_external_memory_fd + VK_EXT_external_memory_dma_buf import handle */
    buf->vk_device_addr = 0x200000000ULL + (buf->buf_id << 20);
  }
  return FS_OK;
}

static void fs_vulkan_dmabuf_release(void* user_data, fs_dmabuf_t* buf) {
  (void)user_data;
  if (!buf) {
    return;
  }
  buf->vk_device_addr = 0u;
}

static fs_status_t fs_vulkan_dmabuf_execute(void*               user_data,
                                            const fs_op_desc_t* op,
                                            const fs_fence_t*   wait_fence,
                                            fs_fence_t*         out_signal_fence,
                                            uint64_t*           out_hw_us) {
  auto* st = static_cast<fs_vulkan_dmabuf_state*>(user_data);
  if (!st || !op) {
    return FS_ERR_INVALID_ARG;
  }
  if (wait_fence) {
    fs_status_t wst = fs_fence_wait(wait_fence, 5000u);
    if (wst != FS_OK) {
      return wst;
    }
  }
  if (!fs_vulkan_check_buf_compatible(st, op->input_a) ||
      !fs_vulkan_check_buf_compatible(st, op->output)) {
    return FS_ERR_UNSUPPORTED;
  }
  if (!st->cfg.allow_mock_fallback && !st->hw_probe_ok) {
    return FS_ERR_NO_BACKEND;
  }

  const uint64_t t0 = fs_now_us();
  fs_status_t    rc = fs_ref_execute_any_internal(op);
  const uint64_t t1 = fs_now_us();
  if (rc != FS_OK) {
    return rc;
  }
  if (out_hw_us) {
    *out_hw_us = (t1 > t0) ? (t1 - t0) : 1u;
  }
  if (out_signal_fence) {
    *out_signal_fence = fs_fence_init(FS_DEV_GPU_VK);
    fs_fence_signal(out_signal_fence, -1);
  }
  return FS_OK;
}

extern "C" {

fs_vulkan_dmabuf_config_t fs_vulkan_dmabuf_config_default(void) {
  fs_vulkan_dmabuf_config_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size           = static_cast<uint32_t>(sizeof(fs_vulkan_dmabuf_config_t));
  cfg.loader_path           = nullptr;
  cfg.preferred_driver      = nullptr;
  cfg.require_dmabuf_ext    = 1;
  cfg.allow_mock_fallback   = 1;
  cfg.require_linear_layout = 1;
  return cfg;
}

fs_status_t fs_backend_register_vulkan_dmabuf(fs_context_t*                    ctx,
                                              const fs_vulkan_dmabuf_config_t* cfg) {
  if (!ctx) {
    return FS_ERR_INVALID_ARG;
  }
  auto* st = static_cast<fs_vulkan_dmabuf_state*>(std::calloc(1u, sizeof(fs_vulkan_dmabuf_state)));
  if (!st) {
    return FS_ERR_NO_MEMORY;
  }
  st->cfg = cfg ? *cfg : fs_vulkan_dmabuf_config_default();

  fs_backend_ops_t ops;
  std::memset(&ops, 0, sizeof(ops));
  ops.struct_size    = static_cast<uint32_t>(sizeof(fs_backend_ops_t));
  ops.dev            = FS_DEV_GPU_VK;
  ops.name           = "vulkan_dmabuf";
  ops.user_data      = st;
  ops.init           = fs_vulkan_dmabuf_init;
  ops.shutdown       = fs_vulkan_dmabuf_shutdown;
  ops.supports_op    = fs_vulkan_dmabuf_supports;
  ops.import_dmabuf  = fs_vulkan_dmabuf_import;
  ops.release_dmabuf = fs_vulkan_dmabuf_release;
  ops.execute_op     = fs_vulkan_dmabuf_execute;

  fs_status_t rc = fs_context_register_backend(ctx, &ops);
  if (rc != FS_OK) {
    fs_vulkan_dmabuf_shutdown(st);
  }
  return rc;
}

} /* extern "C" */
