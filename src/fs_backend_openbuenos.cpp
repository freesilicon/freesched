/*
 * freesched optional OpenBuenos EdgeTPU backend adapter seam.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Clean-room freetpu (libopenbuenos + FreeTile) adapter:
 *   - Supports direct static libopenbuenos linkage (FREESCHED_HAVE_LIBOPENBUENOS=1,
 *     ideal for static ARM64 binaries) as well as dynamic dlopen/dlsym lookup.
 *   - Probes /dev/edgetpu-soc + EDGETPU_MAP_DMABUF (36-bit IOVA) on Linux.
 *   - Registers the clean-room DGC0/FreeTile emitter, immutable weight regions,
 *     and upfront quantized weight prepacking (zero runtime repack/eviction loop).
 *   - Enforces fs_dmabuf_reconcile() upfront and fails closed on tiled layouts.
 *   - Falls back cleanly to exact host reference execution when
 *     allow_mock_fallback == 1 so macOS/Linux CI verifies full parity offline.
 */
#include "fs_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

#ifdef FREESCHED_HAVE_LIBOPENBUENOS
#include "openbuenos.h"
#endif

#ifdef __linux__
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* Clean-room EdgeTPU DMA-BUF ioctl ABI (36-bit IOVA) */
struct fs_edgetpu_map_dmabuf_ioctl {
  uint64_t offset;
  uint64_t size;
  uint64_t device_address;
  int32_t  dmabuf_fd;
  uint32_t flags;
  uint32_t die_index;
};
static_assert(sizeof(fs_edgetpu_map_dmabuf_ioctl) == 40u, "edgetpu_map_dmabuf_ioctl must be 40 B");

#define FS_EDGETPU_IOCTL_BASE        0xED
#define FS_EDGETPU_MAP_DMABUF        _IOWR(FS_EDGETPU_IOCTL_BASE, 17, struct fs_edgetpu_map_dmabuf_ioctl)
#define FS_EDGETPU_UNMAP_DMABUF      _IOW(FS_EDGETPU_IOCTL_BASE, 18, struct fs_edgetpu_map_dmabuf_ioctl)
#define FS_EDGETPU_MAP_BIDIRECTIONAL 0u
#define FS_EDGETPU_MAP_36BIT_IOVA    (1u << 3)
#endif /* __linux__ */

/* 1:1 ABI-compatible mirrors of libopenbuenos/include/openbuenos.h */
struct fs_ob_config {
  uint32_t    struct_size;
  int         mock_mode;
  const char* device_path;
  const char* dma_heap_path;
  uint32_t    max_inflight;
  uint32_t    graph_cache_capacity;
  uint32_t    buffer_pool_max_bytes;
  uint32_t    timeout_ms;
  uint8_t     core_power_state;
  uint8_t     mem_power_state;
  uint8_t     qos_class;
  uint8_t     cluster_id;
  uint8_t     priority;
  uint8_t     perf_mode;
  uint8_t     reserved[2];
};

struct fs_ob_invoke_stats {
  uint64_t hw_execution_us;
  uint64_t wall_time_us;
  uint64_t seq;
  uint32_t model_handle;
  uint16_t vii_resp_code;
  uint16_t flags;
  uint32_t tiles_dispatched;
  uint32_t cache_hits;
};

struct fs_ob_profile_stats {
  uint64_t quant_us;
  uint64_t hash_us;
  uint64_t weight_pack_us;
  uint64_t dma_sync_us;
  uint64_t submit_us;
  uint64_t wait_us;
  uint64_t dequant_us;
  uint64_t hw_us;
  uint64_t total_wall_us;
  uint64_t calls;
  uint64_t tiles_dispatched;
  uint64_t horiz_fusions;
  uint64_t horiz_jobs_saved;
  uint64_t sat_retries;
  uint64_t spin_hits;
  uint64_t poll_waits;
  uint64_t pool_hits;
  uint64_t pool_misses;
  uint64_t weight_cache_hits;
  uint64_t weight_cache_misses;
  uint64_t weight_cache_bytes;
  uint64_t weight_cache_peak_bytes;
  uint64_t weight_cache_budget_bytes;
  uint64_t weight_cache_evictions;
  uint64_t swiglu_calls;
  uint64_t swiglu_chunks_chained;
  uint64_t swiglu_fallbacks;
};

struct fs_ob_prepack_stats {
  uint32_t tiles_checked;
  uint32_t tiles_prepacked;
  uint32_t tiles_already_cached;
  uint32_t tiles_skipped_budget;
  uint32_t graphs_precompiled;
  uint64_t bytes_resident;
};

struct fs_ob_gemm_f32_quant_job {
  const void* b_nk;
  float*      c_mn;
  uint32_t    n_dim;
  uint32_t    activation;
  uint8_t     is_q4_0;
  uint8_t     reserved[3];
};

struct fs_ob_gemm_swiglu_quant_desc {
  const void* w_gate_hk;
  const void* w_up_hk;
  const void* w_down_nh;
  uint32_t    m_dim;
  uint32_t    k_dim;
  uint32_t    h_dim;
  uint32_t    n_dim;
  uint8_t     gate_is_q4_0;
  uint8_t     up_is_q4_0;
  uint8_t     down_is_q4_0;
  uint8_t     reserved;
};

#ifdef FREESCHED_HAVE_LIBOPENBUENOS
static_assert(sizeof(fs_ob_config) == sizeof(ob_config_t), "ob_config_t ABI mismatch");
static_assert(sizeof(fs_ob_invoke_stats) == sizeof(ob_invoke_stats_t), "ob_invoke_stats_t ABI mismatch");
static_assert(sizeof(fs_ob_profile_stats) == sizeof(ob_profile_stats_t), "ob_profile_stats_t ABI mismatch");
static_assert(sizeof(fs_ob_prepack_stats) == sizeof(ob_prepack_stats_t), "ob_prepack_stats_t ABI mismatch");
static_assert(sizeof(fs_ob_gemm_f32_quant_job) == sizeof(ob_gemm_f32_quant_job_t),
              "ob_gemm_f32_quant_job_t ABI mismatch");
static_assert(sizeof(fs_ob_gemm_swiglu_quant_desc) == sizeof(ob_gemm_swiglu_quant_desc_t),
              "ob_gemm_swiglu_quant_desc_t ABI mismatch");
#endif

struct fs_ob_api {
  int (*ctx_create)(const fs_ob_config*, void**);
  void (*ctx_destroy)(void*);
  int (*set_profile)(void*, int);
  int (*get_profile)(const void*, fs_ob_profile_stats*);
  int (*set_freetile)(void*, int);
  int (*get_freetile)(const void*);
  int (*reg_builtin_emitter)(void*, const char*);
  int (*set_tile_limits)(void*, uint32_t, uint32_t, uint32_t);
  int (*reg_immutable)(void*, const void*, size_t, uint64_t);
  int (*unreg_immutable)(void*, const void*, size_t);
  int (*prepack_batch)(void*, uint32_t, uint32_t, const fs_ob_gemm_f32_quant_job*, uint32_t, fs_ob_prepack_stats*);
  int (*matmul_batch)(void*, const float*, uint32_t, uint32_t, const fs_ob_gemm_f32_quant_job*, uint32_t, fs_ob_invoke_stats*);
  int (*swiglu)(void*, const float*, const fs_ob_gemm_swiglu_quant_desc*, float*, fs_ob_invoke_stats*);
  int (*dmabuf_map)(void*, int, uint64_t, uint64_t, uint32_t, uint64_t*);
  int (*dmabuf_unmap)(void*, uint64_t);
};

#ifdef FREESCHED_HAVE_LIBOPENBUENOS
static int fs_wrap_ob_ctx_create(const fs_ob_config* c, void** out) {
  return static_cast<int>(ob_context_create(reinterpret_cast<const ob_config_t*>(c),
                                            reinterpret_cast<ob_context_t**>(out)));
}
static void fs_wrap_ob_ctx_destroy(void* ctx) {
  ob_context_destroy(static_cast<ob_context_t*>(ctx));
}
static int fs_wrap_ob_set_profile(void* ctx, int enable) {
  return static_cast<int>(ob_context_set_profile(static_cast<ob_context_t*>(ctx), enable));
}
static int fs_wrap_ob_get_profile(const void* ctx, fs_ob_profile_stats* out) {
  return static_cast<int>(ob_context_get_profile(static_cast<const ob_context_t*>(ctx),
                                                 reinterpret_cast<ob_profile_stats_t*>(out)));
}
static int fs_wrap_ob_set_freetile(void* ctx, int enable) {
  return static_cast<int>(ob_context_set_freetile(static_cast<ob_context_t*>(ctx), enable));
}
static int fs_wrap_ob_get_freetile(const void* ctx) {
  return ob_context_get_freetile(static_cast<const ob_context_t*>(ctx));
}
static int fs_wrap_ob_reg_builtin_emitter(void* ctx, const char* path) {
  return static_cast<int>(ob_gemm_register_builtin_emitter(static_cast<ob_context_t*>(ctx), path));
}
static int fs_wrap_ob_set_tile_limits(void* ctx, uint32_t m, uint32_t k, uint32_t n) {
  return static_cast<int>(ob_gemm_set_tile_limits(static_cast<ob_context_t*>(ctx), m, k, n));
}
static int fs_wrap_ob_reg_immutable(void* ctx, const void* base, size_t size, uint64_t ver) {
  return static_cast<int>(ob_gemm_register_immutable_weights(static_cast<ob_context_t*>(ctx), base, size, ver));
}
static int fs_wrap_ob_unreg_immutable(void* ctx, const void* base, size_t size) {
  return static_cast<int>(ob_gemm_unregister_immutable_weights(static_cast<ob_context_t*>(ctx), base, size));
}
static int fs_wrap_ob_prepack_batch(void*                           ctx,
                                    uint32_t                        m_hint,
                                    uint32_t                        k_dim,
                                    const fs_ob_gemm_f32_quant_job* jobs,
                                    uint32_t                        num_jobs,
                                    fs_ob_prepack_stats*            out) {
  return static_cast<int>(ob_gemm_prepack_quant_batch(
      static_cast<ob_context_t*>(ctx), m_hint, k_dim,
      reinterpret_cast<const ob_gemm_f32_quant_job_t*>(jobs), num_jobs,
      reinterpret_cast<ob_prepack_stats_t*>(out)));
}
static int fs_wrap_ob_matmul_batch(void*                           ctx,
                                   const float*                    a_mk,
                                   uint32_t                        m_dim,
                                   uint32_t                        k_dim,
                                   const fs_ob_gemm_f32_quant_job* jobs,
                                   uint32_t                        num_jobs,
                                   fs_ob_invoke_stats*             out) {
  return static_cast<int>(ob_gemm_matmul_f32_quant_batch(
      static_cast<ob_context_t*>(ctx), a_mk, m_dim, k_dim,
      reinterpret_cast<const ob_gemm_f32_quant_job_t*>(jobs), num_jobs,
      reinterpret_cast<ob_invoke_stats_t*>(out)));
}
static int fs_wrap_ob_swiglu(void*                               ctx,
                             const float*                        a_mk,
                             const fs_ob_gemm_swiglu_quant_desc* desc,
                             float*                              c_mn,
                             fs_ob_invoke_stats*                 out) {
  return static_cast<int>(ob_gemm_swiglu_f32_quant(
      static_cast<ob_context_t*>(ctx), a_mk,
      reinterpret_cast<const ob_gemm_swiglu_quant_desc_t*>(desc), c_mn,
      reinterpret_cast<ob_invoke_stats_t*>(out)));
}
static int fs_wrap_ob_dmabuf_map(void*     ctx,
                                 int       dmabuf_fd,
                                 uint64_t  offset,
                                 uint64_t  size,
                                 uint32_t  flags,
                                 uint64_t* out_iova) {
  return static_cast<int>(
      ob_dmabuf_map(static_cast<ob_context_t*>(ctx), dmabuf_fd, offset, size, flags, out_iova));
}
static int fs_wrap_ob_dmabuf_unmap(void* ctx, uint64_t iova) {
  return static_cast<int>(ob_dmabuf_unmap(static_cast<ob_context_t*>(ctx), iova));
}
#endif /* FREESCHED_HAVE_LIBOPENBUENOS */

struct fs_ob_immutable_reg {
  const void* base;
  size_t      size;
  uint64_t    version;
};

struct fs_openbuenos_state {
  fs_openbuenos_config_t           cfg;
  int                              tpu_dev_fd;
  void*                            dl_handle;
  void*                            ob_ctx;
  fs_ob_api                        api;
  int                              hw_active;
  int                              emitter_active;
  int                              freetile_active;
  std::mutex                       mu;
  std::vector<fs_ob_immutable_reg> imm_regs;
};

static fs_openbuenos_config_t fs_normalize_ob_cfg(const fs_openbuenos_config_t* cfg) {
  fs_openbuenos_config_t out = fs_openbuenos_config_default();
  if (!cfg) {
    return out;
  }
  if (cfg->struct_size == 0u || cfg->struct_size >= sizeof(fs_openbuenos_config_t)) {
    out             = *cfg;
    out.struct_size = static_cast<uint32_t>(sizeof(fs_openbuenos_config_t));
    if (out.perf_mode == 0u) out.perf_mode = 5u;
    return out;
  }
  const size_t copy_n = std::min<size_t>(cfg->struct_size, sizeof(fs_openbuenos_config_t));
  std::memcpy(&out, cfg, copy_n);
  out.struct_size = static_cast<uint32_t>(sizeof(fs_openbuenos_config_t));
  if (out.perf_mode == 0u) out.perf_mode = 5u;
  return out;
}

static int fs_openbuenos_check_buf_compatible(const fs_dmabuf_t* buf) {
  if (!buf) {
    return 1;
  }
  if (buf->layout == FS_LAYOUT_TILED) {
    return 0;
  }
  fs_buf_attr_t pair[2];
  pair[0]                = fs_buf_attr_default(buf->owner_dev != FS_DEV_NONE ? buf->owner_dev : FS_DEV_GPU_VK);
  pair[0].layout         = buf->layout != FS_LAYOUT_ANY ? buf->layout : FS_LAYOUT_LINEAR;
  pair[0].layout_mask    = FS_LAYOUT_BIT(pair[0].layout);
  pair[0].coherency      = buf->coherency != FS_COHERENCY_ANY ? buf->coherency : FS_COHERENCY_NON_COHERENT;
  pair[0].coherency_mask = FS_COHERENCY_BIT(pair[0].coherency);
  pair[1]                = fs_buf_attr_default(FS_DEV_TPU);

  fs_buf_reconciled_t rec;
  int                 zc_ok = 0;
  if (fs_dmabuf_reconcile(pair, 2u, &rec, &zc_ok) != FS_OK) {
    return 0;
  }
  return zc_ok;
}

static void fs_ob_resolve_api(fs_openbuenos_state* st, int allow_dlopen) {
  std::memset(&st->api, 0, sizeof(st->api));
#ifdef FREESCHED_HAVE_LIBOPENBUENOS
  (void)allow_dlopen;
  st->api.ctx_create          = fs_wrap_ob_ctx_create;
  st->api.ctx_destroy         = fs_wrap_ob_ctx_destroy;
  st->api.set_profile         = fs_wrap_ob_set_profile;
  st->api.get_profile         = fs_wrap_ob_get_profile;
  st->api.set_freetile        = fs_wrap_ob_set_freetile;
  st->api.get_freetile        = fs_wrap_ob_get_freetile;
  st->api.reg_builtin_emitter = fs_wrap_ob_reg_builtin_emitter;
  st->api.set_tile_limits     = fs_wrap_ob_set_tile_limits;
  st->api.reg_immutable       = fs_wrap_ob_reg_immutable;
  st->api.unreg_immutable     = fs_wrap_ob_unreg_immutable;
  st->api.prepack_batch       = fs_wrap_ob_prepack_batch;
  st->api.matmul_batch        = fs_wrap_ob_matmul_batch;
  st->api.swiglu              = fs_wrap_ob_swiglu;
  st->api.dmabuf_map          = fs_wrap_ob_dmabuf_map;
  st->api.dmabuf_unmap        = fs_wrap_ob_dmabuf_unmap;
#elif defined(__linux__)
  void* sym_handle = nullptr;
  if (allow_dlopen) {
    const char* lib_path =
        (st->cfg.library_path && st->cfg.library_path[0] != '\0') ? st->cfg.library_path : "libopenbuenos.so";
    st->dl_handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    sym_handle    = st->dl_handle ? st->dl_handle : RTLD_DEFAULT;
  } else {
    sym_handle = RTLD_DEFAULT;
  }
  auto load_sym = [&](auto& dst, const char* name) {
    using FnT = std::remove_reference_t<decltype(dst)>;
    dst       = reinterpret_cast<FnT>(dlsym(sym_handle, name));
  };
  load_sym(st->api.ctx_create, "ob_context_create");
  load_sym(st->api.ctx_destroy, "ob_context_destroy");
  load_sym(st->api.set_profile, "ob_context_set_profile");
  load_sym(st->api.get_profile, "ob_context_get_profile");
  load_sym(st->api.set_freetile, "ob_context_set_freetile");
  load_sym(st->api.get_freetile, "ob_context_get_freetile");
  load_sym(st->api.reg_builtin_emitter, "ob_gemm_register_builtin_emitter");
  load_sym(st->api.set_tile_limits, "ob_gemm_set_tile_limits");
  load_sym(st->api.reg_immutable, "ob_gemm_register_immutable_weights");
  load_sym(st->api.unreg_immutable, "ob_gemm_unregister_immutable_weights");
  load_sym(st->api.prepack_batch, "ob_gemm_prepack_quant_batch");
  load_sym(st->api.matmul_batch, "ob_gemm_matmul_f32_quant_batch");
  load_sym(st->api.swiglu, "ob_gemm_swiglu_f32_quant");
  load_sym(st->api.dmabuf_map, "ob_dmabuf_map");
  load_sym(st->api.dmabuf_unmap, "ob_dmabuf_unmap");
#else
  (void)st;
  (void)allow_dlopen;
#endif
}

static void fs_ob_ensure_immutable_locked(fs_openbuenos_state* st, const fs_dmabuf_t* buf) {
  if (!st || !buf || !st->ob_ctx || !st->api.reg_immutable ||
      (buf->flags & FS_BUF_IMMUTABLE_WEIGHT) == 0u || !buf->host_ptr || buf->size == 0u) {
    return;
  }
  const uint64_t ver = (buf->generation > 0u) ? buf->generation : 1u;
  for (fs_ob_immutable_reg& reg : st->imm_regs) {
    if (reg.base == buf->host_ptr && reg.size == buf->size) {
      if (reg.version == ver) {
        return;
      }
      if (st->api.reg_immutable(st->ob_ctx, buf->host_ptr, buf->size, ver) == 0) {
        reg.version = ver;
      }
      return;
    }
  }
  if (st->api.reg_immutable(st->ob_ctx, buf->host_ptr, buf->size, ver) == 0) {
    st->imm_regs.push_back({buf->host_ptr, buf->size, ver});
  }
}

static void fs_ob_unreg_immutable_locked(fs_openbuenos_state* st, const fs_dmabuf_t* buf) {
  if (!st || !buf || !st->ob_ctx || !st->api.unreg_immutable || !buf->host_ptr) {
    return;
  }
  for (size_t i = 0u; i < st->imm_regs.size(); ++i) {
    if (st->imm_regs[i].base == buf->host_ptr && st->imm_regs[i].size == buf->size) {
      (void)st->api.unreg_immutable(st->ob_ctx, buf->host_ptr, buf->size);
      st->imm_regs[i] = st->imm_regs.back();
      st->imm_regs.pop_back();
      break;
    }
  }
}

static fs_status_t fs_openbuenos_init(void* user_data, const fs_topology_t* topo) {
  auto* st = static_cast<fs_openbuenos_state*>(user_data);
  if (!st) {
    return FS_ERR_INVALID_ARG;
  }

  const char* ob_mock_env  = std::getenv("OPENBUENOS_MOCK");
  const int   env_mock     = (ob_mock_env && std::strcmp(ob_mock_env, "1") == 0) ? 1 : 0;
  const int   is_mock_topo = ((topo && (topo->global_caps & FS_CAP_MOCK) != 0u) || env_mock) ? 1 : 0;

  const char* dev_path =
      (st->cfg.device_path && st->cfg.device_path[0] != '\0') ? st->cfg.device_path : "/dev/edgetpu-soc";

  fs_ob_resolve_api(st, !is_mock_topo);

  const char* dive_path =
      (st->cfg.dive_interp_path && st->cfg.dive_interp_path[0] != '\0')
          ? st->cfg.dive_interp_path
          : std::getenv("OPENBUENOS_DIVE_INTERP_PATH");
  const int has_explicit_dive = (dive_path && dive_path[0] != '\0') ? 1 : 0;

  if (st->api.ctx_create && st->api.ctx_destroy) {
    fs_ob_config ob_cfg;
    std::memset(&ob_cfg, 0, sizeof(ob_cfg));
    ob_cfg.struct_size          = static_cast<uint32_t>(sizeof(fs_ob_config));
    ob_cfg.mock_mode            = is_mock_topo ? 1 : 0;
    ob_cfg.device_path          = dev_path;
    ob_cfg.dma_heap_path        = st->cfg.dma_heap_path;
    ob_cfg.max_inflight         = 16u;
    ob_cfg.graph_cache_capacity = 2048u;
    ob_cfg.perf_mode            = st->cfg.perf_mode;

    /*
     * Let ob_context_create() own the single /dev/edgetpu-soc descriptor with
     * EDGETPU_CREATE_GROUP / shared PASID instead of opening a duplicate
     * ungrouped fd in freesched.
     */
    int ctx_rc = st->api.ctx_create(&ob_cfg, &st->ob_ctx);
    if ((ctx_rc != 0 || !st->ob_ctx) && !ob_cfg.mock_mode) {
      st->ob_ctx = nullptr;
      if (!st->cfg.allow_mock_fallback) {
        return FS_ERR_NO_BACKEND;
      }
      ob_cfg.mock_mode = 1;
      ctx_rc           = st->api.ctx_create(&ob_cfg, &st->ob_ctx);
    }

    if (ctx_rc == 0 && st->ob_ctx) {
      st->hw_active = (!ob_cfg.mock_mode) ? 1 : 0;
      if (st->api.set_profile) {
        (void)st->api.set_profile(st->ob_ctx, 1);
      }

      int ft_req = st->cfg.enable_freetile ? 1 : 0;
      if (const char* ft_env = std::getenv("OPENBUENOS_FREETILE"); ft_env && ft_env[0] != '\0') {
        ft_req = (std::atoi(ft_env) != 0) ? 1 : 0;
      }
      if (st->api.set_freetile) {
        (void)st->api.set_freetile(st->ob_ctx, ft_req);
      }
      if (st->api.get_freetile) {
        st->freetile_active = st->api.get_freetile(st->ob_ctx) ? 1 : 0;
      }
      if (st->api.set_tile_limits) {
        (void)st->api.set_tile_limits(st->ob_ctx, 64u, 2048u, 2048u);
      }

      if ((ob_cfg.mock_mode == 0 || has_explicit_dive) && st->api.reg_builtin_emitter) {
        const char* p = has_explicit_dive ? dive_path : nullptr;
        if (st->api.reg_builtin_emitter(st->ob_ctx, p) == 0) {
          st->emitter_active = 1;
        } else if (ob_cfg.mock_mode == 0 && !st->cfg.allow_mock_fallback) {
          st->api.ctx_destroy(st->ob_ctx);
          st->ob_ctx    = nullptr;
          st->hw_active = 0;
          return FS_ERR_NO_BACKEND;
        }
      }
    }
  } else {
#ifdef __linux__
    if (!is_mock_topo) {
      st->tpu_dev_fd = open(dev_path, O_RDWR | O_CLOEXEC);
      st->hw_active  = (st->tpu_dev_fd >= 0) ? 1 : 0;
    }
#else
    (void)dev_path;
#endif
  }

  if (!st->hw_active && !st->cfg.allow_mock_fallback) {
    return FS_ERR_NO_BACKEND;
  }
  if (!st->ob_ctx && !st->cfg.allow_mock_fallback) {
    return FS_ERR_NO_BACKEND;
  }
  return FS_OK;
}

static void fs_openbuenos_shutdown(void* user_data) {
  auto* st = static_cast<fs_openbuenos_state*>(user_data);
  if (!st) {
    return;
  }
  {
    std::lock_guard<std::mutex> lk(st->mu);
    if (st->ob_ctx && st->api.unreg_immutable) {
      for (const fs_ob_immutable_reg& reg : st->imm_regs) {
        (void)st->api.unreg_immutable(st->ob_ctx, reg.base, reg.size);
      }
      st->imm_regs.clear();
    }
    if (st->ob_ctx && st->api.ctx_destroy) {
      st->api.ctx_destroy(st->ob_ctx);
      st->ob_ctx = nullptr;
    }
  }
#ifdef __linux__
  if (st->tpu_dev_fd >= 0) {
    close(st->tpu_dev_fd);
    st->tpu_dev_fd = -1;
  }
  if (st->dl_handle) {
    dlclose(st->dl_handle);
    st->dl_handle = nullptr;
  }
#endif
  delete st;
}

static int fs_openbuenos_supports(void* user_data, const fs_op_desc_t* op) {
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
    case FS_OP_ELEMENTWISE:
      return ((op->route_flags & FS_ROUTE_ALLOW_SPLINE) != 0u &&
              (op->activation == FS_ACT_SILU || op->activation == FS_ACT_GELU))
                 ? 1
                 : 0;
    default:
      return 0;
  }
}

static fs_status_t fs_openbuenos_import_dmabuf(void* user_data, fs_dmabuf_t* buf) {
  auto* st = static_cast<fs_openbuenos_state*>(user_data);
  if (!st || !buf) {
    return FS_ERR_INVALID_ARG;
  }

  /* Fail closed on incompatible tiled layouts (TPU requires linear DMA-BUF) */
  if (!fs_openbuenos_check_buf_compatible(buf)) {
    return FS_ERR_UNSUPPORTED;
  }

  {
    std::lock_guard<std::mutex> lk(st->mu);
    fs_ob_ensure_immutable_locked(st, buf);
  }

  if ((buf->backing == FS_BACKING_DMA_HEAP || buf->backing == FS_BACKING_IMPORTED) &&
      buf->dmabuf_fd >= 0) {
    if (st->ob_ctx) {
      if (!st->api.dmabuf_map) {
        if (!st->cfg.allow_mock_fallback) {
          return FS_ERR_IOCTL;
        }
      } else {
        uint64_t mapped_iova = 0u;
        const int map_rc = st->api.dmabuf_map(
            st->ob_ctx, buf->dmabuf_fd, static_cast<uint64_t>(buf->offset),
            static_cast<uint64_t>(buf->size), 0u, &mapped_iova);
        if (map_rc == 0 && mapped_iova != 0u) {
          buf->tpu_iova = mapped_iova;
          return FS_OK;
        }
        if (!st->cfg.allow_mock_fallback) {
          return FS_ERR_IOCTL;
        }
      }
    }
#ifdef __linux__
    else if (st->tpu_dev_fd >= 0) {
      struct fs_edgetpu_map_dmabuf_ioctl req;
      std::memset(&req, 0, sizeof(req));
      req.offset    = static_cast<uint64_t>(buf->offset);
      req.size      = static_cast<uint64_t>(buf->size);
      req.dmabuf_fd = static_cast<int32_t>(buf->dmabuf_fd);
      req.flags     = FS_EDGETPU_MAP_BIDIRECTIONAL | FS_EDGETPU_MAP_36BIT_IOVA;
      req.die_index = 0u;
      int rc        = 0;
      do {
        rc = ioctl(st->tpu_dev_fd, FS_EDGETPU_MAP_DMABUF, &req);
      } while (rc != 0 && errno == EINTR);
      if (rc != 0) {
        if (!st->cfg.allow_mock_fallback) {
          return FS_ERR_IOCTL;
        }
      } else {
        buf->tpu_iova = req.device_address;
        return FS_OK;
      }
    }
#endif
  }

  if (buf->tpu_iova == 0u) {
    buf->tpu_iova = 0x100000000ULL + (buf->buf_id << 20);
  }
  return FS_OK;
}

static void fs_openbuenos_release_dmabuf(void* user_data, fs_dmabuf_t* buf) {
  auto* st = static_cast<fs_openbuenos_state*>(user_data);
  if (!st || !buf) {
    return;
  }
  {
    std::lock_guard<std::mutex> lk(st->mu);
    fs_ob_unreg_immutable_locked(st, buf);
  }
  if ((buf->backing == FS_BACKING_DMA_HEAP || buf->backing == FS_BACKING_IMPORTED) &&
      buf->dmabuf_fd >= 0 && buf->tpu_iova != 0u) {
    if (st->ob_ctx && st->api.dmabuf_unmap) {
      (void)st->api.dmabuf_unmap(st->ob_ctx, buf->tpu_iova);
    }
#ifdef __linux__
    else if (st->tpu_dev_fd >= 0) {
      struct fs_edgetpu_map_dmabuf_ioctl req;
      std::memset(&req, 0, sizeof(req));
      req.offset         = static_cast<uint64_t>(buf->offset);
      req.size           = static_cast<uint64_t>(buf->size);
      req.device_address = buf->tpu_iova;
      req.dmabuf_fd      = static_cast<int32_t>(buf->dmabuf_fd);
      req.flags          = FS_EDGETPU_MAP_BIDIRECTIONAL | FS_EDGETPU_MAP_36BIT_IOVA;
      (void)ioctl(st->tpu_dev_fd, FS_EDGETPU_UNMAP_DMABUF, &req);
    }
#endif
  }
  buf->tpu_iova = 0u;
}

static fs_status_t fs_openbuenos_execute(void*               user_data,
                                         const fs_op_desc_t* op,
                                         const fs_fence_t*   wait_fence,
                                         fs_fence_t*         out_signal_fence,
                                         uint64_t*           out_hw_us) {
  auto* st = static_cast<fs_openbuenos_state*>(user_data);
  if (!st || !op) {
    return FS_ERR_INVALID_ARG;
  }

  if (wait_fence) {
    fs_status_t wst = fs_fence_wait(wait_fence, 5000u);
    if (wst != FS_OK) {
      return wst;
    }
  }

  /* Fail closed if any active buffer is tiled */
  if (!fs_openbuenos_check_buf_compatible(op->input_a) ||
      !fs_openbuenos_check_buf_compatible(op->output)) {
    return FS_ERR_UNSUPPORTED;
  }

  if (st->ob_ctx && op->act_dtype == FS_DTYPE_F32 &&
      (op->weight_dtype == FS_DTYPE_Q8_0 || op->weight_dtype == FS_DTYPE_Q4_0)) {
    std::lock_guard<std::mutex> lk(st->mu);
    fs_ob_ensure_immutable_locked(st, op->input_b);
    fs_ob_ensure_immutable_locked(st, op->input_c);
    fs_ob_ensure_immutable_locked(st, op->input_d);

    fs_ob_invoke_stats ob_stats;
    std::memset(&ob_stats, 0, sizeof(ob_stats));
    const uint8_t is_q4 = (op->weight_dtype == FS_DTYPE_Q4_0) ? 1u : 0u;

    if ((op->type == FS_OP_DECODE_GEMV || op->type == FS_OP_GEMM || op->type == FS_OP_PREFILL_GEMM) &&
        st->api.matmul_batch && op->input_a && op->input_b && op->output) {
      fs_ob_gemm_f32_quant_job job;
      std::memset(&job, 0, sizeof(job));
      job.b_nk       = op->input_b->host_ptr;
      job.c_mn       = static_cast<float*>(op->output->host_ptr);
      job.n_dim      = op->n_dim;
      job.activation = static_cast<uint32_t>(op->activation);
      job.is_q4_0    = is_q4;
      if (st->api.matmul_batch(st->ob_ctx, static_cast<const float*>(op->input_a->host_ptr),
                               op->m_dim, op->k_dim, &job, 1u, &ob_stats) == 0) {
        if (out_hw_us) {
          *out_hw_us = ob_stats.hw_execution_us > 0u ? ob_stats.hw_execution_us : ob_stats.wall_time_us;
        }
        if (out_signal_fence) {
          *out_signal_fence = fs_fence_init(FS_DEV_TPU);
          fs_fence_signal(out_signal_fence, -1);
        }
        return FS_OK;
      }
      if (!st->cfg.allow_mock_fallback) {
        return FS_ERR_DEVICE;
      }
    } else if (op->type == FS_OP_SWIGLU && st->api.swiglu && op->input_a && op->input_b &&
               op->input_c && op->input_d && op->output) {
      fs_ob_gemm_swiglu_quant_desc desc;
      std::memset(&desc, 0, sizeof(desc));
      desc.w_gate_hk    = op->input_b->host_ptr;
      desc.w_up_hk      = op->input_c->host_ptr;
      desc.w_down_nh    = op->input_d->host_ptr;
      desc.m_dim        = op->m_dim;
      desc.k_dim        = op->k_dim;
      desc.h_dim        = op->h_dim;
      desc.n_dim        = op->n_dim;
      desc.gate_is_q4_0 = is_q4;
      desc.up_is_q4_0   = is_q4;
      desc.down_is_q4_0 = is_q4;
      if (st->api.swiglu(st->ob_ctx, static_cast<const float*>(op->input_a->host_ptr), &desc,
                         static_cast<float*>(op->output->host_ptr), &ob_stats) == 0) {
        if (out_hw_us) {
          *out_hw_us = ob_stats.hw_execution_us > 0u ? ob_stats.hw_execution_us : ob_stats.wall_time_us;
        }
        if (out_signal_fence) {
          *out_signal_fence = fs_fence_init(FS_DEV_TPU);
          fs_fence_signal(out_signal_fence, -1);
        }
        return FS_OK;
      }
      if (!st->cfg.allow_mock_fallback) {
        return FS_ERR_DEVICE;
      }
    }
  }

  if (!st->cfg.allow_mock_fallback) {
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
    *out_signal_fence = fs_fence_init(FS_DEV_TPU);
    fs_fence_signal(out_signal_fence, -1);
  }
  return FS_OK;
}

static fs_openbuenos_state* fs_lookup_openbuenos_state(const fs_context_t* ctx) {
  if (!ctx) {
    return nullptr;
  }
  const fs_backend_slot& slot = ctx->backends[FS_DEV_TPU];
  if (!slot.registered || !slot.ops.name ||
      std::strcmp(slot.ops.name, "openbuenos_dmabuf") != 0) {
    return nullptr;
  }
  return static_cast<fs_openbuenos_state*>(slot.ops.user_data);
}

extern "C" {

fs_openbuenos_config_t fs_openbuenos_config_default(void) {
  fs_openbuenos_config_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size         = static_cast<uint32_t>(sizeof(fs_openbuenos_config_t));
  cfg.library_path        = nullptr;
  cfg.device_path         = nullptr;
  cfg.dma_heap_path       = nullptr;
  cfg.allow_mock_fallback = 1;
  cfg.perf_mode           = 5u;
  cfg.dive_interp_path    = nullptr;
  cfg.enable_freetile     = 1;
  cfg.weight_cache_mb     = 0u;
  return cfg;
}

fs_status_t fs_backend_register_openbuenos(fs_context_t*                 ctx,
                                           const fs_openbuenos_config_t* cfg) {
  if (!ctx) {
    return FS_ERR_INVALID_ARG;
  }
  auto* st = new (std::nothrow) fs_openbuenos_state();
  if (!st) {
    return FS_ERR_NO_MEMORY;
  }
  st->cfg             = fs_normalize_ob_cfg(cfg);
  st->tpu_dev_fd      = -1;
  st->dl_handle       = nullptr;
  st->ob_ctx          = nullptr;
  st->hw_active       = 0;
  st->emitter_active  = 0;
  st->freetile_active = 0;
  std::memset(&st->api, 0, sizeof(st->api));

  fs_backend_ops_t ops;
  std::memset(&ops, 0, sizeof(ops));
  ops.struct_size    = static_cast<uint32_t>(sizeof(fs_backend_ops_t));
  ops.dev            = FS_DEV_TPU;
  ops.name           = "openbuenos_dmabuf";
  ops.user_data      = st;
  ops.init           = fs_openbuenos_init;
  ops.shutdown       = fs_openbuenos_shutdown;
  ops.supports_op    = fs_openbuenos_supports;
  ops.import_dmabuf  = fs_openbuenos_import_dmabuf;
  ops.release_dmabuf = fs_openbuenos_release_dmabuf;
  ops.execute_op     = fs_openbuenos_execute;

  fs_status_t rc = fs_context_register_backend(ctx, &ops);
  if (rc != FS_OK) {
    fs_openbuenos_shutdown(st);
  }
  return rc;
}

fs_status_t fs_backend_openbuenos_prepack_op(fs_context_t* ctx, const fs_op_desc_t* op) {
  if (!ctx || !op) {
    return FS_ERR_INVALID_ARG;
  }
  fs_openbuenos_state* st = fs_lookup_openbuenos_state(ctx);
  if (!st) {
    return FS_OK;
  }
  if ((op->type != FS_OP_DECODE_GEMV && op->type != FS_OP_GEMM && op->type != FS_OP_PREFILL_GEMM) ||
      op->act_dtype != FS_DTYPE_F32 ||
      (op->weight_dtype != FS_DTYPE_Q8_0 && op->weight_dtype != FS_DTYPE_Q4_0) ||
      !op->input_b || !op->input_b->host_ptr || op->m_dim == 0u || op->k_dim == 0u ||
      op->n_dim == 0u || (op->k_dim % FS_QK8_0) != 0u) {
    return FS_OK;
  }

  std::lock_guard<std::mutex> lk(st->mu);
  if (!st->ob_ctx) {
    return FS_OK;
  }
  fs_ob_ensure_immutable_locked(st, op->input_b);
  if (!st->emitter_active || !st->api.prepack_batch) {
    return FS_OK;
  }

  fs_ob_gemm_f32_quant_job job;
  std::memset(&job, 0, sizeof(job));
  job.b_nk       = op->input_b->host_ptr;
  job.c_mn       = nullptr;
  job.n_dim      = op->n_dim;
  job.activation = static_cast<uint32_t>(op->activation);
  job.is_q4_0    = (op->weight_dtype == FS_DTYPE_Q4_0) ? 1u : 0u;

  fs_ob_prepack_stats pstats;
  std::memset(&pstats, 0, sizeof(pstats));
  const int rc = st->api.prepack_batch(st->ob_ctx, op->m_dim, op->k_dim, &job, 1u, &pstats);
  return (rc == 0) ? FS_OK : FS_ERR_DEVICE;
}

fs_status_t fs_backend_openbuenos_get_profile(const fs_context_t*      ctx,
                                              fs_openbuenos_profile_t* out_profile) {
  if (!ctx || !out_profile) {
    return FS_ERR_INVALID_ARG;
  }
  std::memset(out_profile, 0, sizeof(*out_profile));
  fs_openbuenos_state* st = fs_lookup_openbuenos_state(ctx);
  if (!st) {
    return FS_ERR_NOT_FOUND;
  }

  std::lock_guard<std::mutex> lk(st->mu);
  out_profile->hw_active       = st->hw_active;
  out_profile->emitter_active  = st->emitter_active;
  out_profile->freetile_active = st->freetile_active;
  if (st->ob_ctx && st->api.get_profile) {
    fs_ob_profile_stats ps;
    std::memset(&ps, 0, sizeof(ps));
    if (st->api.get_profile(st->ob_ctx, &ps) == 0) {
      out_profile->calls                    = ps.calls;
      out_profile->tiles_dispatched         = ps.tiles_dispatched;
      out_profile->hw_us                    = ps.hw_us;
      out_profile->total_wall_us            = ps.total_wall_us;
      out_profile->hash_us                  = ps.hash_us;
      out_profile->weight_pack_us           = ps.weight_pack_us;
      out_profile->weight_cache_hits        = ps.weight_cache_hits;
      out_profile->weight_cache_misses      = ps.weight_cache_misses;
      out_profile->weight_cache_evictions   = ps.weight_cache_evictions;
      out_profile->weight_cache_bytes       = ps.weight_cache_bytes;
      out_profile->weight_cache_peak_bytes  = ps.weight_cache_peak_bytes;
      out_profile->weight_cache_budget_bytes = ps.weight_cache_budget_bytes;
    }
  }
  return FS_OK;
}

} /* extern "C" */
