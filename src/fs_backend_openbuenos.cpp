/*
 * freesched optional OpenBuenos EdgeTPU backend adapter seam.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zero hard vendor or freetpu build dependencies:
 *   - Aligns 1:1 with libopenbuenos/include/openbuenos.h C11 ABI.
 *   - Dynamically probes libopenbuenos.so (dlopen) and /dev/edgetpu-soc
 *     EDGETPU_MAP_DMABUF (36-bit IOVA) on Linux.
 *   - Enforces fs_dmabuf_reconcile() upfront and fails closed on tiled layouts.
 *   - Falls back cleanly to exact host reference execution when
 *     allow_mock_fallback == 1 so macOS/Linux CI verifies full parity offline.
 */
#include "fs_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

typedef int (*fs_ob_ctx_create_fn)(const fs_ob_config*, void**);
typedef void (*fs_ob_ctx_destroy_fn)(void*);
typedef int (*fs_ob_matmul_quant_batch_fn)(void*, const float*, uint32_t, uint32_t,
                                           const fs_ob_gemm_f32_quant_job*, uint32_t,
                                           fs_ob_invoke_stats*);
typedef int (*fs_ob_swiglu_quant_fn)(void*, const float*, const fs_ob_gemm_swiglu_quant_desc*,
                                     float*, fs_ob_invoke_stats*);

struct fs_openbuenos_state {
  fs_openbuenos_config_t      cfg;
  int                         tpu_dev_fd;
  void*                       dl_handle;
  void*                       ob_ctx;
  fs_ob_ctx_destroy_fn        fn_destroy;
  fs_ob_matmul_quant_batch_fn fn_matmul_batch;
  fs_ob_swiglu_quant_fn       fn_swiglu;
  int                         hw_active;
};

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

static fs_status_t fs_openbuenos_init(void* user_data, const fs_topology_t* topo) {
  auto* st = static_cast<fs_openbuenos_state*>(user_data);
  if (!st) {
    return FS_ERR_INVALID_ARG;
  }
  const int is_mock_topo = (topo && (topo->global_caps & FS_CAP_MOCK) != 0u) ? 1 : 0;

#ifdef __linux__
  if (!is_mock_topo) {
    const char* dev_path =
        (st->cfg.device_path && st->cfg.device_path[0] != '\0') ? st->cfg.device_path : "/dev/edgetpu-soc";
    st->tpu_dev_fd = open(dev_path, O_RDWR | O_CLOEXEC);

    const char* lib_path =
        (st->cfg.library_path && st->cfg.library_path[0] != '\0') ? st->cfg.library_path : "libopenbuenos.so";
    st->dl_handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (st->dl_handle) {
      auto fn_create      = reinterpret_cast<fs_ob_ctx_create_fn>(dlsym(st->dl_handle, "ob_context_create"));
      st->fn_destroy      = reinterpret_cast<fs_ob_ctx_destroy_fn>(dlsym(st->dl_handle, "ob_context_destroy"));
      st->fn_matmul_batch = reinterpret_cast<fs_ob_matmul_quant_batch_fn>(
          dlsym(st->dl_handle, "ob_gemm_matmul_f32_quant_batch"));
      st->fn_swiglu =
          reinterpret_cast<fs_ob_swiglu_quant_fn>(dlsym(st->dl_handle, "ob_gemm_swiglu_f32_quant"));

      if (fn_create && st->fn_destroy) {
        fs_ob_config ob_cfg;
        std::memset(&ob_cfg, 0, sizeof(ob_cfg));
        ob_cfg.struct_size   = static_cast<uint32_t>(sizeof(fs_ob_config));
        ob_cfg.mock_mode     = (st->tpu_dev_fd < 0) ? 1 : 0;
        ob_cfg.device_path   = dev_path;
        ob_cfg.dma_heap_path = st->cfg.dma_heap_path;
        ob_cfg.perf_mode     = st->cfg.perf_mode;
        if (fn_create(&ob_cfg, &st->ob_ctx) == 0 && st->ob_ctx) {
          st->hw_active = (st->tpu_dev_fd >= 0) ? 1 : 0;
        }
      }
    }
  }
#else
  (void)is_mock_topo;
#endif

  if (!st->hw_active && !st->ob_ctx && !st->cfg.allow_mock_fallback) {
    return FS_ERR_NO_BACKEND;
  }
  return FS_OK;
}

static void fs_openbuenos_shutdown(void* user_data) {
  auto* st = static_cast<fs_openbuenos_state*>(user_data);
  if (!st) {
    return;
  }
  if (st->ob_ctx && st->fn_destroy) {
    st->fn_destroy(st->ob_ctx);
    st->ob_ctx = nullptr;
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
  std::free(st);
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

#ifdef __linux__
  if (st->tpu_dev_fd >= 0 &&
      (buf->backing == FS_BACKING_DMA_HEAP || buf->backing == FS_BACKING_IMPORTED) &&
      buf->dmabuf_fd >= 0) {
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
#ifdef __linux__
  if (st->tpu_dev_fd >= 0 &&
      (buf->backing == FS_BACKING_DMA_HEAP || buf->backing == FS_BACKING_IMPORTED) &&
      buf->dmabuf_fd >= 0 && buf->tpu_iova != 0u) {
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

  /* Dispatch through dynamically loaded libopenbuenos when active on supported quant ops */
  if (st->ob_ctx && op->act_dtype == FS_DTYPE_F32 &&
      (op->weight_dtype == FS_DTYPE_Q8_0 || op->weight_dtype == FS_DTYPE_Q4_0)) {
    fs_ob_invoke_stats ob_stats;
    std::memset(&ob_stats, 0, sizeof(ob_stats));
    const uint8_t is_q4 = (op->weight_dtype == FS_DTYPE_Q4_0) ? 1u : 0u;

    if ((op->type == FS_OP_DECODE_GEMV || op->type == FS_OP_GEMM || op->type == FS_OP_PREFILL_GEMM) &&
        st->fn_matmul_batch && op->input_a && op->input_b && op->output) {
      fs_ob_gemm_f32_quant_job job;
      std::memset(&job, 0, sizeof(job));
      job.b_nk       = op->input_b->host_ptr;
      job.c_mn       = static_cast<float*>(op->output->host_ptr);
      job.n_dim      = op->n_dim;
      job.activation = static_cast<uint32_t>(op->activation);
      job.is_q4_0    = is_q4;
      if (st->fn_matmul_batch(st->ob_ctx, static_cast<const float*>(op->input_a->host_ptr),
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
    } else if (op->type == FS_OP_SWIGLU && st->fn_swiglu && op->input_a && op->input_b &&
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
      if (st->fn_swiglu(st->ob_ctx, static_cast<const float*>(op->input_a->host_ptr), &desc,
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
  return cfg;
}

fs_status_t fs_backend_register_openbuenos(fs_context_t*                 ctx,
                                           const fs_openbuenos_config_t* cfg) {
  if (!ctx) {
    return FS_ERR_INVALID_ARG;
  }
  auto* st = static_cast<fs_openbuenos_state*>(std::calloc(1u, sizeof(fs_openbuenos_state)));
  if (!st) {
    return FS_ERR_NO_MEMORY;
  }
  st->cfg        = cfg ? *cfg : fs_openbuenos_config_default();
  st->tpu_dev_fd = -1;

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

} /* extern "C" */
