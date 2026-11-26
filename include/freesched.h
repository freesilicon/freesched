/*
 * freesched - Layer 2 Cross-Accelerator Runtime & Zero-Copy DMA-BUF Scheduler
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Frozen Pure C11 / C++17 Public ABI Header.
 *
 * Coordinates heterogeneous unified-memory execution across:
 *   - EdgeTPU (freesilicon/freetpu -> libopenbuenos, EDGETPU_MAP_DMABUF)
 *   - Vulkan GPU (freesilicon/freepvr -> PowerVR Volcanic, VK_KHR_external_memory_fd)
 *   - ARMv9 NEON / SME2 CPU (attention, RoPE, RMSNorm, control glue)
 *
 * Dependencies stay strictly one-way: freesched depends on zero vendor libraries
 * and binds to hardware domains via pluggable backend vtables (fs_backend_ops_t).
 */
#ifndef FREESCHED_H_
#define FREESCHED_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version & Compile-Time Constants
 * ========================================================================= */

#define FS_API_VERSION_MAJOR 0u
#define FS_API_VERSION_MINOR 1u
#define FS_API_VERSION_PATCH 0u

#define FS_QK8_0             32u
#define FS_MAX_NAME_LEN      64u
#define FS_DEFAULT_PREFILL_M 32u

/* =========================================================================
 * Status Codes
 * ========================================================================= */

typedef enum fs_status {
  FS_OK              = 0,
  FS_ERR_INVALID_ARG = -1,
  FS_ERR_NO_MEMORY   = -2,
  FS_ERR_DEVICE      = -3,
  FS_ERR_IOCTL       = -4,
  FS_ERR_TIMEOUT     = -5,
  FS_ERR_OWNERSHIP   = -6,  /* Double-owner, wrong-owner, or use-after-handoff race */
  FS_ERR_BUSY        = -7,
  FS_ERR_NOT_FOUND   = -8,
  FS_ERR_UNSUPPORTED = -9,
  FS_ERR_OVERFLOW    = -10,
  FS_ERR_IO          = -11,
  FS_ERR_FENCE       = -12, /* Unsignaled or errored sync_file fence */
  FS_ERR_NO_BACKEND  = -13
} fs_status_t;

/* =========================================================================
 * Hardware Devices, Capabilities & Profiles
 * ========================================================================= */

typedef enum fs_dev {
  FS_DEV_NONE     = 0,
  FS_DEV_TPU      = 1, /* EdgeTPU (libopenbuenos / buenos_a0 / malibu) */
  FS_DEV_GPU_VK   = 2, /* Vulkan GPU (PowerVR Volcanic DXT / CXTP) */
  FS_DEV_CPU_NEON = 3, /* ARMv9 NEON (or portable host CPU fallback) */
  FS_DEV_CPU_SME2 = 4, /* ARMv9 SME2 matrix-tile CPU extension */
  FS_DEV_COUNT    = 5
} fs_dev_t;

/* Hardware & transport capability bitmask */
enum fs_cap_flags {
  FS_CAP_UNIFIED_MEMORY     = (1u << 0),
  FS_CAP_DMABUF             = (1u << 1),
  FS_CAP_VK_EXT_MEMORY_FD   = (1u << 2), /* VK_KHR_external_memory_fd */
  FS_CAP_EDGETPU_MAP_DMABUF = (1u << 3), /* EDGETPU_MAP_DMABUF (36-bit IOVA) */
  FS_CAP_SYNC_FILE          = (1u << 4), /* Linux sync_file explicit fences */
  FS_CAP_TPU_SPLINE_0X640   = (1u << 5), /* Mode-A 0x640 degree-4 Horner SiLU/GELU */
  FS_CAP_TPU_SWIGLU_CHAIN   = (1u << 6), /* In-SRAM chained SwiGLU MLP */
  FS_CAP_FP16               = (1u << 7),
  FS_CAP_INT8               = (1u << 8),
  FS_CAP_Q8_0               = (1u << 9),
  FS_CAP_Q4_0               = (1u << 10),
  FS_CAP_SME2_MATMUL        = (1u << 11),
  FS_CAP_MOCK               = (1u << 12)
};

typedef enum fs_hw_profile {
  FS_PROFILE_AUTO           = 0,
  FS_PROFILE_HOST_MOCK      = 1,
  FS_PROFILE_PIXEL10_LAGUNA = 2, /* Tensor G5 laguna: BUENOS_A0 + DXT-48-1536 + NEON */
  FS_PROFILE_PIXEL11_MALIBU = 3  /* Tensor G6 malibu: malibu TPU + CXTP-48-1536 + SME2 */
} fs_hw_profile_t;

typedef struct fs_dev_info {
  fs_dev_t dev;
  int      available;           /* 1 = online and schedulable, 0 = offline/disabled */
  uint32_t caps;                /* Bitmask of enum fs_cap_flags */
  char     name[FS_MAX_NAME_LEN];
  float    peak_gflops_fp16;
  float    peak_tops_int8;
  float    mem_bw_gbps;
  float    launch_overhead_us;
  float    handoff_overhead_us; /* Expected sync_file / cache-sync handoff latency */
  float    typical_watts;
  uint32_t sram_kb;
  uint32_t max_inflight;        /* Matches OB_MAX_CREDITS (16) on EdgeTPU */
} fs_dev_info_t;

typedef struct fs_topology {
  uint32_t        struct_size;
  fs_hw_profile_t profile;
  char            soc_name[FS_MAX_NAME_LEN];
  uint64_t        unified_ram_bytes;
  uint32_t        global_caps;         /* Bitmask of enum fs_cap_flags */
  uint32_t        prefill_m_threshold; /* M >= threshold routes PREFILL_GEMM to GPU */
  fs_dev_info_t   devices[FS_DEV_COUNT];
} fs_topology_t;

/* =========================================================================
 * Quantization Blocks & Data Types (1:1 ABI alignment with openbuenos & ggml)
 * ========================================================================= */

typedef enum fs_dtype {
  FS_DTYPE_F32  = 0,
  FS_DTYPE_F16  = 1,
  FS_DTYPE_I8   = 2,
  FS_DTYPE_Q8_0 = 3,
  FS_DTYPE_Q4_0 = 4,
  FS_DTYPE_I32  = 5
} fs_dtype_t;

/* Matches enum ob_activation in openbuenos.h */
typedef enum fs_activation {
  FS_ACT_NONE  = 0,
  FS_ACT_RELU  = 1,
  FS_ACT_RELU6 = 3,
  FS_ACT_SILU  = 4,
  FS_ACT_GELU  = 5
} fs_activation_t;

/* Matches ob_block_q8_0_t (34 B) and ob_block_q4_0_t (18 B) byte-for-byte */
#if defined(__GNUC__) || defined(__clang__)
typedef struct __attribute__((packed)) fs_block_q8_0 {
  uint16_t d;
  int8_t   qs[FS_QK8_0];
} fs_block_q8_0_t;

typedef struct __attribute__((packed)) fs_block_q4_0 {
  uint16_t d;
  uint8_t  qs[FS_QK8_0 / 2];
} fs_block_q4_0_t;
#else
#pragma pack(push, 1)
typedef struct fs_block_q8_0 {
  uint16_t d;
  int8_t   qs[FS_QK8_0];
} fs_block_q8_0_t;

typedef struct fs_block_q4_0 {
  uint16_t d;
  uint8_t  qs[FS_QK8_0 / 2];
} fs_block_q4_0_t;
#pragma pack(pop)
#endif

/* =========================================================================
 * Zero-Copy DMA-BUF Buffer & Explicit Sync Fence Primitives
 * ========================================================================= */

enum fs_buf_flags {
  FS_BUF_BIDIRECTIONAL    = 0u,
  FS_BUF_ZERO_INIT        = (1u << 0),
  FS_BUF_IMMUTABLE_WEIGHT = (1u << 1), /* Shared read-only weight; concurrent reads allowed */
  FS_BUF_NO_POOL          = (1u << 2),
  FS_BUF_FORCE_HOST_COPY  = (1u << 3)  /* Diagnostic flag to test fallback copy path */
};

/* NvSciBuf-style upfront buffer attribute reconciliation primitives */
typedef enum fs_buf_layout {
  FS_LAYOUT_ANY    = 0, /* Engine accepts linear or tiled */
  FS_LAYOUT_LINEAR = 1, /* Pitch-linear row-major */
  FS_LAYOUT_TILED  = 2  /* Hardware-tiled (e.g. PowerVR PBE twiddled/FBCDC) */
} fs_buf_layout_t;

typedef enum fs_buf_coherency {
  FS_COHERENCY_ANY          = 0,
  FS_COHERENCY_COHERENT     = 1, /* Hardware cache-coherent */
  FS_COHERENCY_NON_COHERENT = 2  /* Explicit DMA_BUF_IOCTL_SYNC required */
} fs_buf_coherency_t;

enum fs_buf_access_flags {
  FS_ACCESS_NONE       = 0u,
  FS_ACCESS_READ       = (1u << 0),
  FS_ACCESS_WRITE      = (1u << 1),
  FS_ACCESS_READ_WRITE = (FS_ACCESS_READ | FS_ACCESS_WRITE)
};

#define FS_LAYOUT_BIT(layout_enum)       (1u << (uint32_t)(layout_enum))
#define FS_COHERENCY_BIT(coherency_enum) (1u << (uint32_t)(coherency_enum))

typedef struct fs_buf_attr {
  fs_dev_t           dev;
  size_t             min_align;      /* Minimum base alignment in bytes (0 -> 1) */
  size_t             stride_align;   /* Row stride alignment in bytes (0 -> 1) */
  fs_buf_layout_t    layout;         /* Preferred/required layout (or FS_LAYOUT_ANY) */
  fs_buf_coherency_t coherency;      /* Preferred/required coherency (or FS_COHERENCY_ANY) */
  uint32_t           access;         /* Bitmask of enum fs_buf_access_flags */
  uint32_t           layout_mask;    /* Optional bitmask of FS_LAYOUT_BIT(...) accepted */
  uint32_t           coherency_mask; /* Optional bitmask of FS_COHERENCY_BIT(...) accepted */
} fs_buf_attr_t;

typedef struct fs_buf_reconciled {
  size_t             min_align;            /* Reconciled LCM base alignment in bytes */
  size_t             stride_align;         /* Reconciled LCM row stride alignment in bytes */
  fs_buf_layout_t    layout;               /* Reconciled layout (FS_LAYOUT_ANY on conflict) */
  fs_buf_coherency_t coherency;            /* Reconciled coherency mode */
  uint32_t           combined_access;      /* Union of participant access flags */
  uint32_t           participant_dev_mask; /* Bitmask of (1u << dev) participants */
  int                requires_dma_sync;    /* 1 if explicit DMA_BUF_IOCTL_SYNC is needed */
  int                zero_copy_compatible; /* 1 if valid zero-copy intersection exists */
  char               conflict_reason[FS_MAX_NAME_LEN];
} fs_buf_reconciled_t;

typedef enum fs_fence_state {
  FS_FENCE_UNSIGNALED = 0,
  FS_FENCE_SIGNALED   = 1,
  FS_FENCE_ERROR      = 2
} fs_fence_state_t;

typedef struct fs_fence {
  int              sync_file_fd;   /* Linux sync_file fd, or -1 in mock/host mode */
  uint64_t         seq;            /* Monotonic fence sequence number */
  fs_dev_t         producer_dev;   /* Device that emits this fence */
  fs_fence_state_t state;
  uint64_t         signaled_at_us;
} fs_fence_t;

/*
 * Zero-copy unified-memory DMA-BUF descriptor.
 *
 * Models Linux DMA_BUF_IOCTL_SYNC + Vulkan VK_KHR_external_memory_fd +
 * EdgeTPU EDGETPU_MAP_DMABUF with strict single-owner handoff tracking.
 */
typedef struct fs_dmabuf {
  uint64_t buf_id;          /* Unique non-zero buffer identifier */
  int      dmabuf_fd;       /* Linux /dev/dma_heap fd, or -1 in host mock mode */
  size_t   size;            /* Requested usable byte size */
  size_t   offset;          /* Byte offset into backing allocation */
  fs_dev_t owner_dev;       /* Current exclusive owner device (FS_DEV_NONE if freed) */
  int      sync_file_fd;    /* Latest attached explicit sync_file fd (-1 if none) */
  uint32_t flags;           /* Bitmask of enum fs_buf_flags */
  uint32_t refcount;        /* Current reference count */
  uint64_t generation;      /* Bumped on every fs_buffer_handoff (stale-handle guard) */
  uint64_t tpu_iova;        /* Mapped EdgeTPU 36-bit IOVA (0 if unmapped) */
  uint64_t           vk_device_addr;       /* Imported Vulkan device address (0 if unmapped) */
  void*              host_ptr;             /* Host mmap pointer for CPU/mock access */
  uint64_t           last_handoff_us;      /* Timestamp of last ownership transfer */
  fs_buf_layout_t    layout;               /* Reconciled buffer layout (linear/tiled) */
  fs_buf_coherency_t coherency;            /* Reconciled coherency mode */
  size_t             min_align;            /* Reconciled base alignment in bytes */
  size_t             stride_align;         /* Reconciled row stride alignment in bytes */
  int                zero_copy_compatible; /* 1 if reconciled for zero-copy sharing */
} fs_dmabuf_t;

/* =========================================================================
 * Cost-Model Router: Operation Descriptors & Execution Plans
 * ========================================================================= */

typedef enum fs_op_type {
  FS_OP_GEMM         = 0, /* Generic matrix multiply C[M,N] = A[M,K] @ B[N,K]^T */
  FS_OP_PREFILL_GEMM = 1, /* Prompt prefill projection (routes to GPU when M >= threshold) */
  FS_OP_DECODE_GEMV  = 2, /* Low-watt token decode M=1..small (routes to EdgeTPU) */
  FS_OP_SWIGLU       = 3, /* Fused MLP: (SiLU(A@Wg^T) * (A@Wu^T)) @ Wd^T */
  FS_OP_ATTN_SOFTMAX = 4, /* Attention score scaling + row-wise softmax (CPU/SME2) */
  FS_OP_ROPE         = 5, /* Rotary positional embedding (CPU/SME2) */
  FS_OP_RMS_NORM     = 6, /* Root-mean-square layer norm (CPU/SME2) */
  FS_OP_VISION_CONV  = 7, /* Vision encoder patch/conv GEMM (GPU) */
  FS_OP_ELEMENTWISE  = 8  /* Small control / residual add / spline activation */
} fs_op_type_t;

typedef enum fs_phase {
  FS_PHASE_AUTO    = 0,
  FS_PHASE_PREFILL = 1,
  FS_PHASE_DECODE  = 2,
  FS_PHASE_VISION  = 3,
  FS_PHASE_CONTROL = 4
} fs_phase_t;

enum fs_route_flags {
  FS_ROUTE_AUTO                = 0u,
  FS_ROUTE_FORCE_DEVICE        = (1u << 0), /* Force op->preferred_dev (fails if offline) */
  FS_ROUTE_PREFER_LOW_POWER    = (1u << 1), /* Bias cost model toward minimal millijoules */
  FS_ROUTE_NO_CONTINUITY_BONUS = (1u << 2), /* Disable zero-copy anti-ping-pong bonus */
  FS_ROUTE_ALLOW_SPLINE        = (1u << 3)  /* Allow TPU 0x640 spline fusion */
};

typedef struct fs_op_desc {
  uint32_t        op_id;
  fs_op_type_t    type;
  fs_phase_t      phase;
  uint32_t        m_dim;        /* Sequence / batch rows M */
  uint32_t        k_dim;        /* Inner / hidden dimension K */
  uint32_t        n_dim;        /* Output dimension N (or sequence length for softmax) */
  uint32_t        h_dim;        /* Intermediate MLP dim (SwiGLU) or head dim (RoPE) */
  fs_dtype_t      act_dtype;    /* Activation tensor dtype (F32 or F16) */
  fs_dtype_t      weight_dtype; /* Weight tensor dtype (F32, F16, Q8_0, Q4_0, I8) */
  fs_activation_t activation;   /* Fused or standalone activation */
  uint32_t        route_flags;  /* Bitmask of enum fs_route_flags */
  fs_dev_t        preferred_dev;/* Explicit override or tie-break hint */
  fs_dmabuf_t*    input_a;      /* Primary activation input */
  fs_dmabuf_t*    input_b;      /* Weights (B or W_gate) or secondary input */
  fs_dmabuf_t*    input_c;      /* Optional: W_up (SwiGLU) or residual/weight */
  fs_dmabuf_t*    input_d;      /* Optional: W_down (SwiGLU) */
  fs_dmabuf_t*    output;       /* Destination buffer */
  float           aux_param;        /* RMSNorm eps, softmax scale, RoPE base, etc. */
  uint32_t        pos_offset;       /* Token position offset (RoPE) */
  const char*     label;            /* Optional human-readable op label */
  fs_buf_layout_t input_layout_req; /* Optional layout required on input_a (0 -> dev default) */
  fs_buf_layout_t output_layout;    /* Optional layout emitted on output (0 -> linear) */
} fs_op_desc_t;

typedef struct fs_plan_step {
  uint32_t op_index;
  fs_dev_t assigned_dev;
  fs_dev_t prev_input_owner;
  int      requires_handoff;
  int      zero_copy;
  int      degraded;            /* 1 if primary accelerator was offline */
  float    est_compute_us;
  float    est_handoff_us;
  float    est_total_us;
  float    est_energy_mj;
  char     reason[FS_MAX_NAME_LEN];
} fs_plan_step_t;

typedef struct fs_plan {
  uint32_t        num_steps;
  fs_plan_step_t* steps;
  float           total_est_us;
  float           total_est_energy_mj;
  uint32_t        planned_zero_copy_handoffs;
  uint32_t        planned_fallback_copies;
  uint32_t        degraded_ops;
} fs_plan_t;

/* =========================================================================
 * Telemetry & Numerical Parity Metrics
 * ========================================================================= */

typedef struct fs_telemetry {
  uint64_t tpu_us;
  uint64_t gpu_us;
  uint64_t cpu_us;
  uint64_t handoff_us;
  uint64_t total_wall_us;
  uint64_t zero_copy_handoffs;
  uint64_t fallback_copies;
  uint64_t tpu_ops;
  uint64_t gpu_ops;
  uint64_t cpu_ops;
  float    est_watts;
  float    total_energy_mj;
} fs_telemetry_t;

typedef struct fs_parity {
  float    max_abs_err;
  double   nmse;
  double   rmse;
  uint64_t count;
  int      exact_match; /* 1 if max_abs_err == 0.0f */
} fs_parity_t;

/* =========================================================================
 * Context Configuration & Pluggable Backend Vtable
 * ========================================================================= */

typedef struct fs_context fs_context_t;

typedef struct fs_config {
  uint32_t        struct_size;           /* Optional: sizeof(fs_config_t) or 0 */
  int             mock_mode;             /* 1 = host mock, 0 = auto/probe */
  fs_hw_profile_t profile;               /* FS_PROFILE_AUTO or explicit SoC profile */
  const char*     dma_heap_path;         /* NULL -> "/dev/dma_heap/system" */
  uint32_t        prefill_m_threshold;   /* 0 -> profile default (32) */
  float           continuity_bonus_us;   /* <= 0 -> default 45.0 us */
  uint32_t        buffer_pool_max_bytes; /* 0 -> default 128 MiB */
  int             enable_tpu;            /* -1 = profile default, 0 = disable, 1 = enable */
  int             enable_gpu;            /* -1 = profile default, 0 = disable, 1 = enable */
  int             enable_sme2;           /* -1 = profile default, 0 = disable, 1 = enable */
} fs_config_t;

/*
 * Pluggable accelerator backend vtable.
 *
 * Allows freesilicon/freetpu (libopenbuenos) and freesilicon/freepvr (Vulkan)
 * to plug into freesched without introducing circular build dependencies.
 */
typedef struct fs_backend_ops {
  uint32_t    struct_size;
  fs_dev_t    dev;
  const char* name;
  void*       user_data;

  fs_status_t (*init)(void* user_data, const fs_topology_t* topo);
  void        (*shutdown)(void* user_data);
  int         (*supports_op)(void* user_data, const fs_op_desc_t* op);
  fs_status_t (*import_dmabuf)(void* user_data, fs_dmabuf_t* buf);
  void        (*release_dmabuf)(void* user_data, fs_dmabuf_t* buf);
  fs_status_t (*execute_op)(void*               user_data,
                            const fs_op_desc_t* op,
                            const fs_fence_t*   wait_fence,
                            fs_fence_t*         out_signal_fence,
                            uint64_t*           out_hw_us);
} fs_backend_ops_t;

/* =========================================================================
 * Topology & Diagnostics API
 * ========================================================================= */

fs_status_t fs_topology_probe(fs_hw_profile_t profile_hint, fs_topology_t* out_topo);
fs_status_t fs_topology_set_device_enabled(fs_topology_t* topo, fs_dev_t dev, int enabled);
int         fs_topology_has_device(const fs_topology_t* topo, fs_dev_t dev);

const char* fs_status_string(fs_status_t status);
const char* fs_dev_name(fs_dev_t dev);
const char* fs_profile_name(fs_hw_profile_t profile);
const char* fs_op_type_name(fs_op_type_t type);

/* =========================================================================
 * Context Lifecycle & Backend Registration API
 * ========================================================================= */

fs_config_t            fs_config_default(void);
fs_status_t            fs_context_create(const fs_config_t* cfg, fs_context_t** out_ctx);
void                   fs_context_destroy(fs_context_t* ctx);
const fs_topology_t*   fs_context_topology(const fs_context_t* ctx);
const char*            fs_context_last_error(const fs_context_t* ctx);
int                    fs_context_is_mock(const fs_context_t* ctx);
fs_status_t            fs_context_set_device_enabled(fs_context_t* ctx,
                                                     fs_dev_t      dev,
                                                     int           enabled);
fs_status_t            fs_context_register_backend(fs_context_t*           ctx,
                                                   const fs_backend_ops_t* ops);

/* =========================================================================
 * Zero-Copy DMA-BUF Buffer Pool, Reconciliation & Ownership Handoff API
 * ========================================================================= */

fs_buf_attr_t fs_buf_attr_default(fs_dev_t dev);

/*
 * NvSciBuf-style upfront multi-engine buffer attribute reconciliation.
 *
 * Computes the intersection of layout/coherency constraints and LCM alignment
 * across `count` participating engines before allocation or scheduling.
 * Returns FS_OK even on incompatibility, setting `*out_zero_copy_compatible = 0`
 * and populating `out_reconciled->conflict_reason` so routers can price
 * layout-conversion / copy costs upfront.
 */
fs_status_t fs_dmabuf_reconcile(const fs_buf_attr_t* attrs,
                                uint32_t             count,
                                fs_buf_reconciled_t* out_reconciled,
                                int*                 out_zero_copy_compatible);

fs_status_t fs_buffer_alloc_reconciled(fs_context_t*        ctx,
                                       size_t               size_bytes,
                                       const fs_buf_attr_t* attrs,
                                       uint32_t             attr_count,
                                       fs_dev_t             initial_owner,
                                       uint32_t             flags,
                                       fs_dmabuf_t**        out_buf,
                                       fs_buf_reconciled_t* out_reconciled);

fs_status_t fs_buffer_alloc(fs_context_t* ctx,
                            size_t        size_bytes,
                            fs_dev_t      initial_owner,
                            uint32_t      flags,
                            fs_dmabuf_t** out_buf);

fs_status_t fs_buffer_import_dmabuf(fs_context_t* ctx,
                                    int           dmabuf_fd,
                                    size_t        size_bytes,
                                    size_t        offset,
                                    void*         host_ptr,
                                    fs_dev_t      initial_owner,
                                    uint32_t      flags,
                                    fs_dmabuf_t** out_buf);

fs_status_t fs_buffer_retain(fs_context_t* ctx, fs_dmabuf_t* buf);
void        fs_buffer_release(fs_context_t* ctx, fs_dmabuf_t* buf);

/*
 * Explicit zero-copy ownership transfer between accelerators.
 *
 * Enforces single-owner invariants:
 *   - Returns FS_ERR_OWNERSHIP if buf->owner_dev != from_dev (use-after-handoff).
 *   - Returns FS_ERR_OWNERSHIP if buf is currently locked by an active accessor.
 *   - Validates/signals io_fence (if non-NULL) and updates buf->sync_file_fd.
 *   - Bumps buf->generation on every transfer to invalidate stale handles.
 *   - On Linux with real DMA-BUF, invokes DMA_BUF_IOCTL_SYNC when CPU participates
 *     and transfers fd + sync_file with 0 host copies between GPU and TPU.
 */
fs_status_t fs_buffer_handoff(fs_context_t* ctx,
                              fs_dmabuf_t*  buf,
                              fs_dev_t      from_dev,
                              fs_dev_t      to_dev,
                              fs_fence_t*   io_fence);

/*
 * Scoped device access lock (detects double-owner & stale-generation races).
 * Pass expected_gen == 0 to skip generation pin check.
 */
fs_status_t fs_buffer_acquire_access(fs_context_t* ctx,
                                     fs_dmabuf_t*  buf,
                                     fs_dev_t      dev,
                                     uint64_t      expected_gen);

fs_status_t fs_buffer_release_access(fs_context_t* ctx,
                                     fs_dmabuf_t*  buf,
                                     fs_dev_t      dev);

fs_status_t fs_buffer_pool_stats(const fs_context_t* ctx,
                                 size_t*             out_pooled_bytes,
                                 size_t*             out_pooled_count,
                                 uint64_t*           out_pool_hits,
                                 uint64_t*           out_pool_misses);

/* Fence helpers */
fs_fence_t  fs_fence_init(fs_dev_t producer_dev);
fs_status_t fs_fence_signal(fs_fence_t* fence, int sync_file_fd);
fs_status_t fs_fence_wait(const fs_fence_t* fence, uint32_t timeout_ms);

/* =========================================================================
 * Scheduler Plan, Execution & Telemetry API
 * ========================================================================= */

fs_status_t fs_plan_create(fs_context_t*       ctx,
                           const fs_op_desc_t* ops,
                           uint32_t            num_ops,
                           fs_plan_t**         out_plan);

void        fs_plan_destroy(fs_plan_t* plan);

fs_status_t fs_execute_plan(fs_context_t*       ctx,
                            const fs_plan_t*    plan,
                            const fs_op_desc_t* ops,
                            uint32_t            num_ops);

fs_status_t fs_execute_op(fs_context_t*       ctx,
                          const fs_op_desc_t* op,
                          fs_plan_step_t*     out_step);

fs_status_t fs_context_get_telemetry(const fs_context_t* ctx,
                                     fs_telemetry_t*     out_telemetry);

fs_status_t fs_context_reset_telemetry(fs_context_t* ctx);

/* Numerical parity helper for host verification (max_abs_err, nmse, rmse) */
fs_status_t fs_compute_parity_f32(const float* actual,
                                  const float* reference,
                                  size_t       count,
                                  fs_parity_t* out_parity);

/* IEEE-754 FP16 conversion helpers for portable F16 verification */
uint16_t    fs_f32_to_f16(float value);
float       fs_f16_to_f32(uint16_t bits);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FREESCHED_H_ */
