/*
 * ggml_freesilicon - Thin llama.cpp / ggml Backend Adapter Seam for freesched
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Clean standalone adapter header with ZERO vendored llama.cpp/ggml code.
 * Aligns 1:1 with ggml-backend conventions (modeled after ggml-edgetpu.h) so
 * llama.cpp can link libfreesched directly and dispatch heterogeneous
 * GPU (prefill/vision) + EdgeTPU (decode) + CPU/SME2 (attention/control)
 * graphs over unified zero-copy DMA-BUF memory.
 */
#ifndef GGML_FREESILICON_H_
#define GGML_FREESILICON_H_

#include "freesched.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_fs_backend ggml_fs_backend_t;

/* Mirrors GGML tensor operation codes used in transformer inference */
typedef enum ggml_fs_op {
  GGML_FS_OP_NONE     = 0,
  GGML_FS_OP_MUL_MAT  = 1,
  GGML_FS_OP_SOFT_MAX = 2,
  GGML_FS_OP_ROPE     = 3,
  GGML_FS_OP_RMS_NORM = 4,
  GGML_FS_OP_SILU     = 5,
  GGML_FS_OP_GELU     = 6,
  GGML_FS_OP_SWIGLU   = 7,
  GGML_FS_OP_IM2COL   = 8  /* Vision encoder conv/patch lowering */
} ggml_fs_op_t;

/* Thin GGML-compatible tensor descriptor backed by a freesched DMA-BUF */
typedef struct ggml_fs_tensor_desc {
  const char*  name;
  fs_dtype_t   dtype;
  int64_t      ne[4];      /* GGML column-major element dimensions [ne0, ne1, ne2, ne3] */
  size_t       nb[4];      /* Byte strides */
  fs_dmabuf_t* buf;        /* Zero-copy unified DMA-BUF backing */
} ggml_fs_tensor_desc_t;

/* Thin GGML-compatible graph node binding */
typedef struct ggml_fs_node {
  ggml_fs_op_t                 op;
  fs_phase_t                   phase_hint;
  uint32_t                     route_flags;
  fs_dev_t                     preferred_dev;
  const ggml_fs_tensor_desc_t* src0;      /* Weight / primary input */
  const ggml_fs_tensor_desc_t* src1;      /* Activation / secondary input */
  const ggml_fs_tensor_desc_t* src2;      /* Optional third input (e.g. SwiGLU W_up) */
  const ggml_fs_tensor_desc_t* src3;      /* Optional fourth input (e.g. SwiGLU W_down) */
  ggml_fs_tensor_desc_t*       dst;       /* Destination tensor */
  float                        aux_param; /* Softmax scale, RMSNorm eps, etc. */
  uint32_t                     pos_offset;
} ggml_fs_node_t;

typedef struct ggml_fs_cgraph {
  const ggml_fs_node_t* nodes;
  uint32_t              n_nodes;
} ggml_fs_cgraph_t;

/* Lifecycle (parallels ggml_backend_edgetpu_init / free) */
ggml_fs_backend_t* ggml_fs_backend_init(const fs_config_t* cfg);
void               ggml_fs_backend_free(ggml_fs_backend_t* backend);

const char*        ggml_fs_backend_name(const ggml_fs_backend_t* backend);
int                ggml_fs_backend_is_mock(const ggml_fs_backend_t* backend);
fs_context_t*      ggml_fs_backend_context(ggml_fs_backend_t* backend);

/* Unified zero-copy buffer allocator (GGML backend buffer type seam) */
fs_status_t        ggml_fs_buffer_alloc(ggml_fs_backend_t* backend,
                                        size_t             nbytes,
                                        fs_dev_t           initial_owner,
                                        uint32_t           flags,
                                        fs_dmabuf_t**      out_buf);
void               ggml_fs_buffer_free(ggml_fs_backend_t* backend, fs_dmabuf_t* buf);

/* Capability query, graph planning & execution */
int                ggml_fs_supports_node(const ggml_fs_backend_t* backend,
                                         const ggml_fs_node_t*    node);

fs_status_t        ggml_fs_graph_plan(ggml_fs_backend_t*      backend,
                                      const ggml_fs_cgraph_t* cgraph,
                                      fs_plan_t**             out_plan);

fs_status_t        ggml_fs_graph_compute(ggml_fs_backend_t*      backend,
                                         const ggml_fs_cgraph_t* cgraph,
                                         fs_plan_t**             out_plan_opt);

fs_status_t        ggml_fs_get_telemetry(const ggml_fs_backend_t* backend,
                                         fs_telemetry_t*          out_telemetry);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GGML_FREESILICON_H_ */
