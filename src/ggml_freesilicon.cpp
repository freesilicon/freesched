/*
 * ggml_freesilicon - Thin llama.cpp / ggml backend adapter over freesched.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ggml_freesilicon.h"

#include <cstdlib>
#include <cstring>
#include <vector>

struct ggml_fs_backend {
  fs_context_t* ctx;
};

static fs_status_t ggml_fs_node_to_op(const ggml_fs_backend_t* backend,
                                      uint32_t                 op_id,
                                      const ggml_fs_node_t*    node,
                                      fs_op_desc_t*            out_op) {
  if (!backend || !node || !out_op || !node->dst) {
    return FS_ERR_INVALID_ARG;
  }
  std::memset(out_op, 0, sizeof(*out_op));
  out_op->op_id         = op_id;
  out_op->phase         = node->phase_hint;
  out_op->route_flags   = node->route_flags;
  out_op->preferred_dev = node->preferred_dev;
  out_op->aux_param     = node->aux_param;
  out_op->pos_offset    = node->pos_offset;
  out_op->output        = node->dst->buf;
  out_op->act_dtype     = node->dst->dtype;
  out_op->label         = node->dst->name;

  const fs_topology_t* topo = fs_context_topology(backend->ctx);
  const uint32_t       prefill_thresh =
      (topo && topo->prefill_m_threshold > 0u) ? topo->prefill_m_threshold : FS_DEFAULT_PREFILL_M;

  switch (node->op) {
    case GGML_FS_OP_MUL_MAT: {
      /* GGML convention: src0 is W[K, N] (ne0=K, ne1=N), src1 is X[K, M] (ne0=K, ne1=M) */
      if (!node->src0 || !node->src1) {
        return FS_ERR_INVALID_ARG;
      }
      const uint32_t k_dim = static_cast<uint32_t>(node->src0->ne[0]);
      const uint32_t n_dim = static_cast<uint32_t>(node->src0->ne[1]);
      const uint32_t m_dim = static_cast<uint32_t>(node->src1->ne[1]);
      out_op->m_dim        = m_dim;
      out_op->k_dim        = k_dim;
      out_op->n_dim        = n_dim;
      out_op->act_dtype    = node->src1->dtype;
      out_op->weight_dtype = node->src0->dtype;
      out_op->input_a      = node->src1->buf;
      out_op->input_b      = node->src0->buf;
      if (node->phase_hint == FS_PHASE_PREFILL || m_dim >= prefill_thresh) {
        out_op->type = FS_OP_PREFILL_GEMM;
      } else {
        out_op->type = FS_OP_DECODE_GEMV;
      }
      return FS_OK;
    }

    case GGML_FS_OP_SWIGLU: {
      if (!node->src0 || !node->src1 || !node->src2 || !node->src3) {
        return FS_ERR_INVALID_ARG;
      }
      out_op->type         = FS_OP_SWIGLU;
      out_op->m_dim        = static_cast<uint32_t>(node->src0->ne[1]);
      out_op->k_dim        = static_cast<uint32_t>(node->src0->ne[0]);
      out_op->h_dim        = static_cast<uint32_t>(node->src1->ne[1]);
      out_op->n_dim        = static_cast<uint32_t>(node->src3->ne[1]);
      out_op->act_dtype    = node->src0->dtype;
      out_op->weight_dtype = node->src1->dtype;
      out_op->input_a      = node->src0->buf; /* X */
      out_op->input_b      = node->src1->buf; /* W_gate */
      out_op->input_c      = node->src2->buf; /* W_up */
      out_op->input_d      = node->src3->buf; /* W_down */
      return FS_OK;
    }

    case GGML_FS_OP_SOFT_MAX: {
      if (!node->src0) return FS_ERR_INVALID_ARG;
      out_op->type      = FS_OP_ATTN_SOFTMAX;
      out_op->n_dim     = static_cast<uint32_t>(node->src0->ne[0]);
      out_op->m_dim     = static_cast<uint32_t>(node->src0->ne[1] > 0 ? node->src0->ne[1] : 1);
      out_op->act_dtype = node->src0->dtype;
      out_op->input_a   = node->src0->buf;
      return FS_OK;
    }

    case GGML_FS_OP_ROPE: {
      if (!node->src0) return FS_ERR_INVALID_ARG;
      out_op->type      = FS_OP_ROPE;
      out_op->n_dim     = static_cast<uint32_t>(node->src0->ne[0]);
      out_op->m_dim     = static_cast<uint32_t>(node->src0->ne[1] > 0 ? node->src0->ne[1] : 1);
      out_op->h_dim     = static_cast<uint32_t>(node->src0->ne[2] > 0 ? node->src0->ne[2] : out_op->n_dim);
      out_op->act_dtype = node->src0->dtype;
      out_op->input_a   = node->src0->buf;
      return FS_OK;
    }

    case GGML_FS_OP_RMS_NORM: {
      if (!node->src0) return FS_ERR_INVALID_ARG;
      out_op->type         = FS_OP_RMS_NORM;
      out_op->n_dim        = static_cast<uint32_t>(node->src0->ne[0]);
      out_op->m_dim        = static_cast<uint32_t>(node->src0->ne[1] > 0 ? node->src0->ne[1] : 1);
      out_op->act_dtype    = node->src0->dtype;
      out_op->weight_dtype = node->src1 ? node->src1->dtype : FS_DTYPE_F32;
      out_op->input_a      = node->src0->buf;
      out_op->input_b      = node->src1 ? node->src1->buf : nullptr;
      return FS_OK;
    }

    case GGML_FS_OP_SILU:
    case GGML_FS_OP_GELU: {
      if (!node->src0) return FS_ERR_INVALID_ARG;
      out_op->type       = FS_OP_ELEMENTWISE;
      out_op->n_dim      = static_cast<uint32_t>(node->src0->ne[0]);
      out_op->m_dim      = static_cast<uint32_t>(node->src0->ne[1] > 0 ? node->src0->ne[1] : 1);
      out_op->act_dtype  = node->src0->dtype;
      out_op->activation = (node->op == GGML_FS_OP_SILU) ? FS_ACT_SILU : FS_ACT_GELU;
      out_op->input_a    = node->src0->buf;
      return FS_OK;
    }

    case GGML_FS_OP_IM2COL: {
      if (!node->src0 || !node->src1) return FS_ERR_INVALID_ARG;
      out_op->type         = FS_OP_VISION_CONV;
      out_op->phase        = FS_PHASE_VISION;
      out_op->k_dim        = static_cast<uint32_t>(node->src0->ne[0]);
      out_op->n_dim        = static_cast<uint32_t>(node->src0->ne[1]);
      out_op->m_dim        = static_cast<uint32_t>(node->src1->ne[1]);
      out_op->act_dtype    = node->src1->dtype;
      out_op->weight_dtype = node->src0->dtype;
      out_op->input_a      = node->src1->buf;
      out_op->input_b      = node->src0->buf;
      return FS_OK;
    }

    case GGML_FS_OP_NONE:
    default:
      return FS_ERR_UNSUPPORTED;
  }
}

extern "C" {

ggml_fs_backend_t* ggml_fs_backend_init(const fs_config_t* cfg) {
  auto* backend = static_cast<ggml_fs_backend_t*>(std::calloc(1u, sizeof(ggml_fs_backend_t)));
  if (!backend) {
    return nullptr;
  }
  if (fs_context_create(cfg, &backend->ctx) != FS_OK) {
    std::free(backend);
    return nullptr;
  }
  return backend;
}

void ggml_fs_backend_free(ggml_fs_backend_t* backend) {
  if (!backend) {
    return;
  }
  fs_context_destroy(backend->ctx);
  std::free(backend);
}

const char* ggml_fs_backend_name(const ggml_fs_backend_t* backend) {
  (void)backend;
  return "FreeSilicon";
}

int ggml_fs_backend_is_mock(const ggml_fs_backend_t* backend) {
  return backend ? fs_context_is_mock(backend->ctx) : 0;
}

fs_context_t* ggml_fs_backend_context(ggml_fs_backend_t* backend) {
  return backend ? backend->ctx : nullptr;
}

fs_status_t ggml_fs_buffer_alloc(ggml_fs_backend_t* backend,
                                 size_t             nbytes,
                                 fs_dev_t           initial_owner,
                                 uint32_t           flags,
                                 fs_dmabuf_t**      out_buf) {
  if (!backend) {
    return FS_ERR_INVALID_ARG;
  }
  return fs_buffer_alloc(backend->ctx, nbytes, initial_owner, flags, out_buf);
}

void ggml_fs_buffer_free(ggml_fs_backend_t* backend, fs_dmabuf_t* buf) {
  if (!backend) {
    return;
  }
  fs_buffer_release(backend->ctx, buf);
}

int ggml_fs_supports_node(const ggml_fs_backend_t* backend, const ggml_fs_node_t* node) {
  if (!backend || !node) {
    return 0;
  }
  fs_op_desc_t op;
  return ggml_fs_node_to_op(backend, 0u, node, &op) == FS_OK ? 1 : 0;
}

fs_status_t ggml_fs_graph_plan(ggml_fs_backend_t*      backend,
                               const ggml_fs_cgraph_t* cgraph,
                               fs_plan_t**             out_plan) {
  if (!backend || !cgraph || !cgraph->nodes || cgraph->n_nodes == 0u || !out_plan) {
    return FS_ERR_INVALID_ARG;
  }
  std::vector<fs_op_desc_t> ops(cgraph->n_nodes);
  for (uint32_t i = 0; i < cgraph->n_nodes; ++i) {
    fs_status_t st = ggml_fs_node_to_op(backend, i, &cgraph->nodes[i], &ops[i]);
    if (st != FS_OK) {
      return st;
    }
  }
  return fs_plan_create(backend->ctx, ops.data(), cgraph->n_nodes, out_plan);
}

fs_status_t ggml_fs_graph_compute(ggml_fs_backend_t*      backend,
                                  const ggml_fs_cgraph_t* cgraph,
                                  fs_plan_t**             out_plan_opt) {
  if (!backend || !cgraph || !cgraph->nodes || cgraph->n_nodes == 0u) {
    return FS_ERR_INVALID_ARG;
  }
  std::vector<fs_op_desc_t> ops(cgraph->n_nodes);
  for (uint32_t i = 0; i < cgraph->n_nodes; ++i) {
    fs_status_t st = ggml_fs_node_to_op(backend, i, &cgraph->nodes[i], &ops[i]);
    if (st != FS_OK) {
      return st;
    }
  }

  fs_plan_t*  plan = nullptr;
  fs_status_t st   = fs_plan_create(backend->ctx, ops.data(), cgraph->n_nodes, &plan);
  if (st != FS_OK) {
    return st;
  }

  st = fs_execute_plan(backend->ctx, plan, ops.data(), cgraph->n_nodes);
  if (st != FS_OK) {
    fs_plan_destroy(plan);
    return st;
  }

  if (out_plan_opt) {
    *out_plan_opt = plan;
  } else {
    fs_plan_destroy(plan);
  }
  return FS_OK;
}

fs_status_t ggml_fs_get_telemetry(const ggml_fs_backend_t* backend, fs_telemetry_t* out_telemetry) {
  if (!backend) {
    return FS_ERR_INVALID_ARG;
  }
  return fs_context_get_telemetry(backend->ctx, out_telemetry);
}

} /* extern "C" */
