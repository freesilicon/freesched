/*
 * Pre-baked zero-copy DMA-BUF graph + openbuenos EdgeTPU adapter wiring +
 * DSpark (MAX_DUP=3/4) speculative expert deduplication planner.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pocketcoder_runner.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

struct pc_graph {
  pc_graph_config_t   cfg;
  fs_context_t*       ctx;
  fs_buf_reconciled_t reconciled;
  fs_dmabuf_t*        buf_x;
  fs_dmabuf_t*        buf_rms_w;
  fs_dmabuf_t*        buf_norm;
  fs_dmabuf_t*        buf_w_q4;
  fs_dmabuf_t*        buf_out;
  fs_op_desc_t        ops[2];
  fs_plan_t*          plan;
  size_t              w_q4_bytes;
};

extern "C" {

fs_status_t pc_dspark_plan_experts(const uint32_t*         routed_ids,
                                   uint32_t                batch_size,
                                   uint32_t                top_k,
                                   uint32_t                num_experts,
                                   uint64_t                bytes_per_expert,
                                   pc_dspark_dedup_plan_t* out_plan) {
  if (!routed_ids || !out_plan || batch_size == 0u || batch_size > PC_DSPARK_MAX_DUP ||
      top_k == 0u || top_k > PC_TOP_K_EXPERTS || num_experts == 0u || num_experts > PC_MAX_EXPERTS) {
    return FS_ERR_INVALID_ARG;
  }

  std::memset(out_plan, 0, sizeof(*out_plan));
  out_plan->batch_size         = batch_size;
  out_plan->top_k              = top_k;
  out_plan->bytes_per_expert   = bytes_per_expert;
  out_plan->naive_expert_loads = batch_size * top_k;

  uint8_t expert_to_entry[PC_MAX_EXPERTS];
  std::memset(expert_to_entry, 0xffu, sizeof(expert_to_entry));

  for (uint32_t t = 0; t < batch_size; ++t) {
    for (uint32_t k = 0; k < top_k; ++k) {
      const uint32_t eid = routed_ids[t * top_k + k];
      if (eid >= num_experts) {
        return FS_ERR_INVALID_ARG;
      }
      uint8_t entry_idx = expert_to_entry[eid];
      if (entry_idx == 0xffu) {
        entry_idx                         = static_cast<uint8_t>(out_plan->unique_count++);
        expert_to_entry[eid]              = entry_idx;
        out_plan->entries[entry_idx].expert_id  = eid;
        out_plan->entries[entry_idx].token_mask = 0u;
        out_plan->entries[entry_idx].ref_count  = 0u;
      }
      out_plan->entries[entry_idx].token_mask |= (1u << t);
      out_plan->entries[entry_idx].ref_count++;
      out_plan->slot_to_entry[t][k] = entry_idx;
    }
  }

  out_plan->dedup_expert_loads = out_plan->unique_count;
  out_plan->naive_bytes        = static_cast<uint64_t>(out_plan->naive_expert_loads) * bytes_per_expert;
  out_plan->dedup_bytes        = static_cast<uint64_t>(out_plan->dedup_expert_loads) * bytes_per_expert;
  out_plan->bytes_saved =
      (out_plan->naive_bytes > out_plan->dedup_bytes) ? (out_plan->naive_bytes - out_plan->dedup_bytes) : 0u;
  out_plan->dedup_ratio = (out_plan->dedup_expert_loads > 0u)
                              ? static_cast<double>(out_plan->naive_expert_loads) /
                                    static_cast<double>(out_plan->dedup_expert_loads)
                              : 1.0;

  return FS_OK;
}

fs_status_t pc_dspark_execute_moe_f32(const pc_dspark_dedup_plan_t* plan,
                                      const pc_soa_matrix_t*        experts,
                                      uint32_t                      num_experts,
                                      const float*                  routing_weights,
                                      const float*                  x_batch,
                                      float*                        y_batch) {
  if (!plan || !experts || !routing_weights || !x_batch || !y_batch ||
      plan->batch_size == 0u || plan->batch_size > PC_DSPARK_MAX_DUP ||
      plan->unique_count == 0u || num_experts == 0u) {
    return FS_ERR_INVALID_ARG;
  }

  const uint32_t n_dim = experts[0].n_dim;
  const uint32_t k_dim = experts[0].k_dim;
  if (n_dim == 0u || k_dim == 0u) {
    return FS_ERR_INVALID_ARG;
  }

  std::memset(y_batch, 0, static_cast<size_t>(plan->batch_size) * n_dim * sizeof(float));

  /* Per-(entry, token) cached expert output: evaluated once per unique (expert, active token) */
  std::vector<float> scratch(static_cast<size_t>(plan->unique_count) * plan->batch_size * n_dim, 0.0f);

  for (uint32_t ei = 0; ei < plan->unique_count; ++ei) {
    const pc_dspark_expert_entry_t& ent = plan->entries[ei];
    if (ent.expert_id >= num_experts) {
      return FS_ERR_INVALID_ARG;
    }
    const pc_soa_matrix_t& exp_mat = experts[ent.expert_id];
    if (exp_mat.n_dim != n_dim || exp_mat.k_dim != k_dim) {
      return FS_ERR_INVALID_ARG;
    }
    for (uint32_t t = 0; t < plan->batch_size; ++t) {
      if ((ent.token_mask & (1u << t)) != 0u) {
        float* dst = scratch.data() + (static_cast<size_t>(ei) * plan->batch_size + t) * n_dim;
        fs_status_t rc = pc_gemv_soa_f32(&exp_mat, x_batch + static_cast<size_t>(t) * k_dim, dst);
        if (rc != FS_OK) {
          return rc;
        }
      }
    }
  }

  for (uint32_t t = 0; t < plan->batch_size; ++t) {
    float* yt = y_batch + static_cast<size_t>(t) * n_dim;
    for (uint32_t k = 0; k < plan->top_k; ++k) {
      const uint8_t ei = plan->slot_to_entry[t][k];
      const float   rw = routing_weights[t * plan->top_k + k];
      const float*  eo = scratch.data() + (static_cast<size_t>(ei) * plan->batch_size + t) * n_dim;
      for (uint32_t n = 0; n < n_dim; ++n) {
        yt[n] += rw * eo[n];
      }
    }
  }

  return FS_OK;
}

pc_graph_config_t pc_graph_config_default(void) {
  pc_graph_config_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size    = static_cast<uint32_t>(sizeof(pc_graph_config_t));
  cfg.profile        = FS_PROFILE_PIXEL10_LAGUNA;
  cfg.mock_mode      = 1;
  cfg.hidden_dim     = 2304u;
  cfg.expert_ffn_dim = 1152u;
  cfg.dspark_max_dup = 4u;
  return cfg;
}

fs_status_t pc_graph_create(const pc_graph_config_t* cfg, pc_graph_t** out_graph) {
  if (!out_graph) {
    return FS_ERR_INVALID_ARG;
  }
  *out_graph = nullptr;

  pc_graph_config_t eff = cfg ? *cfg : pc_graph_config_default();
  if (eff.hidden_dim == 0u || eff.expert_ffn_dim == 0u ||
      (eff.hidden_dim % PC_GGUF_BLK_SIZE) != 0u) {
    return FS_ERR_INVALID_ARG;
  }

  fs_config_t fs_cfg = fs_config_default();
  fs_cfg.profile     = eff.profile;
  fs_cfg.mock_mode   = eff.mock_mode;

  fs_context_t* ctx = nullptr;
  fs_status_t   rc  = fs_context_create(&fs_cfg, &ctx);
  if (rc != FS_OK) {
    return rc;
  }

  /* Wire freesched's optional openbuenos EdgeTPU adapter (mock fallback when HW absent) */
  fs_openbuenos_config_t ob_cfg = fs_openbuenos_config_default();
  ob_cfg.allow_mock_fallback    = 1;
  rc = fs_backend_register_openbuenos(ctx, &ob_cfg);
  if (rc != FS_OK) {
    fs_context_destroy(ctx);
    return rc;
  }

  /* Upfront NvSciBuf-style DMA-BUF reconciliation across CPU + EdgeTPU (+ GPU when online) */
  fs_buf_attr_t attrs[3];
  attrs[0] = fs_buf_attr_default(FS_DEV_CPU_NEON);
  attrs[1] = fs_buf_attr_default(FS_DEV_TPU);
  attrs[2] = fs_buf_attr_default(FS_DEV_GPU_VK);

  fs_buf_reconciled_t rec;
  int                 zc_ok = 0;
  rc = fs_dmabuf_reconcile(attrs, 3u, &rec, &zc_ok);
  if (rc != FS_OK || !zc_ok) {
    fs_context_destroy(ctx);
    return (rc != FS_OK) ? rc : FS_ERR_UNSUPPORTED;
  }

  auto* g = new pc_graph();
  std::memset(g, 0, sizeof(*g));
  g->cfg        = eff;
  g->ctx        = ctx;
  g->reconciled = rec;
  g->w_q4_bytes =
      static_cast<size_t>(eff.expert_ffn_dim) * (eff.hidden_dim / FS_QK8_0) * sizeof(fs_block_q4_0_t);

  const size_t vec_bytes = static_cast<size_t>(eff.hidden_dim) * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(eff.expert_ffn_dim) * sizeof(float);

  if (fs_buffer_alloc_reconciled(ctx, vec_bytes, attrs, 3u, FS_DEV_CPU_NEON, FS_BUF_ZERO_INIT,
                                 &g->buf_x, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, vec_bytes, attrs, 3u, FS_DEV_CPU_NEON,
                                 FS_BUF_IMMUTABLE_WEIGHT | FS_BUF_ZERO_INIT, &g->buf_rms_w, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, vec_bytes, attrs, 3u, FS_DEV_CPU_NEON, FS_BUF_ZERO_INIT,
                                 &g->buf_norm, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, g->w_q4_bytes, attrs, 3u, FS_DEV_TPU,
                                 FS_BUF_IMMUTABLE_WEIGHT | FS_BUF_ZERO_INIT, &g->buf_w_q4, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, out_bytes, attrs, 3u, FS_DEV_TPU, FS_BUF_ZERO_INIT,
                                 &g->buf_out, nullptr) != FS_OK) {
    pc_graph_destroy(g);
    return FS_ERR_NO_MEMORY;
  }

  /* Pre-bake heterogeneous execution plan: RMSNorm (CPU) -> Q4_0 GEMV (EdgeTPU/openbuenos) */
  std::memset(g->ops, 0, sizeof(g->ops));
  g->ops[0].op_id     = 0u;
  g->ops[0].type      = FS_OP_RMS_NORM;
  g->ops[0].phase     = FS_PHASE_CONTROL;
  g->ops[0].m_dim     = 1u;
  g->ops[0].n_dim     = eff.hidden_dim;
  g->ops[0].act_dtype = FS_DTYPE_F32;
  g->ops[0].aux_param = 1e-6f;
  g->ops[0].input_a   = g->buf_x;
  g->ops[0].input_b   = g->buf_rms_w;
  g->ops[0].output    = g->buf_norm;
  g->ops[0].label     = "pocketcoder_rms_norm";

  g->ops[1].op_id        = 1u;
  g->ops[1].type         = FS_OP_DECODE_GEMV;
  g->ops[1].phase        = FS_PHASE_DECODE;
  g->ops[1].m_dim        = 1u;
  g->ops[1].k_dim        = eff.hidden_dim;
  g->ops[1].n_dim        = eff.expert_ffn_dim;
  g->ops[1].act_dtype    = FS_DTYPE_F32;
  g->ops[1].weight_dtype = FS_DTYPE_Q4_0;
  g->ops[1].input_a      = g->buf_norm;
  g->ops[1].input_b      = g->buf_w_q4;
  g->ops[1].output       = g->buf_out;
  g->ops[1].label        = "pocketcoder_expert_proj";

  rc = fs_plan_create(ctx, g->ops, 2u, &g->plan);
  if (rc != FS_OK) {
    pc_graph_destroy(g);
    return rc;
  }

  *out_graph = g;
  return FS_OK;
}

void pc_graph_destroy(pc_graph_t* graph) {
  if (!graph) {
    return;
  }
  if (graph->plan) {
    fs_plan_destroy(graph->plan);
  }
  if (graph->ctx) {
    if (graph->buf_x) fs_buffer_release(graph->ctx, graph->buf_x);
    if (graph->buf_rms_w) fs_buffer_release(graph->ctx, graph->buf_rms_w);
    if (graph->buf_norm) fs_buffer_release(graph->ctx, graph->buf_norm);
    if (graph->buf_w_q4) fs_buffer_release(graph->ctx, graph->buf_w_q4);
    if (graph->buf_out) fs_buffer_release(graph->ctx, graph->buf_out);
    fs_context_destroy(graph->ctx);
  }
  delete graph;
}

fs_status_t pc_graph_run_prebaked_step(pc_graph_t*       graph,
                                       const float*      x_in,
                                       const void*       w_q4_0,
                                       const float*      rms_weight,
                                       float*            y_out,
                                       pc_graph_stats_t* out_stats) {
  if (!graph || !x_in || !w_q4_0 || !rms_weight || !y_out) {
    return FS_ERR_INVALID_ARG;
  }

  const uint32_t k_dim = graph->cfg.hidden_dim;
  const uint32_t n_dim = graph->cfg.expert_ffn_dim;

  std::memcpy(graph->buf_x->host_ptr, x_in, static_cast<size_t>(k_dim) * sizeof(float));
  std::memcpy(graph->buf_rms_w->host_ptr, rms_weight, static_cast<size_t>(k_dim) * sizeof(float));
  std::memcpy(graph->buf_w_q4->host_ptr, w_q4_0, graph->w_q4_bytes);

  /* Reset buffer owner to CPU_NEON before executing pre-baked plan so step 1 performs zero-copy handoff */
  if (graph->buf_norm->owner_dev != FS_DEV_CPU_NEON) {
    fs_status_t hrc = fs_buffer_handoff(graph->ctx, graph->buf_norm, graph->buf_norm->owner_dev,
                                        FS_DEV_CPU_NEON, nullptr);
    if (hrc != FS_OK) {
      return hrc;
    }
  }

  fs_context_reset_telemetry(graph->ctx);
  fs_status_t rc = fs_execute_plan(graph->ctx, graph->plan, graph->ops, 2u);
  if (rc != FS_OK) {
    return rc;
  }

  std::memcpy(y_out, graph->buf_out->host_ptr, static_cast<size_t>(n_dim) * sizeof(float));

  if (out_stats) {
    std::vector<float> norm_ref(k_dim);
    double ss = 0.0;
    for (uint32_t i = 0; i < k_dim; ++i) {
      ss += static_cast<double>(x_in[i]) * static_cast<double>(x_in[i]);
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(k_dim) + 1e-6));
    for (uint32_t i = 0; i < k_dim; ++i) {
      norm_ref[i] = x_in[i] * inv * rms_weight[i];
    }
    std::vector<float> y_ref(n_dim);
    rc = pc_gemv_ref_raw_f32(w_q4_0, PC_GGML_TYPE_Q4_0, norm_ref.data(), y_ref.data(), n_dim, k_dim);
    if (rc != FS_OK) {
      return rc;
    }

    fs_parity_t parity;
    fs_compute_parity_f32(y_out, y_ref.data(), n_dim, &parity);

    fs_telemetry_t tel;
    fs_context_get_telemetry(graph->ctx, &tel);

    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->reconciled_zero_copy    = graph->reconciled.zero_copy_compatible;
    out_stats->reconciled_min_align    = graph->reconciled.min_align;
    out_stats->reconciled_stride_align = graph->reconciled.stride_align;
    out_stats->planned_steps           = graph->plan->num_steps;
    out_stats->zero_copy_handoffs      = tel.zero_copy_handoffs;
    out_stats->fallback_copies         = tel.fallback_copies;
    out_stats->tpu_ops                 = tel.tpu_ops;
    out_stats->cpu_ops                 = tel.cpu_ops;
    out_stats->dma_sync_calls          = tel.dma_sync_calls;
    out_stats->max_abs_err             = parity.max_abs_err;
    out_stats->nmse                    = parity.nmse;
  }

  return FS_OK;
}

} /* extern "C" */
