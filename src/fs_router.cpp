/*
 * freesched deterministic cost-model heterogeneous router.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Partitions transformer & vision workloads across GPU (prefill/vision),
 * EdgeTPU (low-watt decode & chained SwiGLU), and ARMv9 NEON/SME2 CPU
 * (attention softmax, RoPE, RMSNorm, small control) with a zero-copy
 * continuity bonus to prevent cross-accelerator ping-ponging and automatic
 * graceful degradation when any accelerator is offline.
 */
#include "fs_internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

static double fs_dtype_bytes_per_elem(fs_dtype_t dtype) {
  switch (dtype) {
    case FS_DTYPE_F32:  return 4.0;
    case FS_DTYPE_I32:  return 4.0;
    case FS_DTYPE_F16:  return 2.0;
    case FS_DTYPE_I8:   return 1.0;
    case FS_DTYPE_Q8_0: return 34.0 / 32.0;
    case FS_DTYPE_Q4_0: return 18.0 / 32.0;
    default:            return 4.0;
  }
}

static fs_dev_t fs_primary_ideal_device(const fs_topology_t* topo, const fs_op_desc_t* op) {
  const uint32_t threshold =
      topo->prefill_m_threshold > 0u ? topo->prefill_m_threshold : FS_DEFAULT_PREFILL_M;

  switch (op->type) {
    case FS_OP_PREFILL_GEMM:
      return (op->m_dim >= threshold) ? FS_DEV_GPU_VK : FS_DEV_TPU;

    case FS_OP_VISION_CONV:
      return FS_DEV_GPU_VK;

    case FS_OP_DECODE_GEMV:
      return FS_DEV_TPU;

    case FS_OP_SWIGLU:
      if (op->phase == FS_PHASE_PREFILL || op->m_dim >= threshold) {
        return FS_DEV_GPU_VK;
      }
      return FS_DEV_TPU;

    case FS_OP_GEMM:
      if (op->phase == FS_PHASE_VISION) {
        return FS_DEV_GPU_VK;
      }
      if (op->phase == FS_PHASE_PREFILL) {
        return (op->m_dim >= threshold) ? FS_DEV_GPU_VK : FS_DEV_TPU;
      }
      if (op->phase == FS_PHASE_DECODE) {
        return FS_DEV_TPU;
      }
      return (op->m_dim >= threshold) ? FS_DEV_GPU_VK : FS_DEV_TPU;

    case FS_OP_ATTN_SOFTMAX:
    case FS_OP_ROPE:
    case FS_OP_RMS_NORM:
      return topo->devices[FS_DEV_CPU_SME2].available ? FS_DEV_CPU_SME2 : FS_DEV_CPU_NEON;

    case FS_OP_ELEMENTWISE:
      if ((op->route_flags & FS_ROUTE_ALLOW_SPLINE) != 0u &&
          (op->activation == FS_ACT_SILU || op->activation == FS_ACT_GELU)) {
        return FS_DEV_TPU;
      }
      return topo->devices[FS_DEV_CPU_SME2].available ? FS_DEV_CPU_SME2 : FS_DEV_CPU_NEON;

    default:
      return FS_DEV_CPU_NEON;
  }
}

static void fs_estimate_op_work(const fs_op_desc_t* op, double* out_flops, double* out_bytes) {
  const double m  = static_cast<double>(op->m_dim > 0u ? op->m_dim : 1u);
  const double k  = static_cast<double>(op->k_dim > 0u ? op->k_dim : 1u);
  const double n  = static_cast<double>(op->n_dim > 0u ? op->n_dim : 1u);
  const double h  = static_cast<double>(op->h_dim > 0u ? op->h_dim : n);
  const double ab = fs_dtype_bytes_per_elem(op->act_dtype);
  const double wb = fs_dtype_bytes_per_elem(op->weight_dtype);

  switch (op->type) {
    case FS_OP_GEMM:
    case FS_OP_PREFILL_GEMM:
    case FS_OP_DECODE_GEMV:
    case FS_OP_VISION_CONV:
      *out_flops = 2.0 * m * k * n;
      *out_bytes = (m * k + m * n) * ab + (n * k) * wb;
      break;
    case FS_OP_SWIGLU:
      *out_flops = 2.0 * m * (2.0 * k * h + h * n) + 4.0 * m * h;
      *out_bytes = (m * k + m * n) * ab + (2.0 * h * k + n * h) * wb;
      break;
    case FS_OP_ATTN_SOFTMAX:
    case FS_OP_ROPE:
    case FS_OP_RMS_NORM:
    case FS_OP_ELEMENTWISE:
    default:
      *out_flops = 6.0 * m * n;
      *out_bytes = 2.0 * m * n * ab;
      break;
  }
}

static float fs_estimate_compute_us(const fs_dev_info_t* dev,
                                    const fs_op_desc_t*  op,
                                    double               flops,
                                    double               bytes) {
  const double is_quant = (op->weight_dtype == FS_DTYPE_I8 || op->weight_dtype == FS_DTYPE_Q8_0 ||
                           op->weight_dtype == FS_DTYPE_Q4_0)
                              ? 1.0
                              : 0.0;
  double peak_gops = static_cast<double>(dev->peak_gflops_fp16);
  if (is_quant > 0.5 && dev->peak_tops_int8 > 0.0f) {
    peak_gops = static_cast<double>(dev->peak_tops_int8) * 1000.0;
  }
  if (peak_gops < 1.0) {
    peak_gops = 100.0;
  }
  double bw_gbps = static_cast<double>(dev->mem_bw_gbps > 1.0f ? dev->mem_bw_gbps : 20.0f);

  /* Roofline: max(compute_time, memory_time) + launch_overhead */
  const double compute_us = (flops / (peak_gops * 1e9)) * 1e6;
  const double mem_us     = (bytes / (bw_gbps * 1e9)) * 1e6;
  double       total_us   = (compute_us > mem_us ? compute_us : mem_us) + dev->launch_overhead_us;

  /* Hardware affinity weights reflecting real architectural strengths */
  if (op->type == FS_OP_ATTN_SOFTMAX || op->type == FS_OP_ROPE || op->type == FS_OP_RMS_NORM) {
    if (!fs_dev_is_cpu(dev->dev)) {
      total_us *= 8.0;
    }
  } else if (op->type == FS_OP_DECODE_GEMV ||
             (op->type == FS_OP_SWIGLU && op->m_dim < FS_DEFAULT_PREFILL_M)) {
    if (dev->dev == FS_DEV_TPU) {
      total_us *= 0.55; /* In-SRAM tiled decode & chained SwiGLU advantage */
    } else if (dev->dev == FS_DEV_GPU_VK) {
      total_us *= 1.60; /* Vulkan command submission & underfilled warp penalty at low M */
    } else if (fs_dev_is_cpu(dev->dev)) {
      total_us += 200.0; /* Keep CPU free for attention/control when TPU/GPU are online */
    }
  } else if (op->type == FS_OP_PREFILL_GEMM || op->type == FS_OP_VISION_CONV ||
             (op->type == FS_OP_SWIGLU && op->m_dim >= FS_DEFAULT_PREFILL_M)) {
    if (dev->dev == FS_DEV_GPU_VK && (op->m_dim >= FS_DEFAULT_PREFILL_M || op->type == FS_OP_VISION_CONV)) {
      total_us *= 0.50; /* High-M GPU shader tile occupancy advantage */
    } else if (dev->dev == FS_DEV_TPU && op->m_dim >= FS_DEFAULT_PREFILL_M) {
      total_us *= 1.45; /* Multi-tile M-chunking overhead on EdgeTPU */
    } else if (fs_dev_is_cpu(dev->dev)) {
      total_us += 250.0; /* Offload bulk prefill/vision GEMMs off CPU when accelerators are online */
    }
  } else if (op->type == FS_OP_GEMM) {
    if (fs_dev_is_cpu(dev->dev)) {
      total_us += 160.0; /* Prefer TPU/GPU for generic GEMM unless CPU is only option */
    }
  } else if (op->type == FS_OP_ELEMENTWISE && (op->route_flags & FS_ROUTE_ALLOW_SPLINE) != 0u &&
             (op->activation == FS_ACT_SILU || op->activation == FS_ACT_GELU)) {
    if (dev->dev == FS_DEV_TPU && (dev->caps & FS_CAP_TPU_SPLINE_0X640) != 0u) {
      /* Mode-A 0x640 degree-4 Horner spline fuses with zero extra launch cost */
      total_us = compute_us + 1.0;
    } else if (fs_dev_is_cpu(dev->dev)) {
      total_us += 20.0;
    }
  }

  return static_cast<float>(total_us);
}

extern "C" {

fs_status_t fs_plan_create(fs_context_t*       ctx,
                           const fs_op_desc_t* ops,
                           uint32_t            num_ops,
                           fs_plan_t**         out_plan) {
  if (!ctx || !ops || num_ops == 0u || !out_plan) {
    return FS_ERR_INVALID_ARG;
  }

  auto* plan = static_cast<fs_plan_t*>(std::calloc(1u, sizeof(fs_plan_t)));
  if (!plan) {
    return FS_ERR_NO_MEMORY;
  }
  plan->steps = static_cast<fs_plan_step_t*>(std::calloc(num_ops, sizeof(fs_plan_step_t)));
  if (!plan->steps) {
    std::free(plan);
    return FS_ERR_NO_MEMORY;
  }
  plan->num_steps = num_ops;

  const float continuity_bonus =
      ctx->config.continuity_bonus_us > 0.0f ? ctx->config.continuity_bonus_us : 45.0f;

  /* Track simulated buffer ownership across multi-op pipelines */
  std::unordered_map<uint64_t, fs_dev_t> sim_owner;

  for (uint32_t i = 0; i < num_ops; ++i) {
    const fs_op_desc_t& op   = ops[i];
    fs_plan_step_t&     step = plan->steps[i];
    step.op_index            = i;

    fs_dev_t prev_owner = FS_DEV_NONE;
    if (op.input_a) {
      auto it = sim_owner.find(op.input_a->buf_id);
      if (it != sim_owner.end()) {
        prev_owner = it->second;
      } else {
        prev_owner = op.input_a->owner_dev;
      }
    }
    step.prev_input_owner = prev_owner;

    double flops = 0.0;
    double bytes = 0.0;
    fs_estimate_op_work(&op, &flops, &bytes);

    const fs_dev_t ideal_dev = fs_primary_ideal_device(&ctx->topo, &op);
    fs_dev_t       chosen    = FS_DEV_NONE;
    float          best_cost = 1e30f;
    float          best_comp = 0.0f;
    float          best_ho   = 0.0f;
    float          best_mj   = 0.0f;
    const char*    reason    = "cost_model";

    /* 1. Explicit device override */
    if ((op.route_flags & FS_ROUTE_FORCE_DEVICE) != 0u) {
      if (!fs_dev_is_valid(op.preferred_dev) || !ctx->topo.devices[op.preferred_dev].available ||
          !fs_backend_supports_op_internal(ctx, op.preferred_dev, &op)) {
        fs_set_error(ctx, "forced device %s unavailable or unsupported for op %u (%s)",
                     fs_dev_name(op.preferred_dev), i, fs_op_type_name(op.type));
        fs_plan_destroy(plan);
        return FS_ERR_NO_BACKEND;
      }
      chosen = op.preferred_dev;
      const fs_dev_info_t& di = ctx->topo.devices[chosen];
      best_comp               = fs_estimate_compute_us(&di, &op, flops, bytes);
      best_ho = (prev_owner != FS_DEV_NONE && prev_owner != chosen) ? di.handoff_overhead_us : 0.0f;
      best_mj = (best_comp * 1e-6f) * di.typical_watts * 1000.0f;
      reason  = "forced_override";
    } else {
      /* 2. Evaluate all online backends supporting this op */
      for (int d = FS_DEV_TPU; d < FS_DEV_COUNT; ++d) {
        const fs_dev_t       dev = static_cast<fs_dev_t>(d);
        const fs_dev_info_t& di  = ctx->topo.devices[d];
        if (!di.available) {
          continue;
        }
        if (!fs_backend_supports_op_internal(ctx, dev, &op)) {
          continue;
        }

        const float comp_us = fs_estimate_compute_us(&di, &op, flops, bytes);
        const int   needs_ho =
            (prev_owner != FS_DEV_NONE && prev_owner != dev) ? 1 : 0;
        const float ho_us = needs_ho ? di.handoff_overhead_us : 0.0f;
        const float energy_mj = (comp_us * 1e-6f) * di.typical_watts * 1000.0f;

        /* Score combines latency, optional low-power weighting, phase affinity, and continuity */
        float score = comp_us + ho_us;
        if ((op.route_flags & FS_ROUTE_PREFER_LOW_POWER) != 0u) {
          score += energy_mj * 120.0f;
        } else {
          score += energy_mj * 15.0f;
        }

        /* Strong architectural phase bias when ideal device is online */
        if (dev == ideal_dev) {
          score *= 0.72f;
        }

        /* Zero-copy continuity bonus prevents ping-ponging across accelerators */
        const int allow_continuity =
            (op.route_flags & FS_ROUTE_NO_CONTINUITY_BONUS) == 0u &&
            prev_owner != FS_DEV_NONE && prev_owner == dev &&
            (!fs_dev_is_cpu(dev) || op.type == FS_OP_ATTN_SOFTMAX || op.type == FS_OP_ROPE ||
             op.type == FS_OP_RMS_NORM || op.type == FS_OP_ELEMENTWISE);
        if (allow_continuity) {
          score -= continuity_bonus;
        }

        /* Optional soft hint */
        if (op.preferred_dev == dev) {
          score -= 5.0f;
        }

        if (score < best_cost) {
          best_cost = score;
          chosen    = dev;
          best_comp = comp_us;
          best_ho   = ho_us;
          best_mj   = energy_mj;
        }
      }

      if (chosen == FS_DEV_NONE) {
        fs_set_error(ctx, "no available backend supports op %u (%s)", i, fs_op_type_name(op.type));
        fs_plan_destroy(plan);
        return FS_ERR_NO_BACKEND;
      }

      if (chosen != ideal_dev && !ctx->topo.devices[ideal_dev].available) {
        step.degraded = 1;
        plan->degraded_ops++;
        reason = "degraded_fallback";
      } else if (prev_owner == chosen && (op.route_flags & FS_ROUTE_NO_CONTINUITY_BONUS) == 0u &&
                 chosen != ideal_dev) {
        reason = "continuity_keep";
      } else if (op.type == FS_OP_PREFILL_GEMM ||
                 (op.type == FS_OP_GEMM && op.m_dim >= ctx->topo.prefill_m_threshold)) {
        reason = "prefill_gpu";
      } else if (op.type == FS_OP_DECODE_GEMV || op.type == FS_OP_SWIGLU ||
                 (op.type == FS_OP_GEMM && op.m_dim < ctx->topo.prefill_m_threshold)) {
        reason = "decode_tpu";
      } else if (op.type == FS_OP_VISION_CONV) {
        reason = "vision_gpu";
      } else if (op.type == FS_OP_ELEMENTWISE && chosen == FS_DEV_TPU) {
        reason = "tpu_spline";
      } else {
        reason = fs_dev_is_cpu(chosen) ? "cpu_control_attn" : "cost_model";
      }
    }

    const int req_handoff = (prev_owner != FS_DEV_NONE && prev_owner != chosen) ? 1 : 0;
    const int force_copy =
        (op.input_a && (op.input_a->flags & FS_BUF_FORCE_HOST_COPY) != 0u) ||
        ((ctx->topo.global_caps & FS_CAP_DMABUF) == 0u);
    const int zero_copy = req_handoff ? (force_copy ? 0 : 1) : 1;

    step.assigned_dev     = chosen;
    step.requires_handoff = req_handoff;
    step.zero_copy        = zero_copy;
    step.est_compute_us   = best_comp;
    step.est_handoff_us   = best_ho;
    step.est_total_us     = best_comp + best_ho;
    step.est_energy_mj    = best_mj;
    std::snprintf(step.reason, sizeof(step.reason), "%s", reason);

    plan->total_est_us        += step.est_total_us;
    plan->total_est_energy_mj += step.est_energy_mj;
    if (req_handoff) {
      if (zero_copy) {
        plan->planned_zero_copy_handoffs++;
      } else {
        plan->planned_fallback_copies++;
      }
    }

    /* Update simulated ownership for subsequent pipeline stages */
    if (op.input_a && (op.input_a->flags & FS_BUF_IMMUTABLE_WEIGHT) == 0u) {
      sim_owner[op.input_a->buf_id] = chosen;
    }
    if (op.output) {
      sim_owner[op.output->buf_id] = chosen;
    }
  }

  *out_plan = plan;
  return FS_OK;
}

void fs_plan_destroy(fs_plan_t* plan) {
  if (!plan) {
    return;
  }
  std::free(plan->steps);
  std::free(plan);
}

} /* extern "C" */
