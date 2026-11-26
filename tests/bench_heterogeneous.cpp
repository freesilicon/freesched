/*
 * Heterogeneous prefill (GPU) + attention (CPU/SME2) + decode (EdgeTPU) benchmark.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "freesched.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int run_profile_bench(fs_hw_profile_t profile) {
  fs_config_t cfg = fs_config_default();
  cfg.mock_mode   = 1;
  cfg.profile     = profile;

  fs_context_t* ctx = nullptr;
  if (fs_context_create(&cfg, &ctx) != FS_OK) {
    std::fprintf(stderr, "Failed to create context for %s\n", fs_profile_name(profile));
    return 1;
  }

  const fs_topology_t* topo = fs_context_topology(ctx);
  const uint32_t       M_PRE = 512u;
  const uint32_t       M_DEC = 1u;
  const uint32_t       D     = 64u;
  const uint32_t       H     = 128u;

  fs_dmabuf_t *b_pre_in = nullptr, *b_w = nullptr, *b_pre_out = nullptr, *b_attn = nullptr;
  fs_dmabuf_t *b_dec_in = nullptr, *b_dec_out = nullptr;
  fs_dmabuf_t *b_wg = nullptr, *b_wu = nullptr, *b_wd = nullptr, *b_mlp = nullptr;

  fs_buffer_alloc(ctx, M_PRE * D * sizeof(float), FS_DEV_CPU_NEON, FS_BUF_ZERO_INIT, &b_pre_in);
  fs_buffer_alloc(ctx, D * D * sizeof(float), FS_DEV_GPU_VK, FS_BUF_IMMUTABLE_WEIGHT, &b_w);
  fs_buffer_alloc(ctx, M_PRE * D * sizeof(float), FS_DEV_GPU_VK, FS_BUF_ZERO_INIT, &b_pre_out);
  fs_buffer_alloc(ctx, M_PRE * D * sizeof(float), FS_DEV_CPU_NEON, FS_BUF_ZERO_INIT, &b_attn);
  fs_buffer_alloc(ctx, M_DEC * D * sizeof(float), FS_DEV_CPU_NEON, FS_BUF_ZERO_INIT, &b_dec_in);
  fs_buffer_alloc(ctx, M_DEC * D * sizeof(float), FS_DEV_TPU, FS_BUF_ZERO_INIT, &b_dec_out);
  fs_buffer_alloc(ctx, H * D * sizeof(float), FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &b_wg);
  fs_buffer_alloc(ctx, H * D * sizeof(float), FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &b_wu);
  fs_buffer_alloc(ctx, D * H * sizeof(float), FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &b_wd);
  fs_buffer_alloc(ctx, M_DEC * D * sizeof(float), FS_DEV_TPU, FS_BUF_ZERO_INIT, &b_mlp);

  auto* pre_in = static_cast<float*>(b_pre_in->host_ptr);
  auto* w      = static_cast<float*>(b_w->host_ptr);
  auto* dec_in = static_cast<float*>(b_dec_in->host_ptr);
  auto* wg     = static_cast<float*>(b_wg->host_ptr);
  auto* wu     = static_cast<float*>(b_wu->host_ptr);
  auto* wd     = static_cast<float*>(b_wd->host_ptr);

  for (uint32_t i = 0; i < M_PRE * D; ++i) pre_in[i] = std::sin(static_cast<float>(i) * 0.01f);
  for (uint32_t i = 0; i < D * D; ++i) w[i]          = std::cos(static_cast<float>(i) * 0.02f) * 0.1f;
  for (uint32_t i = 0; i < D; ++i) dec_in[i]         = std::sin(static_cast<float>(i) * 0.05f);
  for (uint32_t i = 0; i < H * D; ++i) {
    wg[i] = 0.08f * std::sin(static_cast<float>(i) * 0.03f);
    wu[i] = 0.08f * std::cos(static_cast<float>(i) * 0.03f);
    wd[i] = 0.08f * std::sin(static_cast<float>(i) * 0.04f);
  }

  fs_op_desc_t ops[4];
  std::memset(ops, 0, sizeof(ops));
  ops[0].type = FS_OP_PREFILL_GEMM; ops[0].phase = FS_PHASE_PREFILL;
  ops[0].m_dim = M_PRE; ops[0].k_dim = D; ops[0].n_dim = D;
  ops[0].input_a = b_pre_in; ops[0].input_b = b_w; ops[0].output = b_pre_out;
  ops[0].label = "prefill_qkv";

  ops[1].type = FS_OP_ATTN_SOFTMAX; ops[1].phase = FS_PHASE_CONTROL;
  ops[1].m_dim = M_PRE; ops[1].n_dim = D; ops[1].aux_param = 0.125f;
  ops[1].input_a = b_pre_out; ops[1].output = b_attn;
  ops[1].label = "attn_softmax";

  ops[2].type = FS_OP_DECODE_GEMV; ops[2].phase = FS_PHASE_DECODE;
  ops[2].m_dim = M_DEC; ops[2].k_dim = D; ops[2].n_dim = D;
  ops[2].input_a = b_dec_in; ops[2].input_b = b_w; ops[2].output = b_dec_out;
  ops[2].label = "decode_proj";

  ops[3].type = FS_OP_SWIGLU; ops[3].phase = FS_PHASE_DECODE;
  ops[3].m_dim = M_DEC; ops[3].k_dim = D; ops[3].h_dim = H; ops[3].n_dim = D;
  ops[3].input_a = b_dec_out; ops[3].input_b = b_wg; ops[3].input_c = b_wu;
  ops[3].input_d = b_wd; ops[3].output = b_mlp;
  ops[3].label = "decode_swiglu";

  fs_plan_t* plan = nullptr;
  if (fs_plan_create(ctx, ops, 4u, &plan) != FS_OK) {
    std::fprintf(stderr, "Plan failed: %s\n", fs_context_last_error(ctx));
    fs_context_destroy(ctx);
    return 1;
  }

  for (int iter = 0; iter < 8; ++iter) {
    if (fs_execute_plan(ctx, plan, ops, 4u) != FS_OK) {
      std::fprintf(stderr, "Execute failed: %s\n", fs_context_last_error(ctx));
      fs_plan_destroy(plan);
      fs_context_destroy(ctx);
      return 1;
    }
  }

  fs_telemetry_t tel;
  fs_context_get_telemetry(ctx, &tel);

  std::printf("=== freesched profile: %s (RAM: %llu GiB) ===\n", topo->soc_name,
              static_cast<unsigned long long>(topo->unified_ram_bytes >> 30));
  for (uint32_t i = 0; i < plan->num_steps; ++i) {
    const fs_plan_step_t& s = plan->steps[i];
    std::printf("  step[%u] %-14s -> %-8s (handoff=%d zero_copy=%d est=%.1f us reason=%s)\n",
                i, ops[i].label, fs_dev_name(s.assigned_dev), s.requires_handoff, s.zero_copy,
                s.est_total_us, s.reason);
  }
  std::printf("  telemetry (8 iters): gpu_us=%llu tpu_us=%llu cpu_us=%llu handoff_us=%llu "
              "zero_copy=%llu fallback_copies=%llu est_watts=%.2f W\n",
              static_cast<unsigned long long>(tel.gpu_us),
              static_cast<unsigned long long>(tel.tpu_us),
              static_cast<unsigned long long>(tel.cpu_us),
              static_cast<unsigned long long>(tel.handoff_us),
              static_cast<unsigned long long>(tel.zero_copy_handoffs),
              static_cast<unsigned long long>(tel.fallback_copies),
              tel.est_watts);

  fs_plan_destroy(plan);
  fs_context_destroy(ctx);
  return (tel.fallback_copies == 0u) ? 0 : 1;
}

int main() {
  if (run_profile_bench(FS_PROFILE_PIXEL10_LAGUNA) != 0) return 1;
  if (run_profile_bench(FS_PROFILE_PIXEL11_MALIBU) != 0) return 1;
  return 0;
}
