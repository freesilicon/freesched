/*
 * freesched_dmabuf_smoke - Zero-Copy Cross-Accelerator DMA-BUF Smoke Binary
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Runs cleanly on:
 *   - ARM64 Linux (Pixel 10 / OmaMobile): exercises /dev/dma_heap/system,
 *     DMA_BUF_IOCTL_SYNC, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, EDGETPU_MAP_DMABUF,
 *     and Vulkan VK_EXT_external_memory_dma_buf.
 *   - macOS & unprivileged Linux CI: gracefully falls back to memfd/mmap mock
 *     while proving upfront fs_dmabuf_reconcile(), fail-closed tiled layout
 *     rejection, zero-copy handoff invariants, and bit-exact numerical parity.
 */
#include "freesched.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char* backing_name(fs_buf_backing_t b) {
  switch (b) {
    case FS_BACKING_DMA_HEAP:  return "dma_heap";
    case FS_BACKING_MEMFD:     return "memfd";
    case FS_BACKING_IMPORTED:  return "imported";
    case FS_BACKING_ANON_MMAP:
    default:                   return "anon_mmap";
  }
}

static void ref_gemm_f32(const float* a, const float* b, float* c, uint32_t m, uint32_t k,
                         uint32_t n) {
  for (uint32_t mi = 0; mi < m; ++mi) {
    for (uint32_t ni = 0; ni < n; ++ni) {
      double acc = 0.0;
      for (uint32_t ki = 0; ki < k; ++ki) {
        acc += static_cast<double>(a[static_cast<size_t>(mi) * k + ki]) *
               static_cast<double>(b[static_cast<size_t>(ni) * k + ki]);
      }
      c[static_cast<size_t>(mi) * n + ni] = static_cast<float>(acc);
    }
  }
}

static void ref_rms_norm(const float* x, float* y, uint32_t m, uint32_t n, float eps) {
  for (uint32_t r = 0; r < m; ++r) {
    double ss = 0.0;
    for (uint32_t c = 0; c < n; ++c) {
      const double v = static_cast<double>(x[static_cast<size_t>(r) * n + c]);
      ss += v * v;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
    for (uint32_t c = 0; c < n; ++c) {
      y[static_cast<size_t>(r) * n + c] = x[static_cast<size_t>(r) * n + c] * inv;
    }
  }
}

int main(int argc, char** argv) {
  fs_hw_profile_t profile   = FS_PROFILE_PIXEL10_LAGUNA;
  int             force_mock = 0;
  int             json_out   = 0;
  const char*     heap_path  = nullptr;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mock") == 0) {
      force_mock = 1;
    } else if (std::strcmp(argv[i], "--json") == 0) {
      json_out = 1;
    } else if (std::strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
      const char* p = argv[++i];
      if (std::strcmp(p, "laguna") == 0) {
        profile = FS_PROFILE_PIXEL10_LAGUNA;
      } else if (std::strcmp(p, "malibu") == 0) {
        profile = FS_PROFILE_PIXEL11_MALIBU;
      } else {
        profile = FS_PROFILE_AUTO;
      }
    } else if (std::strcmp(argv[i], "--heap") == 0 && i + 1 < argc) {
      heap_path = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf("Usage: freesched_dmabuf_smoke [--profile laguna|malibu|auto] [--mock] [--heap PATH] [--json]\n");
      return 0;
    }
  }

  fs_config_t cfg   = fs_config_default();
  cfg.mock_mode     = force_mock;
  cfg.profile       = profile;
  cfg.dma_heap_path = heap_path;

  fs_context_t* ctx = nullptr;
  fs_status_t   st  = fs_context_create(&cfg, &ctx);
  if (st != FS_OK || !ctx) {
    std::fprintf(stderr, "fs_context_create failed: %s\n", fs_status_string(st));
    return 1;
  }

  /* Ensure accelerators are online for smoke routing even in headless host/CI mode */
  fs_context_set_device_enabled(ctx, FS_DEV_GPU_VK, 1);
  fs_context_set_device_enabled(ctx, FS_DEV_TPU, 1);

  fs_openbuenos_config_t ob_cfg = fs_openbuenos_config_default();
  ob_cfg.dma_heap_path          = heap_path;
  ob_cfg.allow_mock_fallback    = 1;
  if (fs_backend_register_openbuenos(ctx, &ob_cfg) != FS_OK) {
    std::fprintf(stderr, "fs_backend_register_openbuenos failed: %s\n", fs_context_last_error(ctx));
    fs_context_destroy(ctx);
    return 1;
  }

  fs_vulkan_dmabuf_config_t vk_cfg = fs_vulkan_dmabuf_config_default();
  vk_cfg.allow_mock_fallback       = 1;
  vk_cfg.require_linear_layout     = 1;
  if (fs_backend_register_vulkan_dmabuf(ctx, &vk_cfg) != FS_OK) {
    std::fprintf(stderr, "fs_backend_register_vulkan_dmabuf failed: %s\n", fs_context_last_error(ctx));
    fs_context_destroy(ctx);
    return 1;
  }

  /* 1. Upfront 3-engine reconciliation (GPU + TPU + CPU) */
  const fs_topology_t* topo = fs_context_topology(ctx);
  const fs_dev_t       cpu_dev =
      fs_topology_has_device(topo, FS_DEV_CPU_SME2) ? FS_DEV_CPU_SME2 : FS_DEV_CPU_NEON;

  fs_buf_attr_t shared_attrs[3] = {
      fs_buf_attr_default(FS_DEV_GPU_VK),
      fs_buf_attr_default(FS_DEV_TPU),
      fs_buf_attr_default(cpu_dev),
  };
  fs_buf_reconciled_t rec;
  int                 zc_ok = 0;
  if (fs_dmabuf_reconcile(shared_attrs, 3u, &rec, &zc_ok) != FS_OK || !zc_ok ||
      rec.min_align != 128u || rec.layout != FS_LAYOUT_LINEAR || rec.requires_dma_sync != 1) {
    std::fprintf(stderr, "3-engine reconciliation failed\n");
    fs_context_destroy(ctx);
    return 1;
  }

  /* 2. Verify fail-closed rejection on tiled PBE GPU -> linear TPU handoff */
  fs_buf_attr_t tiled_attr = fs_buf_attr_default(FS_DEV_GPU_VK);
  tiled_attr.layout        = FS_LAYOUT_TILED;
  tiled_attr.layout_mask   = FS_LAYOUT_BIT(FS_LAYOUT_TILED);
  fs_dmabuf_t* b_tiled     = nullptr;
  if (fs_buffer_alloc_reconciled(ctx, 4096u, &tiled_attr, 1u, FS_DEV_GPU_VK, FS_BUF_ZERO_INIT,
                                 &b_tiled, nullptr) != FS_OK) {
    std::fprintf(stderr, "failed allocating tiled probe buffer\n");
    fs_context_destroy(ctx);
    return 1;
  }
  const fs_status_t tiled_handoff_st =
      fs_buffer_handoff(ctx, b_tiled, FS_DEV_GPU_VK, FS_DEV_TPU, nullptr);
  const int tiled_fail_closed = (tiled_handoff_st == FS_ERR_UNSUPPORTED) ? 1 : 0;
  fs_buffer_release(ctx, b_tiled);
  if (!tiled_fail_closed) {
    std::fprintf(stderr, "expected FS_ERR_UNSUPPORTED for tiled GPU -> linear TPU handoff, got %s\n",
                 fs_status_string(tiled_handoff_st));
    fs_context_destroy(ctx);
    return 1;
  }

  /* 3. Allocate reconciled zero-copy DMA-BUF tensors & run GPU -> CPU -> TPU pipeline */
  const uint32_t M_PRE = 64u;
  const uint32_t M_DEC = 1u;
  const uint32_t D     = 32u;

  fs_dmabuf_t *b_in = nullptr, *b_w = nullptr, *b_pre = nullptr, *b_rms = nullptr, *b_dec = nullptr;
  if (fs_buffer_alloc_reconciled(ctx, M_PRE * D * sizeof(float), shared_attrs, 3u, cpu_dev, 0u,
                                 &b_in, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, D * D * sizeof(float), shared_attrs, 3u, FS_DEV_GPU_VK,
                                 FS_BUF_IMMUTABLE_WEIGHT, &b_w, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, M_PRE * D * sizeof(float), shared_attrs, 3u, FS_DEV_GPU_VK,
                                 0u, &b_pre, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, M_PRE * D * sizeof(float), shared_attrs, 3u, cpu_dev, 0u,
                                 &b_rms, nullptr) != FS_OK ||
      fs_buffer_alloc_reconciled(ctx, M_DEC * D * sizeof(float), shared_attrs, 3u, FS_DEV_TPU, 0u,
                                 &b_dec, nullptr) != FS_OK) {
    std::fprintf(stderr, "reconciled buffer allocation failed: %s\n", fs_context_last_error(ctx));
    fs_context_destroy(ctx);
    return 1;
  }

  auto* pin = static_cast<float*>(b_in->host_ptr);
  auto* pw  = static_cast<float*>(b_w->host_ptr);
  for (uint32_t i = 0; i < M_PRE * D; ++i) pin[i] = std::sin(static_cast<float>(i) * 0.03f) * 0.5f;
  for (uint32_t i = 0; i < D * D; ++i) pw[i]      = std::cos(static_cast<float>(i) * 0.05f) * 0.25f;

  /* Explicit CPU cache flush + sync_file export check */
  if (fs_buffer_dma_sync(ctx, b_in, static_cast<uint32_t>(FS_DMA_SYNC_END) |
                                        static_cast<uint32_t>(FS_DMA_SYNC_RW)) != FS_OK) {
    std::fprintf(stderr, "fs_buffer_dma_sync failed\n");
    fs_context_destroy(ctx);
    return 1;
  }
  fs_fence_t exported_fence = fs_fence_init(cpu_dev);
  if (fs_buffer_export_sync_file(ctx, b_in, static_cast<uint32_t>(FS_DMA_SYNC_RW),
                                 &exported_fence) != FS_OK) {
    std::fprintf(stderr, "fs_buffer_export_sync_file failed\n");
    fs_context_destroy(ctx);
    return 1;
  }
  fs_fence_close(&exported_fence);

  fs_op_desc_t ops[3];
  std::memset(ops, 0, sizeof(ops));
  /* 0: GPU Prefill GEMM */
  ops[0].type = FS_OP_PREFILL_GEMM; ops[0].phase = FS_PHASE_PREFILL;
  ops[0].m_dim = M_PRE; ops[0].k_dim = D; ops[0].n_dim = D;
  ops[0].act_dtype = FS_DTYPE_F32; ops[0].weight_dtype = FS_DTYPE_F32;
  ops[0].input_a = b_in; ops[0].input_b = b_w; ops[0].output = b_pre;
  /* 1: CPU RMSNorm (zero-copy from GPU) */
  ops[1].type = FS_OP_RMS_NORM; ops[1].phase = FS_PHASE_CONTROL;
  ops[1].m_dim = M_PRE; ops[1].n_dim = D; ops[1].aux_param = 1e-5f;
  ops[1].act_dtype = FS_DTYPE_F32;
  ops[1].input_a = b_pre; ops[1].output = b_rms;
  /* 2: TPU Decode GEMV M=1 (zero-copy from CPU) */
  ops[2].type = FS_OP_DECODE_GEMV; ops[2].phase = FS_PHASE_DECODE;
  ops[2].m_dim = M_DEC; ops[2].k_dim = D; ops[2].n_dim = D;
  ops[2].act_dtype = FS_DTYPE_F32; ops[2].weight_dtype = FS_DTYPE_F32;
  ops[2].input_a = b_rms; ops[2].input_b = b_w; ops[2].output = b_dec;

  fs_plan_t* plan = nullptr;
  if (fs_plan_create(ctx, ops, 3u, &plan) != FS_OK ||
      fs_execute_plan(ctx, plan, ops, 3u) != FS_OK) {
    std::fprintf(stderr, "pipeline execution failed: %s\n", fs_context_last_error(ctx));
    fs_plan_destroy(plan);
    fs_context_destroy(ctx);
    return 1;
  }

  /* Verify exact CPU reference parity */
  std::vector<float> r_pre(M_PRE * D), r_rms(M_PRE * D), r_dec(M_DEC * D);
  ref_gemm_f32(pin, pw, r_pre.data(), M_PRE, D, D);
  ref_rms_norm(r_pre.data(), r_rms.data(), M_PRE, D, 1e-5f);
  ref_gemm_f32(r_rms.data(), pw, r_dec.data(), M_DEC, D, D);

  fs_parity_t par;
  fs_compute_parity_f32(static_cast<const float*>(b_dec->host_ptr), r_dec.data(), M_DEC * D, &par);

  fs_telemetry_t tel;
  fs_context_get_telemetry(ctx, &tel);

  const int pass = (par.exact_match == 1 && tel.fallback_copies == 0u &&
                    tel.zero_copy_handoffs >= 2u && tel.dma_sync_calls > 0u &&
                    plan->steps[0].assigned_dev == FS_DEV_GPU_VK &&
                    plan->steps[2].assigned_dev == FS_DEV_TPU)
                       ? 1
                       : 0;

  if (json_out) {
    std::printf(
        "{\"pass\":%s,\"profile\":\"%s\",\"mock\":%s,\"backing\":\"%s\","
        "\"dmabuf_fd\":%d,\"tiled_fail_closed\":%s,"
        "\"zero_copy_handoffs\":%llu,\"fallback_copies\":%llu,"
        "\"dma_sync_calls\":%llu,\"sync_file_exports\":%llu,"
        "\"max_abs_err\":%.6f,\"nmse\":%.9f}\n",
        pass ? "true" : "false", topo->soc_name, fs_context_is_mock(ctx) ? "true" : "false",
        backing_name(b_in->backing), b_in->dmabuf_fd, tiled_fail_closed ? "true" : "false",
        static_cast<unsigned long long>(tel.zero_copy_handoffs),
        static_cast<unsigned long long>(tel.fallback_copies),
        static_cast<unsigned long long>(tel.dma_sync_calls),
        static_cast<unsigned long long>(tel.sync_file_exports), par.max_abs_err, par.nmse);
  } else {
    std::printf(
        "[%s] freesched_dmabuf_smoke profile=%s mock=%d backing=%s fd=%d "
        "tiled_fail_closed=%d handoffs=%llu fallback_copies=%llu dma_sync=%llu "
        "sync_exports=%llu max_abs_err=%.6f nmse=%.9f\n",
        pass ? "PASS" : "FAIL", topo->soc_name, fs_context_is_mock(ctx),
        backing_name(b_in->backing), b_in->dmabuf_fd, tiled_fail_closed,
        static_cast<unsigned long long>(tel.zero_copy_handoffs),
        static_cast<unsigned long long>(tel.fallback_copies),
        static_cast<unsigned long long>(tel.dma_sync_calls),
        static_cast<unsigned long long>(tel.sync_file_exports), par.max_abs_err, par.nmse);
  }

  fs_plan_destroy(plan);
  fs_context_destroy(ctx);
  return pass ? 0 : 1;
}
