/*
 * Comprehensive C++17 unit & integration test suite for freesched.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "freesched.h"
#include "ggml_freesilicon.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
      return false;                                                                    \
    }                                                                                  \
  } while (0)

/* =========================================================================
 * Independent Reference Math Helpers (for numerical parity verification)
 * ========================================================================= */

static void ref_quantize_q8_0(const float* src, uint32_t rows, uint32_t k_dim,
                              fs_block_q8_0_t* dst) {
  const uint32_t nb = k_dim / FS_QK8_0;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t b = 0; b < nb; ++b) {
      const float* row = src + static_cast<size_t>(r) * k_dim + b * FS_QK8_0;
      float        amax = 0.0f;
      for (uint32_t i = 0; i < FS_QK8_0; ++i) {
        const float v = std::fabs(row[i]);
        if (v > amax) amax = v;
      }
      const float scale = amax > 0.0f ? (amax / 127.0f) : 0.0f;
      const uint16_t d16 = fs_f32_to_f16(scale);
      const float    d   = fs_f16_to_f32(d16);
      const float    inv = d > 0.0f ? (1.0f / d) : 0.0f;
      fs_block_q8_0_t& blk = dst[static_cast<size_t>(r) * nb + b];
      blk.d = d16;
      for (uint32_t i = 0; i < FS_QK8_0; ++i) {
        long q = std::lround(row[i] * inv);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        blk.qs[i] = static_cast<int8_t>(q);
      }
    }
  }
}

static void ref_quantize_q4_0(const float* src, uint32_t rows, uint32_t k_dim,
                              fs_block_q4_0_t* dst) {
  const uint32_t nb = k_dim / FS_QK8_0;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t b = 0; b < nb; ++b) {
      const float* row  = src + static_cast<size_t>(r) * k_dim + b * FS_QK8_0;
      float        amax = 0.0f;
      for (uint32_t i = 0; i < FS_QK8_0; ++i) {
        const float v = std::fabs(row[i]);
        if (v > amax) amax = v;
      }
      const float    scale = amax > 0.0f ? (amax / 7.0f) : 0.0f;
      const uint16_t d16   = fs_f32_to_f16(scale);
      const float    d     = fs_f16_to_f32(d16);
      const float    inv   = d > 0.0f ? (1.0f / d) : 0.0f;
      fs_block_q4_0_t& blk = dst[static_cast<size_t>(r) * nb + b];
      blk.d = d16;
      for (uint32_t j = 0; j < FS_QK8_0 / 2u; ++j) {
        long q0 = std::lround(row[j] * inv) + 8;
        long q1 = std::lround(row[j + 16u] * inv) + 8;
        if (q0 < 0) q0 = 0;
        if (q0 > 15) q0 = 15;
        if (q1 < 0) q1 = 0;
        if (q1 > 15) q1 = 15;
        blk.qs[j] = static_cast<uint8_t>((q0 & 0x0f) | ((q1 & 0x0f) << 4));
      }
    }
  }
}

static void ref_dequant_q8_0(const fs_block_q8_0_t* src, uint32_t rows, uint32_t k_dim,
                             float* dst) {
  const uint32_t nb = k_dim / FS_QK8_0;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t b = 0; b < nb; ++b) {
      const fs_block_q8_0_t& blk = src[static_cast<size_t>(r) * nb + b];
      const float            d   = fs_f16_to_f32(blk.d);
      for (uint32_t i = 0; i < FS_QK8_0; ++i) {
        dst[static_cast<size_t>(r) * k_dim + b * FS_QK8_0 + i] = d * static_cast<float>(blk.qs[i]);
      }
    }
  }
}

static void ref_dequant_q4_0(const fs_block_q4_0_t* src, uint32_t rows, uint32_t k_dim,
                             float* dst) {
  const uint32_t nb = k_dim / FS_QK8_0;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t b = 0; b < nb; ++b) {
      const fs_block_q4_0_t& blk = src[static_cast<size_t>(r) * nb + b];
      const float            d   = fs_f16_to_f32(blk.d);
      for (uint32_t j = 0; j < FS_QK8_0 / 2u; ++j) {
        const int lo = static_cast<int>(blk.qs[j] & 0x0fu) - 8;
        const int hi = static_cast<int>(blk.qs[j] >> 4u) - 8;
        dst[static_cast<size_t>(r) * k_dim + b * FS_QK8_0 + j]       = d * static_cast<float>(lo);
        dst[static_cast<size_t>(r) * k_dim + b * FS_QK8_0 + j + 16u] = d * static_cast<float>(hi);
      }
    }
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

static void ref_rms_norm(const float* x, const float* w, float* y, uint32_t m, uint32_t n,
                         float eps) {
  for (uint32_t r = 0; r < m; ++r) {
    double ss = 0.0;
    for (uint32_t c = 0; c < n; ++c) {
      const double v = static_cast<double>(x[static_cast<size_t>(r) * n + c]);
      ss += v * v;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
    for (uint32_t c = 0; c < n; ++c) {
      y[static_cast<size_t>(r) * n + c] = x[static_cast<size_t>(r) * n + c] * inv * w[c];
    }
  }
}

static void ref_softmax(const float* x, float* y, uint32_t m, uint32_t n, float scale) {
  for (uint32_t r = 0; r < m; ++r) {
    float mx = -1e30f;
    for (uint32_t c = 0; c < n; ++c) {
      const float v = x[static_cast<size_t>(r) * n + c] * scale;
      if (v > mx) mx = v;
    }
    double sum = 0.0;
    for (uint32_t c = 0; c < n; ++c) {
      sum += std::exp(static_cast<double>(x[static_cast<size_t>(r) * n + c] * scale - mx));
    }
    for (uint32_t c = 0; c < n; ++c) {
      const double v = std::exp(static_cast<double>(x[static_cast<size_t>(r) * n + c] * scale - mx));
      y[static_cast<size_t>(r) * n + c] = static_cast<float>(v / sum);
    }
  }
}

static void ref_rope(const float* x, float* y, uint32_t m, uint32_t n, uint32_t pos_offset) {
  for (uint32_t r = 0; r < m; ++r) {
    const double pos = static_cast<double>(pos_offset + r);
    for (uint32_t c = 0; c < n; c += 2u) {
      const double freq  = std::pow(10000.0, -static_cast<double>(c) / static_cast<double>(n));
      const double angle = pos * freq;
      const double ca    = std::cos(angle);
      const double sa    = std::sin(angle);
      const double x0    = x[static_cast<size_t>(r) * n + c];
      const double x1    = x[static_cast<size_t>(r) * n + c + 1u];
      y[static_cast<size_t>(r) * n + c]      = static_cast<float>(x0 * ca - x1 * sa);
      y[static_cast<size_t>(r) * n + c + 1u] = static_cast<float>(x0 * sa + x1 * ca);
    }
  }
}

static void ref_swiglu(const float* x, const float* wg, const float* wu, const float* wd,
                       float* y, uint32_t m, uint32_t k, uint32_t h, uint32_t n) {
  std::vector<float> mid(h, 0.0f);
  for (uint32_t mi = 0; mi < m; ++mi) {
    for (uint32_t hi = 0; hi < h; ++hi) {
      double g = 0.0;
      double u = 0.0;
      for (uint32_t ki = 0; ki < k; ++ki) {
        const double xv = x[static_cast<size_t>(mi) * k + ki];
        g += xv * wg[static_cast<size_t>(hi) * k + ki];
        u += xv * wu[static_cast<size_t>(hi) * k + ki];
      }
      const float gf = static_cast<float>(g);
      const float uf = static_cast<float>(u);
      mid[hi]        = (gf / (1.0f + std::exp(-gf))) * uf;
    }
    for (uint32_t ni = 0; ni < n; ++ni) {
      double acc = 0.0;
      for (uint32_t hi = 0; hi < h; ++hi) {
        acc += static_cast<double>(mid[hi]) * static_cast<double>(wd[static_cast<size_t>(ni) * h + hi]);
      }
      y[static_cast<size_t>(mi) * n + ni] = static_cast<float>(acc);
    }
  }
}

/* =========================================================================
 * Test 1: Hardware Profiles & Topology Capabilities
 * ========================================================================= */
static bool test_topology_profiles() {
  fs_topology_t laguna;
  CHECK(fs_topology_probe(FS_PROFILE_PIXEL10_LAGUNA, &laguna) == FS_OK);
  CHECK(std::strcmp(laguna.soc_name, "laguna") == 0);
  CHECK(laguna.unified_ram_bytes == 16ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK((laguna.global_caps & FS_CAP_UNIFIED_MEMORY) != 0u);
  CHECK((laguna.global_caps & FS_CAP_DMABUF) != 0u);
  CHECK((laguna.devices[FS_DEV_TPU].caps & FS_CAP_EDGETPU_MAP_DMABUF) != 0u);
  CHECK((laguna.devices[FS_DEV_GPU_VK].caps & FS_CAP_VK_EXT_MEMORY_FD) != 0u);
  CHECK(laguna.devices[FS_DEV_TPU].max_inflight == 16u);
  CHECK(laguna.devices[FS_DEV_CPU_SME2].available == 0);
  CHECK(laguna.devices[FS_DEV_TPU].typical_watts < laguna.devices[FS_DEV_GPU_VK].typical_watts);

  fs_topology_t malibu;
  CHECK(fs_topology_probe(FS_PROFILE_PIXEL11_MALIBU, &malibu) == FS_OK);
  CHECK(std::strcmp(malibu.soc_name, "malibu") == 0);
  CHECK(malibu.devices[FS_DEV_CPU_SME2].available == 1);
  CHECK((malibu.devices[FS_DEV_CPU_SME2].caps & FS_CAP_SME2_MATMUL) != 0u);
  CHECK(malibu.devices[FS_DEV_TPU].peak_tops_int8 > laguna.devices[FS_DEV_TPU].peak_tops_int8);
  CHECK(malibu.devices[FS_DEV_GPU_VK].peak_gflops_fp16 > laguna.devices[FS_DEV_GPU_VK].peak_gflops_fp16);

  fs_topology_t mock_topo;
  CHECK(fs_topology_probe(FS_PROFILE_HOST_MOCK, &mock_topo) == FS_OK);
  CHECK(std::strcmp(mock_topo.soc_name, "host_mock") == 0);
  CHECK((mock_topo.global_caps & FS_CAP_MOCK) != 0u);
  CHECK(fs_topology_has_device(&mock_topo, FS_DEV_TPU) == 1);
  CHECK(fs_topology_has_device(&mock_topo, FS_DEV_GPU_VK) == 1);
  CHECK(fs_topology_has_device(&mock_topo, FS_DEV_CPU_SME2) == 1);

  return true;
}

/* =========================================================================
 * Test 2: Zero-Copy DMA-BUF Ownership, Fence & Pool Invariants
 * ========================================================================= */
static bool test_dmabuf_ownership_and_races() {
  fs_config_t cfg = fs_config_default();
  cfg.mock_mode   = 1;
  cfg.profile     = FS_PROFILE_PIXEL11_MALIBU;

  fs_context_t* ctx = nullptr;
  CHECK(fs_context_create(&cfg, &ctx) == FS_OK);

  fs_dmabuf_t* buf = nullptr;
  CHECK(fs_buffer_alloc(ctx, 4096u, FS_DEV_GPU_VK, FS_BUF_ZERO_INIT, &buf) == FS_OK);
  CHECK(buf != nullptr);
  CHECK(buf->owner_dev == FS_DEV_GPU_VK);
  CHECK(buf->refcount == 1u);
  const uint64_t gen1 = buf->generation;
  CHECK(gen1 == 1u);

  /* Refcounting retain/release */
  CHECK(fs_buffer_retain(ctx, buf) == FS_OK);
  CHECK(buf->refcount == 2u);
  fs_buffer_release(ctx, buf);
  CHECK(buf->refcount == 1u);

  /* Detect wrong-owner access (TPU trying to touch GPU-owned buffer) */
  CHECK(fs_buffer_acquire_access(ctx, buf, FS_DEV_TPU, 0u) == FS_ERR_OWNERSHIP);

  /* Detect double-owner lock race while GPU holds exclusive access */
  CHECK(fs_buffer_acquire_access(ctx, buf, FS_DEV_GPU_VK, gen1) == FS_OK);
  CHECK(fs_buffer_acquire_access(ctx, buf, FS_DEV_GPU_VK, gen1) == FS_ERR_OWNERSHIP);

  /* Detect handoff attempted while buffer is actively locked */
  fs_fence_t fence = fs_fence_init(FS_DEV_GPU_VK);
  CHECK(fs_buffer_handoff(ctx, buf, FS_DEV_GPU_VK, FS_DEV_TPU, &fence) == FS_ERR_OWNERSHIP);
  CHECK(fs_buffer_release_access(ctx, buf, FS_DEV_GPU_VK) == FS_OK);

  /* Detect errored fence rejected cleanly */
  fs_fence_t bad_fence = fs_fence_init(FS_DEV_GPU_VK);
  bad_fence.state      = FS_FENCE_ERROR;
  CHECK(fs_buffer_handoff(ctx, buf, FS_DEV_GPU_VK, FS_DEV_TPU, &bad_fence) == FS_ERR_FENCE);
  CHECK(buf->owner_dev == FS_DEV_GPU_VK);

  /* Valid zero-copy GPU -> TPU handoff */
  fence = fs_fence_init(FS_DEV_GPU_VK);
  CHECK(fs_buffer_handoff(ctx, buf, FS_DEV_GPU_VK, FS_DEV_TPU, &fence) == FS_OK);
  CHECK(buf->owner_dev == FS_DEV_TPU);
  CHECK(buf->generation == gen1 + 1u);
  CHECK(buf->tpu_iova != 0u);
  CHECK(fence.state == FS_FENCE_SIGNALED);

  /* Detect use-after-handoff: GPU tries to handoff or access after transferring to TPU */
  CHECK(fs_buffer_handoff(ctx, buf, FS_DEV_GPU_VK, FS_DEV_CPU_SME2, nullptr) == FS_ERR_OWNERSHIP);
  CHECK(fs_buffer_acquire_access(ctx, buf, FS_DEV_GPU_VK, 0u) == FS_ERR_OWNERSHIP);
  /* Detect stale-generation access even by current owner */
  CHECK(fs_buffer_acquire_access(ctx, buf, FS_DEV_TPU, gen1) == FS_ERR_OWNERSHIP);
  /* Valid current-generation access succeeds */
  CHECK(fs_buffer_acquire_access(ctx, buf, FS_DEV_TPU, buf->generation) == FS_OK);
  CHECK(fs_buffer_release_access(ctx, buf, FS_DEV_TPU) == FS_OK);

  /* Verify zero-copy telemetry */
  fs_telemetry_t tel;
  CHECK(fs_context_get_telemetry(ctx, &tel) == FS_OK);
  CHECK(tel.zero_copy_handoffs == 1u);
  CHECK(tel.fallback_copies == 0u);

  /* Immutable weight buffers allow concurrent multi-device reads without handoffs */
  fs_dmabuf_t* wbuf = nullptr;
  CHECK(fs_buffer_alloc(ctx, 4096u, FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &wbuf) == FS_OK);
  CHECK(fs_buffer_acquire_access(ctx, wbuf, FS_DEV_TPU, 0u) == FS_OK);
  CHECK(fs_buffer_acquire_access(ctx, wbuf, FS_DEV_GPU_VK, 0u) == FS_OK);
  CHECK(fs_buffer_release_access(ctx, wbuf, FS_DEV_GPU_VK) == FS_OK);
  CHECK(fs_buffer_release_access(ctx, wbuf, FS_DEV_TPU) == FS_OK);
  fs_buffer_release(ctx, wbuf);

  /* Buffer pool reuse check */
  fs_buffer_release(ctx, buf);
  size_t   pooled_bytes = 0, pooled_count = 0;
  uint64_t hits = 0, misses = 0;
  CHECK(fs_buffer_pool_stats(ctx, &pooled_bytes, &pooled_count, &hits, &misses) == FS_OK);
  CHECK(pooled_count == 2u);
  CHECK(pooled_bytes >= 8192u);

  fs_dmabuf_t* reused = nullptr;
  CHECK(fs_buffer_alloc(ctx, 4096u, FS_DEV_CPU_SME2, FS_BUF_ZERO_INIT, &reused) == FS_OK);
  CHECK(fs_buffer_pool_stats(ctx, &pooled_bytes, &pooled_count, &hits, &misses) == FS_OK);
  CHECK(hits == 1u);
  fs_buffer_release(ctx, reused);

  /* Diagnostic fallback copy path */
  fs_dmabuf_t* fbuf = nullptr;
  CHECK(fs_buffer_alloc(ctx, 1024u, FS_DEV_GPU_VK, FS_BUF_FORCE_HOST_COPY | FS_BUF_NO_POOL, &fbuf) == FS_OK);
  CHECK(fs_buffer_handoff(ctx, fbuf, FS_DEV_GPU_VK, FS_DEV_TPU, nullptr) == FS_OK);
  CHECK(fs_context_get_telemetry(ctx, &tel) == FS_OK);
  CHECK(tel.fallback_copies == 1u);
  fs_buffer_release(ctx, fbuf);

  /* External DMA-BUF import (zero copy wrapper around external fd/ptr) */
  float        ext_mem[16] = {0.0f};
  fs_dmabuf_t* imported    = nullptr;
  CHECK(fs_buffer_import_dmabuf(ctx, -1, sizeof(ext_mem), 0u, ext_mem, FS_DEV_CPU_SME2, 0u,
                                &imported) == FS_OK);
  CHECK(imported->host_ptr == ext_mem);
  fs_buffer_release(ctx, imported);

  fs_context_destroy(ctx);
  return true;
}

/* =========================================================================
 * Test 3: Heterogeneous LLM Pipeline (Prefill GPU -> Attn CPU -> Decode TPU)
 *         + Exact FP32 & FP16 Reference Parity + Zero Fallback Copies
 * ========================================================================= */
static bool test_heterogeneous_llm_pipeline_and_parity() {
  fs_config_t cfg = fs_config_default();
  cfg.mock_mode   = 1;
  cfg.profile     = FS_PROFILE_PIXEL11_MALIBU;

  fs_context_t* ctx = nullptr;
  CHECK(fs_context_create(&cfg, &ctx) == FS_OK);

  const uint32_t M_PRE = 512u;
  const uint32_t M_DEC = 1u;
  const uint32_t K     = 64u;
  const uint32_t N     = 64u;
  const uint32_t H     = 96u;

  /* Allocate unified DMA-BUF tensors */
  fs_dmabuf_t *b_pre_in = nullptr, *b_pre_w = nullptr, *b_pre_out = nullptr;
  fs_dmabuf_t *b_rms_w = nullptr, *b_rms_out = nullptr, *b_sm_out = nullptr, *b_rope_out = nullptr;
  fs_dmabuf_t *b_dec_in = nullptr, *b_dec_w_q8 = nullptr, *b_dec_out = nullptr;
  fs_dmabuf_t *b_wg_q4 = nullptr, *b_wu_q4 = nullptr, *b_wd_q4 = nullptr, *b_swiglu_out = nullptr;

  CHECK(fs_buffer_alloc(ctx, M_PRE * K * sizeof(float), FS_DEV_CPU_SME2, 0u, &b_pre_in) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, N * K * sizeof(float), FS_DEV_GPU_VK, FS_BUF_IMMUTABLE_WEIGHT, &b_pre_w) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, M_PRE * N * sizeof(float), FS_DEV_GPU_VK, 0u, &b_pre_out) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, N * sizeof(float), FS_DEV_CPU_SME2, FS_BUF_IMMUTABLE_WEIGHT, &b_rms_w) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, M_PRE * N * sizeof(float), FS_DEV_CPU_SME2, 0u, &b_rms_out) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, M_PRE * N * sizeof(float), FS_DEV_CPU_SME2, 0u, &b_sm_out) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, M_PRE * N * sizeof(float), FS_DEV_CPU_SME2, 0u, &b_rope_out) == FS_OK);

  const size_t q8_bytes = static_cast<size_t>(N) * (K / FS_QK8_0) * sizeof(fs_block_q8_0_t);
  const size_t q4_hk    = static_cast<size_t>(H) * (K / FS_QK8_0) * sizeof(fs_block_q4_0_t);
  const size_t q4_nh    = static_cast<size_t>(N) * (H / FS_QK8_0) * sizeof(fs_block_q4_0_t);

  CHECK(fs_buffer_alloc(ctx, M_DEC * K * sizeof(float), FS_DEV_CPU_SME2, 0u, &b_dec_in) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, q8_bytes, FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &b_dec_w_q8) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, M_DEC * N * sizeof(float), FS_DEV_TPU, 0u, &b_dec_out) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, q4_hk, FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &b_wg_q4) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, q4_hk, FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &b_wu_q4) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, q4_nh, FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &b_wd_q4) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, M_DEC * N * sizeof(float), FS_DEV_TPU, 0u, &b_swiglu_out) == FS_OK);

  /* Deterministic test data */
  auto* pre_in = static_cast<float*>(b_pre_in->host_ptr);
  auto* pre_w  = static_cast<float*>(b_pre_w->host_ptr);
  auto* rms_w  = static_cast<float*>(b_rms_w->host_ptr);
  auto* dec_in = static_cast<float*>(b_dec_in->host_ptr);

  for (uint32_t i = 0; i < M_PRE * K; ++i) {
    pre_in[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.5f;
  }
  for (uint32_t i = 0; i < N * K; ++i) {
    pre_w[i] = std::cos(static_cast<float>(i) * 0.017f) * 0.25f;
  }
  for (uint32_t i = 0; i < N; ++i) {
    rms_w[i] = 0.8f + 0.005f * static_cast<float>(i);
  }
  for (uint32_t i = 0; i < M_DEC * K; ++i) {
    dec_in[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.75f;
  }

  std::vector<float> dec_w_src(N * K), wg_src(H * K), wu_src(H * K), wd_src(N * H);
  for (uint32_t i = 0; i < N * K; ++i) dec_w_src[i] = std::sin(static_cast<float>(i) * 0.029f);
  for (uint32_t i = 0; i < H * K; ++i) wg_src[i]    = std::cos(static_cast<float>(i) * 0.021f) * 0.4f;
  for (uint32_t i = 0; i < H * K; ++i) wu_src[i]    = std::sin(static_cast<float>(i) * 0.031f) * 0.4f;
  for (uint32_t i = 0; i < N * H; ++i) wd_src[i]    = std::cos(static_cast<float>(i) * 0.019f) * 0.3f;

  ref_quantize_q8_0(dec_w_src.data(), N, K, static_cast<fs_block_q8_0_t*>(b_dec_w_q8->host_ptr));
  ref_quantize_q4_0(wg_src.data(), H, K, static_cast<fs_block_q4_0_t*>(b_wg_q4->host_ptr));
  ref_quantize_q4_0(wu_src.data(), H, K, static_cast<fs_block_q4_0_t*>(b_wu_q4->host_ptr));
  ref_quantize_q4_0(wd_src.data(), N, H, static_cast<fs_block_q4_0_t*>(b_wd_q4->host_ptr));

  /* Build 6-stage heterogeneous transformer graph */
  fs_op_desc_t ops[6];
  std::memset(ops, 0, sizeof(ops));

  /* 0: Prompt Prefill GEMM M=512 -> GPU_VK */
  ops[0].op_id = 0; ops[0].type = FS_OP_PREFILL_GEMM; ops[0].phase = FS_PHASE_PREFILL;
  ops[0].m_dim = M_PRE; ops[0].k_dim = K; ops[0].n_dim = N;
  ops[0].act_dtype = FS_DTYPE_F32; ops[0].weight_dtype = FS_DTYPE_F32;
  ops[0].input_a = b_pre_in; ops[0].input_b = b_pre_w; ops[0].output = b_pre_out;

  /* 1: RMSNorm -> CPU_SME2 (zero-copy handoff from GPU_VK) */
  ops[1].op_id = 1; ops[1].type = FS_OP_RMS_NORM; ops[1].phase = FS_PHASE_CONTROL;
  ops[1].m_dim = M_PRE; ops[1].n_dim = N; ops[1].aux_param = 1e-5f;
  ops[1].act_dtype = FS_DTYPE_F32; ops[1].weight_dtype = FS_DTYPE_F32;
  ops[1].input_a = b_pre_out; ops[1].input_b = b_rms_w; ops[1].output = b_rms_out;

  /* 2: Attention Softmax -> CPU_SME2 */
  ops[2].op_id = 2; ops[2].type = FS_OP_ATTN_SOFTMAX; ops[2].phase = FS_PHASE_CONTROL;
  ops[2].m_dim = M_PRE; ops[2].n_dim = N; ops[2].aux_param = 0.125f;
  ops[2].act_dtype = FS_DTYPE_F32;
  ops[2].input_a = b_rms_out; ops[2].output = b_sm_out;

  /* 3: RoPE -> CPU_SME2 */
  ops[3].op_id = 3; ops[3].type = FS_OP_ROPE; ops[3].phase = FS_PHASE_CONTROL;
  ops[3].m_dim = M_PRE; ops[3].n_dim = N; ops[3].h_dim = N; ops[3].pos_offset = 16u;
  ops[3].act_dtype = FS_DTYPE_F32;
  ops[3].input_a = b_sm_out; ops[3].output = b_rope_out;

  /* 4: Decode GEMV M=1 (Q8_0) -> EdgeTPU */
  ops[4].op_id = 4; ops[4].type = FS_OP_DECODE_GEMV; ops[4].phase = FS_PHASE_DECODE;
  ops[4].m_dim = M_DEC; ops[4].k_dim = K; ops[4].n_dim = N;
  ops[4].act_dtype = FS_DTYPE_F32; ops[4].weight_dtype = FS_DTYPE_Q8_0;
  ops[4].input_a = b_dec_in; ops[4].input_b = b_dec_w_q8; ops[4].output = b_dec_out;

  /* 5: Decode SwiGLU M=1 (Q4_0) -> EdgeTPU (zero handoff from op 4!) */
  ops[5].op_id = 5; ops[5].type = FS_OP_SWIGLU; ops[5].phase = FS_PHASE_DECODE;
  ops[5].m_dim = M_DEC; ops[5].k_dim = K; ops[5].h_dim = H; ops[5].n_dim = N;
  ops[5].act_dtype = FS_DTYPE_F32; ops[5].weight_dtype = FS_DTYPE_Q4_0;
  ops[5].input_a = b_dec_out; ops[5].input_b = b_wg_q4; ops[5].input_c = b_wu_q4;
  ops[5].input_d = b_wd_q4; ops[5].output = b_swiglu_out;

  fs_plan_t* plan = nullptr;
  CHECK(fs_plan_create(ctx, ops, 6u, &plan) == FS_OK);
  CHECK(plan->steps[0].assigned_dev == FS_DEV_GPU_VK);
  CHECK(plan->steps[1].assigned_dev == FS_DEV_CPU_SME2);
  CHECK(plan->steps[2].assigned_dev == FS_DEV_CPU_SME2);
  CHECK(plan->steps[3].assigned_dev == FS_DEV_CPU_SME2);
  CHECK(plan->steps[4].assigned_dev == FS_DEV_TPU);
  CHECK(plan->steps[5].assigned_dev == FS_DEV_TPU);
  CHECK(plan->steps[5].requires_handoff == 0);
  CHECK(plan->planned_fallback_copies == 0u);
  CHECK(plan->degraded_ops == 0u);

  CHECK(fs_execute_plan(ctx, plan, ops, 6u) == FS_OK);

  fs_telemetry_t tel;
  CHECK(fs_context_get_telemetry(ctx, &tel) == FS_OK);
  CHECK(tel.fallback_copies == 0u);
  CHECK(tel.zero_copy_handoffs >= 3u);
  CHECK(tel.gpu_ops == 1u);
  CHECK(tel.cpu_ops == 3u);
  CHECK(tel.tpu_ops == 2u);
  CHECK(tel.gpu_us > 0u && tel.cpu_us > 0u && tel.tpu_us > 0u && tel.handoff_us > 0u);
  CHECK(tel.est_watts > 0.0f);

  /* Independent reference parity verification (bit-exact FP32) */
  std::vector<float> ref_pre(M_PRE * N), ref_rms(M_PRE * N), ref_sm(M_PRE * N), ref_rp(M_PRE * N);
  ref_gemm_f32(pre_in, pre_w, ref_pre.data(), M_PRE, K, N);
  ref_rms_norm(ref_pre.data(), rms_w, ref_rms.data(), M_PRE, N, 1e-5f);
  ref_softmax(ref_rms.data(), ref_sm.data(), M_PRE, N, 0.125f);
  ref_rope(ref_sm.data(), ref_rp.data(), M_PRE, N, 16u);

  fs_parity_t p_rope;
  CHECK(fs_compute_parity_f32(static_cast<const float*>(b_rope_out->host_ptr), ref_rp.data(),
                              M_PRE * N, &p_rope) == FS_OK);
  CHECK(p_rope.exact_match == 1);
  CHECK(p_rope.max_abs_err == 0.0f);
  CHECK(p_rope.nmse == 0.0);

  std::vector<float> w_q8_f32(N * K), wg_f32(H * K), wu_f32(H * K), wd_f32(N * H);
  ref_dequant_q8_0(static_cast<const fs_block_q8_0_t*>(b_dec_w_q8->host_ptr), N, K, w_q8_f32.data());
  ref_dequant_q4_0(static_cast<const fs_block_q4_0_t*>(b_wg_q4->host_ptr), H, K, wg_f32.data());
  ref_dequant_q4_0(static_cast<const fs_block_q4_0_t*>(b_wu_q4->host_ptr), H, K, wu_f32.data());
  ref_dequant_q4_0(static_cast<const fs_block_q4_0_t*>(b_wd_q4->host_ptr), N, H, wd_f32.data());

  std::vector<float> ref_dec(M_DEC * N), ref_sw(M_DEC * N);
  ref_gemm_f32(dec_in, w_q8_f32.data(), ref_dec.data(), M_DEC, K, N);
  ref_swiglu(ref_dec.data(), wg_f32.data(), wu_f32.data(), wd_f32.data(), ref_sw.data(), M_DEC, K,
             H, N);

  fs_parity_t p_swiglu;
  CHECK(fs_compute_parity_f32(static_cast<const float*>(b_swiglu_out->host_ptr), ref_sw.data(),
                              M_DEC * N, &p_swiglu) == FS_OK);
  CHECK(p_swiglu.exact_match == 1);
  CHECK(p_swiglu.max_abs_err == 0.0f);
  CHECK(p_swiglu.nmse == 0.0);

  /* Also verify bit-exact FP16 activation/weight parity */
  fs_dmabuf_t *f16_a = nullptr, *f16_w = nullptr, *f16_c = nullptr;
  CHECK(fs_buffer_alloc(ctx, 4u * 32u * sizeof(uint16_t), FS_DEV_TPU, 0u, &f16_a) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 8u * 32u * sizeof(uint16_t), FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT, &f16_w) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 4u * 8u * sizeof(uint16_t), FS_DEV_TPU, 0u, &f16_c) == FS_OK);
  auto* pa16 = static_cast<uint16_t*>(f16_a->host_ptr);
  auto* pw16 = static_cast<uint16_t*>(f16_w->host_ptr);
  std::vector<float> a_ref16(4u * 32u), w_ref16(8u * 32u), c_ref16(4u * 8u), c_act16(4u * 8u);
  for (uint32_t i = 0; i < 4u * 32u; ++i) {
    pa16[i]    = fs_f32_to_f16(0.25f * static_cast<float>((i % 7u) - 3));
    a_ref16[i] = fs_f16_to_f32(pa16[i]);
  }
  for (uint32_t i = 0; i < 8u * 32u; ++i) {
    pw16[i]    = fs_f32_to_f16(0.5f * static_cast<float>((i % 5u) - 2));
    w_ref16[i] = fs_f16_to_f32(pw16[i]);
  }
  fs_op_desc_t op16;
  std::memset(&op16, 0, sizeof(op16));
  op16.type = FS_OP_DECODE_GEMV; op16.m_dim = 4u; op16.k_dim = 32u; op16.n_dim = 8u;
  op16.act_dtype = FS_DTYPE_F16; op16.weight_dtype = FS_DTYPE_F16;
  op16.input_a = f16_a; op16.input_b = f16_w; op16.output = f16_c;
  CHECK(fs_execute_op(ctx, &op16, nullptr) == FS_OK);
  ref_gemm_f32(a_ref16.data(), w_ref16.data(), c_ref16.data(), 4u, 32u, 8u);
  const auto* pc16 = static_cast<const uint16_t*>(f16_c->host_ptr);
  for (uint32_t i = 0; i < 4u * 8u; ++i) {
    c_ref16[i] = fs_f16_to_f32(fs_f32_to_f16(c_ref16[i]));
    c_act16[i] = fs_f16_to_f32(pc16[i]);
  }
  fs_parity_t p_f16;
  CHECK(fs_compute_parity_f32(c_act16.data(), c_ref16.data(), 4u * 8u, &p_f16) == FS_OK);
  CHECK(p_f16.exact_match == 1);

  fs_plan_destroy(plan);
  fs_context_destroy(ctx);
  return true;
}

/* =========================================================================
 * Test 4: Continuity Bonus, Explicit Overrides & Graceful Degradation
 * ========================================================================= */
static bool test_routing_overrides_continuity_and_degradation() {
  /* 4a: Continuity bonus prevents ping-ponging on small GEMM already on GPU */
  fs_config_t cfg = fs_config_default();
  cfg.mock_mode   = 1;
  cfg.profile     = FS_PROFILE_PIXEL10_LAGUNA;
  cfg.continuity_bonus_us = 120.0f;

  fs_context_t* ctx = nullptr;
  CHECK(fs_context_create(&cfg, &ctx) == FS_OK);

  fs_dmabuf_t *ba = nullptr, *bw = nullptr, *bo = nullptr;
  CHECK(fs_buffer_alloc(ctx, 16u * 32u * sizeof(float), FS_DEV_GPU_VK, FS_BUF_ZERO_INIT, &ba) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 32u * 32u * sizeof(float), FS_DEV_GPU_VK, FS_BUF_IMMUTABLE_WEIGHT | FS_BUF_ZERO_INIT, &bw) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 16u * 32u * sizeof(float), FS_DEV_GPU_VK, FS_BUF_ZERO_INIT, &bo) == FS_OK);

  fs_op_desc_t small_gemm;
  std::memset(&small_gemm, 0, sizeof(small_gemm));
  small_gemm.type = FS_OP_GEMM; small_gemm.m_dim = 16u; small_gemm.k_dim = 32u; small_gemm.n_dim = 32u;
  small_gemm.act_dtype = FS_DTYPE_F32; small_gemm.weight_dtype = FS_DTYPE_F32;
  small_gemm.input_a = ba; small_gemm.input_b = bw; small_gemm.output = bo;

  /* With continuity bonus, stays on GPU_VK (zero handoff) */
  fs_plan_t* p_keep = nullptr;
  CHECK(fs_plan_create(ctx, &small_gemm, 1u, &p_keep) == FS_OK);
  CHECK(p_keep->steps[0].assigned_dev == FS_DEV_GPU_VK);
  CHECK(p_keep->steps[0].requires_handoff == 0);
  fs_plan_destroy(p_keep);

  /* Without continuity bonus, switches to TPU */
  small_gemm.route_flags = FS_ROUTE_NO_CONTINUITY_BONUS;
  fs_plan_t* p_switch = nullptr;
  CHECK(fs_plan_create(ctx, &small_gemm, 1u, &p_switch) == FS_OK);
  CHECK(p_switch->steps[0].assigned_dev == FS_DEV_TPU);
  CHECK(p_switch->steps[0].requires_handoff == 1);
  fs_plan_destroy(p_switch);

  /* Explicit override forces CPU_NEON */
  small_gemm.route_flags   = FS_ROUTE_FORCE_DEVICE;
  small_gemm.preferred_dev = FS_DEV_CPU_NEON;
  fs_plan_t* p_force = nullptr;
  CHECK(fs_plan_create(ctx, &small_gemm, 1u, &p_force) == FS_OK);
  CHECK(p_force->steps[0].assigned_dev == FS_DEV_CPU_NEON);
  fs_plan_destroy(p_force);

  /* Forced offline device (SME2 is offline on laguna) fails cleanly */
  small_gemm.preferred_dev = FS_DEV_CPU_SME2;
  fs_plan_t* p_fail = nullptr;
  CHECK(fs_plan_create(ctx, &small_gemm, 1u, &p_fail) == FS_ERR_NO_BACKEND);

  /* TPU spline routing when FS_ROUTE_ALLOW_SPLINE is set */
  fs_op_desc_t spline_op;
  std::memset(&spline_op, 0, sizeof(spline_op));
  spline_op.type = FS_OP_ELEMENTWISE; spline_op.m_dim = 1u; spline_op.n_dim = 32u;
  spline_op.act_dtype = FS_DTYPE_F32; spline_op.activation = FS_ACT_SILU;
  spline_op.route_flags = FS_ROUTE_ALLOW_SPLINE | FS_ROUTE_NO_CONTINUITY_BONUS;
  spline_op.input_a = ba; spline_op.output = bo;
  fs_plan_t* p_sp = nullptr;
  CHECK(fs_plan_create(ctx, &spline_op, 1u, &p_sp) == FS_OK);
  CHECK(p_sp->steps[0].assigned_dev == FS_DEV_TPU);
  fs_plan_destroy(p_sp);

  /* Vision conv routes to GPU_VK */
  fs_op_desc_t vis_op;
  std::memset(&vis_op, 0, sizeof(vis_op));
  vis_op.type = FS_OP_VISION_CONV; vis_op.phase = FS_PHASE_VISION;
  vis_op.m_dim = 196u; vis_op.k_dim = 64u; vis_op.n_dim = 64u;
  vis_op.act_dtype = FS_DTYPE_F32; vis_op.weight_dtype = FS_DTYPE_F32;
  fs_plan_t* p_vis = nullptr;
  CHECK(fs_plan_create(ctx, &vis_op, 1u, &p_vis) == FS_OK);
  CHECK(p_vis->steps[0].assigned_dev == FS_DEV_GPU_VK);
  fs_plan_destroy(p_vis);

  fs_context_destroy(ctx);

  /* 4b: Graceful degradation across GPU-offline, TPU-offline, and CPU-only modes */
  const int modes[3][2] = {
      {1, 0}, /* TPU + CPU (GPU offline) */
      {0, 1}, /* GPU + CPU (TPU offline) */
      {0, 0}, /* CPU-only (TPU & GPU offline) */
  };

  for (int m = 0; m < 3; ++m) {
    fs_config_t dcfg = fs_config_default();
    dcfg.mock_mode   = 1;
    dcfg.profile     = FS_PROFILE_PIXEL10_LAGUNA;
    dcfg.enable_tpu  = modes[m][0];
    dcfg.enable_gpu  = modes[m][1];

    fs_context_t* dctx = nullptr;
    CHECK(fs_context_create(&dcfg, &dctx) == FS_OK);

    fs_dmabuf_t *xa = nullptr, *xw = nullptr, *xo = nullptr, *xs = nullptr;
    CHECK(fs_buffer_alloc(dctx, 64u * 32u * sizeof(float), FS_DEV_CPU_NEON, 0u, &xa) == FS_OK);
    CHECK(fs_buffer_alloc(dctx, 32u * 32u * sizeof(float), FS_DEV_CPU_NEON, FS_BUF_IMMUTABLE_WEIGHT, &xw) == FS_OK);
    CHECK(fs_buffer_alloc(dctx, 64u * 32u * sizeof(float), FS_DEV_CPU_NEON, 0u, &xo) == FS_OK);
    CHECK(fs_buffer_alloc(dctx, 64u * 32u * sizeof(float), FS_DEV_CPU_NEON, 0u, &xs) == FS_OK);

    auto* pa = static_cast<float*>(xa->host_ptr);
    auto* pw = static_cast<float*>(xw->host_ptr);
    for (uint32_t i = 0; i < 64u * 32u; ++i) pa[i] = 0.1f * static_cast<float>((i % 9u) - 4);
    for (uint32_t i = 0; i < 32u * 32u; ++i) pw[i] = 0.2f * static_cast<float>((i % 7u) - 3);

    fs_op_desc_t dops[2];
    std::memset(dops, 0, sizeof(dops));
    dops[0].type = FS_OP_PREFILL_GEMM; dops[0].m_dim = 64u; dops[0].k_dim = 32u; dops[0].n_dim = 32u;
    dops[0].act_dtype = FS_DTYPE_F32; dops[0].weight_dtype = FS_DTYPE_F32;
    dops[0].input_a = xa; dops[0].input_b = xw; dops[0].output = xo;

    dops[1].type = FS_OP_DECODE_GEMV; dops[1].m_dim = 1u; dops[1].k_dim = 32u; dops[1].n_dim = 32u;
    dops[1].act_dtype = FS_DTYPE_F32; dops[1].weight_dtype = FS_DTYPE_F32;
    dops[1].input_a = xo; dops[1].input_b = xw; dops[1].output = xs;

    fs_plan_t* dp = nullptr;
    CHECK(fs_plan_create(dctx, dops, 2u, &dp) == FS_OK);
    CHECK(dp->degraded_ops >= 1u);
    if (modes[m][0] == 0) {
      CHECK(dp->steps[0].assigned_dev != FS_DEV_TPU);
      CHECK(dp->steps[1].assigned_dev != FS_DEV_TPU);
    }
    if (modes[m][1] == 0) {
      CHECK(dp->steps[0].assigned_dev != FS_DEV_GPU_VK);
      CHECK(dp->steps[1].assigned_dev != FS_DEV_GPU_VK);
    }
    if (modes[m][0] == 0 && modes[m][1] == 0) {
      CHECK(dp->steps[0].assigned_dev == FS_DEV_CPU_NEON);
      CHECK(dp->steps[1].assigned_dev == FS_DEV_CPU_NEON);
    }
    CHECK(fs_execute_plan(dctx, dp, dops, 2u) == FS_OK);

    std::vector<float> ref0(64u * 32u), ref1(32u);
    ref_gemm_f32(pa, pw, ref0.data(), 64u, 32u, 32u);
    ref_gemm_f32(ref0.data(), pw, ref1.data(), 1u, 32u, 32u);

    fs_parity_t par;
    CHECK(fs_compute_parity_f32(static_cast<const float*>(xs->host_ptr), ref1.data(), 32u, &par) == FS_OK);
    CHECK(par.exact_match == 1);

    fs_plan_destroy(dp);
    fs_context_destroy(dctx);
  }

  return true;
}

/* =========================================================================
 * Test 5: Pluggable Custom Backend Vtable + ggml_freesilicon Adapter Seam
 * ========================================================================= */
struct custom_tpu_state {
  uint32_t import_calls;
  uint32_t exec_calls;
};

static int custom_tpu_supports(void* ud, const fs_op_desc_t* op) {
  (void)ud;
  return op && op->type == FS_OP_DECODE_GEMV ? 1 : 0;
}

static fs_status_t custom_tpu_import(void* ud, fs_dmabuf_t* buf) {
  auto* st = static_cast<custom_tpu_state*>(ud);
  st->import_calls++;
  buf->tpu_iova = 0x300000000ULL + buf->buf_id;
  return FS_OK;
}

static fs_status_t custom_tpu_exec(void* ud, const fs_op_desc_t* op, const fs_fence_t* wf,
                                   fs_fence_t* sf, uint64_t* hw_us) {
  (void)wf;
  auto* st = static_cast<custom_tpu_state*>(ud);
  st->exec_calls++;
  auto* out = static_cast<float*>(op->output->host_ptr);
  for (uint32_t i = 0; i < op->m_dim * op->n_dim; ++i) {
    out[i] = 42.0f;
  }
  if (hw_us) *hw_us = 17u;
  if (sf) {
    *sf = fs_fence_init(FS_DEV_TPU);
    fs_fence_signal(sf, -1);
  }
  return FS_OK;
}

static bool test_custom_vtable_and_ggml_seam() {
  /* 5a: Custom pluggable backend vtable (simulates libopenbuenos drop-in) */
  fs_config_t cfg = fs_config_default();
  cfg.mock_mode   = 1;
  cfg.profile     = FS_PROFILE_PIXEL11_MALIBU;

  fs_context_t* ctx = nullptr;
  CHECK(fs_context_create(&cfg, &ctx) == FS_OK);

  custom_tpu_state st = {0u, 0u};
  fs_backend_ops_t ops;
  std::memset(&ops, 0, sizeof(ops));
  ops.struct_size   = static_cast<uint32_t>(sizeof(fs_backend_ops_t));
  ops.dev           = FS_DEV_TPU;
  ops.name          = "custom_openbuenos_plugin";
  ops.user_data     = &st;
  ops.supports_op   = custom_tpu_supports;
  ops.import_dmabuf = custom_tpu_import;
  ops.execute_op    = custom_tpu_exec;
  CHECK(fs_context_register_backend(ctx, &ops) == FS_OK);

  fs_dmabuf_t *a = nullptr, *b = nullptr, *c = nullptr;
  CHECK(fs_buffer_alloc(ctx, 32u * sizeof(float), FS_DEV_CPU_SME2, FS_BUF_ZERO_INIT, &a) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 4u * 32u * sizeof(float), FS_DEV_TPU, FS_BUF_IMMUTABLE_WEIGHT | FS_BUF_ZERO_INIT, &b) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 4u * sizeof(float), FS_DEV_TPU, FS_BUF_ZERO_INIT, &c) == FS_OK);

  fs_op_desc_t dec;
  std::memset(&dec, 0, sizeof(dec));
  dec.type = FS_OP_DECODE_GEMV; dec.m_dim = 1u; dec.k_dim = 32u; dec.n_dim = 4u;
  dec.act_dtype = FS_DTYPE_F32; dec.weight_dtype = FS_DTYPE_F32;
  dec.input_a = a; dec.input_b = b; dec.output = c;
  CHECK(fs_execute_op(ctx, &dec, nullptr) == FS_OK);
  CHECK(st.exec_calls == 1u);
  CHECK(st.import_calls >= 1u);
  CHECK(static_cast<const float*>(c->host_ptr)[0] == 42.0f);
  CHECK(a->tpu_iova == 0x300000000ULL + a->buf_id);

  fs_context_destroy(ctx);

  /* 5b: ggml_freesilicon graph execution seam */
  ggml_fs_backend_t* gg = ggml_fs_backend_init(&cfg);
  CHECK(gg != nullptr);

  const uint32_t M_PRE = 64u, K = 32u, N = 32u;
  fs_dmabuf_t *g_w = nullptr, *g_x = nullptr, *g_pre = nullptr, *g_rms = nullptr, *g_sm = nullptr;
  CHECK(ggml_fs_buffer_alloc(gg, N * K * sizeof(float), FS_DEV_GPU_VK, FS_BUF_IMMUTABLE_WEIGHT, &g_w) == FS_OK);
  CHECK(ggml_fs_buffer_alloc(gg, M_PRE * K * sizeof(float), FS_DEV_CPU_SME2, 0u, &g_x) == FS_OK);
  CHECK(ggml_fs_buffer_alloc(gg, M_PRE * N * sizeof(float), FS_DEV_GPU_VK, 0u, &g_pre) == FS_OK);
  CHECK(ggml_fs_buffer_alloc(gg, M_PRE * N * sizeof(float), FS_DEV_CPU_SME2, 0u, &g_rms) == FS_OK);
  CHECK(ggml_fs_buffer_alloc(gg, M_PRE * N * sizeof(float), FS_DEV_CPU_SME2, 0u, &g_sm) == FS_OK);

  auto* pw = static_cast<float*>(g_w->host_ptr);
  auto* px = static_cast<float*>(g_x->host_ptr);
  for (uint32_t i = 0; i < N * K; ++i) pw[i] = 0.125f * std::cos(static_cast<float>(i));
  for (uint32_t i = 0; i < M_PRE * K; ++i) px[i] = 0.25f * std::sin(static_cast<float>(i));

  ggml_fs_tensor_desc_t tw   = {"W_q", FS_DTYPE_F32, {K, N, 1, 1}, {0, 0, 0, 0}, g_w};
  ggml_fs_tensor_desc_t tx   = {"X_pre", FS_DTYPE_F32, {K, M_PRE, 1, 1}, {0, 0, 0, 0}, g_x};
  ggml_fs_tensor_desc_t tpre = {"Q_pre", FS_DTYPE_F32, {N, M_PRE, 1, 1}, {0, 0, 0, 0}, g_pre};
  ggml_fs_tensor_desc_t trms = {"Q_rms", FS_DTYPE_F32, {N, M_PRE, 1, 1}, {0, 0, 0, 0}, g_rms};
  ggml_fs_tensor_desc_t tsm  = {"Attn", FS_DTYPE_F32, {N, M_PRE, 1, 1}, {0, 0, 0, 0}, g_sm};

  ggml_fs_node_t nodes[3];
  std::memset(nodes, 0, sizeof(nodes));
  nodes[0].op = GGML_FS_OP_MUL_MAT; nodes[0].phase_hint = FS_PHASE_PREFILL;
  nodes[0].src0 = &tw; nodes[0].src1 = &tx; nodes[0].dst = &tpre;

  nodes[1].op = GGML_FS_OP_RMS_NORM; nodes[1].aux_param = 1e-5f;
  nodes[1].src0 = &tpre; nodes[1].dst = &trms;

  nodes[2].op = GGML_FS_OP_SOFT_MAX; nodes[2].aux_param = 0.25f;
  nodes[2].src0 = &trms; nodes[2].dst = &tsm;

  CHECK(ggml_fs_supports_node(gg, &nodes[0]) == 1);
  CHECK(ggml_fs_supports_node(gg, &nodes[1]) == 1);
  CHECK(ggml_fs_supports_node(gg, &nodes[2]) == 1);

  ggml_fs_cgraph_t cgraph = {nodes, 3u};
  fs_plan_t*       gplan  = nullptr;
  CHECK(ggml_fs_graph_compute(gg, &cgraph, &gplan) == FS_OK);
  CHECK(gplan != nullptr);
  CHECK(gplan->steps[0].assigned_dev == FS_DEV_GPU_VK);
  CHECK(gplan->steps[1].assigned_dev == FS_DEV_CPU_SME2);
  CHECK(gplan->steps[2].assigned_dev == FS_DEV_CPU_SME2);

  fs_telemetry_t gtel;
  CHECK(ggml_fs_get_telemetry(gg, &gtel) == FS_OK);
  CHECK(gtel.fallback_copies == 0u);
  CHECK(gtel.zero_copy_handoffs >= 2u);

  std::vector<float> r_pre(M_PRE * N), r_rms(M_PRE * N), r_sm(M_PRE * N), ones(N, 1.0f);
  ref_gemm_f32(px, pw, r_pre.data(), M_PRE, K, N);
  ref_rms_norm(r_pre.data(), ones.data(), r_rms.data(), M_PRE, N, 1e-5f);
  ref_softmax(r_rms.data(), r_sm.data(), M_PRE, N, 0.25f);

  fs_parity_t gpar;
  CHECK(fs_compute_parity_f32(static_cast<const float*>(g_sm->host_ptr), r_sm.data(), M_PRE * N,
                              &gpar) == FS_OK);
  CHECK(gpar.exact_match == 1);

  fs_plan_destroy(gplan);
  ggml_fs_backend_free(gg);
  return true;
}

/* =========================================================================
 * Test 6: NvSciBuf-Style Upfront Buffer Reconciliation & Router Copy Pricing
 * ========================================================================= */
static bool test_dmabuf_reconcile_and_router_pricing() {
  /* 6a: Compatible 3-engine intersection (GPU + TPU + CPU_SME2) -> 128B linear non-coherent */
  fs_buf_attr_t compat[3] = {
      fs_buf_attr_default(FS_DEV_GPU_VK),
      fs_buf_attr_default(FS_DEV_TPU),
      fs_buf_attr_default(FS_DEV_CPU_SME2),
  };
  fs_buf_reconciled_t rec;
  int                 zc_ok = 0;
  CHECK(fs_dmabuf_reconcile(compat, 3u, &rec, &zc_ok) == FS_OK);
  CHECK(zc_ok == 1);
  CHECK(rec.zero_copy_compatible == 1);
  CHECK(rec.min_align == 128u);
  CHECK(rec.stride_align == 64u);
  CHECK(rec.layout == FS_LAYOUT_LINEAR);
  CHECK(rec.coherency == FS_COHERENCY_NON_COHERENT);
  CHECK(rec.requires_dma_sync == 1);
  CHECK(rec.conflict_reason[0] == '\0');

  /* 6b: Incompatible layout (tiled PBE GPU output vs linear TPU input) */
  fs_buf_attr_t tiled_vs_linear[2] = {
      fs_buf_attr_default(FS_DEV_GPU_VK),
      fs_buf_attr_default(FS_DEV_TPU),
  };
  tiled_vs_linear[0].layout      = FS_LAYOUT_TILED;
  tiled_vs_linear[0].layout_mask = FS_LAYOUT_BIT(FS_LAYOUT_TILED);
  CHECK(fs_dmabuf_reconcile(tiled_vs_linear, 2u, &rec, &zc_ok) == FS_OK);
  CHECK(zc_ok == 0);
  CHECK(rec.zero_copy_compatible == 0);
  CHECK(std::strstr(rec.conflict_reason, "layout_mismatch") != nullptr);

  /* 6c: Incompatible coherency (strict coherent vs non-coherent) */
  fs_buf_attr_t coh_conflict[2] = {
      fs_buf_attr_default(FS_DEV_CPU_SME2),
      fs_buf_attr_default(FS_DEV_TPU),
  };
  coh_conflict[0].coherency      = FS_COHERENCY_COHERENT;
  coh_conflict[0].coherency_mask = FS_COHERENCY_BIT(FS_COHERENCY_COHERENT);
  CHECK(fs_dmabuf_reconcile(coh_conflict, 2u, &rec, &zc_ok) == FS_OK);
  CHECK(zc_ok == 0);
  CHECK(rec.zero_copy_compatible == 0);
  CHECK(std::strstr(rec.conflict_reason, "coherency_mismatch") != nullptr);

  /* 6c2: Non-power-of-two LCM alignment (48 & 64 -> 192) */
  fs_buf_attr_t non_pow2[2] = {
      fs_buf_attr_default(FS_DEV_GPU_VK),
      fs_buf_attr_default(FS_DEV_TPU),
  };
  non_pow2[0].min_align = 48u;
  non_pow2[1].min_align = 64u;
  CHECK(fs_dmabuf_reconcile(non_pow2, 2u, &rec, &zc_ok) == FS_OK);
  CHECK(zc_ok == 1);
  CHECK(rec.min_align == 192u);

  /* 6d: Router upfront pricing for tiled PBE GPU output vs linear TPU */
  fs_config_t cfg = fs_config_default();
  cfg.mock_mode   = 1;
  cfg.profile     = FS_PROFILE_PIXEL10_LAGUNA;

  fs_context_t* ctx = nullptr;
  CHECK(fs_context_create(&cfg, &ctx) == FS_OK);

  /* Allocate 1 tiled PBE GPU buffer and 1 reconciled zero-copy GPU+TPU buffer */
  fs_buf_attr_t gpu_tiled_only = fs_buf_attr_default(FS_DEV_GPU_VK);
  gpu_tiled_only.layout        = FS_LAYOUT_TILED;
  gpu_tiled_only.layout_mask   = FS_LAYOUT_BIT(FS_LAYOUT_TILED);

  fs_dmabuf_t *b_tiled = nullptr, *b_linear = nullptr, *b_w = nullptr, *b_out = nullptr;
  CHECK(fs_buffer_alloc_reconciled(ctx, 16u * 32u * sizeof(float), &gpu_tiled_only, 1u,
                                   FS_DEV_GPU_VK, FS_BUF_ZERO_INIT, &b_tiled, &rec) == FS_OK);
  CHECK(b_tiled->layout == FS_LAYOUT_TILED);
  CHECK(b_tiled->zero_copy_compatible == 1);

  CHECK(fs_buffer_alloc_reconciled(ctx, 16u * 32u * sizeof(float), compat, 3u, FS_DEV_GPU_VK,
                                   FS_BUF_ZERO_INIT, &b_linear, &rec) == FS_OK);
  CHECK(b_linear->layout == FS_LAYOUT_LINEAR);
  CHECK(b_linear->min_align == 128u);

  CHECK(fs_buffer_alloc(ctx, 32u * 32u * sizeof(float), FS_DEV_TPU,
                        FS_BUF_IMMUTABLE_WEIGHT | FS_BUF_ZERO_INIT, &b_w) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 16u * 32u * sizeof(float), FS_DEV_GPU_VK, FS_BUF_ZERO_INIT, &b_out) == FS_OK);

  /* Forced TPU on tiled GPU buffer: router flags zero_copy=0 & prices layout-convert copy upfront */
  fs_op_desc_t forced_tpu;
  std::memset(&forced_tpu, 0, sizeof(forced_tpu));
  forced_tpu.type          = FS_OP_GEMM;
  forced_tpu.m_dim         = 16u;
  forced_tpu.k_dim         = 32u;
  forced_tpu.n_dim         = 32u;
  forced_tpu.act_dtype     = FS_DTYPE_F32;
  forced_tpu.weight_dtype  = FS_DTYPE_F32;
  forced_tpu.route_flags   = FS_ROUTE_FORCE_DEVICE;
  forced_tpu.preferred_dev = FS_DEV_TPU;
  forced_tpu.input_a       = b_tiled;
  forced_tpu.input_b       = b_w;
  forced_tpu.output        = b_out;

  fs_plan_t* p_forced = nullptr;
  CHECK(fs_plan_create(ctx, &forced_tpu, 1u, &p_forced) == FS_OK);
  CHECK(p_forced->steps[0].assigned_dev == FS_DEV_TPU);
  CHECK(p_forced->steps[0].requires_handoff == 1);
  CHECK(p_forced->steps[0].zero_copy == 0);
  CHECK(p_forced->planned_fallback_copies == 1u);
  CHECK(p_forced->steps[0].est_handoff_us > 65.0f);
  CHECK(std::strcmp(p_forced->steps[0].reason, "layout_convert_copy") == 0);
  fs_plan_destroy(p_forced);

  /* Auto-routed GEMM (even with NO_CONTINUITY_BONUS) avoids tiled->linear copy by staying on GPU */
  fs_op_desc_t auto_tiled = forced_tpu;
  auto_tiled.route_flags   = FS_ROUTE_NO_CONTINUITY_BONUS;
  auto_tiled.preferred_dev = FS_DEV_NONE;

  fs_plan_t* p_auto_tiled = nullptr;
  CHECK(fs_plan_create(ctx, &auto_tiled, 1u, &p_auto_tiled) == FS_OK);
  CHECK(p_auto_tiled->steps[0].assigned_dev == FS_DEV_GPU_VK);
  CHECK(p_auto_tiled->steps[0].zero_copy == 1);
  CHECK(p_auto_tiled->planned_fallback_copies == 0u);
  fs_plan_destroy(p_auto_tiled);

  /* Same op with reconciled linear DMA-BUF moves cleanly to TPU with zero copies */
  fs_op_desc_t auto_linear = auto_tiled;
  auto_linear.input_a      = b_linear;
  fs_plan_t* p_auto_linear = nullptr;
  CHECK(fs_plan_create(ctx, &auto_linear, 1u, &p_auto_linear) == FS_OK);
  CHECK(p_auto_linear->steps[0].assigned_dev == FS_DEV_TPU);
  CHECK(p_auto_linear->steps[0].requires_handoff == 1);
  CHECK(p_auto_linear->steps[0].zero_copy == 1);
  CHECK(p_auto_linear->planned_fallback_copies == 0u);
  fs_plan_destroy(p_auto_linear);

  fs_context_destroy(ctx);
  return true;
}

int main() {
  struct {
    const char* name;
    bool (*fn)();
  } tests[] = {
      {"topology_profiles", test_topology_profiles},
      {"dmabuf_ownership_and_races", test_dmabuf_ownership_and_races},
      {"heterogeneous_llm_pipeline_and_parity", test_heterogeneous_llm_pipeline_and_parity},
      {"routing_overrides_continuity_and_degradation", test_routing_overrides_continuity_and_degradation},
      {"custom_vtable_and_ggml_seam", test_custom_vtable_and_ggml_seam},
      {"dmabuf_reconcile_and_router_pricing", test_dmabuf_reconcile_and_router_pricing},
  };

  int passed = 0;
  for (const auto& t : tests) {
    if (!t.fn()) {
      std::fprintf(stderr, "[FAIL] %s\n", t.name);
      return 1;
    }
    std::printf("[PASS] %s\n", t.name);
    passed++;
  }
  std::printf("All %d freesched test suites PASSED.\n", passed);
  return 0;
}
