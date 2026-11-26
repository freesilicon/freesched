/*
 * Strict C11 ABI verification test for freesched.h and ggml_freesilicon.h.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "freesched.h"
#include "ggml_freesilicon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(fs_block_q8_0_t) == 34u, "fs_block_q8_0_t must be 34 bytes");
_Static_assert(sizeof(fs_block_q4_0_t) == 18u, "fs_block_q4_0_t must be 18 bytes");
_Static_assert(FS_ACT_SILU == 4, "FS_ACT_SILU must match openbuenos OB_ACT_SILU");
_Static_assert(FS_ACT_GELU == 5, "FS_ACT_GELU must match openbuenos OB_ACT_GELU");

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      return 1;                                                                  \
    }                                                                            \
  } while (0)

int main(void) {
  fs_topology_t topo;
  CHECK(fs_topology_probe(FS_PROFILE_PIXEL10_LAGUNA, &topo) == FS_OK);
  CHECK(strcmp(topo.soc_name, "laguna") == 0);
  CHECK(fs_topology_has_device(&topo, FS_DEV_TPU) == 1);
  CHECK(fs_topology_has_device(&topo, FS_DEV_GPU_VK) == 1);
  CHECK(fs_topology_has_device(&topo, FS_DEV_CPU_NEON) == 1);
  CHECK(fs_topology_has_device(&topo, FS_DEV_CPU_SME2) == 0);

  fs_config_t cfg = fs_config_default();
  cfg.mock_mode   = 1;
  cfg.profile     = FS_PROFILE_PIXEL11_MALIBU;

  fs_context_t* ctx = NULL;
  CHECK(fs_context_create(&cfg, &ctx) == FS_OK);
  CHECK(ctx != NULL);
  CHECK(fs_context_is_mock(ctx) == 1);

  const fs_topology_t* live_topo = fs_context_topology(ctx);
  CHECK(live_topo != NULL);
  CHECK(strcmp(live_topo->soc_name, "malibu") == 0);
  CHECK(fs_topology_has_device(live_topo, FS_DEV_CPU_SME2) == 1);

  fs_dmabuf_t* buf_a = NULL;
  fs_dmabuf_t* buf_b = NULL;
  fs_dmabuf_t* buf_c = NULL;
  CHECK(fs_buffer_alloc(ctx, 32u * sizeof(float), FS_DEV_CPU_SME2, FS_BUF_ZERO_INIT, &buf_a) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 4u * 32u * sizeof(float), FS_DEV_TPU,
                        FS_BUF_IMMUTABLE_WEIGHT | FS_BUF_ZERO_INIT, &buf_b) == FS_OK);
  CHECK(fs_buffer_alloc(ctx, 4u * sizeof(float), FS_DEV_TPU, FS_BUF_ZERO_INIT, &buf_c) == FS_OK);

  float* a = (float*)buf_a->host_ptr;
  float* b = (float*)buf_b->host_ptr;
  for (uint32_t k = 0; k < 32u; ++k) {
    a[k] = 1.0f;
  }
  for (uint32_t n = 0; n < 4u; ++n) {
    for (uint32_t k = 0; k < 32u; ++k) {
      b[n * 32u + k] = (float)(n + 1u);
    }
  }

  fs_op_desc_t op;
  memset(&op, 0, sizeof(op));
  op.type         = FS_OP_DECODE_GEMV;
  op.phase        = FS_PHASE_DECODE;
  op.m_dim        = 1u;
  op.k_dim        = 32u;
  op.n_dim        = 4u;
  op.act_dtype    = FS_DTYPE_F32;
  op.weight_dtype = FS_DTYPE_F32;
  op.input_a      = buf_a;
  op.input_b      = buf_b;
  op.output       = buf_c;

  fs_plan_step_t step;
  CHECK(fs_execute_op(ctx, &op, &step) == FS_OK);
  CHECK(step.assigned_dev == FS_DEV_TPU);
  CHECK(step.requires_handoff == 1);
  CHECK(step.zero_copy == 1);

  float* out = (float*)buf_c->host_ptr;
  for (uint32_t n = 0; n < 4u; ++n) {
    CHECK(out[n] == (float)(32u * (n + 1u)));
  }

  fs_telemetry_t tel;
  CHECK(fs_context_get_telemetry(ctx, &tel) == FS_OK);
  CHECK(tel.zero_copy_handoffs >= 1u);
  CHECK(tel.fallback_copies == 0u);
  CHECK(tel.tpu_ops == 1u);

  fs_buffer_release(ctx, buf_a);
  fs_buffer_release(ctx, buf_b);
  fs_buffer_release(ctx, buf_c);

  /* Verify C11 NvSciBuf-style upfront buffer reconciliation */
  fs_buf_attr_t attrs[3];
  attrs[0] = fs_buf_attr_default(FS_DEV_GPU_VK);
  attrs[1] = fs_buf_attr_default(FS_DEV_TPU);
  attrs[2] = fs_buf_attr_default(FS_DEV_CPU_SME2);
  fs_buf_reconciled_t rec;
  int                 zc_ok = 0;
  CHECK(fs_dmabuf_reconcile(attrs, 3u, &rec, &zc_ok) == FS_OK);
  CHECK(zc_ok == 1);
  CHECK(rec.zero_copy_compatible == 1);
  CHECK(rec.min_align == 128u);
  CHECK(rec.stride_align == 64u);
  CHECK(rec.layout == FS_LAYOUT_LINEAR);
  CHECK(rec.coherency == FS_COHERENCY_NON_COHERENT);
  CHECK(rec.requires_dma_sync == 1);

  fs_dmabuf_t* rec_buf = NULL;
  CHECK(fs_buffer_alloc_reconciled(ctx, 256u, attrs, 3u, FS_DEV_GPU_VK, FS_BUF_ZERO_INIT,
                                   &rec_buf, NULL) == FS_OK);
  CHECK(rec_buf != NULL);
  CHECK(rec_buf->zero_copy_compatible == 1);
  CHECK(rec_buf->min_align == 128u);
  fs_buffer_release(ctx, rec_buf);

  fs_context_destroy(ctx);

  /* Verify ggml_freesilicon C11 init/free seam */
  ggml_fs_backend_t* gg = ggml_fs_backend_init(&cfg);
  CHECK(gg != NULL);
  CHECK(strcmp(ggml_fs_backend_name(gg), "FreeSilicon") == 0);
  CHECK(ggml_fs_backend_is_mock(gg) == 1);
  ggml_fs_backend_free(gg);

  printf("[PASS] test_c11_abi: pure C11 header + runtime ABI verified.\n");
  return 0;
}
