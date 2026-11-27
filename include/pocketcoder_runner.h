/*
 * pocketcoder_runner - Standalone Model-Specific PocketCoder-13B-A3B Engine API
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pure C11 / C++17 header providing:
 *   1. Direct mmap-backed GGUF reader with strict bounds & inkling2_loop checks
 *   2. 1024-token sliding-window KV ring buffer (40/48 looped visits)
 *   3. 16-byte SoA expert repacker (Q4_0, IQ4_NL, IQ4_XS) + ARM64 NEON/sdot GEMV
 *   4. Pre-baked zero-copy DMA-BUF execution graph + openbuenos & DSpark dedup
 */
#ifndef POCKETCODER_RUNNER_H_
#define POCKETCODER_RUNNER_H_

#include "freesched.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * PocketCoder-13B-A3B (inkling2_loop) Architecture Constants
 * ========================================================================= */

#define PC_MAX_PHYS_LAYERS     64u
#define PC_MAX_LOGICAL_VISITS  128u
#define PC_MAX_EXPERTS         64u
#define PC_TOP_K_EXPERTS       8u
#define PC_DSPARK_MAX_DUP      4u
#define PC_DEFAULT_SWA_WINDOW  1024u
#define PC_LANE_ALIGN          128u
#define PC_SOA_TILE_ROWS       4u
#define PC_GGUF_BLK_SIZE       32u
#define PC_GGUF_SUPERBLK_SIZE  256u
#define PC_GGUF_NAME_MAX       128u

/* GGUF tensor quantization type IDs (subset used by PocketCoder-13B-A3B) */
typedef enum pc_ggml_type {
  PC_GGML_TYPE_F32    = 0,
  PC_GGML_TYPE_F16    = 1,
  PC_GGML_TYPE_Q4_0   = 2,
  PC_GGML_TYPE_Q8_0   = 8,
  PC_GGML_TYPE_Q6_K   = 14,
  PC_GGML_TYPE_IQ4_NL = 20,
  PC_GGML_TYPE_IQ4_XS = 23
} pc_ggml_type_t;

typedef struct pc_visit_desc {
  uint32_t visit_idx;
  uint32_t phys_layer;
  uint32_t loop_pass;
  uint32_t is_sliding_window;
  uint32_t window_size;
} pc_visit_desc_t;

typedef struct pc_inkling2_config {
  char            arch[64];
  char            model_name[128];
  uint32_t        file_type;
  uint32_t        block_count;
  uint32_t        context_length;
  uint32_t        embedding_length;
  uint32_t        feed_forward_length;
  uint32_t        expert_feed_forward_length;
  uint32_t        head_count;
  uint32_t        head_count_kv;
  uint32_t        key_length;
  uint32_t        value_length;
  uint32_t        expert_count;
  uint32_t        expert_used_count;
  uint32_t        sliding_window;
  uint8_t         sliding_window_pattern[PC_MAX_PHYS_LAYERS];
  uint32_t        loop_start;
  uint32_t        loop_end;
  uint32_t        loop_count;
  uint32_t        loop_gate_rank;
  uint32_t        loop_varying;
  uint32_t        shortconv_l_cache;
  uint32_t        vocab_size;
  uint32_t        vocab_size_unpadded;
  uint32_t        eos_token_id;
  uint32_t        alignment;
  float           rms_norm_eps;
  float           expert_weights_scale;
  float           situ_beta;
  float           situ_linear_beta;
  float           final_logit_softcapping;
  /* Derived 48-visit looped execution topology */
  uint32_t        logical_visits;
  uint32_t        sliding_window_visits;
  uint32_t        full_context_visits;
  pc_visit_desc_t visits[PC_MAX_LOGICAL_VISITS];
} pc_inkling2_config_t;

typedef struct pc_gguf_tensor {
  char     name[PC_GGUF_NAME_MAX];
  uint32_t n_dims;
  uint64_t ne[4];
  uint32_t ggml_type;
  uint64_t offset;     /* Byte offset relative to GGUF data section */
  uint64_t byte_size;  /* Exact validated byte span in file */
} pc_gguf_tensor_t;

typedef struct pc_gguf_file pc_gguf_file_t;

/* =========================================================================
 * 1. Direct mmap-backed GGUF Reader
 * ========================================================================= */

fs_status_t pc_gguf_open(const char* path, pc_gguf_file_t** out_file);
void        pc_gguf_close(pc_gguf_file_t* file);

const pc_inkling2_config_t* pc_gguf_config(const pc_gguf_file_t* file);
uint64_t                    pc_gguf_file_size(const pc_gguf_file_t* file);
uint64_t                    pc_gguf_data_offset(const pc_gguf_file_t* file);
uint32_t                    pc_gguf_tensor_count(const pc_gguf_file_t* file);
const pc_gguf_tensor_t*     pc_gguf_tensor_at(const pc_gguf_file_t* file, uint32_t idx);
const pc_gguf_tensor_t*     pc_gguf_find_tensor(const pc_gguf_file_t* file, const char* name);
const void*                 pc_gguf_tensor_data(const pc_gguf_file_t* file, const pc_gguf_tensor_t* tensor);

fs_status_t pc_gguf_validate_pocketcoder(const pc_gguf_file_t* file,
                                         char*                 err_buf,
                                         size_t                err_cap);

size_t      pc_ggml_type_block_elems(uint32_t ggml_type);
size_t      pc_ggml_type_block_bytes(uint32_t ggml_type);
uint64_t    pc_ggml_tensor_nbytes(uint32_t ggml_type, uint64_t n_elems);
const char* pc_ggml_type_name(uint32_t ggml_type);

/* =========================================================================
 * 2. Strict 1024-Token Sliding-Window KV Ring Buffer (40/48 Visits)
 * ========================================================================= */

typedef struct pc_kv_footprint {
  uint32_t context_len;
  uint32_t bytes_per_elem;
  uint32_t sliding_visits;
  uint32_t full_visits;
  uint32_t sliding_window_cap;
  uint64_t sliding_kv_bytes;
  uint64_t full_kv_bytes;
  uint64_t total_kv_bytes;
  uint64_t naive_full_kv_bytes;
  uint64_t saved_bytes;
  double   reduction_ratio;
} pc_kv_footprint_t;

typedef struct pc_kv_cache pc_kv_cache_t;

fs_status_t pc_kv_compute_footprint(const pc_inkling2_config_t* cfg,
                                    uint32_t                    context_len,
                                    uint32_t                    bytes_per_elem,
                                    pc_kv_footprint_t*          out_footprint);

fs_status_t pc_kv_cache_create(const pc_inkling2_config_t* cfg,
                               uint32_t                    max_full_context,
                               pc_kv_cache_t**             out_cache);
void        pc_kv_cache_destroy(pc_kv_cache_t* cache);
void        pc_kv_cache_reset(pc_kv_cache_t* cache);

fs_status_t pc_kv_cache_append(pc_kv_cache_t* cache,
                               uint32_t       visit_idx,
                               uint32_t       abs_pos,
                               const float*   k_vec,
                               const float*   v_vec);

uint32_t    pc_kv_cache_visit_capacity(const pc_kv_cache_t* cache, uint32_t visit_idx);
uint32_t    pc_kv_cache_visit_len(const pc_kv_cache_t* cache, uint32_t visit_idx);
uint64_t    pc_kv_cache_total_bytes(const pc_kv_cache_t* cache);

/*
 * Reads chronological KV entries visible at `current_abs_pos` for `visit_idx`.
 * Copies up to `max_out_tokens` chronological (K, V) pairs and absolute token
 * positions into caller buffers.
 */
fs_status_t pc_kv_cache_read_window(const pc_kv_cache_t* cache,
                                    uint32_t             visit_idx,
                                    uint32_t             current_abs_pos,
                                    uint32_t             max_out_tokens,
                                    float*               out_k,
                                    float*               out_v,
                                    uint32_t*            out_abs_positions,
                                    uint32_t*            out_count);

/* =========================================================================
 * 3. 16-Byte SoA Expert Repacker + ARM64 NEON / Scalar GEMV Kernel
 * ========================================================================= */

typedef enum pc_soa_quant {
  PC_SOA_Q4_0  = 0, /* Uniform symmetric 4-bit (nibble - 8) */
  PC_SOA_IQ4   = 1  /* Non-linear IQ4_NL / IQ4_XS 16-entry codebook */
} pc_soa_quant_t;

/*
 * 16-byte SoA tiled matrix [n_dim, k_dim] repacked for 4-row SIMD/FreeTile bursts.
 *
 * Layout:
 *   - tiles = ceil(n_dim / 4), n_blocks = k_dim / 32
 *   - scales: 16-byte aligned [tiles][n_blocks][4] float (pre-decoded d or dl)
 *   - qs:     16-byte aligned [tiles][n_blocks][4][16] uint8_t (32 4-bit nibbles/row)
 */
typedef struct pc_soa_matrix {
  uint32_t       n_dim;
  uint32_t       k_dim;
  uint32_t       n_tiles;
  uint32_t       n_blocks;
  pc_soa_quant_t quant;
  float*         scales;
  uint8_t*       qs;
  size_t         scales_bytes;
  size_t         qs_bytes;
} pc_soa_matrix_t;

fs_status_t pc_soa_repack_q4_0(const void*      src_q4_0,
                               uint32_t         n_dim,
                               uint32_t         k_dim,
                               pc_soa_matrix_t* out_mat);

fs_status_t pc_soa_repack_iq4_nl(const void*      src_iq4_nl,
                                 uint32_t         n_dim,
                                 uint32_t         k_dim,
                                 pc_soa_matrix_t* out_mat);

fs_status_t pc_soa_repack_iq4_xs(const void*      src_iq4_xs,
                                 uint32_t         n_dim,
                                 uint32_t         k_dim,
                                 pc_soa_matrix_t* out_mat);

fs_status_t pc_soa_repack_expert_from_gguf(const pc_gguf_file_t* file,
                                           const char*           tensor_name,
                                           uint32_t              expert_idx,
                                           pc_soa_matrix_t*      out_mat);

void pc_soa_free(pc_soa_matrix_t* mat);

/* Exact FP32 reference GEMV directly from raw GGUF blocks (no SoA repack) */
fs_status_t pc_gemv_ref_raw_f32(const void*  src_blocks,
                                uint32_t     ggml_type,
                                const float* x,
                                float*       y,
                                uint32_t     n_dim,
                                uint32_t     k_dim);

/* Fused 16-byte SoA dequant+GEMV (FP32 activation; NEON on ARM64, scalar fallback) */
fs_status_t pc_gemv_soa_f32(const pc_soa_matrix_t* mat, const float* x, float* y);

/* Force scalar path for A/B micro-benchmark baseline & parity testing */
fs_status_t pc_gemv_soa_scalar_f32(const pc_soa_matrix_t* mat, const float* x, float* y);

/* Dynamic Q8 activation quantization + ARM64 NEON sdot SoA GEMV (scalar fallback) */
fs_status_t pc_gemv_soa_sdot(const pc_soa_matrix_t* mat, const float* x, float* y);

const char* pc_soa_kernel_backend_name(void);

/* =========================================================================
 * 4. Pre-Baked Zero-Copy DMA-BUF Graph + DSpark (MAX_DUP=3/4) Dedup Planner
 * ========================================================================= */

typedef struct pc_dspark_expert_entry {
  uint32_t expert_id;
  uint32_t token_mask; /* Bit `t` set if token `t` in [0..batch_size) routes to expert_id */
  uint32_t ref_count;
} pc_dspark_expert_entry_t;

typedef struct pc_dspark_dedup_plan {
  uint32_t                 batch_size; /* 1..PC_DSPARK_MAX_DUP */
  uint32_t                 top_k;      /* <= PC_TOP_K_EXPERTS */
  uint32_t                 unique_count;
  uint32_t                 naive_expert_loads;
  uint32_t                 dedup_expert_loads;
  uint64_t                 bytes_per_expert;
  uint64_t                 naive_bytes;
  uint64_t                 dedup_bytes;
  uint64_t                 bytes_saved;
  double                   dedup_ratio;
  pc_dspark_expert_entry_t entries[PC_MAX_EXPERTS];
  uint8_t                  slot_to_entry[PC_DSPARK_MAX_DUP][PC_TOP_K_EXPERTS];
} pc_dspark_dedup_plan_t;

fs_status_t pc_dspark_plan_experts(const uint32_t*         routed_ids, /* [batch_size * top_k] */
                                   uint32_t                batch_size,
                                   uint32_t                top_k,
                                   uint32_t                num_experts,
                                   uint64_t                bytes_per_expert,
                                   pc_dspark_dedup_plan_t* out_plan);

/*
 * Executes deduplicated MoE layer for a DSpark batch [batch_size, k_dim] -> [batch_size, n_dim]
 * across `num_experts` SoA matrices, matching naive un-deduplicated weighted top-k sum.
 */
fs_status_t pc_dspark_execute_moe_f32(const pc_dspark_dedup_plan_t* plan,
                                      const pc_soa_matrix_t*        experts,
                                      uint32_t                      num_experts,
                                      const float*                  routing_weights, /* [batch * top_k] */
                                      const float*                  x_batch,         /* [batch * k_dim] */
                                      float*                        y_batch);        /* [batch * n_dim] */

typedef struct pc_graph_config {
  uint32_t        struct_size;
  fs_hw_profile_t profile;
  int             mock_mode;
  uint32_t        hidden_dim;      /* 2304 for PocketCoder-13B-A3B */
  uint32_t        expert_ffn_dim;  /* 1152 for PocketCoder-13B-A3B */
  uint32_t        dspark_max_dup;  /* 3 or 4 */
  const char*     dive_interp_path;/* Optional external RV64 DIVE interpreter path */
  int             require_tpu_hw;  /* 1 = fail closed if /dev/edgetpu-soc or DIVE unavailable */
  uint8_t         perf_mode;       /* 0..5 DVFS perf tier (default 5) */
  uint8_t         reserved[3];
} pc_graph_config_t;

typedef struct pc_graph_stats {
  int      reconciled_zero_copy;
  size_t   reconciled_min_align;
  size_t   reconciled_stride_align;
  uint32_t planned_steps;
  uint64_t zero_copy_handoffs;
  uint64_t fallback_copies;
  uint64_t tpu_ops;
  uint64_t cpu_ops;
  uint64_t dma_sync_calls;
  double   max_abs_err;
  double   nmse;
  int      ob_hw_active;
  int      ob_emitter_active;
  int      ob_freetile_active;
  uint64_t ob_calls;
  uint64_t ob_tiles_dispatched;
  uint64_t ob_hw_us;
  uint64_t ob_w_hits;
  uint64_t ob_w_misses;
  uint64_t ob_w_evictions;
  uint64_t ob_weight_pack_us;
  uint64_t ob_weight_cache_bytes;
} pc_graph_stats_t;

typedef struct pc_graph pc_graph_t;

pc_graph_config_t pc_graph_config_default(void);
fs_status_t       pc_graph_create(const pc_graph_config_t* cfg, pc_graph_t** out_graph);
void              pc_graph_destroy(pc_graph_t* graph);

/*
 * Uploads immutable Q4_0 + RMSNorm weights ONCE into reconciled DMA-BUFs,
 * registers the immutable region, and prepacks EdgeTPU ext[2] tiles/graphs
 * upfront so decode steps incur zero runtime weight copying or repacking.
 */
fs_status_t pc_graph_bind_weights(pc_graph_t*  graph,
                                  const void*  w_q4_0,
                                  const float* rms_weight);

/*
 * Executes one pre-baked step against weights already bound via pc_graph_bind_weights.
 */
fs_status_t pc_graph_step_bound(pc_graph_t*       graph,
                                const float*      x_in,
                                float*            y_out,
                                pc_graph_stats_t* out_stats);

/*
 * Executes one pre-baked heterogeneous layer step:
 *   RMSNorm (CPU_NEON/SME2) -> zero-copy DMA-BUF handoff ->
 *   DECODE_GEMV (EdgeTPU/openbuenos) -> Parity check vs reference
 */
fs_status_t pc_graph_run_prebaked_step(pc_graph_t*       graph,
                                       const float*      x_in,
                                       const void*       w_q4_0,
                                       const float*      rms_weight,
                                       float*            y_out,
                                       pc_graph_stats_t* out_stats);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POCKETCODER_RUNNER_H_ */
