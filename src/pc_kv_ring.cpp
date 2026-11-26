/*
 * Strict 1024-token sliding-window KV ring buffer for PocketCoder-13B-A3B.
 *
 * Separates the 40/48 sliding-window visits (fixed 1024-token ring capacity)
 * from the 8/48 full-context visits so 32K-128K contexts fit within 16 GB RAM.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pocketcoder_runner.h"

#include <cstring>
#include <vector>

struct pc_kv_visit_ring {
  uint32_t              capacity;
  uint32_t              is_sliding;
  uint32_t              total_written;
  std::vector<uint32_t> slot_abs_pos;
  std::vector<float>    k_buf;
  std::vector<float>    v_buf;
};

struct pc_kv_cache {
  uint32_t                      n_visits;
  uint32_t                      kv_dim;
  uint32_t                      max_full_context;
  uint64_t                      total_bytes;
  std::vector<pc_kv_visit_ring> visits;
};

extern "C" {

fs_status_t pc_kv_compute_footprint(const pc_inkling2_config_t* cfg,
                                    uint32_t                    context_len,
                                    uint32_t                    bytes_per_elem,
                                    pc_kv_footprint_t*          out_footprint) {
  if (!cfg || !out_footprint || context_len == 0u || bytes_per_elem == 0u) {
    return FS_ERR_INVALID_ARG;
  }
  if (cfg->logical_visits == 0u || cfg->head_count_kv == 0u || cfg->key_length == 0u) {
    return FS_ERR_INVALID_ARG;
  }

  const uint32_t swa_cap = cfg->sliding_window > 0u ? cfg->sliding_window : PC_DEFAULT_SWA_WINDOW;
  const uint32_t eff_swa = (context_len < swa_cap) ? context_len : swa_cap;
  const uint64_t per_tok_bytes =
      2ULL * static_cast<uint64_t>(cfg->head_count_kv) * static_cast<uint64_t>(cfg->key_length) *
      static_cast<uint64_t>(bytes_per_elem);

  const uint64_t swa_bytes =
      static_cast<uint64_t>(cfg->sliding_window_visits) * eff_swa * per_tok_bytes;
  const uint64_t full_bytes =
      static_cast<uint64_t>(cfg->full_context_visits) * context_len * per_tok_bytes;
  const uint64_t total_bytes = swa_bytes + full_bytes;
  const uint64_t naive_bytes =
      static_cast<uint64_t>(cfg->logical_visits) * context_len * per_tok_bytes;

  std::memset(out_footprint, 0, sizeof(*out_footprint));
  out_footprint->context_len         = context_len;
  out_footprint->bytes_per_elem      = bytes_per_elem;
  out_footprint->sliding_visits      = cfg->sliding_window_visits;
  out_footprint->full_visits         = cfg->full_context_visits;
  out_footprint->sliding_window_cap  = swa_cap;
  out_footprint->sliding_kv_bytes    = swa_bytes;
  out_footprint->full_kv_bytes       = full_bytes;
  out_footprint->total_kv_bytes      = total_bytes;
  out_footprint->naive_full_kv_bytes = naive_bytes;
  out_footprint->saved_bytes         = (naive_bytes > total_bytes) ? (naive_bytes - total_bytes) : 0u;
  out_footprint->reduction_ratio =
      (total_bytes > 0u) ? static_cast<double>(naive_bytes) / static_cast<double>(total_bytes) : 1.0;

  return FS_OK;
}

fs_status_t pc_kv_cache_create(const pc_inkling2_config_t* cfg,
                               uint32_t                    max_full_context,
                               pc_kv_cache_t**             out_cache) {
  if (!cfg || !out_cache || max_full_context == 0u ||
      cfg->logical_visits == 0u || cfg->logical_visits > PC_MAX_LOGICAL_VISITS ||
      cfg->head_count_kv == 0u || cfg->key_length == 0u) {
    return FS_ERR_INVALID_ARG;
  }
  *out_cache = nullptr;

  const uint32_t kv_dim  = cfg->head_count_kv * cfg->key_length;
  const uint32_t swa_cap = cfg->sliding_window > 0u ? cfg->sliding_window : PC_DEFAULT_SWA_WINDOW;

  auto* cache = new pc_kv_cache();
  cache->n_visits         = cfg->logical_visits;
  cache->kv_dim           = kv_dim;
  cache->max_full_context = max_full_context;
  cache->total_bytes      = 0u;
  cache->visits.resize(cache->n_visits);

  for (uint32_t v = 0; v < cache->n_visits; ++v) {
    pc_kv_visit_ring& r = cache->visits[v];
    r.is_sliding        = cfg->visits[v].is_sliding_window;
    r.capacity          = r.is_sliding ? (max_full_context < swa_cap ? max_full_context : swa_cap)
                                       : max_full_context;
    r.total_written     = 0u;
    r.slot_abs_pos.assign(r.capacity, UINT32_MAX);
    r.k_buf.assign(static_cast<size_t>(r.capacity) * kv_dim, 0.0f);
    r.v_buf.assign(static_cast<size_t>(r.capacity) * kv_dim, 0.0f);
    cache->total_bytes += static_cast<uint64_t>(r.capacity) * kv_dim * 2ULL * sizeof(float);
  }

  *out_cache = cache;
  return FS_OK;
}

void pc_kv_cache_destroy(pc_kv_cache_t* cache) {
  delete cache;
}

void pc_kv_cache_reset(pc_kv_cache_t* cache) {
  if (!cache) {
    return;
  }
  for (auto& r : cache->visits) {
    r.total_written = 0u;
    std::fill(r.slot_abs_pos.begin(), r.slot_abs_pos.end(), UINT32_MAX);
  }
}

fs_status_t pc_kv_cache_append(pc_kv_cache_t* cache,
                               uint32_t       visit_idx,
                               uint32_t       abs_pos,
                               const float*   k_vec,
                               const float*   v_vec) {
  if (!cache || visit_idx >= cache->n_visits || !k_vec || !v_vec) {
    return FS_ERR_INVALID_ARG;
  }
  pc_kv_visit_ring& r = cache->visits[visit_idx];
  if (!r.is_sliding && abs_pos >= r.capacity) {
    return FS_ERR_OVERFLOW;
  }
  const uint32_t slot = abs_pos % r.capacity;
  const size_t   off  = static_cast<size_t>(slot) * cache->kv_dim;
  std::memcpy(r.k_buf.data() + off, k_vec, static_cast<size_t>(cache->kv_dim) * sizeof(float));
  std::memcpy(r.v_buf.data() + off, v_vec, static_cast<size_t>(cache->kv_dim) * sizeof(float));
  r.slot_abs_pos[slot] = abs_pos;
  r.total_written++;
  return FS_OK;
}

uint32_t pc_kv_cache_visit_capacity(const pc_kv_cache_t* cache, uint32_t visit_idx) {
  if (!cache || visit_idx >= cache->n_visits) {
    return 0u;
  }
  return cache->visits[visit_idx].capacity;
}

uint32_t pc_kv_cache_visit_len(const pc_kv_cache_t* cache, uint32_t visit_idx) {
  if (!cache || visit_idx >= cache->n_visits) {
    return 0u;
  }
  const pc_kv_visit_ring& r = cache->visits[visit_idx];
  return (r.total_written < r.capacity) ? r.total_written : r.capacity;
}

uint64_t pc_kv_cache_total_bytes(const pc_kv_cache_t* cache) {
  return cache ? cache->total_bytes : 0u;
}

fs_status_t pc_kv_cache_read_window(const pc_kv_cache_t* cache,
                                    uint32_t             visit_idx,
                                    uint32_t             current_abs_pos,
                                    uint32_t             max_out_tokens,
                                    float*               out_k,
                                    float*               out_v,
                                    uint32_t*            out_abs_positions,
                                    uint32_t*            out_count) {
  if (!cache || visit_idx >= cache->n_visits || !out_count) {
    return FS_ERR_INVALID_ARG;
  }
  const pc_kv_visit_ring& r = cache->visits[visit_idx];
  if (!r.is_sliding && current_abs_pos >= r.capacity) {
    return FS_ERR_OVERFLOW;
  }

  const uint32_t total_toks = current_abs_pos + 1u;
  const uint32_t win_len    = (total_toks < r.capacity) ? total_toks : r.capacity;
  const uint32_t start_pos  = total_toks - win_len;
  const uint32_t copy_n     = (win_len < max_out_tokens) ? win_len : max_out_tokens;

  for (uint32_t i = 0; i < copy_n; ++i) {
    const uint32_t pos  = start_pos + i;
    const uint32_t slot = pos % r.capacity;
    if (r.slot_abs_pos[slot] != pos) {
      return FS_ERR_NOT_FOUND;
    }
    const size_t src_off = static_cast<size_t>(slot) * cache->kv_dim;
    const size_t dst_off = static_cast<size_t>(i) * cache->kv_dim;
    if (out_k) {
      std::memcpy(out_k + dst_off, r.k_buf.data() + src_off,
                  static_cast<size_t>(cache->kv_dim) * sizeof(float));
    }
    if (out_v) {
      std::memcpy(out_v + dst_off, r.v_buf.data() + src_off,
                  static_cast<size_t>(cache->kv_dim) * sizeof(float));
    }
    if (out_abs_positions) {
      out_abs_positions[i] = pos;
    }
  }
  *out_count = copy_n;
  return FS_OK;
}

} /* extern "C" */
