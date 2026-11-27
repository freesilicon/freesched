/*
 * Comprehensive C++17 unit tests for PocketCoder-13B-A3B runner deliverables.
 *
 * Covers:
 *   1. mmap GGUF reader + inkling2_loop 24->48 visit parsing & strict bounds checks
 *   2. 1024-token sliding-window KV ring buffer (40/48 visits, wraparound, 32K/128K RAM math)
 *   3. 16-byte SoA repacker (Q4_0, IQ4_NL, IQ4_XS) + ARM64 NEON/sdot vs FP32 parity
 *   4. Pre-baked DMA-BUF graph + openbuenos adapter + DSpark MAX_DUP=3/4 dedup parity
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pocketcoder_runner.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return false;                                                        \
    }                                                                      \
  } while (0)

namespace {

void push_u8(std::vector<uint8_t>* buf, uint8_t v) {
  buf->push_back(v);
}

void push_u32(std::vector<uint8_t>* buf, uint32_t v) {
  uint8_t b[4];
  std::memcpy(b, &v, 4);
  buf->insert(buf->end(), b, b + 4);
}

void push_u64(std::vector<uint8_t>* buf, uint64_t v) {
  uint8_t b[8];
  std::memcpy(b, &v, 8);
  buf->insert(buf->end(), b, b + 8);
}

void push_str(std::vector<uint8_t>* buf, const char* s) {
  const uint64_t len = static_cast<uint64_t>(std::strlen(s));
  push_u64(buf, len);
  buf->insert(buf->end(), reinterpret_cast<const uint8_t*>(s),
              reinterpret_cast<const uint8_t*>(s) + len);
}

void push_kv_str(std::vector<uint8_t>* buf, const char* key, const char* val) {
  push_str(buf, key);
  push_u32(buf, 8u);
  push_str(buf, val);
}

void push_kv_u32(std::vector<uint8_t>* buf, const char* key, uint32_t val) {
  push_str(buf, key);
  push_u32(buf, 4u);
  push_u32(buf, val);
}

void push_kv_bool_arr(std::vector<uint8_t>* buf, const char* key, const std::vector<uint8_t>& vals) {
  push_str(buf, key);
  push_u32(buf, 9u);
  push_u32(buf, 7u);
  push_u64(buf, vals.size());
  for (uint8_t v : vals) {
    push_u8(buf, v);
  }
}

void build_synthetic_inkling2_gguf(std::vector<uint8_t>* out) {
  out->clear();
  /* GGUF magic + v3 + 3 tensors + 17 KVs */
  push_u32(out, 0x46554747u);
  push_u32(out, 3u);
  push_u64(out, 3u);
  push_u64(out, 17u);

  push_kv_str(out, "general.architecture", "inkling2_loop");
  push_kv_str(out, "general.name", "PocketCoder-Synthetic-Test");
  push_kv_u32(out, "inkling2_loop.block_count", 24u);
  push_kv_u32(out, "inkling2_loop.context_length", 1048576u);
  push_kv_u32(out, "inkling2_loop.embedding_length", 2304u);
  push_kv_u32(out, "inkling2_loop.feed_forward_length", 1152u);
  push_kv_u32(out, "inkling2_loop.expert_feed_forward_length", 1152u);
  push_kv_u32(out, "inkling2_loop.attention.head_count", 24u);
  push_kv_u32(out, "inkling2_loop.attention.head_count_kv", 8u);
  push_kv_u32(out, "inkling2_loop.attention.key_length", 128u);
  push_kv_u32(out, "inkling2_loop.attention.value_length", 128u);
  push_kv_u32(out, "inkling2_loop.expert_count", 64u);
  push_kv_u32(out, "inkling2_loop.expert_used_count", 8u);
  push_kv_u32(out, "inkling2_loop.attention.sliding_window", 1024u);
  push_kv_u32(out, "inkling2_loop.loop.start", 6u);
  push_kv_u32(out, "inkling2_loop.loop.end", 18u);
  push_kv_u32(out, "inkling2_loop.loop.count", 3u);

  /* Wait: patch n_kv to 18 so we include sliding_window_pattern */
  const uint64_t n_kv = 18u;
  std::memcpy(out->data() + 16u, &n_kv, sizeof(n_kv));
  std::vector<uint8_t> swa(24u, 1u);
  swa[5] = swa[11] = swa[17] = swa[23] = 0u;
  push_kv_bool_arr(out, "inkling2_loop.attention.sliding_window_pattern", swa);

  /* Tensors (tiny 3D/2D shapes for unit test file size, keeping required names) */
  const uint64_t t0_bytes = 32u * 18u; /* Q4_0: 32 elems x 1 row x 1 exp -> 18 B */
  const uint64_t t1_bytes = 256u / 256u * 136u; /* IQ4_XS: 256 elems -> 136 B */
  const uint64_t t2_bytes = 64u * 4u; /* F32: 64 elems -> 256 B */

  auto push_tensor = [&](const char* name, uint32_t nd, uint64_t d0, uint64_t d1, uint64_t d2,
                         uint32_t type, uint64_t off) {
    push_str(out, name);
    push_u32(out, nd);
    if (nd >= 1u) push_u64(out, d0);
    if (nd >= 2u) push_u64(out, d1);
    if (nd >= 3u) push_u64(out, d2);
    push_u32(out, type);
    push_u64(out, off);
  };
  push_tensor("blk.0.ffn_down_exps.weight", 3u, 32u, 1u, 1u, PC_GGML_TYPE_Q4_0, 0u);
  push_tensor("blk.0.ffn_gate_up_exps.weight", 3u, 256u, 1u, 1u, PC_GGML_TYPE_IQ4_XS, 32u);
  push_tensor("blk.6.loop_ffn_ctrl_gain.weight", 2u, 64u, 1u, 1u, PC_GGML_TYPE_F32, 32u + 160u);

  while ((out->size() & 31u) != 0u) {
    out->push_back(0u);
  }
  const size_t data_len = 32u + 160u + t2_bytes;
  out->resize(out->size() + data_len, 0x3au);
  (void)t0_bytes;
  (void)t1_bytes;
}

std::string write_temp_file(const std::vector<uint8_t>& bytes, const char* tag) {
  std::string path = std::string("/tmp/pc_test_") + tag + "_" + std::to_string(getpid()) + ".gguf";
  FILE*       fp   = std::fopen(path.c_str(), "wb");
  if (fp) {
    std::fwrite(bytes.data(), 1u, bytes.size(), fp);
    std::fclose(fp);
  }
  return path;
}

bool test_gguf_reader_and_bounds() {
  std::vector<uint8_t> raw;
  build_synthetic_inkling2_gguf(&raw);
  std::string path = write_temp_file(raw, "valid");

  pc_gguf_file_t* gf = nullptr;
  CHECK(pc_gguf_open(path.c_str(), &gf) == FS_OK);
  CHECK(gf != nullptr);

  char err[256];
  CHECK(pc_gguf_validate_pocketcoder(gf, err, sizeof(err)) == FS_OK);
  const pc_inkling2_config_t* cfg = pc_gguf_config(gf);
  CHECK(cfg->block_count == 24u);
  CHECK(cfg->logical_visits == 48u);
  CHECK(cfg->sliding_window_visits == 40u);
  CHECK(cfg->full_context_visits == 8u);
  CHECK(cfg->loop_start == 6u && cfg->loop_end == 18u && cfg->loop_count == 3u);
  CHECK(cfg->embedding_length == 2304u && cfg->expert_feed_forward_length == 1152u);
  CHECK(cfg->expert_count == 64u && cfg->expert_used_count == 8u);

  /* Verify looped visit mapping: visit 5 is phys 5 (full), visit 6 is phys 6 pass 0 (SWA),
   * visit 18 is phys 6 pass 1 (SWA), visit 47 is phys 23 (full) */
  CHECK(cfg->visits[5].phys_layer == 5u && cfg->visits[5].is_sliding_window == 0u);
  CHECK(cfg->visits[6].phys_layer == 6u && cfg->visits[6].loop_pass == 0u &&
        cfg->visits[6].is_sliding_window == 1u);
  CHECK(cfg->visits[18].phys_layer == 6u && cfg->visits[18].loop_pass == 1u);
  CHECK(cfg->visits[47].phys_layer == 23u && cfg->visits[47].is_sliding_window == 0u);

  const pc_gguf_tensor_t* t = pc_gguf_find_tensor(gf, "blk.0.ffn_down_exps.weight");
  CHECK(t != nullptr);
  CHECK(pc_gguf_tensor_data(gf, t) != nullptr);
  pc_gguf_close(gf);
  std::remove(path.c_str());

  /* Bounds-check test 1: truncated file */
  std::vector<uint8_t> trunc(raw.begin(), raw.begin() + 64);
  std::string          tpath = write_temp_file(trunc, "trunc");
  CHECK(pc_gguf_open(tpath.c_str(), &gf) != FS_OK);
  std::remove(tpath.c_str());

  /* Bounds-check test 2: corrupted magic */
  std::vector<uint8_t> bad_magic = raw;
  bad_magic[0]                   = 'X';
  std::string mpath              = write_temp_file(bad_magic, "magic");
  CHECK(pc_gguf_open(mpath.c_str(), &gf) != FS_OK);
  std::remove(mpath.c_str());

  /* Bounds-check test 3: truncated tensor data payload */
  std::vector<uint8_t> short_data(raw.begin(), raw.end() - 64);
  std::string          spath = write_temp_file(short_data, "shortdata");
  CHECK(pc_gguf_open(spath.c_str(), &gf) == FS_ERR_OVERFLOW);
  std::remove(spath.c_str());

  std::printf("[PASS] test_gguf_reader_and_bounds\n");
  return true;
}

bool test_kv_sliding_window_ring() {
  std::vector<uint8_t> raw;
  build_synthetic_inkling2_gguf(&raw);
  std::string path = write_temp_file(raw, "kv");

  pc_gguf_file_t* gf = nullptr;
  CHECK(pc_gguf_open(path.c_str(), &gf) == FS_OK);
  const pc_inkling2_config_t* cfg = pc_gguf_config(gf);

  /* Exact memory math check at 32K and 128K (FP16 = 2 bytes/elem) */
  pc_kv_footprint_t fp32k{}, fp128k{};
  CHECK(pc_kv_compute_footprint(cfg, 32768u, 2u, &fp32k) == FS_OK);
  CHECK(fp32k.sliding_visits == 40u && fp32k.full_visits == 8u);
  /* 40 * 1024 * 4096 + 8 * 32768 * 4096 = 160 MiB + 1024 MiB = 1184 MiB */
  CHECK(fp32k.total_kv_bytes == (160ULL + 1024ULL) * 1024ULL * 1024ULL);
  CHECK(fp32k.naive_full_kv_bytes == 6ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(fp32k.reduction_ratio > 5.18 && fp32k.reduction_ratio < 5.20);

  CHECK(pc_kv_compute_footprint(cfg, 131072u, 2u, &fp128k) == FS_OK);
  CHECK(fp128k.naive_full_kv_bytes == 24ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(fp128k.total_kv_bytes == (160ULL + 4096ULL) * 1024ULL * 1024ULL);
  CHECK(fp128k.reduction_ratio > 5.75);

  /* Live ring buffer wraparound test across sliding visit 0 (cap=1024) and full visit 5 (cap=2500) */
  pc_kv_cache_t* cache = nullptr;
  CHECK(pc_kv_cache_create(cfg, 2500u, &cache) == FS_OK);
  CHECK(pc_kv_cache_visit_capacity(cache, 0u) == 1024u);
  CHECK(pc_kv_cache_visit_capacity(cache, 5u) == 2500u);

  const uint32_t     kv_dim = cfg->head_count_kv * cfg->key_length; /* 1024 */
  std::vector<float> k_in(kv_dim);
  std::vector<float> v_in(kv_dim);

  const uint32_t N_TOKS = 2200u;
  for (uint32_t pos = 0; pos < N_TOKS; ++pos) {
    k_in[0] = static_cast<float>(pos);
    k_in[1] = static_cast<float>(pos) * 0.5f;
    v_in[0] = static_cast<float>(pos) * -1.0f;
    CHECK(pc_kv_cache_append(cache, 0u, pos, k_in.data(), v_in.data()) == FS_OK);
    CHECK(pc_kv_cache_append(cache, 5u, pos, k_in.data(), v_in.data()) == FS_OK);
  }

  CHECK(pc_kv_cache_visit_len(cache, 0u) == 1024u);
  CHECK(pc_kv_cache_visit_len(cache, 5u) == 2200u);

  std::vector<float>    k_out(1024u * kv_dim);
  std::vector<float>    v_out(1024u * kv_dim);
  std::vector<uint32_t> pos_out(1024u);
  uint32_t              cnt = 0u;

  CHECK(pc_kv_cache_read_window(cache, 0u, N_TOKS - 1u, 1024u, k_out.data(), v_out.data(),
                                pos_out.data(), &cnt) == FS_OK);
  CHECK(cnt == 1024u);
  const uint32_t expected_start = N_TOKS - 1024u;
  for (uint32_t i = 0; i < cnt; ++i) {
    const uint32_t ep = expected_start + i;
    CHECK(pos_out[i] == ep);
    CHECK(std::fabs(k_out[static_cast<size_t>(i) * kv_dim] - static_cast<float>(ep)) < 1e-6f);
    CHECK(std::fabs(v_out[static_cast<size_t>(i) * kv_dim] + static_cast<float>(ep)) < 1e-6f);
  }

  /* Verify full-context visit 5 retained token 0 intact */
  std::vector<float>    k_full(N_TOKS * kv_dim);
  std::vector<uint32_t> pos_full(N_TOKS);
  CHECK(pc_kv_cache_read_window(cache, 5u, N_TOKS - 1u, N_TOKS, k_full.data(), nullptr,
                                pos_full.data(), &cnt) == FS_OK);
  CHECK(cnt == N_TOKS);
  CHECK(pos_full[0] == 0u && std::fabs(k_full[0]) < 1e-6f);
  CHECK(pos_full[N_TOKS - 1u] == N_TOKS - 1u);

  pc_kv_cache_destroy(cache);
  pc_gguf_close(gf);
  std::remove(path.c_str());

  std::printf("[PASS] test_kv_sliding_window_ring\n");
  return true;
}

bool test_soa_repack_and_neon_parity() {
  const uint32_t n_dim = 2304u;
  const uint32_t k_dim = 1152u;

  std::vector<float> x(k_dim);
  for (uint32_t i = 0; i < k_dim; ++i) {
    x[i] = std::sin(static_cast<float>(i) * 0.05f);
  }

  /* 1. Q4_0 SoA parity */
  {
    const uint32_t       nb = k_dim / 32u;
    std::vector<uint8_t> q4(static_cast<size_t>(n_dim) * nb * 18u);
    for (uint32_t r = 0; r < n_dim; ++r) {
      for (uint32_t b = 0; b < nb; ++b) {
        uint8_t*       blk = q4.data() + (static_cast<size_t>(r) * nb + b) * 18u;
        const uint16_t d16 = fs_f32_to_f16(0.015f + 0.002f * static_cast<float>((r + b) % 9u));
        std::memcpy(blk, &d16, sizeof(d16));
        for (uint32_t j = 0; j < 16u; ++j) {
          blk[2u + j] = static_cast<uint8_t>(((r + j) & 0x0fu) | (((b + j * 3u) & 0x0fu) << 4u));
        }
      }
    }

    pc_soa_matrix_t mat{};
    CHECK(pc_soa_repack_q4_0(q4.data(), n_dim, k_dim, &mat) == FS_OK);
    CHECK((reinterpret_cast<uintptr_t>(mat.scales) & 15u) == 0u);
    CHECK((reinterpret_cast<uintptr_t>(mat.qs) & 15u) == 0u);

    std::vector<float> y_ref(n_dim), y_sc(n_dim), y_neon(n_dim), y_sdot(n_dim);
    CHECK(pc_gemv_ref_raw_f32(q4.data(), PC_GGML_TYPE_Q4_0, x.data(), y_ref.data(), n_dim, k_dim) ==
          FS_OK);
    CHECK(pc_gemv_soa_scalar_f32(&mat, x.data(), y_sc.data()) == FS_OK);
    CHECK(pc_gemv_soa_f32(&mat, x.data(), y_neon.data()) == FS_OK);
    CHECK(pc_gemv_soa_sdot(&mat, x.data(), y_sdot.data()) == FS_OK);

    fs_parity_t p_sc{}, p_neon{}, p_sdot{};
    fs_compute_parity_f32(y_sc.data(), y_ref.data(), n_dim, &p_sc);
    fs_compute_parity_f32(y_neon.data(), y_ref.data(), n_dim, &p_neon);
    fs_compute_parity_f32(y_sdot.data(), y_ref.data(), n_dim, &p_sdot);
    CHECK(p_sc.nmse < 1e-11 && p_neon.nmse < 1e-11 && p_sdot.nmse < 5e-4);
    pc_soa_free(&mat);
  }

  /* 2. IQ4_NL SoA parity */
  {
    const uint32_t       nb = k_dim / 32u;
    std::vector<uint8_t> iq4nl(static_cast<size_t>(n_dim) * nb * 18u);
    for (uint32_t r = 0; r < n_dim; ++r) {
      for (uint32_t b = 0; b < nb; ++b) {
        uint8_t*       blk = iq4nl.data() + (static_cast<size_t>(r) * nb + b) * 18u;
        const uint16_t d16 = fs_f32_to_f16(0.002f + 0.0004f * static_cast<float>((r + b) % 5u));
        std::memcpy(blk, &d16, sizeof(d16));
        for (uint32_t j = 0; j < 16u; ++j) {
          blk[2u + j] = static_cast<uint8_t>(((r * 5u + j) & 0x0fu) | (((b * 7u + j) & 0x0fu) << 4u));
        }
      }
    }

    pc_soa_matrix_t mat{};
    CHECK(pc_soa_repack_iq4_nl(iq4nl.data(), n_dim, k_dim, &mat) == FS_OK);
    std::vector<float> y_ref(n_dim), y_neon(n_dim), y_sdot(n_dim);
    CHECK(pc_gemv_ref_raw_f32(iq4nl.data(), PC_GGML_TYPE_IQ4_NL, x.data(), y_ref.data(), n_dim,
                              k_dim) == FS_OK);
    CHECK(pc_gemv_soa_f32(&mat, x.data(), y_neon.data()) == FS_OK);
    CHECK(pc_gemv_soa_sdot(&mat, x.data(), y_sdot.data()) == FS_OK);

    fs_parity_t p_neon{}, p_sdot{};
    fs_compute_parity_f32(y_neon.data(), y_ref.data(), n_dim, &p_neon);
    fs_compute_parity_f32(y_sdot.data(), y_ref.data(), n_dim, &p_sdot);
    CHECK(p_neon.nmse < 1e-11 && p_sdot.nmse < 5e-4);
    pc_soa_free(&mat);
  }

  /* 3. IQ4_XS (2304x2304 super-block) SoA parity */
  {
    const uint32_t       k_xs    = 2304u;
    const uint32_t       n_xs    = 256u;
    const uint32_t       n_super = k_xs / 256u;
    std::vector<float>   x_xs(k_xs);
    for (uint32_t i = 0; i < k_xs; ++i) x_xs[i] = std::cos(static_cast<float>(i) * 0.02f);

    std::vector<uint8_t> iq4xs(static_cast<size_t>(n_xs) * n_super * 136u);
    for (uint32_t r = 0; r < n_xs; ++r) {
      for (uint32_t sb = 0; sb < n_super; ++sb) {
        uint8_t*       sblk = iq4xs.data() + (static_cast<size_t>(r) * n_super + sb) * 136u;
        const uint16_t d16  = fs_f32_to_f16(0.001f);
        const uint16_t sh   = static_cast<uint16_t>((r + sb) & 0xffffu);
        std::memcpy(sblk, &d16, 2u);
        std::memcpy(sblk + 2u, &sh, 2u);
        for (uint32_t i = 0; i < 4u; ++i) sblk[4u + i] = static_cast<uint8_t>((r + i * 17u) & 0xffu);
        for (uint32_t i = 0; i < 128u; ++i) sblk[8u + i] = static_cast<uint8_t>((r + sb + i) & 0xffu);
      }
    }

    pc_soa_matrix_t mat{};
    CHECK(pc_soa_repack_iq4_xs(iq4xs.data(), n_xs, k_xs, &mat) == FS_OK);
    std::vector<float> y_ref(n_xs), y_neon(n_xs), y_sdot(n_xs);
    CHECK(pc_gemv_ref_raw_f32(iq4xs.data(), PC_GGML_TYPE_IQ4_XS, x_xs.data(), y_ref.data(), n_xs,
                              k_xs) == FS_OK);
    CHECK(pc_gemv_soa_f32(&mat, x_xs.data(), y_neon.data()) == FS_OK);
    CHECK(pc_gemv_soa_sdot(&mat, x_xs.data(), y_sdot.data()) == FS_OK);

    fs_parity_t p_neon{}, p_sdot{};
    fs_compute_parity_f32(y_neon.data(), y_ref.data(), n_xs, &p_neon);
    fs_compute_parity_f32(y_sdot.data(), y_ref.data(), n_xs, &p_sdot);
    CHECK(p_neon.nmse < 1e-11 && p_sdot.nmse < 5e-4);
    pc_soa_free(&mat);
  }

  std::printf("[PASS] test_soa_repack_and_neon_parity (%s)\n", pc_soa_kernel_backend_name());
  return true;
}

bool test_dspark_dedup_and_dmabuf_graph() {
  /* 1. DSpark MAX_DUP=3 and MAX_DUP=4 deduplication + exact MoE parity */
  const uint32_t num_experts = 16u;
  const uint32_t n_dim       = 32u;
  const uint32_t k_dim       = 64u;
  const uint32_t nb          = k_dim / 32u;

  std::vector<pc_soa_matrix_t> experts(num_experts);
  for (uint32_t e = 0; e < num_experts; ++e) {
    std::vector<uint8_t> raw(static_cast<size_t>(n_dim) * nb * 18u);
    for (uint32_t r = 0; r < n_dim; ++r) {
      for (uint32_t b = 0; b < nb; ++b) {
        uint8_t*       blk = raw.data() + (static_cast<size_t>(r) * nb + b) * 18u;
        const uint16_t d16 = fs_f32_to_f16(0.02f + 0.003f * static_cast<float>(e));
        std::memcpy(blk, &d16, 2u);
        for (uint32_t j = 0; j < 16u; ++j) {
          blk[2u + j] = static_cast<uint8_t>(((e + r + j) & 0x0fu) | (((e * 3u + b + j) & 0x0fu) << 4u));
        }
      }
    }
    CHECK(pc_soa_repack_q4_0(raw.data(), n_dim, k_dim, &experts[e]) == FS_OK);
  }

  const uint32_t routed_dup4[4u * 8u] = {
      0, 1, 2, 3, 4, 5, 6, 7,
      0, 1, 2, 3, 4, 5, 6, 8,
      0, 1, 2, 3, 4, 9, 6, 7,
      0, 1, 2, 10, 4, 5, 6, 7,
  };
  float rweights[4u * 8u];
  for (uint32_t i = 0; i < 32u; ++i) rweights[i] = 0.125f;

  pc_dspark_dedup_plan_t plan3{}, plan4{};
  CHECK(pc_dspark_plan_experts(routed_dup4, 3u, 8u, num_experts, 1492992ULL, &plan3) == FS_OK);
  CHECK(plan3.naive_expert_loads == 24u && plan3.dedup_expert_loads == 10u);
  CHECK(plan3.dedup_ratio > 2.3);

  CHECK(pc_dspark_plan_experts(routed_dup4, 4u, 8u, num_experts, 1492992ULL, &plan4) == FS_OK);
  CHECK(plan4.naive_expert_loads == 32u && plan4.dedup_expert_loads == 11u);
  CHECK(plan4.bytes_saved == 21ULL * 1492992ULL);

  std::vector<float> x_batch(4u * k_dim);
  for (uint32_t i = 0; i < 4u * k_dim; ++i) x_batch[i] = std::sin(static_cast<float>(i) * 0.07f);

  std::vector<float> y_dedup(4u * n_dim, 0.0f);
  CHECK(pc_dspark_execute_moe_f32(&plan4, experts.data(), num_experts, rweights, x_batch.data(),
                                  y_dedup.data()) == FS_OK);

  /* Independent un-deduplicated reference sum */
  std::vector<float> y_naive(4u * n_dim, 0.0f);
  std::vector<float> tmp(n_dim);
  for (uint32_t t = 0; t < 4u; ++t) {
    for (uint32_t k = 0; k < 8u; ++k) {
      CHECK(pc_gemv_soa_f32(&experts[routed_dup4[t * 8u + k]], x_batch.data() + t * k_dim,
                            tmp.data()) == FS_OK);
      for (uint32_t n = 0; n < n_dim; ++n) {
        y_naive[t * n_dim + n] += rweights[t * 8u + k] * tmp[n];
      }
    }
  }
  fs_parity_t p_moe{};
  fs_compute_parity_f32(y_dedup.data(), y_naive.data(), 4u * n_dim, &p_moe);
  CHECK(p_moe.max_abs_err < 1e-6f);

  for (uint32_t e = 0; e < num_experts; ++e) pc_soa_free(&experts[e]);

  /* 2. Pre-baked DMA-BUF graph + openbuenos contract verification */
  pc_graph_config_t gcfg = pc_graph_config_default();
  gcfg.hidden_dim        = 128u;
  gcfg.expert_ffn_dim    = 64u;
  pc_graph_t* graph      = nullptr;
  CHECK(pc_graph_create(&gcfg, &graph) == FS_OK);

  std::vector<float>   gx(128u, 0.5f), grms(128u, 1.0f), gy(64u, 0.0f);
  std::vector<uint8_t> gw(64u * 4u * 18u, 0u);
  for (uint32_t r = 0; r < 64u; ++r) {
    for (uint32_t b = 0; b < 4u; ++b) {
      uint8_t*       blk = gw.data() + (r * 4u + b) * 18u;
      const uint16_t d16 = fs_f32_to_f16(0.05f);
      std::memcpy(blk, &d16, 2u);
      for (uint32_t j = 0; j < 16u; ++j) blk[2u + j] = 0x97u;
    }
  }

  pc_graph_stats_t gstats{};
  CHECK(pc_graph_run_prebaked_step(graph, gx.data(), gw.data(), grms.data(), gy.data(), &gstats) ==
        FS_OK);
  CHECK(gstats.reconciled_zero_copy == 1);
  CHECK(gstats.reconciled_min_align >= 128u);
  CHECK(gstats.planned_steps == 2u);
  CHECK(gstats.zero_copy_handoffs >= 1u);
  CHECK(gstats.fallback_copies == 0u);
  CHECK(gstats.tpu_ops == 1u && gstats.cpu_ops == 1u);
  CHECK(gstats.nmse < 1e-10);

  pc_graph_destroy(graph);

  /* 3. Bound weights + upfront prepack + zero-repack steady-state loop + DIVE emitter wiring */
  char dive_tmp_path[256];
  std::snprintf(dive_tmp_path, sizeof(dive_tmp_path), "/tmp/freesched_dive_test_%ld.bin",
                static_cast<long>(getpid()));
  {
    FILE* df = std::fopen(dive_tmp_path, "wb");
    CHECK(df != nullptr);
    std::vector<uint8_t> synth_dive(65536u, 0x13u);
    CHECK(std::fwrite(synth_dive.data(), 1u, synth_dive.size(), df) == synth_dive.size());
    std::fclose(df);
  }

  pc_graph_config_t gcfg2 = pc_graph_config_default();
  gcfg2.hidden_dim        = 128u;
  gcfg2.expert_ffn_dim    = 64u;
  gcfg2.dive_interp_path  = dive_tmp_path;
  pc_graph_t* graph2      = nullptr;
  CHECK(pc_graph_create(&gcfg2, &graph2) == FS_OK);

  /* Step before binding weights must fail closed */
  CHECK(pc_graph_step_bound(graph2, gx.data(), gy.data(), nullptr) == FS_ERR_INVALID_ARG);

  CHECK(pc_graph_bind_weights(graph2, gw.data(), grms.data()) == FS_OK);

  pc_graph_stats_t gstats0{};
  pc_graph_stats_t gstats8{};
  CHECK(pc_graph_step_bound(graph2, gx.data(), gy.data(), &gstats0) == FS_OK);
  CHECK(gstats0.fallback_copies == 0u);
  CHECK(gstats0.nmse < 1e-10);

  std::vector<float> gx_step(128u, 0.0f);
  for (uint32_t s = 0u; s < 8u; ++s) {
    for (uint32_t j = 0u; j < 128u; ++j) {
      gx_step[j] = 0.25f + 0.03f * static_cast<float>((s + j) & 7u);
    }
    CHECK(pc_graph_step_bound(graph2, gx_step.data(), gy.data(), (s == 7u) ? &gstats8 : nullptr) ==
          FS_OK);
  }
  CHECK(gstats8.fallback_copies == 0u);
  CHECK(gstats8.nmse < 1e-10);
  CHECK(gstats8.ob_w_misses == gstats0.ob_w_misses);
  CHECK(gstats8.ob_w_evictions == gstats0.ob_w_evictions);
  CHECK(gstats8.ob_weight_pack_us == gstats0.ob_weight_pack_us);
  if (gstats8.ob_emitter_active == 1) {
    CHECK(gstats8.ob_freetile_active == 1);
    CHECK(gstats8.ob_calls >= 9u);
    CHECK(gstats8.ob_tiles_dispatched >= 9u);
    CHECK(gstats8.ob_w_hits > gstats0.ob_w_hits);
    CHECK(gstats8.ob_weight_cache_bytes > 0u);
  }

  /* Verify weight re-binding updates generation without stale cache hits */
  std::vector<uint8_t> gw2 = gw;
  for (uint32_t r = 0; r < 64u; ++r) {
    for (uint32_t b = 0; b < 4u; ++b) {
      uint8_t*       blk = gw2.data() + (r * 4u + b) * 18u;
      const uint16_t d16 = fs_f32_to_f16(0.11f);
      std::memcpy(blk, &d16, 2u);
    }
  }
  CHECK(pc_graph_bind_weights(graph2, gw2.data(), grms.data()) == FS_OK);
  pc_graph_stats_t gstats_rebind{};
  CHECK(pc_graph_step_bound(graph2, gx.data(), gy.data(), &gstats_rebind) == FS_OK);
  CHECK(gstats_rebind.nmse < 1e-10);

  pc_graph_destroy(graph2);
  (void)std::remove(dive_tmp_path);

  /* Strict-HW gate must fail closed when /dev/edgetpu-soc is absent */
  if (access("/dev/edgetpu-soc", F_OK) != 0) {
    pc_graph_config_t strict_cfg = pc_graph_config_default();
    strict_cfg.hidden_dim        = 128u;
    strict_cfg.expert_ffn_dim    = 64u;
    strict_cfg.mock_mode         = 0;
    strict_cfg.require_tpu_hw    = 1;
    pc_graph_t* strict_g         = nullptr;
    CHECK(pc_graph_create(&strict_cfg, &strict_g) == FS_ERR_NO_BACKEND);
    CHECK(strict_g == nullptr);
  }

  std::printf("[PASS] test_dspark_dedup_and_dmabuf_graph\n");
  return true;
}

} /* namespace */

int main(void) {
  if (!test_gguf_reader_and_bounds()) return 1;
  if (!test_kv_sliding_window_ring()) return 1;
  if (!test_soa_repack_and_neon_parity()) return 1;
  if (!test_dspark_dedup_and_dmabuf_graph()) return 1;
  std::printf("All 4 PocketCoder runner test suites PASSED.\n");
  return 0;
}
