/*
 * pocketcoder-run - Standalone PocketCoder-13B-A3B (inkling2_loop) CLI & Benchmark
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pocketcoder_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct pc_run_options {
  bool        inspect         = false;
  bool        bench           = false;
  bool        bench_synthetic = false;
  const char* gguf_path       = nullptr;
  uint32_t    iters           = 200u;
  int         mock_mode       = 1;
  int         strict_hw       = 0;
  const char* dive_interp     = nullptr;
  uint8_t     perf_mode       = 5u;
};

void print_usage(const char* argv0) {
  std::fprintf(
      stderr,
      "Usage:\n"
      "  %s --inspect <model.gguf>\n"
      "  %s --bench <model.gguf> [--iters N] [--live|--mock] [--strict-hw] [--dive-interp PATH] [--perf-mode 0..5]\n"
      "  %s --bench-synthetic [--iters N] [--live|--mock] [--strict-hw] [--dive-interp PATH] [--perf-mode 0..5]\n",
      argv0, argv0, argv0);
}

void build_synthetic_q4_0(uint32_t n_dim, uint32_t k_dim, std::vector<uint8_t>* out) {
  const uint32_t nb = k_dim / 32u;
  out->assign(static_cast<size_t>(n_dim) * nb * 18u, 0u);
  for (uint32_t r = 0; r < n_dim; ++r) {
    for (uint32_t b = 0; b < nb; ++b) {
      uint8_t*       blk = out->data() + (static_cast<size_t>(r) * nb + b) * 18u;
      const uint16_t d16 = fs_f32_to_f16(0.02f + 0.005f * static_cast<float>((r + b) % 7u));
      std::memcpy(blk, &d16, sizeof(d16));
      for (uint32_t j = 0; j < 16u; ++j) {
        const uint8_t lo = static_cast<uint8_t>((r * 3u + b * 5u + j) & 0x0fu);
        const uint8_t hi = static_cast<uint8_t>((r * 7u + b * 11u + j + 3u) & 0x0fu);
        blk[2u + j]      = static_cast<uint8_t>(lo | (hi << 4u));
      }
    }
  }
}

void print_kv_table(const pc_inkling2_config_t* cfg) {
  const uint32_t ctxs[4] = {4096u, 16384u, 32768u, 131072u};
  std::printf("\nKV Cache Footprint (40/%u sliding @ %u tok + %u/%u full, %u KV heads x %u dim):\n",
              cfg->logical_visits, cfg->sliding_window, cfg->full_context_visits,
              cfg->logical_visits, cfg->head_count_kv, cfg->key_length);
  std::printf("  %-9s | %-11s | %-11s | %-11s | %-9s\n",
              "Context", "Naive Full", "Sliding KV", "Saved", "Reduction");
  for (uint32_t c : ctxs) {
    pc_kv_footprint_t fp16{};
    if (pc_kv_compute_footprint(cfg, c, 2u, &fp16) == FS_OK) {
      std::printf("  %-9u | %8.2f MB | %8.2f MB | %8.2f MB | %7.2fx\n",
                  c,
                  static_cast<double>(fp16.naive_full_kv_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(fp16.total_kv_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(fp16.saved_bytes) / (1024.0 * 1024.0),
                  fp16.reduction_ratio);
    }
  }
}

double median_us(std::vector<double> samples) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  return (n & 1u) ? samples[n / 2u] : 0.5 * (samples[n / 2u - 1u] + samples[n / 2u]);
}

template <typename Fn>
double run_timed_median_us(uint32_t iters, Fn&& fn, volatile float* sink) {
  /* Warmup */
  for (uint32_t i = 0; i < 10u; ++i) {
    fn();
  }
  std::vector<double> samples;
  samples.reserve(iters);
  for (uint32_t i = 0; i < iters; ++i) {
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    samples.push_back(us);
    *sink += 0.0f;
  }
  return median_us(samples);
}

int run_inspect(const char* path) {
  pc_gguf_file_t* gf = nullptr;
  fs_status_t     rc = pc_gguf_open(path, &gf);
  if (rc != FS_OK) {
    std::fprintf(stderr, "Failed to open GGUF '%s': %s\n", path, fs_status_string(rc));
    return 1;
  }

  char err[256];
  rc = pc_gguf_validate_pocketcoder(gf, err, sizeof(err));
  const pc_inkling2_config_t* cfg = pc_gguf_config(gf);

  std::printf("GGUF: %s (%.2f GiB, %u tensors, data_offset=%llu)\n",
              cfg->model_name[0] ? cfg->model_name : path,
              static_cast<double>(pc_gguf_file_size(gf)) / (1024.0 * 1024.0 * 1024.0),
              pc_gguf_tensor_count(gf),
              static_cast<unsigned long long>(pc_gguf_data_offset(gf)));
  std::printf("  arch=%s blocks=%u loop=[%u..%u)x%u -> %u visits (%u SWA-%u, %u full)\n",
              cfg->arch, cfg->block_count, cfg->loop_start, cfg->loop_end, cfg->loop_count,
              cfg->logical_visits, cfg->sliding_window_visits, cfg->sliding_window,
              cfg->full_context_visits);
  std::printf("  geom: d_model=%u d_ff=%u experts=%u (top-%u) heads=%u/%u head_dim=%u\n",
              cfg->embedding_length, cfg->expert_feed_forward_length, cfg->expert_count,
              cfg->expert_used_count, cfg->head_count, cfg->head_count_kv, cfg->key_length);

  const pc_gguf_tensor_t* down = pc_gguf_find_tensor(gf, "blk.0.ffn_down_exps.weight");
  const pc_gguf_tensor_t* gate = pc_gguf_find_tensor(gf, "blk.0.ffn_gate_up_exps.weight");
  if (down) {
    std::printf("  blk.0.ffn_down_exps.weight: (%llu, %llu, %llu) type=%s\n",
                static_cast<unsigned long long>(down->ne[0]),
                static_cast<unsigned long long>(down->ne[1]),
                static_cast<unsigned long long>(down->ne[2]),
                pc_ggml_type_name(down->ggml_type));
  }
  if (gate) {
    std::printf("  blk.0.ffn_gate_up_exps.weight: (%llu, %llu, %llu) type=%s\n",
                static_cast<unsigned long long>(gate->ne[0]),
                static_cast<unsigned long long>(gate->ne[1]),
                static_cast<unsigned long long>(gate->ne[2]),
                pc_ggml_type_name(gate->ggml_type));
  }

  print_kv_table(cfg);
  if (rc != FS_OK) {
    std::fprintf(stderr, "Validation FAILED: %s\n", err);
    pc_gguf_close(gf);
    return 1;
  }
  std::printf("\nValidation: PASS (100%% inkling2_loop PocketCoder-13B-A3B invariants)\n");
  pc_gguf_close(gf);
  return 0;
}

int run_bench(const pc_run_options& opts) {
  const char*          gguf_path = opts.bench ? opts.gguf_path : nullptr;
  const uint32_t       iters     = opts.iters;
  pc_inkling2_config_t cfg{};
  pc_soa_matrix_t      mat{};
  std::vector<uint8_t> synth_q4;
  const void*          raw_ptr   = nullptr;
  uint32_t             raw_type  = PC_GGML_TYPE_Q4_0;
  uint32_t             n_dim     = 2304u;
  uint32_t             k_dim     = 1152u;
  const char*          src_label = "synthetic_q4_0";

  pc_gguf_file_t* gf = nullptr;
  if (gguf_path) {
    if (pc_gguf_open(gguf_path, &gf) != FS_OK) {
      std::fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path);
      return 1;
    }
    char err[256];
    if (pc_gguf_validate_pocketcoder(gf, err, sizeof(err)) != FS_OK) {
      std::fprintf(stderr, "Validation failed: %s\n", err);
      pc_gguf_close(gf);
      return 1;
    }
    cfg = *pc_gguf_config(gf);

    const pc_gguf_tensor_t* down = pc_gguf_find_tensor(gf, "blk.0.ffn_down_exps.weight");
    if (!down || pc_soa_repack_expert_from_gguf(gf, "blk.0.ffn_down_exps.weight", 0u, &mat) != FS_OK) {
      std::fprintf(stderr, "Failed to repack blk.0.ffn_down_exps.weight[0]\n");
      pc_gguf_close(gf);
      return 1;
    }
    k_dim     = static_cast<uint32_t>(down->ne[0]);
    n_dim     = static_cast<uint32_t>(down->ne[1]);
    raw_ptr   = pc_gguf_tensor_data(gf, down);
    raw_type  = down->ggml_type;
    src_label = pc_ggml_type_name(raw_type);
  } else {
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.block_count                = 24u;
    cfg.context_length             = 1048576u;
    cfg.embedding_length           = 2304u;
    cfg.expert_feed_forward_length = 1152u;
    cfg.head_count                 = 24u;
    cfg.head_count_kv              = 8u;
    cfg.key_length                 = 128u;
    cfg.value_length               = 128u;
    cfg.expert_count               = 64u;
    cfg.expert_used_count          = 8u;
    cfg.sliding_window             = 1024u;
    cfg.logical_visits             = 48u;
    cfg.sliding_window_visits      = 40u;
    cfg.full_context_visits        = 8u;
    build_synthetic_q4_0(n_dim, k_dim, &synth_q4);
    raw_ptr = synth_q4.data();
    if (pc_soa_repack_q4_0(raw_ptr, n_dim, k_dim, &mat) != FS_OK) {
      return 1;
    }
  }

  std::vector<float> x(k_dim);
  for (uint32_t i = 0; i < k_dim; ++i) {
    x[i] = std::sin(static_cast<float>(i) * 0.03125f);
  }

  std::vector<float> y_ref(n_dim, 0.0f);
  std::vector<float> y_scalar(n_dim, 0.0f);
  std::vector<float> y_neon_f32(n_dim, 0.0f);
  std::vector<float> y_neon_sdot(n_dim, 0.0f);

  pc_gemv_ref_raw_f32(raw_ptr, raw_type, x.data(), y_ref.data(), n_dim, k_dim);
  pc_gemv_soa_scalar_f32(&mat, x.data(), y_scalar.data());
  pc_gemv_soa_f32(&mat, x.data(), y_neon_f32.data());
  pc_gemv_soa_sdot(&mat, x.data(), y_neon_sdot.data());

  fs_parity_t p_f32{}, p_sdot{};
  fs_compute_parity_f32(y_neon_f32.data(), y_ref.data(), n_dim, &p_f32);
  fs_compute_parity_f32(y_neon_sdot.data(), y_ref.data(), n_dim, &p_sdot);

  /* Control -> Candidate -> Control benchmark */
  volatile float sink = 0.0f;
  const double c1_us = run_timed_median_us(
      iters, [&]() { pc_gemv_soa_scalar_f32(&mat, x.data(), y_scalar.data()); }, &sink);
  const double neon_f32_us = run_timed_median_us(
      iters, [&]() { pc_gemv_soa_f32(&mat, x.data(), y_neon_f32.data()); }, &sink);
  const double neon_sdot_us = run_timed_median_us(
      iters, [&]() { pc_gemv_soa_sdot(&mat, x.data(), y_neon_sdot.data()); }, &sink);
  const double c2_us = run_timed_median_us(
      iters, [&]() { pc_gemv_soa_scalar_f32(&mat, x.data(), y_scalar.data()); }, &sink);
  const double ctrl_us = 0.5 * (c1_us + c2_us);

  const double bytes_read = static_cast<double>(mat.scales_bytes + mat.qs_bytes + k_dim * 4u);
  const double bw_ctrl    = (bytes_read / (ctrl_us * 1e-6)) / 1e9;
  const double bw_f32     = (bytes_read / (neon_f32_us * 1e-6)) / 1e9;
  const double bw_sdot    = (bytes_read / (neon_sdot_us * 1e-6)) / 1e9;

  /* DSpark MAX_DUP=4 speculative expert dedup sample (realistic coding sequence overlap) */
  const uint32_t routed_dup4[4u * 8u] = {
      0, 3, 7, 12, 19, 25, 41, 58,
      0, 3, 7, 12, 19, 25, 41, 60,
      0, 3, 7, 12, 22, 25, 41, 58,
      0, 5, 7, 12, 19, 25, 41, 58,
  };
  pc_dspark_dedup_plan_t dplan{};
  pc_dspark_plan_experts(routed_dup4, 4u, 8u, 64u, mat.scales_bytes + mat.qs_bytes, &dplan);

  /* Pre-baked DMA-BUF graph step check (uses Q4_0 payload for EdgeTPU/openbuenos contract) */
  pc_graph_config_t gcfg = pc_graph_config_default();
  gcfg.hidden_dim        = k_dim;
  gcfg.expert_ffn_dim    = n_dim;
  gcfg.mock_mode         = opts.mock_mode;
  gcfg.dive_interp_path  = opts.dive_interp;
  gcfg.require_tpu_hw    = opts.strict_hw;
  gcfg.perf_mode         = opts.perf_mode;

  pc_graph_t*      graph = nullptr;
  pc_graph_stats_t gstats_first{};
  pc_graph_stats_t gstats_end{};
  std::vector<uint8_t> dmabuf_w_q4;
  const void* graph_w_ptr = nullptr;
  if (raw_type == PC_GGML_TYPE_Q4_0 && raw_ptr != nullptr) {
    graph_w_ptr = raw_ptr;
  } else {
    build_synthetic_q4_0(n_dim, k_dim, &dmabuf_w_q4);
    graph_w_ptr = dmabuf_w_q4.data();
  }
  std::vector<float> rms_w(k_dim, 1.0f);
  std::vector<float> y_graph(n_dim, 0.0f);
  std::vector<float> x_step(x);

  fs_status_t grc = pc_graph_create(&gcfg, &graph);
  if (grc != FS_OK) {
    std::fprintf(stderr, "pc_graph_create failed (mock_mode=%d, strict_hw=%d): %s\n",
                 gcfg.mock_mode, gcfg.require_tpu_hw, fs_status_string(grc));
    pc_soa_free(&mat);
    if (gf) pc_gguf_close(gf);
    return 1;
  }

  fs_status_t brc = pc_graph_bind_weights(graph, graph_w_ptr, rms_w.data());
  if (brc == FS_OK) {
    brc = pc_graph_step_bound(graph, x.data(), y_graph.data(), &gstats_first);
  }
  if (brc != FS_OK) {
    std::fprintf(stderr, "pc_graph bind/first step failed: %s\n", fs_status_string(brc));
    pc_graph_destroy(graph);
    pc_soa_free(&mat);
    if (gf) pc_gguf_close(gf);
    return 1;
  }

  const uint32_t bound_steps = std::max<uint32_t>(2u, std::min<uint32_t>(iters, 16u));
  for (uint32_t s = 0; s < bound_steps; ++s) {
    const float scale = 1.0f + 0.005f * static_cast<float>(s + 1u);
    for (uint32_t j = 0; j < k_dim; ++j) {
      x_step[j] = x[j] * scale;
    }
    pc_graph_stats_t* dst_stats = (s + 1u == bound_steps) ? &gstats_end : nullptr;
    brc = pc_graph_step_bound(graph, x_step.data(), y_graph.data(), dst_stats);
    if (brc != FS_OK) {
      std::fprintf(stderr, "pc_graph_step_bound failed at step %u: %s\n", s, fs_status_string(brc));
      pc_graph_destroy(graph);
      pc_soa_free(&mat);
      if (gf) pc_gguf_close(gf);
      return 1;
    }
  }
  const double graph_bound_us = run_timed_median_us(
      std::min<uint32_t>(iters, 20u),
      [&]() { (void)pc_graph_step_bound(graph, x.data(), y_graph.data(), nullptr); }, &sink);
  (void)pc_graph_step_bound(graph, x_step.data(), y_graph.data(), &gstats_end);
  pc_graph_destroy(graph);

  const uint64_t delta_misses = (gstats_end.ob_w_misses >= gstats_first.ob_w_misses)
                                    ? (gstats_end.ob_w_misses - gstats_first.ob_w_misses)
                                    : 0u;
  const uint64_t delta_evict = (gstats_end.ob_w_evictions >= gstats_first.ob_w_evictions)
                                   ? (gstats_end.ob_w_evictions - gstats_first.ob_w_evictions)
                                   : 0u;
  const uint64_t delta_pack_us = (gstats_end.ob_weight_pack_us >= gstats_first.ob_weight_pack_us)
                                     ? (gstats_end.ob_weight_pack_us - gstats_first.ob_weight_pack_us)
                                     : 0u;

  std::printf("PocketCoder-13B-A3B Micro-Benchmark [%s, %ux%u, backend=%s, iters=%u, mode=%s]\n",
              src_label, n_dim, k_dim, pc_soa_kernel_backend_name(), iters,
              opts.mock_mode ? "mock" : (gstats_end.ob_hw_active ? "live-hw" : "live-fallback"));
  std::printf("Parity vs FP32 ref: SoA-F32 max_abs=%.3e nmse=%.3e | SoA-SDOT max_abs=%.3e nmse=%.3e\n",
              p_f32.max_abs_err, p_f32.nmse, p_sdot.max_abs_err, p_sdot.nmse);
  std::printf("DSpark (MAX_DUP=4): %u -> %u unique experts (%.2fx dedup, %.2f MB saved/layer)\n",
              dplan.naive_expert_loads, dplan.dedup_expert_loads, dplan.dedup_ratio,
              static_cast<double>(dplan.bytes_saved) / (1024.0 * 1024.0));
  std::printf(
      "DMA-BUF graph: zero_copy=%d min_align=%zu handoffs=%llu fallback_copies=%llu "
      "hw_active=%d emitter=%d freetile=%d ob_calls=%llu ob_tiles=%llu ob_hw_us=%llu "
      "w_hits=%llu w_misses=%llu(delta=%llu) w_evict=%llu(delta=%llu) pack_us=%llu(delta=%llu) "
      "w_bytes=%llu nmse=%.3e\n\n",
      gstats_end.reconciled_zero_copy, gstats_end.reconciled_min_align,
      static_cast<unsigned long long>(gstats_end.zero_copy_handoffs),
      static_cast<unsigned long long>(gstats_end.fallback_copies),
      gstats_end.ob_hw_active, gstats_end.ob_emitter_active, gstats_end.ob_freetile_active,
      static_cast<unsigned long long>(gstats_end.ob_calls),
      static_cast<unsigned long long>(gstats_end.ob_tiles_dispatched),
      static_cast<unsigned long long>(gstats_end.ob_hw_us),
      static_cast<unsigned long long>(gstats_end.ob_w_hits),
      static_cast<unsigned long long>(gstats_end.ob_w_misses),
      static_cast<unsigned long long>(delta_misses),
      static_cast<unsigned long long>(gstats_end.ob_w_evictions),
      static_cast<unsigned long long>(delta_evict),
      static_cast<unsigned long long>(gstats_end.ob_weight_pack_us),
      static_cast<unsigned long long>(delta_pack_us),
      static_cast<unsigned long long>(gstats_end.ob_weight_cache_bytes),
      gstats_end.nmse);

  std::printf("| Metric | Baseline (Scalar SoA) | Now (NEON SoA F32) | Now (NEON SoA sdot) | How measured |\n");
  std::printf("|---|---:|---:|---:|---|\n");
  std::printf("| Expert GEMV %ux%u (%s) | %.2f us (%.2f us / %.2f us) | %.2f us (%.2fx) | %.2f us (%.2fx) | C-C-C median (%u iters) |\n",
              n_dim, k_dim, src_label, ctrl_us, c1_us, c2_us,
              neon_f32_us, ctrl_us / neon_f32_us,
              neon_sdot_us, ctrl_us / neon_sdot_us, iters);
  std::printf("| Effective Weight BW | %.2f GB/s | %.2f GB/s | %.2f GB/s | (scales+qs+x)/median_us |\n",
              bw_ctrl, bw_f32, bw_sdot);
  std::printf("| Pre-baked Bound Step %ux%u | - | %.2f us median | hw=%d emitter=%d | pc_graph_step_bound (%u iters) |\n",
              n_dim, k_dim, graph_bound_us, gstats_end.ob_hw_active, gstats_end.ob_emitter_active,
              std::min<uint32_t>(iters, 20u));

  pc_soa_free(&mat);
  if (gf) {
    pc_gguf_close(gf);
  }
  const double graph_nmse_limit = gstats_end.ob_hw_active ? 5e-3 : 1e-10;
  const bool   dmabuf_ok        = (gstats_end.fallback_copies == 0u) ||
                                  (!opts.mock_mode && !opts.strict_hw && gstats_end.ob_hw_active == 0);
  const bool   zero_repack_ok   = (delta_misses == 0u && delta_evict == 0u && delta_pack_us == 0u);
  const bool   graph_parity_ok  = (gstats_first.nmse < graph_nmse_limit && gstats_end.nmse < graph_nmse_limit);
  return (p_f32.nmse < 1e-10 && p_sdot.nmse < 1e-2 && dmabuf_ok &&
          zero_repack_ok && graph_parity_ok)
             ? 0
             : 1;
}

} /* namespace */

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  pc_run_options opts{};
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--inspect") == 0 && i + 1 < argc) {
      opts.inspect   = true;
      opts.gguf_path = argv[++i];
    } else if (std::strcmp(arg, "--bench") == 0 && i + 1 < argc) {
      opts.bench     = true;
      opts.gguf_path = argv[++i];
    } else if (std::strcmp(arg, "--bench-synthetic") == 0) {
      opts.bench_synthetic = true;
    } else if (std::strcmp(arg, "--iters") == 0 && i + 1 < argc) {
      opts.iters = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
      if (opts.iters == 0u) opts.iters = 1u;
    } else if (std::strcmp(arg, "--live") == 0) {
      opts.mock_mode = 0;
    } else if (std::strcmp(arg, "--mock") == 0) {
      opts.mock_mode = 1;
    } else if (std::strcmp(arg, "--strict-hw") == 0) {
      opts.mock_mode = 0;
      opts.strict_hw = 1;
    } else if (std::strcmp(arg, "--dive-interp") == 0 && i + 1 < argc) {
      opts.dive_interp = argv[++i];
    } else if (std::strcmp(arg, "--perf-mode") == 0 && i + 1 < argc) {
      unsigned long pm = std::strtoul(argv[++i], nullptr, 10);
      opts.perf_mode   = static_cast<uint8_t>(pm <= 5ul ? pm : 5ul);
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg);
      print_usage(argv[0]);
      return 1;
    }
  }

  if (opts.inspect && !opts.bench && !opts.bench_synthetic && opts.gguf_path) {
    return run_inspect(opts.gguf_path);
  }
  if (opts.bench && !opts.inspect && !opts.bench_synthetic && opts.gguf_path) {
    return run_bench(opts);
  }
  if (opts.bench_synthetic && !opts.inspect && !opts.bench) {
    return run_bench(opts);
  }

  print_usage(argv[0]);
  return 1;
}
