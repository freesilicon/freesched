/*
 * freesched executor, pluggable backend vtable, and portable mock/CPU backends.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "fs_internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* =========================================================================
 * IEEE-754 FP16 <-> FP32 & Quantized Block Math (1:1 with openbuenos / ggml)
 * ========================================================================= */

static float fs_f16_to_f32_impl(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1fu;
  uint32_t mant = static_cast<uint32_t>(h & 0x03ffu);
  uint32_t bits = 0u;

  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;
    } else {
      exp = 1u;
      while ((mant & 0x0400u) == 0u) {
        mant <<= 1u;
        exp--;
      }
      mant &= 0x03ffu;
      bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
  } else if (exp == 31u) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static uint16_t fs_f32_to_f16_impl(float f) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &f, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t        exp  = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t       mant = bits & 0x007fffffu;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant = (mant | 0x00800000u) >> static_cast<uint32_t>(1 - exp);
    if ((mant & 0x00001000u) != 0u) {
      mant += 0x00002000u;
    }
    return static_cast<uint16_t>(sign | (mant >> 13));
  }
  if (exp >= 31) {
    if (exp == 128 + 15 && mant != 0u) {
      return static_cast<uint16_t>(sign | 0x7e00u | (mant >> 13));
    }
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  if ((mant & 0x00001000u) != 0u) {
    mant += 0x00002000u;
    if ((mant & 0x00800000u) != 0u) {
      mant = 0u;
      exp++;
      if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static float fs_apply_activation(float x, fs_activation_t act) {
  switch (act) {
    case FS_ACT_RELU:
      return x > 0.0f ? x : 0.0f;
    case FS_ACT_RELU6:
      if (x <= 0.0f) return 0.0f;
      return x >= 6.0f ? 6.0f : x;
    case FS_ACT_SILU:
      return x / (1.0f + std::exp(-x));
    case FS_ACT_GELU: {
      const float k = 0.7978845608f; /* sqrt(2 / pi) */
      return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
    }
    case FS_ACT_NONE:
    default:
      return x;
  }
}

static float fs_read_act(const void* ptr, fs_dtype_t dtype, size_t idx) {
  if (dtype == FS_DTYPE_F16) {
    return fs_f16_to_f32_impl(static_cast<const uint16_t*>(ptr)[idx]);
  }
  return static_cast<const float*>(ptr)[idx];
}

static void fs_write_act(void* ptr, fs_dtype_t dtype, size_t idx, float val) {
  if (dtype == FS_DTYPE_F16) {
    static_cast<uint16_t*>(ptr)[idx] = fs_f32_to_f16_impl(val);
  } else {
    static_cast<float*>(ptr)[idx] = val;
  }
}

static float fs_dot_row(const void* a_row,
                        fs_dtype_t  a_dtype,
                        const void* b_base,
                        fs_dtype_t  b_dtype,
                        uint32_t    n_row,
                        uint32_t    k_dim) {
  double acc = 0.0;
  if (b_dtype == FS_DTYPE_F32) {
    const float* b = static_cast<const float*>(b_base) + static_cast<size_t>(n_row) * k_dim;
    for (uint32_t k = 0; k < k_dim; ++k) {
      acc += static_cast<double>(fs_read_act(a_row, a_dtype, k)) * static_cast<double>(b[k]);
    }
  } else if (b_dtype == FS_DTYPE_F16) {
    const uint16_t* b = static_cast<const uint16_t*>(b_base) + static_cast<size_t>(n_row) * k_dim;
    for (uint32_t k = 0; k < k_dim; ++k) {
      acc += static_cast<double>(fs_read_act(a_row, a_dtype, k)) *
             static_cast<double>(fs_f16_to_f32_impl(b[k]));
    }
  } else if (b_dtype == FS_DTYPE_I8) {
    const int8_t* b = static_cast<const int8_t*>(b_base) + static_cast<size_t>(n_row) * k_dim;
    for (uint32_t k = 0; k < k_dim; ++k) {
      acc += static_cast<double>(fs_read_act(a_row, a_dtype, k)) * static_cast<double>(b[k]);
    }
  } else if (b_dtype == FS_DTYPE_Q8_0) {
    const uint32_t         nblocks = k_dim / FS_QK8_0;
    const fs_block_q8_0_t* row_blk =
        static_cast<const fs_block_q8_0_t*>(b_base) + static_cast<size_t>(n_row) * nblocks;
    for (uint32_t blk = 0; blk < nblocks; ++blk) {
      const float scale = fs_f16_to_f32_impl(row_blk[blk].d);
      const uint32_t base_k = blk * FS_QK8_0;
      for (uint32_t j = 0; j < FS_QK8_0; ++j) {
        const float w = scale * static_cast<float>(row_blk[blk].qs[j]);
        acc += static_cast<double>(fs_read_act(a_row, a_dtype, base_k + j)) * static_cast<double>(w);
      }
    }
  } else if (b_dtype == FS_DTYPE_Q4_0) {
    const uint32_t         nblocks = k_dim / FS_QK8_0;
    const fs_block_q4_0_t* row_blk =
        static_cast<const fs_block_q4_0_t*>(b_base) + static_cast<size_t>(n_row) * nblocks;
    for (uint32_t blk = 0; blk < nblocks; ++blk) {
      const float    scale  = fs_f16_to_f32_impl(row_blk[blk].d);
      const uint32_t base_k = blk * FS_QK8_0;
      for (uint32_t j = 0; j < FS_QK8_0 / 2u; ++j) {
        const uint8_t packed = row_blk[blk].qs[j];
        const int     lo     = static_cast<int>(packed & 0x0fu) - 8;
        const int     hi     = static_cast<int>(packed >> 4u) - 8;
        acc += static_cast<double>(fs_read_act(a_row, a_dtype, base_k + j)) *
               static_cast<double>(scale * static_cast<float>(lo));
        acc += static_cast<double>(fs_read_act(a_row, a_dtype, base_k + j + 16u)) *
               static_cast<double>(scale * static_cast<float>(hi));
      }
    }
  }
  return static_cast<float>(acc);
}

static fs_status_t fs_ref_execute_gemm(const fs_op_desc_t* op) {
  if (!op->input_a || !op->input_b || !op->output || !op->input_a->host_ptr ||
      !op->input_b->host_ptr || !op->output->host_ptr) {
    return FS_ERR_INVALID_ARG;
  }
  if ((op->weight_dtype == FS_DTYPE_Q8_0 || op->weight_dtype == FS_DTYPE_Q4_0) &&
      (op->k_dim % FS_QK8_0) != 0u) {
    return FS_ERR_INVALID_ARG;
  }
  const size_t a_stride =
      static_cast<size_t>(op->k_dim) * (op->act_dtype == FS_DTYPE_F16 ? 2u : 4u);
  const auto* a_bytes = static_cast<const uint8_t*>(op->input_a->host_ptr);

  for (uint32_t m = 0; m < op->m_dim; ++m) {
    const void* a_row = a_bytes + static_cast<size_t>(m) * a_stride;
    for (uint32_t n = 0; n < op->n_dim; ++n) {
      float val = fs_dot_row(a_row, op->act_dtype, op->input_b->host_ptr, op->weight_dtype, n,
                             op->k_dim);
      val       = fs_apply_activation(val, op->activation);
      fs_write_act(op->output->host_ptr, op->act_dtype, static_cast<size_t>(m) * op->n_dim + n, val);
    }
  }
  return FS_OK;
}

static fs_status_t fs_ref_execute_swiglu(const fs_op_desc_t* op) {
  if (!op->input_a || !op->input_b || !op->input_c || !op->input_d || !op->output ||
      !op->input_a->host_ptr || !op->input_b->host_ptr || !op->input_c->host_ptr ||
      !op->input_d->host_ptr || !op->output->host_ptr) {
    return FS_ERR_INVALID_ARG;
  }
  if (op->h_dim == 0u) {
    return FS_ERR_INVALID_ARG;
  }
  if ((op->weight_dtype == FS_DTYPE_Q8_0 || op->weight_dtype == FS_DTYPE_Q4_0) &&
      ((op->k_dim % FS_QK8_0) != 0u || (op->h_dim % FS_QK8_0) != 0u)) {
    return FS_ERR_INVALID_ARG;
  }

  std::vector<float> mid(op->h_dim, 0.0f);
  const size_t       a_stride =
      static_cast<size_t>(op->k_dim) * (op->act_dtype == FS_DTYPE_F16 ? 2u : 4u);
  const auto* a_bytes = static_cast<const uint8_t*>(op->input_a->host_ptr);

  for (uint32_t m = 0; m < op->m_dim; ++m) {
    const void* a_row = a_bytes + static_cast<size_t>(m) * a_stride;
    for (uint32_t h = 0; h < op->h_dim; ++h) {
      const float g = fs_dot_row(a_row, op->act_dtype, op->input_b->host_ptr, op->weight_dtype, h,
                                 op->k_dim);
      const float u = fs_dot_row(a_row, op->act_dtype, op->input_c->host_ptr, op->weight_dtype, h,
                                 op->k_dim);
      mid[h]        = fs_apply_activation(g, FS_ACT_SILU) * u;
    }
    for (uint32_t n = 0; n < op->n_dim; ++n) {
      const float out = fs_dot_row(mid.data(), FS_DTYPE_F32, op->input_d->host_ptr,
                                   op->weight_dtype, n, op->h_dim);
      fs_write_act(op->output->host_ptr, op->act_dtype, static_cast<size_t>(m) * op->n_dim + n, out);
    }
  }
  return FS_OK;
}

static fs_status_t fs_ref_execute_softmax(const fs_op_desc_t* op) {
  if (!op->input_a || !op->output || !op->input_a->host_ptr || !op->output->host_ptr ||
      op->n_dim == 0u) {
    return FS_ERR_INVALID_ARG;
  }
  const float scale = (op->aux_param > 0.0f) ? op->aux_param : 1.0f;
  const uint32_t rows = op->m_dim > 0u ? op->m_dim : 1u;

  for (uint32_t r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * op->n_dim;
    float        mx   = -1e30f;
    for (uint32_t c = 0; c < op->n_dim; ++c) {
      const float v = fs_read_act(op->input_a->host_ptr, op->act_dtype, base + c) * scale;
      if (v > mx) {
        mx = v;
      }
    }
    double sum = 0.0;
    for (uint32_t c = 0; c < op->n_dim; ++c) {
      const float v = fs_read_act(op->input_a->host_ptr, op->act_dtype, base + c) * scale;
      sum += std::exp(static_cast<double>(v - mx));
    }
    const double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
    for (uint32_t c = 0; c < op->n_dim; ++c) {
      const float v = fs_read_act(op->input_a->host_ptr, op->act_dtype, base + c) * scale;
      const float p = static_cast<float>(std::exp(static_cast<double>(v - mx)) * inv);
      fs_write_act(op->output->host_ptr, op->act_dtype, base + c, p);
    }
  }
  return FS_OK;
}

static fs_status_t fs_ref_execute_rms_norm(const fs_op_desc_t* op) {
  if (!op->input_a || !op->output || !op->input_a->host_ptr || !op->output->host_ptr ||
      op->n_dim == 0u) {
    return FS_ERR_INVALID_ARG;
  }
  const float    eps    = (op->aux_param > 0.0f) ? op->aux_param : 1e-5f;
  const uint32_t rows   = op->m_dim > 0u ? op->m_dim : 1u;
  const void*    weight = (op->input_b && op->input_b->host_ptr) ? op->input_b->host_ptr : nullptr;

  for (uint32_t r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * op->n_dim;
    double       ss   = 0.0;
    for (uint32_t c = 0; c < op->n_dim; ++c) {
      const double v = static_cast<double>(fs_read_act(op->input_a->host_ptr, op->act_dtype, base + c));
      ss += v * v;
    }
    const float inv_rms =
        static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(op->n_dim) + eps));
    for (uint32_t c = 0; c < op->n_dim; ++c) {
      const float x = fs_read_act(op->input_a->host_ptr, op->act_dtype, base + c);
      const float w = weight ? fs_read_act(weight, op->weight_dtype, c) : 1.0f;
      fs_write_act(op->output->host_ptr, op->act_dtype, base + c, x * inv_rms * w);
    }
  }
  return FS_OK;
}

static fs_status_t fs_ref_execute_rope(const fs_op_desc_t* op) {
  if (!op->input_a || !op->output || !op->input_a->host_ptr || !op->output->host_ptr ||
      op->n_dim == 0u || (op->n_dim % 2u) != 0u) {
    return FS_ERR_INVALID_ARG;
  }
  const double   theta_base = (op->aux_param > 1.0f) ? static_cast<double>(op->aux_param) : 10000.0;
  const uint32_t rows       = op->m_dim > 0u ? op->m_dim : 1u;
  const uint32_t head_dim   = (op->h_dim > 0u && (op->h_dim % 2u) == 0u) ? op->h_dim : op->n_dim;

  for (uint32_t r = 0; r < rows; ++r) {
    const double pos  = static_cast<double>(op->pos_offset + r);
    const size_t base = static_cast<size_t>(r) * op->n_dim;
    for (uint32_t c = 0; c < op->n_dim; c += 2u) {
      const uint32_t pair_idx = (c % head_dim);
      const double   freq     = std::pow(theta_base, -static_cast<double>(pair_idx) / head_dim);
      const double   angle    = pos * freq;
      const double   ca       = std::cos(angle);
      const double   sa       = std::sin(angle);
      const double   x0       = fs_read_act(op->input_a->host_ptr, op->act_dtype, base + c);
      const double   x1       = fs_read_act(op->input_a->host_ptr, op->act_dtype, base + c + 1u);
      fs_write_act(op->output->host_ptr, op->act_dtype, base + c,
                   static_cast<float>(x0 * ca - x1 * sa));
      fs_write_act(op->output->host_ptr, op->act_dtype, base + c + 1u,
                   static_cast<float>(x0 * sa + x1 * ca));
    }
  }
  return FS_OK;
}

static fs_status_t fs_ref_execute_elementwise(const fs_op_desc_t* op) {
  if (!op->input_a || !op->output || !op->input_a->host_ptr || !op->output->host_ptr) {
    return FS_ERR_INVALID_ARG;
  }
  const size_t count = static_cast<size_t>(op->m_dim > 0u ? op->m_dim : 1u) *
                       static_cast<size_t>(op->n_dim > 0u ? op->n_dim : 1u);
  const void* b_ptr = (op->input_b && op->input_b->host_ptr) ? op->input_b->host_ptr : nullptr;

  for (size_t i = 0; i < count; ++i) {
    float v = fs_read_act(op->input_a->host_ptr, op->act_dtype, i);
    if (b_ptr) {
      v += fs_read_act(b_ptr, op->act_dtype, i);
    }
    v = fs_apply_activation(v, op->activation);
    fs_write_act(op->output->host_ptr, op->act_dtype, i, v);
  }
  return FS_OK;
}

static fs_status_t fs_ref_execute_any(const fs_op_desc_t* op) {
  switch (op->type) {
    case FS_OP_GEMM:
    case FS_OP_PREFILL_GEMM:
    case FS_OP_DECODE_GEMV:
    case FS_OP_VISION_CONV:
      return fs_ref_execute_gemm(op);
    case FS_OP_SWIGLU:
      return fs_ref_execute_swiglu(op);
    case FS_OP_ATTN_SOFTMAX:
      return fs_ref_execute_softmax(op);
    case FS_OP_RMS_NORM:
      return fs_ref_execute_rms_norm(op);
    case FS_OP_ROPE:
      return fs_ref_execute_rope(op);
    case FS_OP_ELEMENTWISE:
      return fs_ref_execute_elementwise(op);
    default:
      return FS_ERR_UNSUPPORTED;
  }
}

fs_status_t fs_ref_execute_any_internal(const fs_op_desc_t* op) {
  return fs_ref_execute_any(op);
}

/* =========================================================================
 * Built-in Mock TPU / Mock Vulkan GPU / Real CPU Reference Backends
 * ========================================================================= */

static int fs_builtin_tpu_supports(void* user_data, const fs_op_desc_t* op) {
  (void)user_data;
  if (!op) return 0;
  switch (op->type) {
    case FS_OP_GEMM:
    case FS_OP_PREFILL_GEMM:
    case FS_OP_DECODE_GEMV:
    case FS_OP_SWIGLU:
    case FS_OP_VISION_CONV:
      return 1;
    case FS_OP_ELEMENTWISE:
      return ((op->route_flags & FS_ROUTE_ALLOW_SPLINE) != 0u &&
              (op->activation == FS_ACT_SILU || op->activation == FS_ACT_GELU))
                 ? 1
                 : 0;
    default:
      return 0;
  }
}

static fs_status_t fs_builtin_tpu_import(void* user_data, fs_dmabuf_t* buf) {
  (void)user_data;
  if (!buf) return FS_ERR_INVALID_ARG;
  /* Models 36-bit EdgeTPU IOVA mapping via EDGETPU_MAP_DMABUF */
  if (buf->tpu_iova == 0u) {
    buf->tpu_iova = 0x100000000ULL + (buf->buf_id << 20);
  }
  return FS_OK;
}

static int fs_builtin_gpu_supports(void* user_data, const fs_op_desc_t* op) {
  (void)user_data;
  if (!op) return 0;
  switch (op->type) {
    case FS_OP_GEMM:
    case FS_OP_PREFILL_GEMM:
    case FS_OP_DECODE_GEMV:
    case FS_OP_SWIGLU:
    case FS_OP_VISION_CONV:
      return 1;
    default:
      return 0;
  }
}

static fs_status_t fs_builtin_gpu_import(void* user_data, fs_dmabuf_t* buf) {
  (void)user_data;
  if (!buf) return FS_ERR_INVALID_ARG;
  /* Models Vulkan VK_KHR_external_memory_fd device-address import */
  if (buf->vk_device_addr == 0u) {
    buf->vk_device_addr = 0x200000000ULL + (buf->buf_id << 20);
  }
  return FS_OK;
}

static int fs_builtin_cpu_supports(void* user_data, const fs_op_desc_t* op) {
  (void)user_data;
  return op != nullptr ? 1 : 0;
}

static fs_status_t fs_builtin_execute(void*               user_data,
                                      const fs_op_desc_t* op,
                                      const fs_fence_t*   wait_fence,
                                      fs_fence_t*         out_signal_fence,
                                      uint64_t*           out_hw_us) {
  const fs_dev_t dev = static_cast<fs_dev_t>(reinterpret_cast<uintptr_t>(user_data));
  if (wait_fence) {
    fs_status_t wst = fs_fence_wait(wait_fence, 5000u);
    if (wst != FS_OK) {
      return wst;
    }
  }
  const uint64_t t0 = fs_now_us();
  fs_status_t    st = fs_ref_execute_any(op);
  const uint64_t t1 = fs_now_us();
  if (st != FS_OK) {
    return st;
  }
  if (out_hw_us) {
    *out_hw_us = (t1 > t0) ? (t1 - t0) : 1u;
  }
  if (out_signal_fence) {
    *out_signal_fence = fs_fence_init(dev);
    fs_fence_signal(out_signal_fence, -1);
  }
  return FS_OK;
}

fs_status_t fs_install_builtin_backends(fs_context_t* ctx) {
  if (!ctx) {
    return FS_ERR_INVALID_ARG;
  }

  const struct {
    fs_dev_t    dev;
    const char* name;
    int (*supports)(void*, const fs_op_desc_t*);
    fs_status_t (*import_fn)(void*, fs_dmabuf_t*);
  } specs[] = {
      {FS_DEV_TPU, "builtin_mock_tpu", fs_builtin_tpu_supports, fs_builtin_tpu_import},
      {FS_DEV_GPU_VK, "builtin_mock_gpu_vk", fs_builtin_gpu_supports, fs_builtin_gpu_import},
      {FS_DEV_CPU_NEON, "builtin_cpu_neon_ref", fs_builtin_cpu_supports, nullptr},
      {FS_DEV_CPU_SME2, "builtin_cpu_sme2_ref", fs_builtin_cpu_supports, nullptr},
  };

  for (const auto& s : specs) {
    fs_backend_slot& slot = ctx->backends[s.dev];
    std::memset(&slot, 0, sizeof(slot));
    slot.ops.struct_size   = static_cast<uint32_t>(sizeof(fs_backend_ops_t));
    slot.ops.dev           = s.dev;
    slot.ops.name          = s.name;
    slot.ops.user_data     = reinterpret_cast<void*>(static_cast<uintptr_t>(s.dev));
    slot.ops.supports_op   = s.supports;
    slot.ops.import_dmabuf = s.import_fn;
    slot.ops.execute_op    = fs_builtin_execute;
    slot.registered        = 1;
    slot.is_builtin        = 1;
  }

  return FS_OK;
}

void fs_shutdown_backends(fs_context_t* ctx) {
  if (!ctx) {
    return;
  }
  for (int d = FS_DEV_TPU; d < FS_DEV_COUNT; ++d) {
    if (ctx->backends[d].registered && ctx->backends[d].ops.shutdown) {
      ctx->backends[d].ops.shutdown(ctx->backends[d].ops.user_data);
    }
    ctx->backends[d].registered = 0;
  }
}

int fs_backend_supports_op_internal(const fs_context_t* ctx,
                                    fs_dev_t            dev,
                                    const fs_op_desc_t* op) {
  if (!ctx || !fs_dev_is_valid(dev) || !op) {
    return 0;
  }
  const fs_backend_slot& slot = ctx->backends[dev];
  if (!slot.registered || !slot.ops.execute_op) {
    return 0;
  }
  if (!slot.ops.supports_op) {
    return 1;
  }
  return slot.ops.supports_op(slot.ops.user_data, op) ? 1 : 0;
}

static fs_status_t fs_prepare_input_buffer(fs_context_t* ctx,
                                           fs_dmabuf_t*  buf,
                                           fs_dev_t      target_dev,
                                           fs_fence_t*   io_fence) {
  if (!buf) {
    return FS_OK;
  }
  if ((buf->flags & FS_BUF_IMMUTABLE_WEIGHT) != 0u) {
    return fs_dmabuf_ensure_mapped(ctx, buf, target_dev);
  }
  if (buf->owner_dev != target_dev) {
    return fs_buffer_handoff(ctx, buf, buf->owner_dev, target_dev, io_fence);
  }
  return fs_dmabuf_ensure_mapped(ctx, buf, target_dev);
}

static fs_status_t fs_execute_single_step(fs_context_t*       ctx,
                                          const fs_op_desc_t* op,
                                          fs_dev_t            target_dev) {
  if (!ctx || !op || !fs_dev_is_valid(target_dev)) {
    return FS_ERR_INVALID_ARG;
  }
  if (!ctx->topo.devices[target_dev].available || !ctx->backends[target_dev].registered ||
      !ctx->backends[target_dev].ops.execute_op) {
    fs_set_error(ctx, "backend %s is offline or unregistered", fs_dev_name(target_dev));
    return FS_ERR_NO_BACKEND;
  }

  const uint64_t wall_t0 = fs_now_us();

  fs_fence_t wait_fence = fs_fence_init(FS_DEV_NONE);
  wait_fence.state      = FS_FENCE_SIGNALED;
  int has_wait_fence    = 0;

  /* Transfer mutable inputs via zero-copy DMA-BUF handoff if owned by another accelerator */
  fs_dmabuf_t* inputs[4] = {op->input_a, op->input_b, op->input_c, op->input_d};
  for (fs_dmabuf_t* in_buf : inputs) {
    if (!in_buf) continue;
    if ((in_buf->flags & FS_BUF_IMMUTABLE_WEIGHT) == 0u && in_buf->owner_dev != target_dev) {
      fs_fence_t ho_fence = fs_fence_init(in_buf->owner_dev);
      fs_status_t st      = fs_prepare_input_buffer(ctx, in_buf, target_dev, &ho_fence);
      if (st != FS_OK) {
        return st;
      }
      wait_fence     = ho_fence;
      has_wait_fence = 1;
    } else {
      fs_status_t st = fs_dmabuf_ensure_mapped(ctx, in_buf, target_dev);
      if (st != FS_OK) {
        return st;
      }
    }
  }

  /* Ensure output buffer is owned by and exclusively locked to target_dev */
  if (op->output) {
    if (op->output->owner_dev != target_dev) {
      fs_fence_t  out_fence = fs_fence_init(op->output->owner_dev);
      fs_status_t st =
          fs_buffer_handoff(ctx, op->output, op->output->owner_dev, target_dev, &out_fence);
      if (st != FS_OK) {
        return st;
      }
    } else {
      fs_status_t st = fs_dmabuf_ensure_mapped(ctx, op->output, target_dev);
      if (st != FS_OK) {
        return st;
      }
    }

    fs_status_t acq = fs_buffer_acquire_access(ctx, op->output, target_dev, op->output->generation);
    if (acq != FS_OK) {
      return acq;
    }
  }

  fs_fence_t  sig_fence = fs_fence_init(target_dev);
  uint64_t    hw_us     = 0u;
  fs_status_t exec_st   = ctx->backends[target_dev].ops.execute_op(
      ctx->backends[target_dev].ops.user_data, op, has_wait_fence ? &wait_fence : nullptr,
      &sig_fence, &hw_us);

  if (op->output) {
    fs_buffer_release_access(ctx, op->output, target_dev);
  }

  if (exec_st != FS_OK) {
    fs_set_error(ctx, "backend %s execute_op failed (%s)", fs_dev_name(target_dev),
                 fs_status_string(exec_st));
    return exec_st;
  }

  const uint64_t wall_t1 = fs_now_us();
  const uint64_t wall_us = (wall_t1 > wall_t0) ? (wall_t1 - wall_t0) : 1u;
  const uint64_t dev_us  = (hw_us > 0u) ? hw_us : wall_us;

  {
    std::lock_guard<std::mutex> lock(ctx->mu);
    ctx->telemetry.total_wall_us += wall_us;
    const float watts = ctx->topo.devices[target_dev].typical_watts;
    ctx->telemetry.total_energy_mj += (static_cast<float>(dev_us) * 1e-6f) * watts * 1000.0f;

    if (target_dev == FS_DEV_TPU) {
      ctx->telemetry.tpu_us += dev_us;
      ctx->telemetry.tpu_ops++;
    } else if (target_dev == FS_DEV_GPU_VK) {
      ctx->telemetry.gpu_us += dev_us;
      ctx->telemetry.gpu_ops++;
    } else {
      ctx->telemetry.cpu_us += dev_us;
      ctx->telemetry.cpu_ops++;
    }

    const uint64_t active_us =
        ctx->telemetry.tpu_us + ctx->telemetry.gpu_us + ctx->telemetry.cpu_us;
    if (active_us > 0u) {
      /* Average power (W) = total_energy_mj / active_ms = total_energy_mj / (active_us * 1e-3) */
      ctx->telemetry.est_watts =
          ctx->telemetry.total_energy_mj / (static_cast<float>(active_us) * 1e-3f);
    }
  }

  return FS_OK;
}

extern "C" {

fs_config_t fs_config_default(void) {
  fs_config_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size           = static_cast<uint32_t>(sizeof(fs_config_t));
  cfg.mock_mode             = 0;
  cfg.profile               = FS_PROFILE_AUTO;
  cfg.dma_heap_path         = nullptr;
  cfg.prefill_m_threshold   = 0u;
  cfg.continuity_bonus_us   = 45.0f;
  cfg.buffer_pool_max_bytes = 128u * 1024u * 1024u;
  cfg.enable_tpu            = -1;
  cfg.enable_gpu            = -1;
  cfg.enable_sme2           = -1;
  return cfg;
}

fs_status_t fs_context_create(const fs_config_t* cfg, fs_context_t** out_ctx) {
  if (!out_ctx) {
    return FS_ERR_INVALID_ARG;
  }

  fs_config_t eff = cfg ? *cfg : fs_config_default();
  if (eff.buffer_pool_max_bytes == 0u) {
    eff.buffer_pool_max_bytes = 128u * 1024u * 1024u;
  }
  if (eff.continuity_bonus_us <= 0.0f) {
    eff.continuity_bonus_us = 45.0f;
  }

  auto* ctx = new (std::nothrow) fs_context_t();
  if (!ctx) {
    return FS_ERR_NO_MEMORY;
  }

  ctx->config         = eff;
  ctx->next_buf_id    = 0u;
  ctx->next_fence_seq = 0u;
  ctx->pool_max_bytes = eff.buffer_pool_max_bytes;
  ctx->pooled_bytes   = 0u;
  ctx->pool_hits      = 0u;
  ctx->pool_misses    = 0u;
  ctx->last_error[0]  = '\0';
  std::memset(&ctx->telemetry, 0, sizeof(ctx->telemetry));
  std::memset(&ctx->backends, 0, sizeof(ctx->backends));

  fs_hw_profile_t hint = eff.profile;
  if (eff.mock_mode && hint == FS_PROFILE_AUTO) {
    hint = FS_PROFILE_HOST_MOCK;
  }

  fs_status_t st = fs_topology_probe_mode(hint, eff.mock_mode ? 1 : -1, &ctx->topo);
  if (st != FS_OK) {
    delete ctx;
    return st;
  }

  if (eff.mock_mode) {
    ctx->mock_mode = 1;
    ctx->topo.global_caps |= FS_CAP_MOCK;
  } else {
    ctx->mock_mode = ((ctx->topo.global_caps & FS_CAP_MOCK) != 0u) ? 1 : 0;
  }

  if (eff.prefill_m_threshold > 0u) {
    ctx->topo.prefill_m_threshold = eff.prefill_m_threshold;
  }
  if (eff.enable_tpu >= 0) {
    ctx->topo.devices[FS_DEV_TPU].available = eff.enable_tpu ? 1 : 0;
  }
  if (eff.enable_gpu >= 0) {
    ctx->topo.devices[FS_DEV_GPU_VK].available = eff.enable_gpu ? 1 : 0;
  }
  if (eff.enable_sme2 >= 0) {
    ctx->topo.devices[FS_DEV_CPU_SME2].available = eff.enable_sme2 ? 1 : 0;
  }

  st = fs_install_builtin_backends(ctx);
  if (st != FS_OK) {
    delete ctx;
    return st;
  }

  *out_ctx = ctx;
  return FS_OK;
}

void fs_context_destroy(fs_context_t* ctx) {
  if (!ctx) {
    return;
  }
  fs_dmabuf_destroy_all(ctx);
  fs_shutdown_backends(ctx);
  delete ctx;
}

const fs_topology_t* fs_context_topology(const fs_context_t* ctx) {
  return ctx ? &ctx->topo : nullptr;
}

const char* fs_context_last_error(const fs_context_t* ctx) {
  return ctx ? ctx->last_error : "";
}

int fs_context_is_mock(const fs_context_t* ctx) {
  return ctx ? ctx->mock_mode : 0;
}

fs_status_t fs_context_set_device_enabled(fs_context_t* ctx, fs_dev_t dev, int enabled) {
  if (!ctx || !fs_dev_is_valid(dev)) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  ctx->topo.devices[dev].available = enabled ? 1 : 0;
  return FS_OK;
}

fs_status_t fs_context_register_backend(fs_context_t* ctx, const fs_backend_ops_t* ops) {
  if (!ctx || !ops || !fs_dev_is_valid(ops->dev) || !ops->execute_op) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_backend_slot&            slot    = ctx->backends[ops->dev];
  const uint32_t              dev_bit = (1u << static_cast<uint32_t>(ops->dev));
  if (slot.registered) {
    for (fs_dmabuf_record* rec : ctx->live_buffers) {
      if ((rec->mapped_dev_mask & dev_bit) != 0u) {
        if (slot.ops.release_dmabuf) {
          slot.ops.release_dmabuf(slot.ops.user_data, &rec->pub);
        }
        rec->mapped_dev_mask &= ~dev_bit;
      }
    }
    for (fs_dmabuf_record* rec : ctx->pooled_buffers) {
      if ((rec->mapped_dev_mask & dev_bit) != 0u) {
        if (slot.ops.release_dmabuf) {
          slot.ops.release_dmabuf(slot.ops.user_data, &rec->pub);
        }
        rec->mapped_dev_mask &= ~dev_bit;
      }
    }
    if (slot.ops.shutdown) {
      slot.ops.shutdown(slot.ops.user_data);
    }
  }
  slot.ops        = *ops;
  slot.registered = 1;
  slot.is_builtin = 0;

  if (slot.ops.init) {
    fs_status_t st = slot.ops.init(slot.ops.user_data, &ctx->topo);
    if (st != FS_OK) {
      slot.registered = 0;
      fs_set_error(ctx, "backend %s init failed (%s)", ops->name ? ops->name : "custom",
                   fs_status_string(st));
      return st;
    }
  }
  ctx->topo.devices[ops->dev].available = 1;
  return FS_OK;
}

fs_status_t fs_execute_plan(fs_context_t*       ctx,
                            const fs_plan_t*    plan,
                            const fs_op_desc_t* ops,
                            uint32_t            num_ops) {
  if (!ctx || !plan || !ops || num_ops == 0u || plan->num_steps != num_ops) {
    return FS_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0; i < num_ops; ++i) {
    const fs_plan_step_t& step = plan->steps[i];
    if (step.op_index >= num_ops) {
      return FS_ERR_INVALID_ARG;
    }
    fs_status_t st = fs_execute_single_step(ctx, &ops[step.op_index], step.assigned_dev);
    if (st != FS_OK) {
      return st;
    }
  }
  return FS_OK;
}

fs_status_t fs_execute_op(fs_context_t* ctx, const fs_op_desc_t* op, fs_plan_step_t* out_step) {
  if (!ctx || !op) {
    return FS_ERR_INVALID_ARG;
  }
  fs_plan_t*  plan = nullptr;
  fs_status_t st   = fs_plan_create(ctx, op, 1u, &plan);
  if (st != FS_OK) {
    return st;
  }
  if (out_step) {
    *out_step = plan->steps[0];
  }
  st = fs_execute_plan(ctx, plan, op, 1u);
  fs_plan_destroy(plan);
  return st;
}

fs_status_t fs_context_get_telemetry(const fs_context_t* ctx, fs_telemetry_t* out_telemetry) {
  if (!ctx || !out_telemetry) {
    return FS_ERR_INVALID_ARG;
  }
  auto*                       mctx = const_cast<fs_context_t*>(ctx);
  std::lock_guard<std::mutex> lock(mctx->mu);
  *out_telemetry = ctx->telemetry;
  return FS_OK;
}

fs_status_t fs_context_reset_telemetry(fs_context_t* ctx) {
  if (!ctx) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  std::memset(&ctx->telemetry, 0, sizeof(ctx->telemetry));
  return FS_OK;
}

fs_status_t fs_compute_parity_f32(const float* actual,
                                  const float* reference,
                                  size_t       count,
                                  fs_parity_t* out_parity) {
  if (!actual || !reference || count == 0u || !out_parity) {
    return FS_ERR_INVALID_ARG;
  }
  float  max_err  = 0.0f;
  double sum_sq   = 0.0;
  double ref_sq   = 0.0;

  for (size_t i = 0; i < count; ++i) {
    const double diff = static_cast<double>(actual[i]) - static_cast<double>(reference[i]);
    const float  aerr = static_cast<float>(std::fabs(diff));
    if (aerr > max_err) {
      max_err = aerr;
    }
    sum_sq += diff * diff;
    ref_sq += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
  }

  out_parity->max_abs_err = max_err;
  out_parity->rmse        = std::sqrt(sum_sq / static_cast<double>(count));
  out_parity->nmse        = (ref_sq > 0.0) ? (sum_sq / ref_sq) : sum_sq;
  out_parity->count       = static_cast<uint64_t>(count);
  out_parity->exact_match = (max_err == 0.0f) ? 1 : 0;
  return FS_OK;
}

uint16_t fs_f32_to_f16(float value) {
  return fs_f32_to_f16_impl(value);
}

float fs_f16_to_f32(uint16_t bits) {
  return fs_f16_to_f32_impl(bits);
}

} /* extern "C" */
