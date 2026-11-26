/*
 * 16-byte SoA expert repacker (Q4_0, IQ4_NL, IQ4_XS) and fused ARM64 NEON/sdot GEMV.
 *
 * Separates scales into a 16-byte-aligned FP32 stream and nibble payloads into
 * 4-row x 16-byte SoA tiles so ARM64 NEON / FreeTile VMEM transfers issue
 * contiguous 16-byte vector loads with zero unaligned 18B/136B block strides.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pocketcoder_runner.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

constexpr int8_t kValuesIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void* alloc_aligned_64(size_t bytes) {
  void*  ptr       = nullptr;
  size_t alloc_sz  = (bytes == 0u) ? 64u : ((bytes + 63u) & ~static_cast<size_t>(63u));
  if (posix_memalign(&ptr, 64u, alloc_sz) != 0) {
    return nullptr;
  }
  std::memset(ptr, 0, alloc_sz);
  return ptr;
}

fs_status_t init_soa_storage(uint32_t         n_dim,
                             uint32_t         k_dim,
                             pc_soa_quant_t   quant,
                             pc_soa_matrix_t* out_mat) {
  if (!out_mat || n_dim == 0u || k_dim == 0u || (k_dim % PC_GGUF_BLK_SIZE) != 0u) {
    return FS_ERR_INVALID_ARG;
  }
  if (quant == PC_SOA_IQ4 && (k_dim % PC_GGUF_BLK_SIZE) != 0u) {
    return FS_ERR_INVALID_ARG;
  }

  std::memset(out_mat, 0, sizeof(*out_mat));
  const uint32_t n_tiles  = (n_dim + PC_SOA_TILE_ROWS - 1u) / PC_SOA_TILE_ROWS;
  const uint32_t n_blocks = k_dim / PC_GGUF_BLK_SIZE;

  const size_t scales_elems =
      static_cast<size_t>(n_tiles) * static_cast<size_t>(n_blocks) * PC_SOA_TILE_ROWS;
  const size_t scales_bytes = scales_elems * sizeof(float);
  const size_t qs_bytes =
      static_cast<size_t>(n_tiles) * static_cast<size_t>(n_blocks) * PC_SOA_TILE_ROWS * 16u;

  float*   scales = static_cast<float*>(alloc_aligned_64(scales_bytes));
  uint8_t* qs     = static_cast<uint8_t*>(alloc_aligned_64(qs_bytes));
  if (!scales || !qs) {
    std::free(scales);
    std::free(qs);
    return FS_ERR_NO_MEMORY;
  }

  out_mat->n_dim        = n_dim;
  out_mat->k_dim        = k_dim;
  out_mat->n_tiles      = n_tiles;
  out_mat->n_blocks     = n_blocks;
  out_mat->quant        = quant;
  out_mat->scales       = scales;
  out_mat->qs           = qs;
  out_mat->scales_bytes = scales_bytes;
  out_mat->qs_bytes     = qs_bytes;
  return FS_OK;
}

inline uint16_t read_u16_le(const uint8_t* p) {
  uint16_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline float iq4_xs_subblock_scale(const uint8_t* sblk, uint32_t ib) {
  const float    d        = fs_f16_to_f32(read_u16_le(sblk));
  const uint16_t scales_h = read_u16_le(sblk + 2u);
  const uint8_t* scales_l = sblk + 4u;
  const uint32_t ls       = ((scales_l[ib >> 1u] >> (4u * (ib & 1u))) & 0x0fu) |
                            (((scales_h >> (2u * ib)) & 0x03u) << 4u);
  return d * static_cast<float>(static_cast<int>(ls) - 32);
}

inline void store_tile_block(pc_soa_matrix_t* mat,
                             uint32_t         row,
                             uint32_t         blk,
                             float            scale,
                             const uint8_t*   qs16) {
  const uint32_t tile     = row / PC_SOA_TILE_ROWS;
  const uint32_t sub_row  = row % PC_SOA_TILE_ROWS;
  const size_t   sc_idx   = (static_cast<size_t>(tile) * mat->n_blocks + blk) * PC_SOA_TILE_ROWS + sub_row;
  const size_t   qs_off   = sc_idx * 16u;
  mat->scales[sc_idx]     = scale;
  std::memcpy(mat->qs + qs_off, qs16, 16u);
}

fs_status_t repack_18b_blocks(const void*      src,
                              uint32_t         n_dim,
                              uint32_t         k_dim,
                              pc_soa_quant_t   quant,
                              pc_soa_matrix_t* out_mat) {
  if (!src) {
    return FS_ERR_INVALID_ARG;
  }
  fs_status_t rc = init_soa_storage(n_dim, k_dim, quant, out_mat);
  if (rc != FS_OK) {
    return rc;
  }
  const uint8_t* raw = static_cast<const uint8_t*>(src);
  for (uint32_t r = 0; r < n_dim; ++r) {
    for (uint32_t b = 0; b < out_mat->n_blocks; ++b) {
      const uint8_t* blk = raw + (static_cast<size_t>(r) * out_mat->n_blocks + b) * 18u;
      store_tile_block(out_mat, r, b, fs_f16_to_f32(read_u16_le(blk)), blk + 2u);
    }
  }
  return FS_OK;
}

void quantize_x_q8_split(const float* x,
                         uint32_t     n_blocks,
                         int8_t*      xq_split,
                         float*       xd) {
  for (uint32_t b = 0; b < n_blocks; ++b) {
    const float* xb   = x + static_cast<size_t>(b) * 32u;
    float        amax = 0.0f;
    for (uint32_t i = 0; i < 32u; ++i) {
      const float av = std::fabs(xb[i]);
      if (av > amax) amax = av;
    }
    const float d   = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
    const float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
    xd[b]           = d;

    int8_t* dst = xq_split + static_cast<size_t>(b) * 32u;
    for (uint32_t j = 0; j < 16u; ++j) {
      long q0 = std::lround(xb[j] * inv);
      long q1 = std::lround(xb[j + 16u] * inv);
      if (q0 < -127) q0 = -127;
      if (q0 > 127) q0 = 127;
      if (q1 < -127) q1 = -127;
      if (q1 > 127) q1 = 127;
      dst[j]       = static_cast<int8_t>(q0);
      dst[j + 16u] = static_cast<int8_t>(q1);
    }
  }
}

void gemv_soa_scalar_f32_impl(const pc_soa_matrix_t* mat, const float* x, float* y) {
  std::memset(y, 0, static_cast<size_t>(mat->n_dim) * sizeof(float));
  for (uint32_t t = 0; t < mat->n_tiles; ++t) {
    float acc[PC_SOA_TILE_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = 0; b < mat->n_blocks; ++b) {
      const float* xb      = x + static_cast<size_t>(b) * 32u;
      const size_t base_sc = (static_cast<size_t>(t) * mat->n_blocks + b) * PC_SOA_TILE_ROWS;
      for (uint32_t r = 0; r < PC_SOA_TILE_ROWS; ++r) {
        const float    d  = mat->scales[base_sc + r];
        const uint8_t* qs = mat->qs + (base_sc + r) * 16u;
        float          sum = 0.0f;
        if (mat->quant == PC_SOA_Q4_0) {
          for (uint32_t j = 0; j < 16u; ++j) {
            const int lo = static_cast<int>(qs[j] & 0x0fu) - 8;
            const int hi = static_cast<int>(qs[j] >> 4u) - 8;
            sum += static_cast<float>(lo) * xb[j] + static_cast<float>(hi) * xb[j + 16u];
          }
        } else {
          for (uint32_t j = 0; j < 16u; ++j) {
            const int lo = static_cast<int>(kValuesIq4Nl[qs[j] & 0x0fu]);
            const int hi = static_cast<int>(kValuesIq4Nl[qs[j] >> 4u]);
            sum += static_cast<float>(lo) * xb[j] + static_cast<float>(hi) * xb[j + 16u];
          }
        }
        acc[r] += d * sum;
      }
    }
    const uint32_t row_base = t * PC_SOA_TILE_ROWS;
    for (uint32_t r = 0; r < PC_SOA_TILE_ROWS && (row_base + r) < mat->n_dim; ++r) {
      y[row_base + r] = acc[r];
    }
  }
}

#if defined(__ARM_NEON) || defined(__aarch64__)

inline void decode_nibbles_neon(uint8x16_t q,
                                uint8x16_t mask_0f,
                                int8x16_t  bias_8,
                                int8x16_t  iq4_tbl,
                                int        is_iq4,
                                int8x16_t* out_lo,
                                int8x16_t* out_hi) {
  uint8x16_t lo = vandq_u8(q, mask_0f);
  uint8x16_t hi = vshrq_n_u8(q, 4);
  if (is_iq4) {
    *out_lo = vqtbl1q_s8(iq4_tbl, lo);
    *out_hi = vqtbl1q_s8(iq4_tbl, hi);
  } else {
    *out_lo = vsubq_s8(vreinterpretq_s8_u8(lo), bias_8);
    *out_hi = vsubq_s8(vreinterpretq_s8_u8(hi), bias_8);
  }
}

inline float32x4_t dot_s8x16_f32(int8x16_t w,
                                  float32x4_t x0,
                                  float32x4_t x1,
                                  float32x4_t x2,
                                  float32x4_t x3) {
  int16x8_t w_lo16 = vmovl_s8(vget_low_s8(w));
  int16x8_t w_hi16 = vmovl_s8(vget_high_s8(w));
  float32x4_t f0   = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_lo16)));
  float32x4_t f1   = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_lo16)));
  float32x4_t f2   = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_hi16)));
  float32x4_t f3   = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_hi16)));
  float32x4_t acc  = vmulq_f32(f0, x0);
  acc              = vfmaq_f32(acc, f1, x1);
  acc              = vfmaq_f32(acc, f2, x2);
  acc              = vfmaq_f32(acc, f3, x3);
  return acc;
}

void gemv_soa_neon_f32_impl(const pc_soa_matrix_t* mat, const float* x, float* y) {
  const uint8x16_t mask_0f = vdupq_n_u8(0x0fu);
  const int8x16_t  bias_8  = vdupq_n_s8(8);
  const int8x16_t  iq4_tbl = vld1q_s8(kValuesIq4Nl);
  const int        is_iq4  = (mat->quant == PC_SOA_IQ4) ? 1 : 0;

  for (uint32_t t = 0; t < mat->n_tiles; ++t) {
    float32x4_t row_acc = vdupq_n_f32(0.0f);
    for (uint32_t b = 0; b < mat->n_blocks; ++b) {
      const float* xb   = x + static_cast<size_t>(b) * 32u;
      float32x4_t  xl0  = vld1q_f32(xb + 0u);
      float32x4_t  xl1  = vld1q_f32(xb + 4u);
      float32x4_t  xl2  = vld1q_f32(xb + 8u);
      float32x4_t  xl3  = vld1q_f32(xb + 12u);
      float32x4_t  xh0  = vld1q_f32(xb + 16u);
      float32x4_t  xh1  = vld1q_f32(xb + 20u);
      float32x4_t  xh2  = vld1q_f32(xb + 24u);
      float32x4_t  xh3  = vld1q_f32(xb + 28u);

      const size_t   base_sc = (static_cast<size_t>(t) * mat->n_blocks + b) * PC_SOA_TILE_ROWS;
      float32x4_t    sc      = vld1q_f32(mat->scales + base_sc);
      const uint8_t* qs_base = mat->qs + base_sc * 16u;

      float32x4_t p0, p1, p2, p3;
      auto eval_row = [&](uint32_t r) -> float32x4_t {
        int8x16_t w_lo, w_hi;
        decode_nibbles_neon(vld1q_u8(qs_base + r * 16u), mask_0f, bias_8, iq4_tbl, is_iq4, &w_lo,
                            &w_hi);
        return vaddq_f32(dot_s8x16_f32(w_lo, xl0, xl1, xl2, xl3),
                         dot_s8x16_f32(w_hi, xh0, xh1, xh2, xh3));
      };

      p0 = eval_row(0u);
      p1 = eval_row(1u);
      p2 = eval_row(2u);
      p3 = eval_row(3u);

      /* Horizontal reduce 4 x float32x4_t into one float32x4_t via vpaddq_f32 */
      float32x4_t s01 = vpaddq_f32(p0, p1);
      float32x4_t s23 = vpaddq_f32(p2, p3);
      float32x4_t dots = vpaddq_f32(s01, s23);
      row_acc          = vfmaq_f32(row_acc, dots, sc);
    }

    float vals[4];
    vst1q_f32(vals, row_acc);
    const uint32_t row_base = t * PC_SOA_TILE_ROWS;
    for (uint32_t r = 0; r < PC_SOA_TILE_ROWS && (row_base + r) < mat->n_dim; ++r) {
      y[row_base + r] = vals[r];
    }
  }
}

#if !defined(__ARM_FEATURE_DOTPROD)
inline int32x4_t sdot_or_neon_i8(int8x16_t a, int8x16_t b) {
  int16x8_t prod_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
  int16x8_t prod_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
  int32x4_t sum     = vpaddlq_s16(prod_lo);
  return vpadalq_s16(sum, prod_hi);
}
#endif

void gemv_soa_neon_sdot_impl(const pc_soa_matrix_t* mat,
                             const int8_t*          xq_split,
                             const float*           xd,
                             float*                 y) {
  const uint8x16_t mask_0f = vdupq_n_u8(0x0fu);
  const int8x16_t  bias_8  = vdupq_n_s8(8);
  const int8x16_t  iq4_tbl = vld1q_s8(kValuesIq4Nl);
  const int        is_iq4  = (mat->quant == PC_SOA_IQ4) ? 1 : 0;

  for (uint32_t t = 0; t < mat->n_tiles; ++t) {
    float32x4_t row_acc = vdupq_n_f32(0.0f);
    for (uint32_t b = 0; b < mat->n_blocks; ++b) {
      const int8_t*  xb_q    = xq_split + static_cast<size_t>(b) * 32u;
      const int8x16_t x_lo   = vld1q_s8(xb_q);
      const int8x16_t x_hi   = vld1q_s8(xb_q + 16u);
      const float     x_sc   = xd[b];

      const size_t   base_sc = (static_cast<size_t>(t) * mat->n_blocks + b) * PC_SOA_TILE_ROWS;
      float32x4_t    w_sc    = vmulq_f32(vld1q_f32(mat->scales + base_sc), vdupq_n_f32(x_sc));
      const uint8_t* qs_base = mat->qs + base_sc * 16u;

      auto dot_row = [&](uint32_t r) -> int32x4_t {
        int8x16_t w_lo, w_hi;
        decode_nibbles_neon(vld1q_u8(qs_base + r * 16u), mask_0f, bias_8, iq4_tbl, is_iq4, &w_lo,
                            &w_hi);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), w_lo, x_lo);
        return vdotq_s32(acc, w_hi, x_hi);
#else
        return vaddq_s32(sdot_or_neon_i8(w_lo, x_lo), sdot_or_neon_i8(w_hi, x_hi));
#endif
      };

      int32x4_t d0  = dot_row(0u);
      int32x4_t d1  = dot_row(1u);
      int32x4_t d2  = dot_row(2u);
      int32x4_t d3  = dot_row(3u);
      int32x4_t d01 = vpaddq_s32(d0, d1);
      int32x4_t d23 = vpaddq_s32(d2, d3);
      int32x4_t idot = vpaddq_s32(d01, d23);
      row_acc        = vfmaq_f32(row_acc, vcvtq_f32_s32(idot), w_sc);
    }

    float vals[4];
    vst1q_f32(vals, row_acc);
    const uint32_t row_base = t * PC_SOA_TILE_ROWS;
    for (uint32_t r = 0; r < PC_SOA_TILE_ROWS && (row_base + r) < mat->n_dim; ++r) {
      y[row_base + r] = vals[r];
    }
  }
}

#endif /* __ARM_NEON || __aarch64__ */

#if !defined(__ARM_NEON) && !defined(__aarch64__)
void gemv_soa_scalar_sdot_impl(const pc_soa_matrix_t* mat,
                               const int8_t*          xq_split,
                               const float*           xd,
                               float*                 y) {
  std::memset(y, 0, static_cast<size_t>(mat->n_dim) * sizeof(float));
  for (uint32_t t = 0; t < mat->n_tiles; ++t) {
    float acc[PC_SOA_TILE_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = 0; b < mat->n_blocks; ++b) {
      const int8_t* xb_q    = xq_split + static_cast<size_t>(b) * 32u;
      const float   x_sc    = xd[b];
      const size_t  base_sc = (static_cast<size_t>(t) * mat->n_blocks + b) * PC_SOA_TILE_ROWS;
      for (uint32_t r = 0; r < PC_SOA_TILE_ROWS; ++r) {
        const float    scale = mat->scales[base_sc + r] * x_sc;
        const uint8_t* qs    = mat->qs + (base_sc + r) * 16u;
        int32_t        isum  = 0;
        if (mat->quant == PC_SOA_Q4_0) {
          for (uint32_t j = 0; j < 16u; ++j) {
            const int lo = static_cast<int>(qs[j] & 0x0fu) - 8;
            const int hi = static_cast<int>(qs[j] >> 4u) - 8;
            isum += lo * static_cast<int>(xb_q[j]) + hi * static_cast<int>(xb_q[j + 16u]);
          }
        } else {
          for (uint32_t j = 0; j < 16u; ++j) {
            const int lo = static_cast<int>(kValuesIq4Nl[qs[j] & 0x0fu]);
            const int hi = static_cast<int>(kValuesIq4Nl[qs[j] >> 4u]);
            isum += lo * static_cast<int>(xb_q[j]) + hi * static_cast<int>(xb_q[j + 16u]);
          }
        }
        acc[r] += scale * static_cast<float>(isum);
      }
    }
    const uint32_t row_base = t * PC_SOA_TILE_ROWS;
    for (uint32_t r = 0; r < PC_SOA_TILE_ROWS && (row_base + r) < mat->n_dim; ++r) {
      y[row_base + r] = acc[r];
    }
  }
}
#endif /* !__ARM_NEON && !__aarch64__ */

} /* namespace */

extern "C" {

fs_status_t pc_soa_repack_q4_0(const void*      src_q4_0,
                               uint32_t         n_dim,
                               uint32_t         k_dim,
                               pc_soa_matrix_t* out_mat) {
  return repack_18b_blocks(src_q4_0, n_dim, k_dim, PC_SOA_Q4_0, out_mat);
}

fs_status_t pc_soa_repack_iq4_nl(const void*      src_iq4_nl,
                                 uint32_t         n_dim,
                                 uint32_t         k_dim,
                                 pc_soa_matrix_t* out_mat) {
  return repack_18b_blocks(src_iq4_nl, n_dim, k_dim, PC_SOA_IQ4, out_mat);
}

fs_status_t pc_soa_repack_iq4_xs(const void*      src_iq4_xs,
                                 uint32_t         n_dim,
                                 uint32_t         k_dim,
                                 pc_soa_matrix_t* out_mat) {
  if (!src_iq4_xs || (k_dim % PC_GGUF_SUPERBLK_SIZE) != 0u) {
    return FS_ERR_INVALID_ARG;
  }
  fs_status_t rc = init_soa_storage(n_dim, k_dim, PC_SOA_IQ4, out_mat);
  if (rc != FS_OK) {
    return rc;
  }
  const uint32_t n_super = k_dim / PC_GGUF_SUPERBLK_SIZE;
  const uint8_t* raw     = static_cast<const uint8_t*>(src_iq4_xs);
  for (uint32_t r = 0; r < n_dim; ++r) {
    for (uint32_t sb = 0; sb < n_super; ++sb) {
      const uint8_t* sblk   = raw + (static_cast<size_t>(r) * n_super + sb) * 136u;
      const uint8_t* qs_all = sblk + 8u;
      for (uint32_t ib = 0; ib < 8u; ++ib) {
        store_tile_block(out_mat, r, sb * 8u + ib, iq4_xs_subblock_scale(sblk, ib),
                         qs_all + ib * 16u);
      }
    }
  }
  return FS_OK;
}

fs_status_t pc_soa_repack_expert_from_gguf(const pc_gguf_file_t* file,
                                           const char*           tensor_name,
                                           uint32_t              expert_idx,
                                           pc_soa_matrix_t*      out_mat) {
  if (!file || !tensor_name || !out_mat) {
    return FS_ERR_INVALID_ARG;
  }
  const pc_gguf_tensor_t* td = pc_gguf_find_tensor(file, tensor_name);
  if (!td) {
    return FS_ERR_NOT_FOUND;
  }
  if (td->n_dims != 3u || expert_idx >= td->ne[2]) {
    return FS_ERR_INVALID_ARG;
  }
  const uint32_t k_dim = static_cast<uint32_t>(td->ne[0]);
  const uint32_t n_dim = static_cast<uint32_t>(td->ne[1]);
  const uint64_t exp_bytes = pc_ggml_tensor_nbytes(td->ggml_type, static_cast<uint64_t>(k_dim) * n_dim);
  if (exp_bytes == 0u || (static_cast<uint64_t>(expert_idx) + 1u) * exp_bytes > td->byte_size) {
    return FS_ERR_OVERFLOW;
  }
  const uint8_t* base = static_cast<const uint8_t*>(pc_gguf_tensor_data(file, td));
  if (!base) {
    return FS_ERR_IO;
  }
  const void* exp_ptr = base + static_cast<size_t>(expert_idx) * static_cast<size_t>(exp_bytes);
  switch (td->ggml_type) {
    case PC_GGML_TYPE_Q4_0:
      return pc_soa_repack_q4_0(exp_ptr, n_dim, k_dim, out_mat);
    case PC_GGML_TYPE_IQ4_NL:
      return pc_soa_repack_iq4_nl(exp_ptr, n_dim, k_dim, out_mat);
    case PC_GGML_TYPE_IQ4_XS:
      return pc_soa_repack_iq4_xs(exp_ptr, n_dim, k_dim, out_mat);
    default:
      return FS_ERR_UNSUPPORTED;
  }
}

void pc_soa_free(pc_soa_matrix_t* mat) {
  if (!mat) {
    return;
  }
  std::free(mat->scales);
  std::free(mat->qs);
  std::memset(mat, 0, sizeof(*mat));
}

fs_status_t pc_gemv_ref_raw_f32(const void*  src_blocks,
                                uint32_t     ggml_type,
                                const float* x,
                                float*       y,
                                uint32_t     n_dim,
                                uint32_t     k_dim) {
  if (!src_blocks || !x || !y || n_dim == 0u || k_dim == 0u || (k_dim % PC_GGUF_BLK_SIZE) != 0u) {
    return FS_ERR_INVALID_ARG;
  }
  const uint8_t* raw = static_cast<const uint8_t*>(src_blocks);

  if (ggml_type == PC_GGML_TYPE_Q4_0 || ggml_type == PC_GGML_TYPE_IQ4_NL) {
    const uint32_t nb     = k_dim / PC_GGUF_BLK_SIZE;
    const int      is_iq4 = (ggml_type == PC_GGML_TYPE_IQ4_NL);
    for (uint32_t r = 0; r < n_dim; ++r) {
      double acc = 0.0;
      for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = raw + (static_cast<size_t>(r) * nb + b) * 18u;
        const float    d   = fs_f16_to_f32(read_u16_le(blk));
        const uint8_t* qs  = blk + 2u;
        const float*   xb  = x + static_cast<size_t>(b) * 32u;
        double         bs  = 0.0;
        for (uint32_t j = 0; j < 16u; ++j) {
          const int lo = is_iq4 ? static_cast<int>(kValuesIq4Nl[qs[j] & 0x0fu])
                                : (static_cast<int>(qs[j] & 0x0fu) - 8);
          const int hi = is_iq4 ? static_cast<int>(kValuesIq4Nl[qs[j] >> 4u])
                                : (static_cast<int>(qs[j] >> 4u) - 8);
          bs += static_cast<double>(lo) * static_cast<double>(xb[j]) +
                static_cast<double>(hi) * static_cast<double>(xb[j + 16u]);
        }
        acc += static_cast<double>(d) * bs;
      }
      y[r] = static_cast<float>(acc);
    }
    return FS_OK;
  }

  if (ggml_type == PC_GGML_TYPE_IQ4_XS) {
    if ((k_dim % PC_GGUF_SUPERBLK_SIZE) != 0u) {
      return FS_ERR_INVALID_ARG;
    }
    const uint32_t n_super = k_dim / PC_GGUF_SUPERBLK_SIZE;
    for (uint32_t r = 0; r < n_dim; ++r) {
      double acc = 0.0;
      for (uint32_t sb = 0; sb < n_super; ++sb) {
        const uint8_t* sblk   = raw + (static_cast<size_t>(r) * n_super + sb) * 136u;
        const uint8_t* qs_all = sblk + 8u;
        for (uint32_t ib = 0; ib < 8u; ++ib) {
          const float    dl = iq4_xs_subblock_scale(sblk, ib);
          const uint8_t* qs = qs_all + ib * 16u;
          const float*   xb = x + static_cast<size_t>(sb * 8u + ib) * 32u;
          double         bs = 0.0;
          for (uint32_t j = 0; j < 16u; ++j) {
            const int lo = static_cast<int>(kValuesIq4Nl[qs[j] & 0x0fu]);
            const int hi = static_cast<int>(kValuesIq4Nl[qs[j] >> 4u]);
            bs += static_cast<double>(lo) * static_cast<double>(xb[j]) +
                  static_cast<double>(hi) * static_cast<double>(xb[j + 16u]);
          }
          acc += static_cast<double>(dl) * bs;
        }
      }
      y[r] = static_cast<float>(acc);
    }
    return FS_OK;
  }

  return FS_ERR_UNSUPPORTED;
}

fs_status_t pc_gemv_soa_scalar_f32(const pc_soa_matrix_t* mat, const float* x, float* y) {
  if (!mat || !mat->scales || !mat->qs || !x || !y) {
    return FS_ERR_INVALID_ARG;
  }
  gemv_soa_scalar_f32_impl(mat, x, y);
  return FS_OK;
}

fs_status_t pc_gemv_soa_f32(const pc_soa_matrix_t* mat, const float* x, float* y) {
  if (!mat || !mat->scales || !mat->qs || !x || !y) {
    return FS_ERR_INVALID_ARG;
  }
#if defined(__ARM_NEON) || defined(__aarch64__)
  gemv_soa_neon_f32_impl(mat, x, y);
#else
  gemv_soa_scalar_f32_impl(mat, x, y);
#endif
  return FS_OK;
}

fs_status_t pc_gemv_soa_sdot(const pc_soa_matrix_t* mat, const float* x, float* y) {
  if (!mat || !mat->scales || !mat->qs || !x || !y) {
    return FS_ERR_INVALID_ARG;
  }
  std::vector<int8_t> xq_split(static_cast<size_t>(mat->n_blocks) * 32u);
  std::vector<float>  xd(mat->n_blocks);
  quantize_x_q8_split(x, mat->n_blocks, xq_split.data(), xd.data());
#if defined(__ARM_NEON) || defined(__aarch64__)
  gemv_soa_neon_sdot_impl(mat, xq_split.data(), xd.data(), y);
#else
  gemv_soa_scalar_sdot_impl(mat, xq_split.data(), xd.data(), y);
#endif
  return FS_OK;
}

const char* pc_soa_kernel_backend_name(void) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  return "arm64_neon_sdot";
#elif defined(__ARM_NEON) || defined(__aarch64__)
  return "arm64_neon_v8";
#else
  return "scalar_portable";
#endif
}

} /* extern "C" */
