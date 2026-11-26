/*
 * Direct mmap-backed GGUF v2/v3 reader for PocketCoder-13B-A3B (inkling2_loop).
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pocketcoder_runner.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kGgufMagic   = 0x46554747u; /* "GGUF" LE */
constexpr uint32_t kGgufTypeU8  = 0u;
constexpr uint32_t kGgufTypeI8  = 1u;
constexpr uint32_t kGgufTypeU16 = 2u;
constexpr uint32_t kGgufTypeI16 = 3u;
constexpr uint32_t kGgufTypeU32 = 4u;
constexpr uint32_t kGgufTypeI32 = 5u;
constexpr uint32_t kGgufTypeF32 = 6u;
constexpr uint32_t kGgufTypeBool = 7u;
constexpr uint32_t kGgufTypeStr = 8u;
constexpr uint32_t kGgufTypeArr = 9u;
constexpr uint32_t kGgufTypeU64 = 10u;
constexpr uint32_t kGgufTypeI64 = 11u;
constexpr uint32_t kGgufTypeF64 = 12u;

struct Cursor {
  const uint8_t* base;
  size_t         size;
  size_t         pos;

  bool can_read(size_t n) const {
    return pos <= size && n <= (size - pos);
  }

  bool read_bytes(void* dst, size_t n) {
    if (!can_read(n)) {
      return false;
    }
    if (n > 0u) {
      std::memcpy(dst, base + pos, n);
      pos += n;
    }
    return true;
  }

  template <typename T>
  bool read_le(T* out) {
    if (!can_read(sizeof(T))) {
      return false;
    }
    std::memcpy(out, base + pos, sizeof(T));
    pos += sizeof(T);
    return true;
  }

  bool skip(size_t n) {
    if (!can_read(n)) {
      return false;
    }
    pos += n;
    return true;
  }

  bool read_str(char* out, size_t cap, uint64_t* out_len) {
    uint64_t len = 0u;
    if (!read_le(&len)) {
      return false;
    }
    if (len > size - pos) {
      return false;
    }
    if (out_len) {
      *out_len = len;
    }
    if (out && cap > 0u) {
      const size_t copy_n = (len < cap - 1u) ? static_cast<size_t>(len) : (cap - 1u);
      if (copy_n > 0u) {
        std::memcpy(out, base + pos, copy_n);
      }
      out[copy_n] = '\0';
    }
    pos += static_cast<size_t>(len);
    return true;
  }
};

bool gguf_scalar_size(uint32_t vtype, size_t* out_sz) {
  switch (vtype) {
    case kGgufTypeU8:
    case kGgufTypeI8:
    case kGgufTypeBool:
      *out_sz = 1u;
      return true;
    case kGgufTypeU16:
    case kGgufTypeI16:
      *out_sz = 2u;
      return true;
    case kGgufTypeU32:
    case kGgufTypeI32:
    case kGgufTypeF32:
      *out_sz = 4u;
      return true;
    case kGgufTypeU64:
    case kGgufTypeI64:
    case kGgufTypeF64:
      *out_sz = 8u;
      return true;
    default:
      return false;
  }
}

bool read_u32_like(Cursor* c, uint32_t vtype, uint32_t* out) {
  switch (vtype) {
    case kGgufTypeU8: {
      uint8_t v = 0;
      if (!c->read_le(&v)) return false;
      *out = v;
      return true;
    }
    case kGgufTypeI8: {
      int8_t v = 0;
      if (!c->read_le(&v) || v < 0) return false;
      *out = static_cast<uint32_t>(v);
      return true;
    }
    case kGgufTypeU16: {
      uint16_t v = 0;
      if (!c->read_le(&v)) return false;
      *out = v;
      return true;
    }
    case kGgufTypeI16: {
      int16_t v = 0;
      if (!c->read_le(&v) || v < 0) return false;
      *out = static_cast<uint32_t>(v);
      return true;
    }
    case kGgufTypeU32:
      return c->read_le(out);
    case kGgufTypeI32: {
      int32_t v = 0;
      if (!c->read_le(&v) || v < 0) return false;
      *out = static_cast<uint32_t>(v);
      return true;
    }
    case kGgufTypeU64: {
      uint64_t v = 0;
      if (!c->read_le(&v) || v > 0xffffffffu) return false;
      *out = static_cast<uint32_t>(v);
      return true;
    }
    case kGgufTypeI64: {
      int64_t v = 0;
      if (!c->read_le(&v) || v < 0 || v > 0xffffffffLL) return false;
      *out = static_cast<uint32_t>(v);
      return true;
    }
    case kGgufTypeBool: {
      uint8_t v = 0;
      if (!c->read_le(&v)) return false;
      *out = v ? 1u : 0u;
      return true;
    }
    default:
      return false;
  }
}

bool read_f32_like(Cursor* c, uint32_t vtype, float* out) {
  if (vtype == kGgufTypeF32) {
    return c->read_le(out);
  }
  if (vtype == kGgufTypeF64) {
    double v = 0.0;
    if (!c->read_le(&v)) return false;
    *out = static_cast<float>(v);
    return true;
  }
  return false;
}

bool skip_gguf_value(Cursor* c, uint32_t vtype, uint32_t depth) {
  if (depth > 8u) {
    return false;
  }
  size_t scalar_sz = 0u;
  if (gguf_scalar_size(vtype, &scalar_sz)) {
    return c->skip(scalar_sz);
  }
  if (vtype == kGgufTypeStr) {
    return c->read_str(nullptr, 0u, nullptr);
  }
  if (vtype == kGgufTypeArr) {
    uint32_t elem_type = 0u;
    uint64_t count     = 0u;
    if (!c->read_le(&elem_type) || !c->read_le(&count)) {
      return false;
    }
    size_t elem_sz = 0u;
    if (gguf_scalar_size(elem_type, &elem_sz)) {
      if (count > (c->size - c->pos) / elem_sz) {
        return false;
      }
      return c->skip(static_cast<size_t>(count) * elem_sz);
    }
    for (uint64_t i = 0; i < count; ++i) {
      if (!skip_gguf_value(c, elem_type, depth + 1u)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool suffix_eq(const char* key, const char* suffix) {
  const size_t kl = std::strlen(key);
  const size_t sl = std::strlen(suffix);
  if (kl < sl) {
    return false;
  }
  return std::strcmp(key + (kl - sl), suffix) == 0;
}

bool handle_kv(Cursor* c, const char* key, uint32_t vtype, pc_inkling2_config_t* cfg) {
  if (std::strcmp(key, "general.architecture") == 0 && vtype == kGgufTypeStr) {
    return c->read_str(cfg->arch, sizeof(cfg->arch), nullptr);
  }
  if (std::strcmp(key, "general.name") == 0 && vtype == kGgufTypeStr) {
    return c->read_str(cfg->model_name, sizeof(cfg->model_name), nullptr);
  }
  if (std::strcmp(key, "general.file_type") == 0) {
    return read_u32_like(c, vtype, &cfg->file_type);
  }
  if (std::strcmp(key, "general.alignment") == 0) {
    return read_u32_like(c, vtype, &cfg->alignment);
  }

  /* Array metadata */
  if (vtype == kGgufTypeArr) {
    if (suffix_eq(key, ".attention.head_count") || suffix_eq(key, ".attention.head_count_kv")) {
      uint32_t et = 0;
      uint64_t n  = 0;
      if (!c->read_le(&et) || !c->read_le(&n)) return false;
      uint32_t first = 0u;
      for (uint64_t i = 0; i < n; ++i) {
        uint32_t v = 0u;
        if (!read_u32_like(c, et, &v)) return false;
        if (i == 0u) first = v;
      }
      if (suffix_eq(key, ".attention.head_count")) {
        cfg->head_count = first;
      } else {
        cfg->head_count_kv = first;
      }
      return true;
    }
    if (suffix_eq(key, ".attention.sliding_window_pattern")) {
      uint32_t et = 0;
      uint64_t n  = 0;
      if (!c->read_le(&et) || !c->read_le(&n)) return false;
      for (uint64_t i = 0; i < n; ++i) {
        uint32_t v = 0u;
        if (!read_u32_like(c, et, &v)) return false;
        if (i < PC_MAX_PHYS_LAYERS) {
          cfg->sliding_window_pattern[i] = v ? 1u : 0u;
        }
      }
      return true;
    }
    return skip_gguf_value(c, vtype, 0u);
  }

  /* Table-driven scalar architecture metadata */
  struct U32Field {
    const char* suffix;
    uint32_t*   dst;
  };
  const U32Field u32_fields[] = {
      {".block_count", &cfg->block_count},
      {".context_length", &cfg->context_length},
      {".embedding_length", &cfg->embedding_length},
      {".feed_forward_length", &cfg->feed_forward_length},
      {".expert_feed_forward_length", &cfg->expert_feed_forward_length},
      {".attention.key_length", &cfg->key_length},
      {".attention.value_length", &cfg->value_length},
      {".attention.head_count", &cfg->head_count},
      {".attention.head_count_kv", &cfg->head_count_kv},
      {".expert_count", &cfg->expert_count},
      {".expert_used_count", &cfg->expert_used_count},
      {".attention.sliding_window", &cfg->sliding_window},
      {".loop.start", &cfg->loop_start},
      {".loop.end", &cfg->loop_end},
      {".loop.count", &cfg->loop_count},
      {".loop.gate_rank", &cfg->loop_gate_rank},
      {".loop.varying", &cfg->loop_varying},
      {".shortconv.l_cache", &cfg->shortconv_l_cache},
      {".vocab_size", &cfg->vocab_size},
      {".vocab_size_unpadded", &cfg->vocab_size_unpadded},
  };
  for (const auto& f : u32_fields) {
    if (suffix_eq(key, f.suffix)) {
      return read_u32_like(c, vtype, f.dst);
    }
  }
  if (std::strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
    return read_u32_like(c, vtype, &cfg->eos_token_id);
  }

  struct F32Field {
    const char* suffix;
    float*      dst;
  };
  const F32Field f32_fields[] = {
      {".attention.layer_norm_rms_epsilon", &cfg->rms_norm_eps},
      {".expert_weights_scale", &cfg->expert_weights_scale},
      {".activation.situ_beta", &cfg->situ_beta},
      {".activation.situ_linear_beta", &cfg->situ_linear_beta},
      {".final_logit_softcapping", &cfg->final_logit_softcapping},
  };
  for (const auto& f : f32_fields) {
    if (suffix_eq(key, f.suffix)) {
      return read_f32_like(c, vtype, f.dst);
    }
  }

  return skip_gguf_value(c, vtype, 0u);
}

bool build_loop_schedule(pc_inkling2_config_t* cfg) {
  if (cfg->block_count == 0u || cfg->block_count > PC_MAX_PHYS_LAYERS) {
    return false;
  }
  const uint32_t lstart = cfg->loop_start;
  const uint32_t lend   = cfg->loop_end;
  const uint32_t lcnt   = (cfg->loop_count > 0u) ? cfg->loop_count : 1u;
  if (lstart > lend || lend > cfg->block_count) {
    return false;
  }

  uint32_t v = 0u;
  auto push_layer = [&](uint32_t phys, uint32_t pass) -> bool {
    if (v >= PC_MAX_LOGICAL_VISITS) {
      return false;
    }
    const uint32_t is_swa = (cfg->sliding_window > 0u && cfg->sliding_window_pattern[phys] != 0u) ? 1u : 0u;
    pc_visit_desc_t& vd   = cfg->visits[v];
    vd.visit_idx          = v;
    vd.phys_layer         = phys;
    vd.loop_pass          = pass;
    vd.is_sliding_window  = is_swa;
    vd.window_size        = is_swa ? cfg->sliding_window : cfg->context_length;
    if (is_swa) {
      cfg->sliding_window_visits++;
    } else {
      cfg->full_context_visits++;
    }
    v++;
    return true;
  };

  for (uint32_t i = 0; i < lstart; ++i) {
    if (!push_layer(i, 0u)) return false;
  }
  for (uint32_t pass = 0; pass < lcnt; ++pass) {
    for (uint32_t i = lstart; i < lend; ++i) {
      if (!push_layer(i, pass)) return false;
    }
  }
  for (uint32_t i = lend; i < cfg->block_count; ++i) {
    if (!push_layer(i, 0u)) return false;
  }

  cfg->logical_visits = v;
  return true;
}

} /* namespace */

struct pc_gguf_file {
  int                           fd;
  const uint8_t*                map_base;
  size_t                        file_size;
  uint32_t                      version;
  uint64_t                      data_offset;
  pc_inkling2_config_t          cfg;
  std::vector<pc_gguf_tensor_t> tensors;
};

extern "C" {

size_t pc_ggml_type_block_elems(uint32_t ggml_type) {
  switch (ggml_type) {
    case PC_GGML_TYPE_F32:    return 1u;
    case PC_GGML_TYPE_F16:    return 1u;
    case PC_GGML_TYPE_Q4_0:   return 32u;
    case PC_GGML_TYPE_Q8_0:   return 32u;
    case PC_GGML_TYPE_Q6_K:   return 256u;
    case PC_GGML_TYPE_IQ4_NL: return 32u;
    case PC_GGML_TYPE_IQ4_XS: return 256u;
    default:                  return 0u;
  }
}

size_t pc_ggml_type_block_bytes(uint32_t ggml_type) {
  switch (ggml_type) {
    case PC_GGML_TYPE_F32:    return 4u;
    case PC_GGML_TYPE_F16:    return 2u;
    case PC_GGML_TYPE_Q4_0:   return 18u;
    case PC_GGML_TYPE_Q8_0:   return 34u;
    case PC_GGML_TYPE_Q6_K:   return 210u;
    case PC_GGML_TYPE_IQ4_NL: return 18u;
    case PC_GGML_TYPE_IQ4_XS: return 136u;
    default:                  return 0u;
  }
}

uint64_t pc_ggml_tensor_nbytes(uint32_t ggml_type, uint64_t n_elems) {
  const size_t be = pc_ggml_type_block_elems(ggml_type);
  const size_t bb = pc_ggml_type_block_bytes(ggml_type);
  if (be == 0u || bb == 0u || (n_elems % be) != 0u) {
    return 0u;
  }
  return (n_elems / be) * bb;
}

const char* pc_ggml_type_name(uint32_t ggml_type) {
  switch (ggml_type) {
    case PC_GGML_TYPE_F32:    return "F32";
    case PC_GGML_TYPE_F16:    return "F16";
    case PC_GGML_TYPE_Q4_0:   return "Q4_0";
    case PC_GGML_TYPE_Q8_0:   return "Q8_0";
    case PC_GGML_TYPE_Q6_K:   return "Q6_K";
    case PC_GGML_TYPE_IQ4_NL: return "IQ4_NL";
    case PC_GGML_TYPE_IQ4_XS: return "IQ4_XS";
    default:                  return "UNKNOWN";
  }
}

fs_status_t pc_gguf_open(const char* path, pc_gguf_file_t** out_file) {
  if (!path || !out_file) {
    return FS_ERR_INVALID_ARG;
  }
  *out_file = nullptr;

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return FS_ERR_IO;
  }

  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 24) {
    close(fd);
    return FS_ERR_IO;
  }

  const size_t fsize = static_cast<size_t>(st.st_size);
  void*        mapped = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    return FS_ERR_NO_MEMORY;
  }

  auto* gf = new pc_gguf_file();
  gf->fd        = fd;
  gf->map_base  = static_cast<const uint8_t*>(mapped);
  gf->file_size = fsize;
  std::memset(&gf->cfg, 0, sizeof(gf->cfg));
  gf->cfg.alignment = 32u;

  Cursor c{gf->map_base, fsize, 0u};
  uint32_t magic = 0u;
  uint64_t n_tensors = 0u;
  uint64_t n_kv      = 0u;
  if (!c.read_le(&magic) || magic != kGgufMagic ||
      !c.read_le(&gf->version) || (gf->version != 2u && gf->version != 3u) ||
      !c.read_le(&n_tensors) || !c.read_le(&n_kv)) {
    pc_gguf_close(gf);
    return FS_ERR_INVALID_ARG;
  }

  if (n_tensors > 1000000ULL || n_kv > 1000000ULL) {
    pc_gguf_close(gf);
    return FS_ERR_OVERFLOW;
  }

  for (uint64_t i = 0; i < n_kv; ++i) {
    char     key[256];
    uint64_t klen  = 0u;
    uint32_t vtype = 0u;
    if (!c.read_str(key, sizeof(key), &klen) || !c.read_le(&vtype)) {
      pc_gguf_close(gf);
      return FS_ERR_INVALID_ARG;
    }
    if (!handle_kv(&c, key, vtype, &gf->cfg)) {
      pc_gguf_close(gf);
      return FS_ERR_INVALID_ARG;
    }
  }

  if (gf->cfg.alignment == 0u || (gf->cfg.alignment & (gf->cfg.alignment - 1u)) != 0u) {
    pc_gguf_close(gf);
    return FS_ERR_INVALID_ARG;
  }

  gf->tensors.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    pc_gguf_tensor_t td;
    std::memset(&td, 0, sizeof(td));
    uint64_t tlen = 0u;
    if (!c.read_str(td.name, sizeof(td.name), &tlen) || tlen == 0u ||
        !c.read_le(&td.n_dims) || td.n_dims == 0u || td.n_dims > 4u) {
      pc_gguf_close(gf);
      return FS_ERR_INVALID_ARG;
    }
    uint64_t n_elems = 1u;
    for (uint32_t d = 0; d < td.n_dims; ++d) {
      if (!c.read_le(&td.ne[d]) || td.ne[d] == 0u) {
        pc_gguf_close(gf);
        return FS_ERR_INVALID_ARG;
      }
      if (n_elems > UINT64_MAX / td.ne[d]) {
        pc_gguf_close(gf);
        return FS_ERR_OVERFLOW;
      }
      n_elems *= td.ne[d];
    }
    for (uint32_t d = td.n_dims; d < 4u; ++d) {
      td.ne[d] = 1u;
    }
    if (!c.read_le(&td.ggml_type) || !c.read_le(&td.offset)) {
      pc_gguf_close(gf);
      return FS_ERR_INVALID_ARG;
    }
    td.byte_size = pc_ggml_tensor_nbytes(td.ggml_type, n_elems);
    if (td.byte_size == 0u) {
      pc_gguf_close(gf);
      return FS_ERR_UNSUPPORTED;
    }
    gf->tensors.push_back(td);
  }

  const size_t align    = gf->cfg.alignment;
  const size_t data_off = (c.pos + (align - 1u)) & ~(align - 1u);
  if (data_off > fsize) {
    pc_gguf_close(gf);
    return FS_ERR_OVERFLOW;
  }
  gf->data_offset = static_cast<uint64_t>(data_off);

  for (const auto& td : gf->tensors) {
    if (td.offset > static_cast<uint64_t>(fsize) - gf->data_offset ||
        td.byte_size > static_cast<uint64_t>(fsize) - gf->data_offset - td.offset) {
      pc_gguf_close(gf);
      return FS_ERR_OVERFLOW;
    }
  }

  if (gf->cfg.block_count > 0u) {
    if (!build_loop_schedule(&gf->cfg)) {
      pc_gguf_close(gf);
      return FS_ERR_INVALID_ARG;
    }
  }

  *out_file = gf;
  return FS_OK;
}

void pc_gguf_close(pc_gguf_file_t* file) {
  if (!file) {
    return;
  }
  if (file->map_base && file->map_base != MAP_FAILED) {
    munmap(const_cast<uint8_t*>(file->map_base), file->file_size);
  }
  if (file->fd >= 0) {
    close(file->fd);
  }
  delete file;
}

const pc_inkling2_config_t* pc_gguf_config(const pc_gguf_file_t* file) {
  return file ? &file->cfg : nullptr;
}

uint64_t pc_gguf_file_size(const pc_gguf_file_t* file) {
  return file ? static_cast<uint64_t>(file->file_size) : 0u;
}

uint64_t pc_gguf_data_offset(const pc_gguf_file_t* file) {
  return file ? file->data_offset : 0u;
}

uint32_t pc_gguf_tensor_count(const pc_gguf_file_t* file) {
  return file ? static_cast<uint32_t>(file->tensors.size()) : 0u;
}

const pc_gguf_tensor_t* pc_gguf_tensor_at(const pc_gguf_file_t* file, uint32_t idx) {
  if (!file || idx >= file->tensors.size()) {
    return nullptr;
  }
  return &file->tensors[idx];
}

const pc_gguf_tensor_t* pc_gguf_find_tensor(const pc_gguf_file_t* file, const char* name) {
  if (!file || !name) {
    return nullptr;
  }
  for (const auto& td : file->tensors) {
    if (std::strcmp(td.name, name) == 0) {
      return &td;
    }
  }
  return nullptr;
}

const void* pc_gguf_tensor_data(const pc_gguf_file_t* file, const pc_gguf_tensor_t* tensor) {
  if (!file || !tensor) {
    return nullptr;
  }
  if (tensor->offset > static_cast<uint64_t>(file->file_size) - file->data_offset ||
      tensor->byte_size > static_cast<uint64_t>(file->file_size) - file->data_offset - tensor->offset) {
    return nullptr;
  }
  return file->map_base + file->data_offset + tensor->offset;
}

fs_status_t pc_gguf_validate_pocketcoder(const pc_gguf_file_t* file,
                                         char*                 err_buf,
                                         size_t                err_cap) {
  auto set_err = [&](const char* msg) {
    if (err_buf && err_cap > 0u) {
      std::snprintf(err_buf, err_cap, "%s", msg);
    }
  };

  if (!file) {
    set_err("null gguf file");
    return FS_ERR_INVALID_ARG;
  }
  const pc_inkling2_config_t& c = file->cfg;
  if (std::strcmp(c.arch, "inkling2_loop") != 0) {
    set_err("general.architecture != inkling2_loop");
    return FS_ERR_UNSUPPORTED;
  }
  if (c.block_count != 24u || c.loop_start != 6u || c.loop_end != 18u || c.loop_count != 3u) {
    set_err("unexpected layer/loop geometry (expected 24 blocks, loop 6..18 x3)");
    return FS_ERR_UNSUPPORTED;
  }
  if (c.logical_visits != 48u || c.sliding_window_visits != 40u || c.full_context_visits != 8u) {
    set_err("unexpected derived visit counts (expected 48 total, 40 SWA, 8 full)");
    return FS_ERR_UNSUPPORTED;
  }
  if (c.sliding_window != 1024u) {
    set_err("expected 1024-token sliding window");
    return FS_ERR_UNSUPPORTED;
  }
  if (c.embedding_length != 2304u || c.expert_feed_forward_length != 1152u ||
      (c.embedding_length % PC_LANE_ALIGN) != 0u ||
      (c.expert_feed_forward_length % PC_LANE_ALIGN) != 0u) {
    set_err("expected 2304x1152 128-lane geometry");
    return FS_ERR_UNSUPPORTED;
  }
  if (c.expert_count != 64u || c.expert_used_count != 8u) {
    set_err("expected 64 routed experts with top-8 active");
    return FS_ERR_UNSUPPORTED;
  }
  if (!pc_gguf_find_tensor(file, "blk.0.ffn_down_exps.weight") ||
      !pc_gguf_find_tensor(file, "blk.0.ffn_gate_up_exps.weight") ||
      !pc_gguf_find_tensor(file, "blk.6.loop_ffn_ctrl_gain.weight")) {
    set_err("missing required PocketCoder expert or loop tensors");
    return FS_ERR_NOT_FOUND;
  }
  if (err_buf && err_cap > 0u) {
    err_buf[0] = '\0';
  }
  return FS_OK;
}

} /* extern "C" */
