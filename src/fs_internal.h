/*
 * freesched internal C++17 declarations.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FS_INTERNAL_H_
#define FS_INTERNAL_H_

#include "freesched.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

static_assert(sizeof(fs_block_q8_0_t) == 34u, "fs_block_q8_0_t must be 34 bytes");
static_assert(sizeof(fs_block_q4_0_t) == 18u, "fs_block_q4_0_t must be 18 bytes");

#define FS_BUF_MAGIC 0x4653444255463031ULL /* "FSDBUF01" */

struct fs_dmabuf_record {
  fs_dmabuf_t pub; /* Must remain first member for pointer interconvertibility */
  uint64_t    magic;
  fs_context_t* owner_ctx;
  size_t      alloc_size;
  int         is_imported;
  int         is_mmap;
  int         in_pool;
  fs_dev_t    active_accessor;
  uint32_t    active_access_count;
  uint32_t    mapped_dev_mask;
  fs_fence_t  last_fence;
};

struct fs_backend_slot {
  fs_backend_ops_t ops;
  int              registered;
  int              is_builtin;
};

struct fs_context {
  fs_config_t     config;
  fs_topology_t   topo;
  fs_telemetry_t  telemetry;
  char            last_error[256];
  int             mock_mode;
  uint64_t        next_buf_id;
  uint64_t        next_fence_seq;
  size_t          pool_max_bytes;
  size_t          pooled_bytes;
  uint64_t        pool_hits;
  uint64_t        pool_misses;
  std::mutex      mu;
  std::vector<fs_dmabuf_record*> live_buffers;
  std::vector<fs_dmabuf_record*> pooled_buffers;
  fs_backend_slot backends[FS_DEV_COUNT];
};

/* Utility helpers implemented in fs_topology.cpp / fs_executor.cpp */
uint64_t fs_now_us(void);
void     fs_set_error(fs_context_t* ctx, const char* fmt, ...);
int      fs_dev_is_valid(fs_dev_t dev);
int      fs_dev_is_cpu(fs_dev_t dev);
size_t   fs_align_up(size_t value, size_t alignment);

/* Internal buffer pool teardown called from fs_context_destroy */
void fs_dmabuf_destroy_all(fs_context_t* ctx);
fs_status_t fs_dmabuf_ensure_mapped(fs_context_t* ctx, fs_dmabuf_t* buf, fs_dev_t dev);

/* Internal built-in backend installation & reference math */
fs_status_t fs_install_builtin_backends(fs_context_t* ctx);
void        fs_shutdown_backends(fs_context_t* ctx);
fs_status_t fs_ref_execute_any_internal(const fs_op_desc_t* op);

/* Router & topology internal helpers */
fs_status_t fs_topology_probe_mode(fs_hw_profile_t profile_hint,
                                   int             force_mock,
                                   fs_topology_t*  out_topo);
int fs_backend_supports_op_internal(const fs_context_t* ctx,
                                    fs_dev_t            dev,
                                    const fs_op_desc_t* op);

#endif /* FS_INTERNAL_H_ */
