/*
 * freesched zero-copy DMA-BUF buffer pool & refcounted ownership state machine.
 *
 * Copyright 2026 FreeSilicon Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portable POSIX mmap backing in host mock mode (macOS/Linux); real Linux
 * /dev/dma_heap/system + DMA_BUF_IOCTL_SYNC path gated cleanly under #ifdef __linux__.
 */
#include "fs_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __linux__
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

/* Portable Linux UAPI structs (avoid requiring kernel >= 6.0 headers) */
struct fs_dma_heap_allocation_data {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};

struct fs_dma_buf_sync {
  uint64_t flags;
};

struct fs_dma_buf_export_sync_file {
  uint32_t flags;
  int32_t  fd;
};

#define FS_DMA_HEAP_IOC_MAGIC   'H'
#define FS_DMA_HEAP_IOCTL_ALLOC _IOWR(FS_DMA_HEAP_IOC_MAGIC, 0x0, struct fs_dma_heap_allocation_data)

#define FS_DMA_BUF_BASE                   'b'
#define FS_DMA_BUF_IOCTL_SYNC             _IOW(FS_DMA_BUF_BASE, 0, struct fs_dma_buf_sync)
#define FS_DMA_BUF_IOCTL_EXPORT_SYNC_FILE _IOWR(FS_DMA_BUF_BASE, 2, struct fs_dma_buf_export_sync_file)
#define FS_DMA_BUF_SYNC_READ              (1ULL << 0)
#define FS_DMA_BUF_SYNC_WRITE             (2ULL << 0)
#define FS_DMA_BUF_SYNC_RW                (FS_DMA_BUF_SYNC_READ | FS_DMA_BUF_SYNC_WRITE)
#define FS_DMA_BUF_SYNC_START             (0ULL << 2)
#define FS_DMA_BUF_SYNC_END               (1ULL << 2)
#endif /* __linux__ */

static fs_dmabuf_record* fs_record_from_pub(fs_dmabuf_t* buf) {
  if (!buf) {
    return nullptr;
  }
  auto* rec = reinterpret_cast<fs_dmabuf_record*>(buf);
  if (rec->magic != FS_BUF_MAGIC) {
    return nullptr;
  }
  return rec;
}

static size_t fs_page_size(void) {
  long pg = sysconf(_SC_PAGESIZE);
  return (pg > 0) ? static_cast<size_t>(pg) : 4096u;
}

static void fs_release_backend_mappings_locked(fs_context_t* ctx, fs_dmabuf_record* rec) {
  if (!ctx || !rec) {
    return;
  }
  for (int d = FS_DEV_TPU; d < FS_DEV_COUNT; ++d) {
    if ((rec->mapped_dev_mask & (1u << d)) != 0u) {
      if (ctx->backends[d].registered && ctx->backends[d].ops.release_dmabuf) {
        ctx->backends[d].ops.release_dmabuf(ctx->backends[d].ops.user_data, &rec->pub);
      }
      rec->mapped_dev_mask &= ~(1u << d);
    }
  }
  rec->pub.tpu_iova       = 0u;
  rec->pub.vk_device_addr = 0u;
}

static void fs_free_record_backing(fs_context_t* ctx, fs_dmabuf_record* rec) {
  if (!rec) {
    return;
  }
  fs_release_backend_mappings_locked(ctx, rec);
  if (!rec->is_imported) {
    if (rec->is_mmap && rec->pub.host_ptr && rec->pub.host_ptr != MAP_FAILED) {
      munmap(rec->pub.host_ptr, rec->alloc_size);
    }
    if (rec->pub.dmabuf_fd >= 0) {
      close(rec->pub.dmabuf_fd);
    }
  }
  rec->pub.host_ptr  = nullptr;
  rec->pub.dmabuf_fd = -1;
  rec->pub.owner_dev = FS_DEV_NONE;
  rec->magic         = 0u;
  std::free(rec);
}

#ifdef __linux__
static fs_status_t fs_linux_dma_buf_sync(int dmabuf_fd, uint64_t flags) {
  if (dmabuf_fd < 0) {
    return FS_OK;
  }
  struct fs_dma_buf_sync sync_arg;
  sync_arg.flags = flags;
  int rc         = 0;
  do {
    rc = ioctl(dmabuf_fd, FS_DMA_BUF_IOCTL_SYNC, &sync_arg);
  } while (rc != 0 && errno == EINTR);
  if (rc != 0) {
    return FS_ERR_IOCTL;
  }
  return FS_OK;
}

static fs_status_t fs_linux_export_sync_file(int dmabuf_fd, uint32_t rw_flags, int* out_sync_fd) {
  if (!out_sync_fd) {
    return FS_ERR_INVALID_ARG;
  }
  *out_sync_fd = -1;
  if (dmabuf_fd < 0) {
    return FS_OK;
  }
  struct fs_dma_buf_export_sync_file req;
  std::memset(&req, 0, sizeof(req));
  req.flags = (rw_flags != 0u) ? (rw_flags & static_cast<uint32_t>(FS_DMA_SYNC_RW))
                               : static_cast<uint32_t>(FS_DMA_SYNC_RW);
  req.fd    = -1;
  int rc    = 0;
  do {
    rc = ioctl(dmabuf_fd, FS_DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &req);
  } while (rc != 0 && errno == EINTR);
  if (rc != 0) {
    /* Older kernels (< 6.0) without DMA_BUF_IOCTL_EXPORT_SYNC_FILE return ENOTTY/EOPNOTSUPP */
    if (errno == ENOTTY || errno == EOPNOTSUPP || errno == EINVAL) {
      return FS_OK;
    }
    return FS_ERR_IOCTL;
  }
  *out_sync_fd = req.fd;
  return FS_OK;
}

static fs_status_t fs_linux_alloc_dma_heap(const char* heap_path,
                                           size_t      alloc_size,
                                           int*        out_fd,
                                           void**      out_ptr) {
  const char* path    = (heap_path && heap_path[0] != '\0') ? heap_path : "/dev/dma_heap/system";
  int         heap_fd = open(path, O_RDONLY | O_CLOEXEC);
  if (heap_fd < 0) {
    return FS_ERR_DEVICE;
  }
  struct fs_dma_heap_allocation_data req;
  std::memset(&req, 0, sizeof(req));
  req.len      = static_cast<uint64_t>(alloc_size);
  req.fd_flags = O_RDWR | O_CLOEXEC;
  int rc       = 0;
  do {
    rc = ioctl(heap_fd, FS_DMA_HEAP_IOCTL_ALLOC, &req);
  } while (rc != 0 && errno == EINTR);
  close(heap_fd);
  if (rc != 0) {
    return FS_ERR_IOCTL;
  }
  void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   static_cast<int>(req.fd), 0);
  if (ptr == MAP_FAILED) {
    close(static_cast<int>(req.fd));
    return FS_ERR_NO_MEMORY;
  }
  *out_fd  = static_cast<int>(req.fd);
  *out_ptr = ptr;
  return FS_OK;
}

static fs_status_t fs_linux_alloc_memfd(size_t alloc_size, int* out_fd, void** out_ptr) {
#ifdef SYS_memfd_create
  long fd_res = syscall(SYS_memfd_create, "freesched_dmabuf", static_cast<unsigned int>(MFD_CLOEXEC));
  if (fd_res < 0) {
    return FS_ERR_DEVICE;
  }
  int fd = static_cast<int>(fd_res);
  if (ftruncate(fd, static_cast<off_t>(alloc_size)) != 0) {
    close(fd);
    return FS_ERR_NO_MEMORY;
  }
  void* ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    return FS_ERR_NO_MEMORY;
  }
  *out_fd  = fd;
  *out_ptr = ptr;
  return FS_OK;
#else
  (void)alloc_size;
  (void)out_fd;
  (void)out_ptr;
  return FS_ERR_UNSUPPORTED;
#endif
}
#endif /* __linux__ */

static fs_status_t fs_dma_sync_record_locked(fs_context_t*     ctx,
                                             fs_dmabuf_record* rec,
                                             uint32_t          sync_flags) {
  if (!ctx || !rec) {
    return FS_ERR_INVALID_ARG;
  }
  const uint32_t valid_mask =
      static_cast<uint32_t>(FS_DMA_SYNC_RW) | static_cast<uint32_t>(FS_DMA_SYNC_END);
  if ((sync_flags & ~valid_mask) != 0u || (sync_flags & static_cast<uint32_t>(FS_DMA_SYNC_RW)) == 0u) {
    return FS_ERR_INVALID_ARG;
  }
#ifdef __linux__
  if ((rec->pub.backing == FS_BACKING_DMA_HEAP || rec->pub.backing == FS_BACKING_IMPORTED) &&
      rec->pub.dmabuf_fd >= 0) {
    uint64_t kflags = 0ULL;
    if ((sync_flags & static_cast<uint32_t>(FS_DMA_SYNC_READ)) != 0u) {
      kflags |= FS_DMA_BUF_SYNC_READ;
    }
    if ((sync_flags & static_cast<uint32_t>(FS_DMA_SYNC_WRITE)) != 0u) {
      kflags |= FS_DMA_BUF_SYNC_WRITE;
    }
    if ((sync_flags & static_cast<uint32_t>(FS_DMA_SYNC_END)) != 0u) {
      kflags |= FS_DMA_BUF_SYNC_END;
    } else {
      kflags |= FS_DMA_BUF_SYNC_START;
    }
    fs_status_t st = fs_linux_dma_buf_sync(rec->pub.dmabuf_fd, kflags);
    if (st != FS_OK) {
      return st;
    }
  }
#endif
  rec->pub.dma_sync_count++;
  ctx->telemetry.dma_sync_calls++;
  return FS_OK;
}

fs_status_t fs_dmabuf_ensure_mapped(fs_context_t* ctx, fs_dmabuf_t* buf, fs_dev_t dev) {
  if (!ctx || !buf || !fs_dev_is_valid(dev)) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_dmabuf_record*           rec = fs_record_from_pub(buf);
  if (!rec || rec->owner_ctx != ctx) {
    return FS_ERR_INVALID_ARG;
  }
  const uint32_t bit = (1u << static_cast<uint32_t>(dev));
  if ((rec->mapped_dev_mask & bit) != 0u) {
    return FS_OK;
  }
  if (ctx->backends[dev].registered && ctx->backends[dev].ops.import_dmabuf) {
    fs_status_t st = ctx->backends[dev].ops.import_dmabuf(ctx->backends[dev].ops.user_data, buf);
    if (st != FS_OK) {
      return st;
    }
  }
  rec->mapped_dev_mask |= bit;
  return FS_OK;
}

void fs_dmabuf_destroy_all(fs_context_t* ctx) {
  if (!ctx) {
    return;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  for (fs_dmabuf_record* rec : ctx->pooled_buffers) {
    fs_free_record_backing(ctx, rec);
  }
  ctx->pooled_buffers.clear();
  ctx->pooled_bytes = 0u;

  for (fs_dmabuf_record* rec : ctx->live_buffers) {
    fs_free_record_backing(ctx, rec);
  }
  ctx->live_buffers.clear();
}

static size_t fs_gcd(size_t a, size_t b) {
  while (b != 0u) {
    const size_t t = a % b;
    a              = b;
    b              = t;
  }
  return a;
}

static int fs_lcm_checked(size_t a, size_t b, size_t* out) {
  if (a == 0u) a = 1u;
  if (b == 0u) b = 1u;
  const size_t g = fs_gcd(a, b);
  const size_t q = a / g;
  if (b > (static_cast<size_t>(-1) / q)) {
    return 0;
  }
  *out = q * b;
  return 1;
}

static uint32_t fs_effective_layout_mask(const fs_buf_attr_t* attr) {
  if (attr->layout_mask != 0u) {
    return attr->layout_mask & (FS_LAYOUT_BIT(FS_LAYOUT_LINEAR) | FS_LAYOUT_BIT(FS_LAYOUT_TILED));
  }
  if (attr->layout == FS_LAYOUT_LINEAR) {
    return FS_LAYOUT_BIT(FS_LAYOUT_LINEAR);
  }
  if (attr->layout == FS_LAYOUT_TILED) {
    return FS_LAYOUT_BIT(FS_LAYOUT_TILED);
  }
  return FS_LAYOUT_BIT(FS_LAYOUT_LINEAR) | FS_LAYOUT_BIT(FS_LAYOUT_TILED);
}

static uint32_t fs_effective_coherency_mask(const fs_buf_attr_t* attr) {
  if (attr->coherency_mask != 0u) {
    return attr->coherency_mask &
           (FS_COHERENCY_BIT(FS_COHERENCY_COHERENT) | FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT));
  }
  if (attr->coherency == FS_COHERENCY_COHERENT) {
    return FS_COHERENCY_BIT(FS_COHERENCY_COHERENT);
  }
  if (attr->coherency == FS_COHERENCY_NON_COHERENT) {
    return FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT);
  }
  return FS_COHERENCY_BIT(FS_COHERENCY_COHERENT) | FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT);
}

extern "C" {

fs_buf_attr_t fs_buf_attr_default(fs_dev_t dev) {
  fs_buf_attr_t a;
  std::memset(&a, 0, sizeof(a));
  a.dev    = dev;
  a.access = FS_ACCESS_READ_WRITE;
  switch (dev) {
    case FS_DEV_TPU:
      /* Tensor G5/G6 EdgeTPU requires 128B-aligned pitch-linear non-coherent DMA-BUF */
      a.min_align      = 128u;
      a.stride_align   = 64u;
      a.layout         = FS_LAYOUT_LINEAR;
      a.coherency      = FS_COHERENCY_NON_COHERENT;
      a.layout_mask    = FS_LAYOUT_BIT(FS_LAYOUT_LINEAR);
      a.coherency_mask = FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT);
      break;
    case FS_DEV_GPU_VK:
      /* PowerVR Volcanic Vulkan supports both linear and tiled PBE, non-coherent DMA-BUF */
      a.min_align      = 64u;
      a.stride_align   = 64u;
      a.layout         = FS_LAYOUT_ANY;
      a.coherency      = FS_COHERENCY_NON_COHERENT;
      a.layout_mask    = FS_LAYOUT_BIT(FS_LAYOUT_LINEAR) | FS_LAYOUT_BIT(FS_LAYOUT_TILED);
      a.coherency_mask = FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT);
      break;
    case FS_DEV_CPU_NEON:
    case FS_DEV_CPU_SME2:
    default:
      /* ARMv9 CPU expects pitch-linear and handles coherent or explicit DMA_BUF_IOCTL_SYNC */
      a.min_align      = 64u;
      a.stride_align   = 16u;
      a.layout         = FS_LAYOUT_LINEAR;
      a.coherency      = FS_COHERENCY_ANY;
      a.layout_mask    = FS_LAYOUT_BIT(FS_LAYOUT_LINEAR);
      a.coherency_mask = FS_COHERENCY_BIT(FS_COHERENCY_COHERENT) |
                         FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT);
      break;
  }
  return a;
}

fs_status_t fs_dmabuf_reconcile(const fs_buf_attr_t* attrs,
                                uint32_t             count,
                                fs_buf_reconciled_t* out_reconciled,
                                int*                 out_zero_copy_compatible) {
  if (!attrs || count == 0u || !out_reconciled) {
    return FS_ERR_INVALID_ARG;
  }

  std::memset(out_reconciled, 0, sizeof(*out_reconciled));

  size_t   min_align       = 1u;
  size_t   stride_align    = 1u;
  uint32_t layout_inter    = FS_LAYOUT_BIT(FS_LAYOUT_LINEAR) | FS_LAYOUT_BIT(FS_LAYOUT_TILED);
  uint32_t coherency_inter = FS_COHERENCY_BIT(FS_COHERENCY_COHERENT) |
                             FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT);
  uint32_t combined_access = 0u;
  uint32_t dev_mask        = 0u;

  for (uint32_t i = 0; i < count; ++i) {
    const fs_buf_attr_t& a = attrs[i];
    if (a.dev != FS_DEV_NONE && !fs_dev_is_valid(a.dev)) {
      return FS_ERR_INVALID_ARG;
    }
    if (!fs_lcm_checked(min_align, a.min_align > 0u ? a.min_align : 1u, &min_align)) {
      return FS_ERR_OVERFLOW;
    }
    if (!fs_lcm_checked(stride_align, a.stride_align > 0u ? a.stride_align : 1u, &stride_align)) {
      return FS_ERR_OVERFLOW;
    }
    layout_inter    &= fs_effective_layout_mask(&a);
    coherency_inter &= fs_effective_coherency_mask(&a);
    combined_access |= a.access;
    if (fs_dev_is_valid(a.dev)) {
      dev_mask |= (1u << static_cast<uint32_t>(a.dev));
    }
  }

  out_reconciled->min_align            = min_align;
  out_reconciled->stride_align         = stride_align;
  out_reconciled->combined_access      = combined_access;
  out_reconciled->participant_dev_mask = dev_mask;

  if (layout_inter == 0u) {
    out_reconciled->layout               = FS_LAYOUT_ANY;
    out_reconciled->coherency            = FS_COHERENCY_NON_COHERENT;
    out_reconciled->requires_dma_sync    = 1;
    out_reconciled->zero_copy_compatible = 0;
    std::snprintf(out_reconciled->conflict_reason, sizeof(out_reconciled->conflict_reason),
                  "layout_mismatch (tiled vs linear)");
    if (out_zero_copy_compatible) {
      *out_zero_copy_compatible = 0;
    }
    return FS_OK;
  }

  if (coherency_inter == 0u) {
    out_reconciled->layout =
        ((layout_inter & FS_LAYOUT_BIT(FS_LAYOUT_LINEAR)) != 0u) ? FS_LAYOUT_LINEAR : FS_LAYOUT_TILED;
    out_reconciled->coherency            = FS_COHERENCY_ANY;
    out_reconciled->requires_dma_sync    = 1;
    out_reconciled->zero_copy_compatible = 0;
    std::snprintf(out_reconciled->conflict_reason, sizeof(out_reconciled->conflict_reason),
                  "coherency_mismatch (coherent vs non_coherent)");
    if (out_zero_copy_compatible) {
      *out_zero_copy_compatible = 0;
    }
    return FS_OK;
  }

  /* Prefer pitch-linear when both linear & tiled are in the valid intersection */
  out_reconciled->layout =
      ((layout_inter & FS_LAYOUT_BIT(FS_LAYOUT_LINEAR)) != 0u) ? FS_LAYOUT_LINEAR : FS_LAYOUT_TILED;

  /* Prefer non-coherent DMA-BUF with explicit DMA_BUF_IOCTL_SYNC when permitted */
  if ((coherency_inter & FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT)) != 0u) {
    out_reconciled->coherency         = FS_COHERENCY_NON_COHERENT;
    out_reconciled->requires_dma_sync = 1;
  } else {
    out_reconciled->coherency         = FS_COHERENCY_COHERENT;
    out_reconciled->requires_dma_sync = 0;
  }

  out_reconciled->zero_copy_compatible = 1;
  out_reconciled->conflict_reason[0]   = '\0';
  if (out_zero_copy_compatible) {
    *out_zero_copy_compatible = 1;
  }
  return FS_OK;
}

fs_status_t fs_buffer_alloc_reconciled(fs_context_t*        ctx,
                                       size_t               size_bytes,
                                       const fs_buf_attr_t* attrs,
                                       uint32_t             attr_count,
                                       fs_dev_t             initial_owner,
                                       uint32_t             flags,
                                       fs_dmabuf_t**        out_buf,
                                       fs_buf_reconciled_t* out_reconciled) {
  if (!ctx || size_bytes == 0u || !attrs || attr_count == 0u || !out_buf) {
    return FS_ERR_INVALID_ARG;
  }
  fs_buf_reconciled_t rec;
  int                 zc_ok = 0;
  fs_status_t         st    = fs_dmabuf_reconcile(attrs, attr_count, &rec, &zc_ok);
  if (st != FS_OK) {
    return st;
  }

  const size_t align = rec.min_align > 0u ? rec.min_align : 1u;
  const size_t aligned_bytes = fs_align_up(size_bytes, align);
  if (aligned_bytes < size_bytes) {
    return FS_ERR_OVERFLOW;
  }
  const uint32_t eff_flags =
      zc_ok ? flags : (flags | static_cast<uint32_t>(FS_BUF_FORCE_HOST_COPY));
  fs_dmabuf_t* buf = nullptr;
  st               = fs_buffer_alloc(ctx, aligned_bytes, initial_owner, eff_flags, &buf);
  if (st != FS_OK) {
    return st;
  }

  const uint32_t first_lmask = fs_effective_layout_mask(&attrs[0]);
  const fs_buf_layout_t fallback_layout =
      ((first_lmask & FS_LAYOUT_BIT(FS_LAYOUT_LINEAR)) != 0u)
          ? FS_LAYOUT_LINEAR
          : (((first_lmask & FS_LAYOUT_BIT(FS_LAYOUT_TILED)) != 0u) ? FS_LAYOUT_TILED
                                                                    : FS_LAYOUT_ANY);
  const uint32_t first_cmask = fs_effective_coherency_mask(&attrs[0]);
  const fs_buf_coherency_t fallback_coh =
      ((first_cmask & FS_COHERENCY_BIT(FS_COHERENCY_NON_COHERENT)) != 0u)
          ? FS_COHERENCY_NON_COHERENT
          : (((first_cmask & FS_COHERENCY_BIT(FS_COHERENCY_COHERENT)) != 0u)
                 ? FS_COHERENCY_COHERENT
                 : FS_COHERENCY_ANY);

  buf->size                 = size_bytes;
  buf->layout               = zc_ok ? rec.layout : fallback_layout;
  buf->coherency            = zc_ok ? rec.coherency : fallback_coh;
  buf->min_align            = rec.min_align;
  buf->stride_align         = rec.stride_align;
  buf->zero_copy_compatible = zc_ok;

  if (out_reconciled) {
    *out_reconciled = rec;
  }
  *out_buf = buf;
  return FS_OK;
}

fs_status_t fs_buffer_dma_sync(fs_context_t* ctx, fs_dmabuf_t* buf, uint32_t sync_flags) {
  if (!ctx || !buf) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_dmabuf_record*           rec = fs_record_from_pub(buf);
  if (!rec || rec->owner_ctx != ctx || rec->in_pool) {
    return FS_ERR_INVALID_ARG;
  }
  return fs_dma_sync_record_locked(ctx, rec, sync_flags);
}

fs_status_t fs_buffer_export_sync_file(fs_context_t* ctx,
                                       fs_dmabuf_t*  buf,
                                       uint32_t      rw_flags,
                                       fs_fence_t*   out_fence) {
  if (!ctx || !buf || !out_fence) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_dmabuf_record*           rec = fs_record_from_pub(buf);
  if (!rec || rec->owner_ctx != ctx || rec->in_pool) {
    return FS_ERR_INVALID_ARG;
  }

  int sync_fd = -1;
#ifdef __linux__
  if ((rec->pub.backing == FS_BACKING_DMA_HEAP || rec->pub.backing == FS_BACKING_IMPORTED) &&
      rec->pub.dmabuf_fd >= 0) {
    fs_status_t st = fs_linux_export_sync_file(rec->pub.dmabuf_fd, rw_flags, &sync_fd);
    if (st != FS_OK) {
      fs_set_error(ctx, "DMA_BUF_IOCTL_EXPORT_SYNC_FILE failed on buf %llu",
                   static_cast<unsigned long long>(rec->pub.buf_id));
      return st;
    }
  }
#else
  (void)rw_flags;
#endif

  *out_fence                = fs_fence_init(rec->pub.owner_dev);
  out_fence->seq            = ++ctx->next_fence_seq;
  out_fence->sync_file_fd   = sync_fd;
  out_fence->state          = FS_FENCE_SIGNALED;
  out_fence->signaled_at_us = fs_now_us();
  ctx->telemetry.sync_file_exports++;
  return FS_OK;
}

fs_fence_t fs_fence_init(fs_dev_t producer_dev) {
  fs_fence_t f;
  std::memset(&f, 0, sizeof(f));
  f.sync_file_fd = -1;
  f.producer_dev = producer_dev;
  f.state        = FS_FENCE_UNSIGNALED;
  return f;
}

fs_status_t fs_fence_signal(fs_fence_t* fence, int sync_file_fd) {
  if (!fence) {
    return FS_ERR_INVALID_ARG;
  }
  if (fence->state == FS_FENCE_ERROR) {
    return FS_ERR_FENCE;
  }
  fence->sync_file_fd   = sync_file_fd;
  fence->state          = FS_FENCE_SIGNALED;
  fence->signaled_at_us = fs_now_us();
  return FS_OK;
}

fs_status_t fs_fence_wait(const fs_fence_t* fence, uint32_t timeout_ms) {
  if (!fence) {
    return FS_OK;
  }
  if (fence->state == FS_FENCE_ERROR) {
    return FS_ERR_FENCE;
  }
#ifdef __linux__
  if (fence->sync_file_fd >= 0) {
    struct pollfd pfd;
    pfd.fd      = fence->sync_file_fd;
    pfd.events  = POLLIN;
    pfd.revents = 0;
    int rc      = 0;
    do {
      rc = poll(&pfd, 1, static_cast<int>(timeout_ms));
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
      return FS_ERR_IO;
    }
    if (rc == 0) {
      return FS_ERR_TIMEOUT;
    }
    if ((pfd.revents & (POLLERR | POLLNVAL)) != 0) {
      return FS_ERR_FENCE;
    }
    return FS_OK;
  }
#else
  (void)timeout_ms;
#endif
  if (fence->state != FS_FENCE_SIGNALED) {
    return FS_ERR_FENCE;
  }
  return FS_OK;
}

void fs_fence_close(fs_fence_t* fence) {
  if (!fence) {
    return;
  }
  if (fence->sync_file_fd >= 0) {
    close(fence->sync_file_fd);
    fence->sync_file_fd = -1;
  }
}

fs_status_t fs_buffer_alloc(fs_context_t* ctx,
                            size_t        size_bytes,
                            fs_dev_t      initial_owner,
                            uint32_t      flags,
                            fs_dmabuf_t** out_buf) {
  if (!ctx || size_bytes == 0u || !fs_dev_is_valid(initial_owner) || !out_buf) {
    return FS_ERR_INVALID_ARG;
  }

  const size_t pg         = fs_page_size();
  const size_t alloc_size = fs_align_up(size_bytes, pg);

  std::lock_guard<std::mutex> lock(ctx->mu);

  /* Try buffer pool first unless FS_BUF_NO_POOL is requested */
  if ((flags & FS_BUF_NO_POOL) == 0u) {
    for (auto it = ctx->pooled_buffers.begin(); it != ctx->pooled_buffers.end(); ++it) {
      fs_dmabuf_record* rec = *it;
      if (rec->alloc_size >= alloc_size && rec->alloc_size <= alloc_size * 2u) {
        ctx->pooled_buffers.erase(it);
        ctx->pooled_bytes -= rec->alloc_size;
        ctx->pool_hits++;

        rec->in_pool             = 0;
        rec->active_accessor     = FS_DEV_NONE;
        rec->active_access_count = 0u;
        rec->pub.buf_id          = ++ctx->next_buf_id;
        rec->pub.size            = size_bytes;
        rec->pub.offset          = 0u;
        rec->pub.owner_dev       = initial_owner;
        rec->pub.sync_file_fd    = -1;
        rec->pub.flags           = flags;
        rec->pub.refcount        = 1u;
        rec->pub.generation           = 1u;
        rec->pub.last_handoff_us      = fs_now_us();
        rec->pub.layout               = FS_LAYOUT_LINEAR;
        rec->pub.coherency            = FS_COHERENCY_NON_COHERENT;
        rec->pub.min_align            = 64u;
        rec->pub.stride_align         = 64u;
        rec->pub.zero_copy_compatible = ((flags & FS_BUF_FORCE_HOST_COPY) == 0u) ? 1 : 0;
        rec->pub.dma_sync_count       = 0u;

        if ((flags & FS_BUF_ZERO_INIT) != 0u && rec->pub.host_ptr) {
          std::memset(rec->pub.host_ptr, 0, size_bytes);
        }

        ctx->live_buffers.push_back(rec);
        *out_buf = &rec->pub;
        return FS_OK;
      }
    }
  }

  ctx->pool_misses++;

  auto* rec = static_cast<fs_dmabuf_record*>(std::calloc(1u, sizeof(fs_dmabuf_record)));
  if (!rec) {
    fs_set_error(ctx, "out of memory allocating fs_dmabuf_record");
    return FS_ERR_NO_MEMORY;
  }

  int              dmabuf_fd = -1;
  void*            host_ptr  = nullptr;
  fs_buf_backing_t backing   = FS_BACKING_ANON_MMAP;

#ifdef __linux__
  if (!ctx->mock_mode) {
    fs_status_t st =
        fs_linux_alloc_dma_heap(ctx->config.dma_heap_path, alloc_size, &dmabuf_fd, &host_ptr);
    if (st == FS_OK) {
      backing = FS_BACKING_DMA_HEAP;
    } else if (fs_linux_alloc_memfd(alloc_size, &dmabuf_fd, &host_ptr) == FS_OK) {
      /* Graceful unprivileged Linux fallback to fd-backed shared memory */
      backing = FS_BACKING_MEMFD;
    }
  }
#endif

  if (!host_ptr) {
    host_ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_ptr == MAP_FAILED) {
      std::free(rec);
      fs_set_error(ctx, "mmap failed allocating %zu bytes", alloc_size);
      return FS_ERR_NO_MEMORY;
    }
    backing = FS_BACKING_ANON_MMAP;
  }

  if ((flags & FS_BUF_ZERO_INIT) != 0u) {
    std::memset(host_ptr, 0, size_bytes);
  }

  rec->magic               = FS_BUF_MAGIC;
  rec->owner_ctx           = ctx;
  rec->alloc_size          = alloc_size;
  rec->is_imported         = 0;
  rec->is_mmap             = 1;
  rec->in_pool             = 0;
  rec->active_accessor     = FS_DEV_NONE;
  rec->active_access_count = 0u;
  rec->mapped_dev_mask     = 0u;
  rec->last_fence          = fs_fence_init(initial_owner);
  rec->last_fence.state    = FS_FENCE_SIGNALED;

  rec->pub.buf_id          = ++ctx->next_buf_id;
  rec->pub.dmabuf_fd       = dmabuf_fd;
  rec->pub.size            = size_bytes;
  rec->pub.offset          = 0u;
  rec->pub.owner_dev       = initial_owner;
  rec->pub.sync_file_fd    = -1;
  rec->pub.flags           = flags;
  rec->pub.refcount        = 1u;
  rec->pub.generation      = 1u;
  rec->pub.tpu_iova        = 0u;
  rec->pub.vk_device_addr  = 0u;
  rec->pub.host_ptr             = host_ptr;
  rec->pub.last_handoff_us      = fs_now_us();
  rec->pub.layout               = FS_LAYOUT_LINEAR;
  rec->pub.coherency            = FS_COHERENCY_NON_COHERENT;
  rec->pub.min_align            = 64u;
  rec->pub.stride_align         = 64u;
  rec->pub.zero_copy_compatible = ((flags & FS_BUF_FORCE_HOST_COPY) == 0u) ? 1 : 0;
  rec->pub.backing              = backing;
  rec->pub.dma_sync_count       = 0u;

  ctx->live_buffers.push_back(rec);
  *out_buf = &rec->pub;
  return FS_OK;
}

fs_status_t fs_buffer_import_dmabuf(fs_context_t* ctx,
                                    int           dmabuf_fd,
                                    size_t        size_bytes,
                                    size_t        offset,
                                    void*         host_ptr,
                                    fs_dev_t      initial_owner,
                                    uint32_t      flags,
                                    fs_dmabuf_t** out_buf) {
  if (!ctx || size_bytes == 0u || !fs_dev_is_valid(initial_owner) || !out_buf) {
    return FS_ERR_INVALID_ARG;
  }
  if (dmabuf_fd < 0 && !host_ptr) {
    return FS_ERR_INVALID_ARG;
  }

  auto* rec = static_cast<fs_dmabuf_record*>(std::calloc(1u, sizeof(fs_dmabuf_record)));
  if (!rec) {
    return FS_ERR_NO_MEMORY;
  }

  std::lock_guard<std::mutex> lock(ctx->mu);
  rec->magic               = FS_BUF_MAGIC;
  rec->owner_ctx           = ctx;
  rec->alloc_size          = size_bytes;
  rec->is_imported         = 1;
  rec->is_mmap             = 0;
  rec->in_pool             = 0;
  rec->active_accessor     = FS_DEV_NONE;
  rec->active_access_count = 0u;
  rec->mapped_dev_mask     = 0u;
  rec->last_fence          = fs_fence_init(initial_owner);
  rec->last_fence.state    = FS_FENCE_SIGNALED;

  rec->pub.buf_id          = ++ctx->next_buf_id;
  rec->pub.dmabuf_fd       = dmabuf_fd;
  rec->pub.size            = size_bytes;
  rec->pub.offset          = offset;
  rec->pub.owner_dev       = initial_owner;
  rec->pub.sync_file_fd    = -1;
  rec->pub.flags           = flags | FS_BUF_NO_POOL;
  rec->pub.refcount        = 1u;
  rec->pub.generation      = 1u;
  rec->pub.host_ptr             = host_ptr;
  rec->pub.last_handoff_us      = fs_now_us();
  rec->pub.layout               = FS_LAYOUT_LINEAR;
  rec->pub.coherency            = FS_COHERENCY_NON_COHERENT;
  rec->pub.min_align            = 64u;
  rec->pub.stride_align         = 64u;
  rec->pub.zero_copy_compatible = ((flags & FS_BUF_FORCE_HOST_COPY) == 0u) ? 1 : 0;
  rec->pub.backing              = FS_BACKING_IMPORTED;
  rec->pub.dma_sync_count       = 0u;

  ctx->live_buffers.push_back(rec);
  *out_buf = &rec->pub;
  return FS_OK;
}

fs_status_t fs_buffer_retain(fs_context_t* ctx, fs_dmabuf_t* buf) {
  if (!ctx || !buf) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_dmabuf_record*           rec = fs_record_from_pub(buf);
  if (!rec || rec->owner_ctx != ctx || rec->in_pool) {
    return FS_ERR_INVALID_ARG;
  }
  rec->pub.refcount++;
  return FS_OK;
}

void fs_buffer_release(fs_context_t* ctx, fs_dmabuf_t* buf) {
  if (!ctx || !buf) {
    return;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_dmabuf_record*           rec = fs_record_from_pub(buf);
  if (!rec || rec->owner_ctx != ctx || rec->in_pool) {
    return;
  }
  if (rec->pub.refcount > 1u) {
    rec->pub.refcount--;
    return;
  }

  auto it = std::find(ctx->live_buffers.begin(), ctx->live_buffers.end(), rec);
  if (it != ctx->live_buffers.end()) {
    ctx->live_buffers.erase(it);
  }

  rec->pub.refcount        = 0u;
  rec->pub.owner_dev       = FS_DEV_NONE;
  rec->active_accessor     = FS_DEV_NONE;
  rec->active_access_count = 0u;

  const int can_pool = !rec->is_imported && ((rec->pub.flags & FS_BUF_NO_POOL) == 0u) &&
                       (ctx->pooled_bytes + rec->alloc_size <= ctx->pool_max_bytes);
  if (can_pool) {
    rec->in_pool = 1;
    ctx->pooled_bytes += rec->alloc_size;
    ctx->pooled_buffers.push_back(rec);
  } else {
    fs_free_record_backing(ctx, rec);
  }
}

fs_status_t fs_buffer_handoff(fs_context_t* ctx,
                              fs_dmabuf_t*  buf,
                              fs_dev_t      from_dev,
                              fs_dev_t      to_dev,
                              fs_fence_t*   io_fence) {
  if (!ctx || !buf || !fs_dev_is_valid(from_dev) || !fs_dev_is_valid(to_dev)) {
    return FS_ERR_INVALID_ARG;
  }

  const uint64_t t0 = fs_now_us();

  {
    std::lock_guard<std::mutex> lock(ctx->mu);
    fs_dmabuf_record*           rec = fs_record_from_pub(buf);
    if (!rec || rec->owner_ctx != ctx || rec->in_pool) {
      fs_set_error(ctx, "handoff called on invalid or released buffer");
      return FS_ERR_INVALID_ARG;
    }

    /* Strict single-owner invariant: catches use-after-handoff and wrong owner */
    if (rec->pub.owner_dev != from_dev) {
      fs_set_error(ctx, "ownership violation on buf %llu: expected owner %s, actual %s",
                   static_cast<unsigned long long>(rec->pub.buf_id), fs_dev_name(from_dev),
                   fs_dev_name(rec->pub.owner_dev));
      return FS_ERR_OWNERSHIP;
    }

    /* Reject handoff while any device holds an active exclusive lock (double-owner guard) */
    if (rec->active_access_count > 0u) {
      fs_set_error(ctx, "double-owner race on buf %llu: active lock held by %s during handoff",
                   static_cast<unsigned long long>(rec->pub.buf_id),
                   fs_dev_name(rec->active_accessor));
      return FS_ERR_OWNERSHIP;
    }

    if (from_dev == to_dev) {
      return FS_OK;
    }

    const int force_copy = ((rec->pub.flags & FS_BUF_FORCE_HOST_COPY) != 0u) ||
                           ((ctx->topo.global_caps & FS_CAP_DMABUF) == 0u);

    /* Upfront reconciliation guard: fail closed on incompatible tiled layouts unless copy-marked */
    if (!force_copy && rec->pub.layout != FS_LAYOUT_ANY) {
      fs_buf_attr_t pair[2];
      pair[0]                = fs_buf_attr_default(from_dev);
      pair[0].layout         = rec->pub.layout;
      pair[0].layout_mask    = FS_LAYOUT_BIT(rec->pub.layout);
      pair[0].coherency      = rec->pub.coherency;
      pair[0].coherency_mask = (rec->pub.coherency != FS_COHERENCY_ANY)
                                   ? FS_COHERENCY_BIT(rec->pub.coherency)
                                   : 0u;
      pair[1]                = fs_buf_attr_default(to_dev);

      fs_buf_reconciled_t rcheck;
      int                 zc_ok = 1;
      if (fs_dmabuf_reconcile(pair, 2u, &rcheck, &zc_ok) != FS_OK || !zc_ok) {
        fs_set_error(ctx, "handoff %s -> %s rejected on buf %llu: %s", fs_dev_name(from_dev),
                     fs_dev_name(to_dev), static_cast<unsigned long long>(rec->pub.buf_id),
                     rcheck.conflict_reason[0] != '\0' ? rcheck.conflict_reason : "incompatible");
        return FS_ERR_UNSUPPORTED;
      }
    }

    /* Validate / advance explicit sync_file fence */
    if (io_fence) {
      if (io_fence->state == FS_FENCE_ERROR) {
        fs_set_error(ctx, "cannot handoff buf %llu with errored fence",
                     static_cast<unsigned long long>(rec->pub.buf_id));
        return FS_ERR_FENCE;
      }
      if (io_fence->sync_file_fd >= 0) {
        fs_status_t wst = fs_fence_wait(io_fence, 5000u);
        if (wst != FS_OK) {
          fs_set_error(ctx, "sync_file fence wait failed on buf %llu (%s)",
                       static_cast<unsigned long long>(rec->pub.buf_id), fs_status_string(wst));
          return wst;
        }
      }
      if (io_fence->state == FS_FENCE_UNSIGNALED) {
        io_fence->producer_dev   = from_dev;
        io_fence->seq            = ++ctx->next_fence_seq;
        io_fence->state          = FS_FENCE_SIGNALED;
        io_fence->signaled_at_us = fs_now_us();
      }
      rec->last_fence       = *io_fence;
      rec->pub.sync_file_fd = io_fence->sync_file_fd;
    } else {
      rec->last_fence                = fs_fence_init(from_dev);
      rec->last_fence.seq            = ++ctx->next_fence_seq;
      rec->last_fence.state          = FS_FENCE_SIGNALED;
      rec->last_fence.signaled_at_us = fs_now_us();
      rec->pub.sync_file_fd          = -1;
    }

    /* Non-coherent DMA-BUF cache maintenance (critical for PowerVR Volcanic GPU <-> TPU <-> CPU) */
    if (rec->pub.coherency != FS_COHERENCY_COHERENT) {
      if (fs_dev_is_cpu(from_dev)) {
        fs_status_t st = fs_dma_sync_record_locked(
            ctx, rec,
            static_cast<uint32_t>(FS_DMA_SYNC_END) | static_cast<uint32_t>(FS_DMA_SYNC_RW));
        if (st != FS_OK) {
          fs_set_error(ctx, "DMA_BUF_IOCTL_SYNC(END) failed on buf %llu",
                       static_cast<unsigned long long>(rec->pub.buf_id));
          return st;
        }
      }
      if (fs_dev_is_cpu(to_dev)) {
        fs_status_t st = fs_dma_sync_record_locked(
            ctx, rec,
            static_cast<uint32_t>(FS_DMA_SYNC_START) | static_cast<uint32_t>(FS_DMA_SYNC_RW));
        if (st != FS_OK) {
          fs_set_error(ctx, "DMA_BUF_IOCTL_SYNC(START) failed on buf %llu",
                       static_cast<unsigned long long>(rec->pub.buf_id));
          return st;
        }
      }
      if (!fs_dev_is_cpu(from_dev) && !fs_dev_is_cpu(to_dev)) {
        /* Accelerator-to-accelerator non-coherent boundary sync + optional sync_file export */
        fs_status_t st = fs_dma_sync_record_locked(
            ctx, rec,
            static_cast<uint32_t>(FS_DMA_SYNC_END) | static_cast<uint32_t>(FS_DMA_SYNC_RW));
        if (st == FS_OK) {
          st = fs_dma_sync_record_locked(
              ctx, rec,
              static_cast<uint32_t>(FS_DMA_SYNC_START) | static_cast<uint32_t>(FS_DMA_SYNC_RW));
        }
        if (st != FS_OK) {
          fs_set_error(ctx, "accelerator DMA_BUF_IOCTL_SYNC failed on buf %llu",
                       static_cast<unsigned long long>(rec->pub.buf_id));
          return st;
        }
#ifdef __linux__
        if ((rec->pub.backing == FS_BACKING_DMA_HEAP || rec->pub.backing == FS_BACKING_IMPORTED) &&
            rec->pub.dmabuf_fd >= 0 && io_fence && io_fence->sync_file_fd < 0) {
          int exp_fd = -1;
          if (fs_linux_export_sync_file(rec->pub.dmabuf_fd, static_cast<uint32_t>(FS_DMA_SYNC_RW),
                                        &exp_fd) == FS_OK &&
              exp_fd >= 0) {
            io_fence->sync_file_fd = exp_fd;
            rec->pub.sync_file_fd  = exp_fd;
            ctx->telemetry.sync_file_exports++;
          }
        }
#endif
      }
    }

    if (force_copy) {
      /* Simulate host bounce copy when DMA-BUF is unavailable or forced for diagnostics */
      if (rec->pub.host_ptr && rec->pub.size > 0u) {
        void* tmp = std::malloc(rec->pub.size);
        if (!tmp) {
          return FS_ERR_NO_MEMORY;
        }
        std::memcpy(tmp, rec->pub.host_ptr, rec->pub.size);
        std::memcpy(rec->pub.host_ptr, tmp, rec->pub.size);
        std::free(tmp);
      }
      ctx->telemetry.fallback_copies++;
    } else {
      ctx->telemetry.zero_copy_handoffs++;
    }

    rec->pub.owner_dev       = to_dev;
    rec->pub.generation     += 1u;
    rec->pub.last_handoff_us = fs_now_us();

    const uint64_t dt = rec->pub.last_handoff_us >= t0 ? (rec->pub.last_handoff_us - t0) : 0u;
    ctx->telemetry.handoff_us += (dt > 0u) ? dt : 1u;
  }

  /* Ensure target device has zero-copy mapping (EDGETPU_MAP_DMABUF / VK_KHR_external_memory_fd) */
  fs_status_t map_st = fs_dmabuf_ensure_mapped(ctx, buf, to_dev);
  if (map_st != FS_OK) {
    /* Roll back ownership atomically if backend import rejects the buffer */
    std::lock_guard<std::mutex> lock(ctx->mu);
    fs_dmabuf_record*           rec = fs_record_from_pub(buf);
    if (rec && rec->owner_ctx == ctx) {
      rec->pub.owner_dev = from_dev;
    }
  }
  return map_st;
}

fs_status_t fs_buffer_acquire_access(fs_context_t* ctx,
                                     fs_dmabuf_t*  buf,
                                     fs_dev_t      dev,
                                     uint64_t      expected_gen) {
  if (!ctx || !buf || !fs_dev_is_valid(dev)) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_dmabuf_record*           rec = fs_record_from_pub(buf);
  if (!rec || rec->owner_ctx != ctx || rec->in_pool) {
    return FS_ERR_INVALID_ARG;
  }

  if (expected_gen != 0u && rec->pub.generation != expected_gen) {
    fs_set_error(ctx, "stale generation on buf %llu: expected %llu, current %llu (use-after-handoff)",
                 static_cast<unsigned long long>(rec->pub.buf_id),
                 static_cast<unsigned long long>(expected_gen),
                 static_cast<unsigned long long>(rec->pub.generation));
    return FS_ERR_OWNERSHIP;
  }

  /* Immutable weight buffers permit concurrent read-only access across all devices */
  if ((rec->pub.flags & FS_BUF_IMMUTABLE_WEIGHT) != 0u) {
    rec->active_access_count++;
    rec->active_accessor = dev;
    return FS_OK;
  }

  if (rec->pub.owner_dev != dev) {
    fs_set_error(ctx, "use-after-handoff / wrong-owner access on buf %llu: %s tried to use %s's buffer",
                 static_cast<unsigned long long>(rec->pub.buf_id), fs_dev_name(dev),
                 fs_dev_name(rec->pub.owner_dev));
    return FS_ERR_OWNERSHIP;
  }

  if (rec->active_access_count > 0u) {
    fs_set_error(ctx, "double-owner access on buf %llu: already locked by %s",
                 static_cast<unsigned long long>(rec->pub.buf_id),
                 fs_dev_name(rec->active_accessor));
    return FS_ERR_OWNERSHIP;
  }

  rec->active_accessor     = dev;
  rec->active_access_count = 1u;
  return FS_OK;
}

fs_status_t fs_buffer_release_access(fs_context_t* ctx, fs_dmabuf_t* buf, fs_dev_t dev) {
  if (!ctx || !buf || !fs_dev_is_valid(dev)) {
    return FS_ERR_INVALID_ARG;
  }
  std::lock_guard<std::mutex> lock(ctx->mu);
  fs_dmabuf_record*           rec = fs_record_from_pub(buf);
  if (!rec || rec->owner_ctx != ctx || rec->in_pool) {
    return FS_ERR_INVALID_ARG;
  }
  if (rec->active_access_count == 0u) {
    return FS_ERR_OWNERSHIP;
  }
  if ((rec->pub.flags & FS_BUF_IMMUTABLE_WEIGHT) == 0u && rec->active_accessor != dev) {
    return FS_ERR_OWNERSHIP;
  }
  rec->active_access_count--;
  if (rec->active_access_count == 0u) {
    rec->active_accessor = FS_DEV_NONE;
  }
  return FS_OK;
}

fs_status_t fs_buffer_pool_stats(const fs_context_t* ctx,
                                 size_t*             out_pooled_bytes,
                                 size_t*             out_pooled_count,
                                 uint64_t*           out_pool_hits,
                                 uint64_t*           out_pool_misses) {
  if (!ctx) {
    return FS_ERR_INVALID_ARG;
  }
  auto*                       mctx = const_cast<fs_context_t*>(ctx);
  std::lock_guard<std::mutex> lock(mctx->mu);
  if (out_pooled_bytes) {
    *out_pooled_bytes = ctx->pooled_bytes;
  }
  if (out_pooled_count) {
    *out_pooled_count = ctx->pooled_buffers.size();
  }
  if (out_pool_hits) {
    *out_pool_hits = ctx->pool_hits;
  }
  if (out_pool_misses) {
    *out_pool_misses = ctx->pool_misses;
  }
  return FS_OK;
}

} /* extern "C" */
