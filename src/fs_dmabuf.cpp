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
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/ioctl.h>
#include <poll.h>

/* Portable Linux UAPI structs (avoid requiring kernel >= 5.6 headers) */
struct fs_dma_heap_allocation_data {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};

struct fs_dma_buf_sync {
  uint64_t flags;
};

#define FS_DMA_HEAP_IOC_MAGIC   'H'
#define FS_DMA_HEAP_IOCTL_ALLOC _IOWR(FS_DMA_HEAP_IOC_MAGIC, 0x0, struct fs_dma_heap_allocation_data)

#define FS_DMA_BUF_BASE         'b'
#define FS_DMA_BUF_IOCTL_SYNC   _IOW(FS_DMA_BUF_BASE, 0, struct fs_dma_buf_sync)
#define FS_DMA_BUF_SYNC_READ    (1ULL << 0)
#define FS_DMA_BUF_SYNC_WRITE   (2ULL << 0)
#define FS_DMA_BUF_SYNC_RW      (FS_DMA_BUF_SYNC_READ | FS_DMA_BUF_SYNC_WRITE)
#define FS_DMA_BUF_SYNC_START   (0ULL << 2)
#define FS_DMA_BUF_SYNC_END     (1ULL << 2)
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
  if (ioctl(dmabuf_fd, FS_DMA_BUF_IOCTL_SYNC, &sync_arg) != 0) {
    return FS_ERR_IOCTL;
  }
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
  int rc       = ioctl(heap_fd, FS_DMA_HEAP_IOCTL_ALLOC, &req);
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
#endif /* __linux__ */

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

extern "C" {

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
    int rc      = poll(&pfd, 1, static_cast<int>(timeout_ms));
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
        rec->pub.generation      = 1u;
        rec->pub.last_handoff_us = fs_now_us();

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

  int   dmabuf_fd = -1;
  void* host_ptr  = nullptr;

#ifdef __linux__
  if (!ctx->mock_mode) {
    fs_status_t st =
        fs_linux_alloc_dma_heap(ctx->config.dma_heap_path, alloc_size, &dmabuf_fd, &host_ptr);
    if (st != FS_OK) {
      std::free(rec);
      fs_set_error(ctx, "DMA-BUF heap allocation failed (%s)", fs_status_string(st));
      return st;
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
  rec->pub.host_ptr        = host_ptr;
  rec->pub.last_handoff_us = fs_now_us();

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
  rec->pub.host_ptr        = host_ptr;
  rec->pub.last_handoff_us = fs_now_us();

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

    /* Validate / advance explicit sync_file fence */
    if (io_fence) {
      if (io_fence->state == FS_FENCE_ERROR) {
        fs_set_error(ctx, "cannot handoff buf %llu with errored fence",
                     static_cast<unsigned long long>(rec->pub.buf_id));
        return FS_ERR_FENCE;
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

#ifdef __linux__
    if (!ctx->mock_mode && rec->pub.dmabuf_fd >= 0) {
      if (fs_dev_is_cpu(from_dev)) {
        fs_status_t st =
            fs_linux_dma_buf_sync(rec->pub.dmabuf_fd, FS_DMA_BUF_SYNC_END | FS_DMA_BUF_SYNC_RW);
        if (st != FS_OK) {
          fs_set_error(ctx, "DMA_BUF_IOCTL_SYNC(END) failed on buf %llu",
                       static_cast<unsigned long long>(rec->pub.buf_id));
          return st;
        }
      }
      if (fs_dev_is_cpu(to_dev)) {
        fs_status_t st =
            fs_linux_dma_buf_sync(rec->pub.dmabuf_fd, FS_DMA_BUF_SYNC_START | FS_DMA_BUF_SYNC_RW);
        if (st != FS_OK) {
          fs_set_error(ctx, "DMA_BUF_IOCTL_SYNC(START) failed on buf %llu",
                       static_cast<unsigned long long>(rec->pub.buf_id));
          return st;
        }
      }
    }
#endif

    const int force_copy = ((rec->pub.flags & FS_BUF_FORCE_HOST_COPY) != 0u) ||
                           ((ctx->topo.global_caps & FS_CAP_DMABUF) == 0u);
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
  return fs_dmabuf_ensure_mapped(ctx, buf, to_dev);
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
