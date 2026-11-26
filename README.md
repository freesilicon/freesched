# `freesched`

**Layer 2 Cross-Accelerator Runtime & Zero-Copy `DMA-BUF` Scheduler** for [`freesilicon`](https://github.com/freesilicon).

`freesched` coordinates heterogeneous unified-memory inference across **Vulkan GPU** (long-prompt prefill & vision), **EdgeTPU** (low-watt autoregressive token decode & in-SRAM SwiGLU), and **ARMv9 NEON / SME2 CPU** (attention softmax, RoPE, RMSNorm, control glue) on the same unified 16 GB RAM with **zero host copies**.

## Layering & One-Way Dependencies

```text
+-----------------------------------------------------------------------+
|                       llama.cpp / ggml / apps                         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|  freesilicon/freesched  (Layer 2: Scheduler, DMA-BUF & ggml Adapter)  |
|  - include/freesched.h         (Frozen pure C11 / C++17 ABI)          |
|  - include/ggml_freesilicon.h  (Thin ggml backend adapter seam)       |
|  - twin/freesilicon-twin       (Unified Computer 2 Digital Twin CLI)  |
|  - twin/REWARD-SCHEMA.json     (freesilicon.twin.v1 GRPO RL contract) |
|  - Zero-copy DMA-BUF ownership + sync_file fence state machine        |
|  - Cost-model router + pluggable fs_backend_ops_t vtable              |
+------------------+----------------------------------+-----------------+
                   | (runtime plugin vtable)          | (runtime plugin vtable)
                   v                                  v
+-----------------------------------+  +-----------------------------------+
|  freesilicon/freetpu              |  |  freesilicon/freepvr              |
|  EdgeTPU (libopenbuenos/FreeTile) |  |  PowerVR Volcanic (FreeUSC/FreeCS)|
|  EDGETPU_MAP_DMABUF (36-bit IOVA) |  |  VK_KHR_external_memory_fd        |
+-----------------------------------+  +-----------------------------------+
```

Dependencies stay strictly one-way: `freesched` carries zero vendor libraries, zero vendored `llama.cpp` source, and binds hardware backends through the C11 `fs_backend_ops_t` vtable.

## Hardware Profiles

| Profile | SoC | EdgeTPU | Vulkan GPU | CPU Extensions | Unified RAM |
| --- | --- | --- | --- | --- | --- |
| `FS_PROFILE_PIXEL10_LAGUNA` | `laguna` (Pixel 10) | `BUENOS_A0` (12 TOPS INT8, 1.8 W, 16 VII credits) | PowerVR `DXT-48-1536` (2.4 TFLOPS FP16, 4.5 W) | ARMv9 NEON | 16 GiB LPDDR5X |
| `FS_PROFILE_PIXEL11_MALIBU` | `malibu` (Pixel 11) | `malibu` TPU (24 TOPS INT8, 1.6 W, 16 credits) | PowerVR `CXTP-48-1536` (3.6 TFLOPS FP16, 4.2 W) | ARMv9 NEON + `SME2` | 16 GiB LPDDR5X |
| `FS_PROFILE_HOST_MOCK` | `host_mock` | Built-in mock TPU (exact reference parity) | Built-in mock Vulkan GPU | Host reference CPU | Portable macOS/Linux |

## Routing Policy & Zero-Copy Invariants

- **Prefill & Vision (`M >= threshold`, default `32`)**: Routes `FS_OP_PREFILL_GEMM` / `FS_OP_VISION_CONV` to `FS_DEV_GPU_VK`.
- **Token Decode (`M < threshold`)**: Routes `FS_OP_DECODE_GEMV` / `FS_OP_SWIGLU` (Q8_0, Q4_0, FP16, FP32) to `FS_DEV_TPU`.
- **Attention & Control**: Routes `FS_OP_ATTN_SOFTMAX`, `FS_OP_ROPE`, `FS_OP_RMS_NORM` to `FS_DEV_CPU_SME2` (or `FS_DEV_CPU_NEON`).
- **Continuity Bonus**: Subtracts `continuity_bonus_us` when an op's input is already resident on a capable device to prevent cross-chip ping-ponging.
- **Zero-Copy `DMA-BUF` Ownership**: `fs_buffer_handoff()` enforces single-owner invariants, advances `generation` on every transfer to trap use-after-handoff / double-owner races (`FS_ERR_OWNERSHIP`), models `DMA_BUF_IOCTL_SYNC` on Linux (`#ifdef __linux__`), and maintains `fallback_copies == 0` across GPU <-> CPU <-> TPU pipelines.
- **Graceful Degradation**: Disabling GPU, TPU, or both degrades cleanly to the remaining online accelerators with 100% numerical parity.

## 60-Second Laptop Build & Test

Works out of the box on macOS and Linux using `FREESCHED_MOCK=1`:

```bash
# Make (zero external dependencies beyond clang/make)
make verify
```

Or with CMake + Ninja:

```bash
cmake -S . -B build-cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake
FREESCHED_MOCK=1 ctest --test-dir build-cmake --output-on-failure
FREESCHED_MOCK=1 ./build-cmake/bench_heterogeneous
```

`make verify` runs:
1. Strict `-std=c11 -Wall -Wextra -Werror` and `-std=c++17 -Wall -Wextra -Werror` syntax checks on `include/freesched.h` and `include/ggml_freesilicon.h`.
2. Pure C11 ABI suite (`test_c11_abi`).
3. Full C++17 unit & integration suite (`test_freesched`) covering topology, `DMA-BUF` ownership/race detection, heterogeneous prefill/attention/decode routing, bit-exact FP32/FP16/Q8_0/Q4_0 parity, graceful degradation, custom backend vtables, and the `ggml_freesilicon` adapter.
4. Heterogeneous laptop benchmark (`bench_heterogeneous`) + `twin/freesilicon-twin` Digital Twin suite & `>= 1,000 evals/min` throughput gate.
5. Clean-room boundary audit (`./scripts/clean-room-audit.sh`).

## Computer 2 Digital Twin (`twin/`)

`freesched` ships with the unified Computer 2 Digital Twin CLI ([twin/freesilicon-twin](twin/freesilicon-twin)) and `freesilicon.twin.v1` GRPO/RL reward contract ([twin/REWARD-SCHEMA.json](twin/REWARD-SCHEMA.json)):

```bash
# Run offline verification across built-in FreeSched + any sibling FreePVR / FreeTile twins
./twin/freesilicon-twin self-test

# Evaluate mixed JSONL batches and validate against REWARD-SCHEMA.json
cat <<'EOF' | ./twin/freesilicon-twin eval --jsonl | ./twin/freesilicon-twin validate-verdict -
{"domain":"freesched","target":"laguna","workload":"prefill-decode-attn","m_prefill":512,"m_decode":1}
{"domain":"freesched","target":"malibu","workload":"prefill-decode-sme2","m_prefill":512,"m_decode":1}
EOF
```

See [twin/README.md](twin/README.md) for `FREEPVR_TWIN_DIR` and `FREETILE_TWIN_BIN` plugin discovery.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
