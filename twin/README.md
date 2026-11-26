# FreeSilicon Computer 2 Digital Twin (`twin/`)

Unified offline verifier & GRPO RL reward oracle (`freesilicon.twin.v1`) coordinating:
- **FreeSched (built-in)**: Cross-accelerator GPU + EdgeTPU + ARMv9/SME2 CPU `DMA-BUF` schedule verification & native `bench_heterogeneous` proof.
- **FreePVR Volcanic GPU** (`FREEPVR_TWIN_DIR` / `--freepvr-dir` / sibling `../freepvr`): `freecs-validate` & `freeusc-sim`.
- **FreeTPU / FreeTile** (`FREETILE_TWIN_BIN` / `--freetile-bin` / sibling `../freetpu`): `freetile-twin` & `.dgc0` validation.

## Quickstart

```sh
# Standalone self-test (works out of the box with zero phone & zero pip installs)
./twin/freesilicon-twin self-test

# Evaluate Layer 2 heterogeneous schedules against native compiled bench_heterogeneous
make bench
./twin/freesilicon-twin sched --target laguna --native --pretty
./twin/freesilicon-twin sched --target malibu --native --pretty

# High-throughput JSONL RL / GRPO batch mode
cat <<'EOF' | ./twin/freesilicon-twin eval --jsonl | ./twin/freesilicon-twin validate-verdict -
{"domain":"freesched","target":"laguna","workload":"prefill-decode-attn","m_prefill":512,"m_decode":1}
{"domain":"freesched","target":"malibu","workload":"prefill-decode-sme2","m_prefill":512,"m_decode":1}
EOF

# Combined throughput benchmark (>= 1,000 evals/min gate)
./twin/freesilicon-twin bench

# Verify Speed-of-Light (SOL) calibration receipt (Tensor G5 Batch-1 SwiGLU A/B)
./twin/freesilicon-twin verify-receipt twin/receipts/tensor-g5-swiglu-b1.json
```

## Python SDK & 3-Line GRPO Reward Hook

```sh
pip install -e .
```

```python
from freesilicon_twin import score_kernel, score_completion, list_targets, load_profile

res = score_kernel(
    {"op": "SWIGLU", "m": 1, "k": 576, "h": 1536, "n": 576, "chained_in_sram": True},
    target="tensor_g5",
)
# res.latency_us, res.sol_pct, res.valid, res.reward, res.diagnostics

# Drop-in TRL GRPOTrainer / NeMo-RL reward hook (bounded in [-1.0, 1.0]):
rewards = score_completion(completions, target="tensor_g5")
```

See [examples/trl_grpo_reward.py](../examples/trl_grpo_reward.py) and [examples/nemo_rl_reward.py](../examples/nemo_rl_reward.py).

## Plugging In Sibling Hardware Twins

Dependencies stay strictly one-way — point the CLI at local `freepvr` / `freetpu` twin builds via env vars or flags:

```sh
export FREEPVR_TWIN_DIR="../freepvr/twin"
export FREETILE_TWIN_BIN="../freetpu/build/freetile-twin"
./twin/freesilicon-twin self-test
```

All emitted verdicts conform to [REWARD-SCHEMA.json](REWARD-SCHEMA.json) (`schema_version: "freesilicon.twin.v1"`).
