#!/usr/bin/env python3
"""Minimal 3-line Hugging Face TRL GRPO reward hook using freesilicon-twin.

Install:
    pip install -e .        # from repo root (or pip install freesilicon-twin)
    pip install trl         # optional, only needed to launch GRPOTrainer
"""

from __future__ import annotations

import json
from freesilicon_twin import score_completion, score_kernel


# --- 3-line TRL GRPO hook --------------------------------------------------
# from freesilicon_twin import score_completion
# from trl import GRPOTrainer
# trainer = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", reward_funcs=score_completion, train_dataset=dataset)
# ---------------------------------------------------------------------------


def main() -> None:
    # Score a single kernel spec with full SOLAR diagnostics
    receipt = score_kernel(
        {
            "op": "SWIGLU",
            "m": 1,
            "k": 576,
            "h": 1536,
            "n": 576,
            "chained_in_sram": True,
        },
        target="tensor_g5",
    )
    print(f"Single kernel: valid={receipt.valid} latency_us={receipt.latency_us} sol_pct={receipt.sol_pct}% reward={receipt.reward}")

    # Score a batch of LLM completions exactly as TRL GRPOTrainer passes them
    completions = [
        json.dumps({"op": "SWIGLU", "m": 1, "k": 576, "h": 1536, "n": 576, "chained_in_sram": True}),
        "```json\n{\"op\": \"SWIGLU\", \"m\": 1, \"k\": 576, \"h\": 1536, \"n\": 576, \"chained_in_sram\": false}\n```",
        json.dumps({"workload": "sram-overflow", "sram_bytes": 8 * 1024 * 1024}),
        "not valid json at all",
    ]
    rewards = score_completion(completions, target="tensor_g5", prompts=["Optimize SwiGLU"] * len(completions))
    print("GRPO batch rewards:", rewards)


if __name__ == "__main__":
    main()
