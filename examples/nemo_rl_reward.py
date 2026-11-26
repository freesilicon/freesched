#!/usr/bin/env python3
"""Minimal NVIDIA NeMo-RL GRPO reward hook using freesilicon-twin."""

from __future__ import annotations

import json
from typing import Any, Sequence
from freesilicon_twin import score_completion


def nemo_rl_reward_fn(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """NeMo-RL compatible reward callable returning bounded floats in [-1.0, 1.0]."""
    return score_completion(completions, target="tensor_g5", **kwargs)


def main() -> None:
    batch = [
        json.dumps({"workload": "prefill-decode-attn", "m_prefill": 512, "m_decode": 1}),
        json.dumps({"workload": "swiglu-b1-chained", "m_decode": 1, "k": 576, "h": 1536, "n": 576, "chained_in_sram": True}),
        json.dumps({"workload": "copy-violation", "fallback_copies": 2}),
        "{malformed",
    ]
    rewards = nemo_rl_reward_fn(batch)
    for idx, reward in enumerate(rewards):
        print(f"sample[{idx}]: reward={reward}")


if __name__ == "__main__":
    main()
