# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FreeSilicon Contributors
"""Structured SOLAR evaluation receipt returned by score_kernel()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import SCHEMA_VERSION


@dataclass(frozen=True)
class SolarReceipt:
    """Deterministic SOLAR roofline & DMA-BUF verification receipt.

    Attributes:
        latency_us: Twin-predicted hardware latency in microseconds.
        sol_pct: Speed-of-Light attainment percentage in [0.0, 100.0].
        valid: True when all hard correctness, SRAM, and zero-copy gates pass.
        reward: Deterministic bounded GRPO RL reward in [-1.0, 1.0].
        diagnostics: Machine-readable diagnostic dicts (code, severity, message, ...).
        target: Requested or resolved hardware target name.
        domain: Twin evaluator domain (freesched, freetile, dgc0, freecs, freeusc).
        workload: Workload or kernel identifier.
        sol_floor_us: Theoretical Speed-of-Light floor in microseconds.
        pct_sol: Speed-of-Light attainment ratio in [0.0, 1.0] (REWARD-SCHEMA.json).
        bytes_moved: Modeled DRAM/SRAM byte traffic.
        flops: Modeled arithmetic operations (FLOPs / INT8 OPs).
        sram_bytes: Estimated active SRAM footprint in bytes.
        sram_limit_bytes: Target SRAM capacity limit in bytes.
        measured_hw_us: Optional real-silicon latency measurement in microseconds.
        twin_error_pct: Relative twin error vs measured_hw_us when provided.
        fault_reason: Concise failure reason when valid is False, else None.
        checks: Gate check dictionary.
        reward_breakdown: Component scores contributing to reward.
        eval_ms: Host evaluation wall-clock time in milliseconds.
        schema_target: Canonical REWARD-SCHEMA.json target enum string.
    """

    latency_us: float
    sol_pct: float
    valid: bool
    reward: float
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    target: str = "tensor_g5"
    domain: str = "freesched"
    workload: str = "kernel"
    sol_floor_us: float = 0.0
    pct_sol: float = 0.0
    bytes_moved: Optional[int] = None
    flops: Optional[int] = None
    sram_bytes: Optional[int] = None
    sram_limit_bytes: Optional[int] = None
    nmse: Optional[float] = None
    max_abs_err: Optional[float] = None
    measured_hw_us: Optional[float] = None
    twin_error_pct: Optional[float] = None
    fault_reason: Optional[str] = None
    checks: Dict[str, Any] = field(default_factory=dict)
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    eval_ms: float = 0.0
    schema_target: str = "laguna"

    def to_verdict(self) -> Dict[str, Any]:
        """Convert to a REWARD-SCHEMA.json (freesilicon.twin.v1) compliant verdict dict."""
        verdict: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "domain": self.domain,
            "target": self.schema_target,
            "workload": self.workload,
            "pass": self.valid,
            "reward": self.reward,
            "nmse": self.nmse,
            "max_abs_err": self.max_abs_err,
            "sram_bytes": self.sram_bytes,
            "sram_limit_bytes": self.sram_limit_bytes,
            "est_hw_us": self.latency_us,
            "bytes_moved": self.bytes_moved,
            "flops": self.flops,
            "sol_floor_us": self.sol_floor_us,
            "twin_pred_us": self.latency_us,
            "measured_hw_us": self.measured_hw_us,
            "pct_sol": self.pct_sol,
            "twin_error_pct": self.twin_error_pct,
            "eval_ms": self.eval_ms,
            "fault_reason": self.fault_reason,
            "diagnostics": list(self.diagnostics),
            "checks": dict(self.checks),
        }
        if self.reward_breakdown:
            verdict["reward_breakdown"] = dict(self.reward_breakdown)
        return verdict

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary dict including SDK and schema fields."""
        data = self.to_verdict()
        data["sdk_target"] = self.target
        data["latency_us"] = self.latency_us
        data["sol_pct"] = self.sol_pct
        data["valid"] = self.valid
        return data
