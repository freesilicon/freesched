# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FreeSilicon Contributors
"""REWARD-SCHEMA.json (freesilicon.twin.v1) constants and strict validator."""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_VERSION = "freesilicon.twin.v1"

REQUIRED_VERDICT_KEYS = (
    "schema_version",
    "domain",
    "target",
    "workload",
    "pass",
    "reward",
    "fault_reason",
    "eval_ms",
    "checks",
)

VALID_DOMAINS = {"freecs", "freeusc", "freetile", "dgc0", "freesched"}

VALID_TARGETS = {
    "volcanic-dxt-48-1536",
    "volcanic-cxtp-48-1536",
    "buenos-a0",
    "malibu",
    "laguna",
    "host-mock",
}

_DATA_DIR = Path(__file__).resolve().parent / "data"
SCHEMA_PATH = _DATA_DIR / "REWARD-SCHEMA.json"


@functools.lru_cache(maxsize=1)
def load_reward_schema() -> Dict[str, Any]:
    """Load the bundled freesilicon.twin.v1 JSON schema."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_verdict(verdict: Any) -> List[str]:
    """Strictly validate a verdict dict against REWARD-SCHEMA.json (freesilicon.twin.v1)."""
    errors: List[str] = []
    if not isinstance(verdict, dict):
        return ["verdict must be a JSON object"]

    for key in REQUIRED_VERDICT_KEYS:
        if key not in verdict:
            errors.append(f"missing required key: {key}")

    if verdict.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"invalid schema_version: {verdict.get('schema_version')!r}")
    if verdict.get("domain") not in VALID_DOMAINS:
        errors.append(f"invalid domain: {verdict.get('domain')!r}")
    if verdict.get("target") not in VALID_TARGETS:
        errors.append(f"invalid target: {verdict.get('target')!r}")

    wl = verdict.get("workload")
    if not isinstance(wl, str) or not wl:
        errors.append("workload must be a non-empty string")
    if not isinstance(verdict.get("pass"), bool):
        errors.append("pass must be boolean")

    reward = verdict.get("reward")
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or not math.isfinite(float(reward))
    ):
        errors.append("reward must be a finite number")
    elif not (-1.0 <= float(reward) <= 1.0):
        errors.append(f"reward out of [-1.0, 1.0]: {reward}")

    eval_ms = verdict.get("eval_ms")
    if (
        not isinstance(eval_ms, (int, float))
        or isinstance(eval_ms, bool)
        or not math.isfinite(float(eval_ms))
        or float(eval_ms) < 0.0
    ):
        errors.append(f"eval_ms must be a non-negative finite number: {eval_ms!r}")

    fr = verdict.get("fault_reason")
    if fr is not None and not isinstance(fr, str):
        errors.append("fault_reason must be string or null")
    if verdict.get("pass") is True and fr is not None:
        errors.append("passing verdict must have fault_reason == null")
    if verdict.get("pass") is False and (not isinstance(fr, str) or not fr):
        errors.append("failing verdict must provide non-empty fault_reason")

    for num_key in (
        "nmse",
        "max_abs_err",
        "est_hw_us",
        "sol_floor_us",
        "twin_pred_us",
        "measured_hw_us",
        "twin_error_pct",
    ):
        if num_key in verdict and verdict[num_key] is not None:
            v = verdict[num_key]
            if (
                not isinstance(v, (int, float))
                or isinstance(v, bool)
                or not math.isfinite(float(v))
                or float(v) < 0.0
            ):
                errors.append(f"{num_key} must be non-negative finite number or null")

    if "pct_sol" in verdict and verdict["pct_sol"] is not None:
        ps = verdict["pct_sol"]
        if (
            not isinstance(ps, (int, float))
            or isinstance(ps, bool)
            or not math.isfinite(float(ps))
            or not (0.0 <= float(ps) <= 1.0)
        ):
            errors.append(f"pct_sol must be finite in [0.0, 1.0] or null: {ps!r}")

    for int_key in ("sram_bytes", "sram_limit_bytes", "bytes_moved", "flops"):
        if int_key in verdict and verdict[int_key] is not None:
            v = verdict[int_key]
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                errors.append(f"{int_key} must be non-negative integer or null")

    if not isinstance(verdict.get("checks"), dict):
        errors.append("checks must be an object")

    diags = verdict.get("diagnostics")
    if diags is not None:
        if not isinstance(diags, list):
            errors.append("diagnostics must be an array")
        else:
            for idx, d in enumerate(diags):
                if not isinstance(d, dict):
                    errors.append(f"diagnostics[{idx}] must be an object")
                    continue
                for dk in ("code", "severity", "message"):
                    if dk not in d or not isinstance(d[dk], str):
                        errors.append(f"diagnostics[{idx}].{dk} must be a string")
                if d.get("severity") not in {"error", "warning", "info"}:
                    errors.append(f"diagnostics[{idx}].severity invalid: {d.get('severity')!r}")

    return errors
