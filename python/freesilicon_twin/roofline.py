# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FreeSilicon Contributors
"""Pure-Python SOLAR roofline evaluator, fail-closed spec parser, and GRPO reward hook."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .adapters import (
    engine_missing_verdict,
    eval_freepvr_item,
    eval_freetile_item,
    find_freepvr_dir,
    find_freetile_bin,
    get_registered_adapter,
)
from .profiles import resolve_profile_or_none
from .receipt import SolarReceipt
from .schema import SCHEMA_VERSION, VALID_DOMAINS, validate_verdict

_DATA_RECEIPTS_DIR = Path(__file__).resolve().parent / "data" / "receipts"
DEFAULT_TENSOR_G5_RECEIPT = _DATA_RECEIPTS_DIR / "tensor-g5-swiglu-b1.json"

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE)
_TAG_BLOCK_RE = re.compile(r"<(?:kernel|spec|json)>\s*(.*?)\s*</(?:kernel|spec|json)>", re.DOTALL | re.IGNORECASE)

_KNOWN_SPEC_KEYS = frozenset(
    {
        "domain",
        "target",
        "workload",
        "name",
        "op",
        "m",
        "m_prefill",
        "m_decode",
        "k",
        "n",
        "h",
        "dtype",
        "chained_in_sram",
        "num_chunks",
        "sram_bytes",
        "gpu_online",
        "tpu_online",
        "fallback_copies",
        "ownership_race",
        "require_zero_copy",
        "nmse",
        "max_abs_err",
        "measured_hw_us",
        "prefill_m_threshold",
        "fixture",
        "hex",
        "bin",
        "payload",
        "input",
    }
)

_SINGLE_KERNEL_OPS = frozenset(
    {"GEMM", "PREFILL_GEMM", "DECODE_GEMV", "VISION_CONV", "ATTN_SOFTMAX", "ROPE", "RMS_NORM"}
)

_DTYPE_BYTES = {
    "F32": 4.0,
    "FP32": 4.0,
    "I32": 4.0,
    "F16": 2.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "I8": 1.0,
    "INT8": 1.0,
    "INT8_EXT2_Q8_0": 1.0,
    "Q8_0": 34.0 / 32.0,
    "Q4_0": 18.0 / 32.0,
}


def _extract_text_from_completion(completion: Any) -> Any:
    """Unwrap TRL conversational message lists into raw text when needed."""
    if isinstance(completion, list):
        if not completion:
            return ""
        # Handle TRL conversational format: [{"role": "assistant", "content": "..."}]
        if all(isinstance(msg, dict) and "content" in msg for msg in completion):
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return completion[-1].get("content", "")
        return completion
    return completion


def _try_parse_json_dict(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from raw text, fenced markdown, or trailing object."""
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    for pattern in (_TAG_BLOCK_RE, _FENCED_BLOCK_RE):
        matches = pattern.findall(stripped)
        for candidate in reversed(matches):
            cand = candidate.strip()
            if cand.startswith("{"):
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue

    # Scan for the last valid complete JSON dict in mixed reasoning + output text
    decoder = json.JSONDecoder()
    for idx in range(len(stripped) - 1, -1, -1):
        if stripped[idx] != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and not stripped[idx + end :].strip():
            return obj
    return None


def parse_spec(spec: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse and structurally validate a kernel/schedule spec; returns (dict, err_msg)."""
    unwrapped = _extract_text_from_completion(spec)
    parsed: Optional[Dict[str, Any]] = None

    if isinstance(unwrapped, dict):
        parsed = unwrapped
    elif isinstance(unwrapped, str):
        parsed = _try_parse_json_dict(unwrapped)
        if parsed is None:
            return None, "INVALID_SPEC: completion is not a valid JSON kernel/schedule object"
    else:
        return None, f"INVALID_SPEC: expected dict or JSON string, got {type(spec).__name__}"

    if not parsed:
        return None, "INVALID_SPEC: empty kernel/schedule specification"

    if not (_KNOWN_SPEC_KEYS & set(parsed.keys())):
        return None, "INVALID_SPEC: specification contains no recognized kernel or schedule fields"

    return parsed, None


def _parse_int_field(
    item: Dict[str, Any],
    key: str,
    default: int,
    *,
    min_value: int = 1,
) -> Tuple[Optional[int], Optional[str]]:
    if key not in item or item[key] is None:
        return default, None
    val = item[key]
    if isinstance(val, bool) or not isinstance(val, int):
        return None, f"INVALID_SPEC: field '{key}' must be an integer, got {val!r}"
    if val < min_value:
        return None, f"INVALID_SPEC: field '{key}' must be >= {min_value}, got {val}"
    return val, None


def _parse_bool_field(
    item: Dict[str, Any],
    key: str,
    default: bool,
) -> Tuple[Optional[bool], Optional[str]]:
    if key not in item or item[key] is None:
        return default, None
    val = item[key]
    if not isinstance(val, bool):
        return None, f"INVALID_SPEC: field '{key}' must be a boolean, got {val!r}"
    return val, None


def _parse_float_field(
    item: Dict[str, Any],
    key: str,
    default: Optional[float],
    *,
    allow_nan: bool = False,
    min_exclusive: Optional[float] = None,
    min_inclusive: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    if key not in item or item[key] is None:
        return default, None
    val = item[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None, f"INVALID_SPEC: field '{key}' must be numeric, got {val!r}"
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval):
        if allow_nan:
            return fval, None
        return None, f"INVALID_SPEC: field '{key}' must be finite, got {val!r}"
    if min_exclusive is not None and fval <= min_exclusive:
        return None, f"INVALID_SPEC: field '{key}' must be > {min_exclusive}, got {val!r}"
    if min_inclusive is not None and fval < min_inclusive:
        return None, f"INVALID_SPEC: field '{key}' must be >= {min_inclusive}, got {val!r}"
    return fval, None


def _fail_verdict(
    *,
    domain: str,
    target: str,
    workload: str,
    code: str,
    message: str,
    field: Optional[str],
    reward: float,
    t0: float,
    checks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safe_domain = domain if domain in VALID_DOMAINS else "freesched"
    prof = resolve_profile_or_none(target)
    schema_target = prof["schema_target"] if prof is not None else "laguna"
    eval_ms = round((time.perf_counter() - t0) * 1000.0, 4)
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": safe_domain,
        "target": schema_target,
        "sdk_target": target,
        "workload": workload or "invalid-spec",
        "pass": False,
        "reward": round(max(-1.0, min(-0.1, float(reward))), 6),
        "nmse": None,
        "max_abs_err": None,
        "sram_bytes": None,
        "sram_limit_bytes": prof["sram_limit_bytes"] if prof is not None else None,
        "est_hw_us": None,
        "bytes_moved": None,
        "flops": None,
        "sol_floor_us": None,
        "twin_pred_us": None,
        "measured_hw_us": None,
        "pct_sol": None,
        "twin_error_pct": None,
        "eval_ms": eval_ms,
        "fault_reason": message,
        "diagnostics": [
            {
                "code": code,
                "severity": "error",
                "offset": None,
                "field": field,
                "message": message,
            }
        ],
        "checks": checks if checks is not None else {"spec_valid": False},
        "reward_breakdown": {
            "gate_pass": 0.0,
            "accuracy_score": 0.0,
            "sram_efficiency": 0.0,
            "latency_score": 0.0,
        },
    }


def eval_freesched_item(
    item: Dict[str, Any],
    *,
    default_target: str = "tensor_g5",
) -> Dict[str, Any]:
    """Evaluate a Layer 2 freesched kernel or schedule candidate deterministically."""
    t0 = time.perf_counter()

    raw_target = item.get("target", default_target)
    if not isinstance(raw_target, str) or not raw_target.strip():
        return _fail_verdict(
            domain="freesched",
            target=default_target,
            workload=str(item.get("workload") or item.get("op") or "invalid-target"),
            code="INVALID_TARGET",
            message=f"Unsupported freesched target {raw_target!r}",
            field="target",
            reward=-1.0,
            t0=t0,
            checks={"target_valid": False},
        )

    prof = resolve_profile_or_none(raw_target)
    if prof is None:
        return _fail_verdict(
            domain="freesched",
            target=default_target,
            workload=str(item.get("workload") or item.get("op") or "invalid-target"),
            code="INVALID_TARGET",
            message=f"Unsupported freesched target {raw_target!r}",
            field="target",
            reward=-1.0,
            t0=t0,
            checks={"target_valid": False},
        )

    schema_target = prof["schema_target"]
    sdk_target = raw_target

    raw_workload = item.get("workload") or item.get("name") or item.get("op") or "prefill-decode-attn"
    if not isinstance(raw_workload, str) or not raw_workload.strip():
        return _fail_verdict(
            domain="freesched",
            target=sdk_target,
            workload="invalid-workload",
            code="INVALID_SPEC",
            message="workload must be a non-empty string",
            field="workload",
            reward=-1.0,
            t0=t0,
        )
    workload = raw_workload.strip()

    op_raw = item.get("op", "")
    if not isinstance(op_raw, str):
        return _fail_verdict(
            domain="freesched",
            target=sdk_target,
            workload=workload,
            code="INVALID_SPEC",
            message=f"op must be a string, got {op_raw!r}",
            field="op",
            reward=-1.0,
            t0=t0,
        )
    op_upper = op_raw.strip().upper()

    is_swiglu = (
        workload.lower().startswith("swiglu")
        or op_upper == "SWIGLU"
        or ("chained_in_sram" in item)
    )
    is_single_op = (not is_swiglu) and (op_upper in _SINGLE_KERNEL_OPS)

    # Validate dimensions & flags (fail closed on malformed/negative/bool inputs)
    m_generic, err = _parse_int_field(item, "m", 1, min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="m", reward=-1.0, t0=t0)
    assert m_generic is not None

    default_m_prefill = m_generic if ("m" in item and not is_swiglu) else 512
    default_m_decode = m_generic if "m" in item else 1

    m_prefill, err = _parse_int_field(item, "m_prefill", default_m_prefill, min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="m_prefill", reward=-1.0, t0=t0)
    m_decode, err = _parse_int_field(item, "m_decode", default_m_decode, min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="m_decode", reward=-1.0, t0=t0)
    k_dim, err = _parse_int_field(item, "k", 64, min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="k", reward=-1.0, t0=t0)
    n_dim, err = _parse_int_field(item, "n", 64, min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="n", reward=-1.0, t0=t0)
    h_dim, err = _parse_int_field(item, "h", 128, min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="h", reward=-1.0, t0=t0)
    threshold, err = _parse_int_field(item, "prefill_m_threshold", int(prof["prefill_m_threshold"]), min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="prefill_m_threshold", reward=-1.0, t0=t0)
    num_chunks, err = _parse_int_field(item, "num_chunks", 1, min_value=1)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="num_chunks", reward=-1.0, t0=t0)
    fallback_copies, err = _parse_int_field(item, "fallback_copies", 0, min_value=0)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="fallback_copies", reward=-1.0, t0=t0)

    gpu_online, err = _parse_bool_field(item, "gpu_online", True)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="gpu_online", reward=-1.0, t0=t0)
    default_tpu_online = bool(prof.get("tpu_online_default", True))
    tpu_online, err = _parse_bool_field(item, "tpu_online", default_tpu_online)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="tpu_online", reward=-1.0, t0=t0)
    ownership_race, err = _parse_bool_field(item, "ownership_race", False)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="ownership_race", reward=-1.0, t0=t0)
    require_zero_copy, err = _parse_bool_field(item, "require_zero_copy", True)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="require_zero_copy", reward=-1.0, t0=t0)
    chained_in_sram, err = _parse_bool_field(item, "chained_in_sram", True)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="chained_in_sram", reward=-1.0, t0=t0)

    nmse, err = _parse_float_field(item, "nmse", 0.0, allow_nan=True, min_inclusive=0.0)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="nmse", reward=-1.0, t0=t0)
    max_abs_err, err = _parse_float_field(item, "max_abs_err", 0.0, allow_nan=True, min_inclusive=0.0)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="max_abs_err", reward=-1.0, t0=t0)
    measured_hw_us, err = _parse_float_field(item, "measured_hw_us", None, allow_nan=False, min_exclusive=0.0)
    if err:
        return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="measured_hw_us", reward=-1.0, t0=t0)

    assert (
        m_prefill is not None
        and m_decode is not None
        and k_dim is not None
        and n_dim is not None
        and h_dim is not None
        and threshold is not None
        and num_chunks is not None
        and fallback_copies is not None
        and gpu_online is not None
        and tpu_online is not None
        and ownership_race is not None
        and require_zero_copy is not None
        and chained_in_sram is not None
        and nmse is not None
        and max_abs_err is not None
    )

    dtype_str = str(item.get("dtype", "INT8_EXT2_Q8_0")).upper()
    if "dtype" in item and dtype_str not in _DTYPE_BYTES:
        return _fail_verdict(
            domain="freesched",
            target=sdk_target,
            workload=workload,
            code="INVALID_SPEC",
            message=f"Unsupported dtype {item['dtype']!r}",
            field="dtype",
            reward=-1.0,
            t0=t0,
        )
    weight_elem_bytes = _DTYPE_BYTES.get(dtype_str, 1.0)

    sram_limit = int(prof["sram_limit_bytes"])
    bw_bps = float(prof["mem_bw_gbps"]) * 1e9
    eff_bw_bps = bw_bps * float(prof["eff_bw_ratio"])

    if is_swiglu:
        m_val = max(1, m_decode)
        flops_int = int(2 * m_val * (2 * k_dim * h_dim + h_dim * n_dim) + 4 * m_val * h_dim)
        weight_bytes = int(round(((2 * h_dim * k_dim) + (n_dim * h_dim)) * weight_elem_bytes))
        bias_bytes = (2 * h_dim + n_dim) * 4
        io_bytes = m_val * (k_dim + n_dim)
        inter_dram_bytes = 0 if chained_in_sram else (6 * m_val * h_dim)
        bytes_moved_int = int(weight_bytes + bias_bytes + io_bytes + inter_dram_bytes)
        default_sram = (m_val * (k_dim + h_dim + n_dim) * 4) + ((h_dim // num_chunks) * k_dim // 4)
        sram_bytes, err = _parse_int_field(item, "sram_bytes", default_sram, min_value=0)
        if err:
            return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="sram_bytes", reward=-1.0, t0=t0)
        assert sram_bytes is not None

        prefill_dev = "TPU" if tpu_online else ("CPU_SME2" if prof["sme2"] else "CPU_NEON")
        attn_dev = prefill_dev
        decode_dev = prefill_dev
        handoffs = 0

        peak_ops = (float(prof["tpu_tops"]) * 1e12) if (tpu_online and prof["tpu_tops"] > 0) else (float(prof["cpu_gflops"]) * 1e9)
        compute_sol_us = (flops_int / peak_ops) * 1e6
        mem_sol_us = (bytes_moved_int / bw_bps) * 1e6
        sol_floor_us = max(compute_sol_us, mem_sol_us)

        num_launches = num_chunks if chained_in_sram else (3 * num_chunks)
        sync_gap_us = 0.0 if chained_in_sram else (2.0 * num_chunks * 11.0)
        launch_us = float(prof["tpu_launch_us"]) if (tpu_online and prof["tpu_tops"] > 0) else float(prof["cpu_launch_us"])
        twin_pred_us = num_launches * launch_us + sync_gap_us + max((bytes_moved_int / eff_bw_bps) * 1e6, compute_sol_us)
        est_hw_us = twin_pred_us
    elif is_single_op:
        m_val = max(1, m_generic)
        if op_upper in ("ATTN_SOFTMAX", "ROPE", "RMS_NORM"):
            flops_int = int(6 * m_val * n_dim)
            bytes_moved_int = int(2 * m_val * n_dim * 4)
            default_sram = m_val * n_dim * 4
            assigned = "CPU_SME2" if prof["sme2"] else "CPU_NEON"
        else:
            flops_int = int(2 * m_val * k_dim * n_dim)
            bytes_moved_int = int(round((m_val * (k_dim + n_dim)) * 2.0 + (n_dim * k_dim) * weight_elem_bytes))
            default_sram = (m_val * (k_dim + n_dim) * 2) + (k_dim * n_dim // 4)
            if op_upper == "VISION_CONV" or (op_upper in ("GEMM", "PREFILL_GEMM") and m_val >= threshold):
                if gpu_online:
                    assigned = "GPU_VK"
                elif tpu_online and prof["tpu_tops"] > 0:
                    assigned = "TPU"
                else:
                    assigned = "CPU_SME2" if prof["sme2"] else "CPU_NEON"
            else:
                if tpu_online and prof["tpu_tops"] > 0:
                    assigned = "TPU"
                elif gpu_online and m_val >= threshold:
                    assigned = "GPU_VK"
                else:
                    assigned = "CPU_SME2" if prof["sme2"] else "CPU_NEON"

        sram_bytes, err = _parse_int_field(item, "sram_bytes", default_sram, min_value=0)
        if err:
            return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="sram_bytes", reward=-1.0, t0=t0)
        assert sram_bytes is not None

        prefill_dev = assigned
        attn_dev = assigned
        decode_dev = assigned
        handoffs = 0

        if assigned == "GPU_VK":
            peak_ops = float(prof["gpu_gflops"]) * 1e9
            launch_us = float(prof["gpu_launch_us"])
        elif assigned == "TPU":
            peak_ops = float(prof["tpu_tops"]) * 1e12
            launch_us = float(prof["tpu_launch_us"])
        else:
            peak_ops = float(prof["cpu_gflops"]) * 1e9
            launch_us = float(prof["cpu_launch_us"])

        compute_sol_us = (flops_int / peak_ops) * 1e6
        mem_sol_us = (bytes_moved_int / bw_bps) * 1e6
        sol_floor_us = max(compute_sol_us, mem_sol_us)
        twin_pred_us = max((bytes_moved_int / eff_bw_bps) * 1e6, compute_sol_us) + launch_us
        est_hw_us = twin_pred_us
    else:
        default_sram = (m_decode * (k_dim + h_dim + n_dim) * 4) + (h_dim * k_dim)
        sram_bytes, err = _parse_int_field(item, "sram_bytes", default_sram, min_value=0)
        if err:
            return _fail_verdict(domain="freesched", target=sdk_target, workload=workload, code="INVALID_SPEC", message=err, field="sram_bytes", reward=-1.0, t0=t0)
        assert sram_bytes is not None

        if m_prefill >= threshold and gpu_online:
            prefill_dev = "GPU_VK"
        elif tpu_online and prof["tpu_tops"] > 0:
            prefill_dev = "TPU"
        else:
            prefill_dev = "CPU_SME2" if prof["sme2"] else "CPU_NEON"

        attn_dev = "CPU_SME2" if prof["sme2"] else "CPU_NEON"

        if tpu_online and prof["tpu_tops"] > 0:
            decode_dev = "TPU"
        elif gpu_online and m_decode >= threshold:
            decode_dev = "GPU_VK"
        else:
            decode_dev = "CPU_SME2" if prof["sme2"] else "CPU_NEON"

        prefill_flops = 2.0 * max(m_prefill, 1) * k_dim * n_dim
        attn_flops = 6.0 * max(m_prefill, 1) * n_dim
        decode_flops = 2.0 * max(m_decode, 1) * (k_dim * n_dim + 2.0 * k_dim * h_dim + h_dim * n_dim)
        flops_int = int(prefill_flops + attn_flops + decode_flops)

        prefill_bytes = (max(m_prefill, 1) * (k_dim + n_dim) + n_dim * k_dim) * 4
        attn_bytes = 2 * max(m_prefill, 1) * n_dim * 4
        decode_bytes = (max(m_decode, 1) * (k_dim + n_dim) * 4) + (n_dim * k_dim + 2 * h_dim * k_dim + n_dim * h_dim)
        bytes_moved_int = int(prefill_bytes + attn_bytes + decode_bytes)

        gpu_peak = float(prof["gpu_gflops"]) * 1e9
        tpu_peak = max(float(prof["tpu_tops"]) * 1e12, 1.0)
        cpu_peak = float(prof["cpu_gflops"]) * 1e9

        def _stage_us(dev: str, stage_flops: float, stage_bytes: float) -> float:
            if dev == "GPU_VK":
                return max((stage_flops / gpu_peak) * 1e6, (stage_bytes / eff_bw_bps) * 1e6) + float(prof["gpu_launch_us"])
            if dev == "TPU":
                return max((stage_flops / tpu_peak) * 1e6, (stage_bytes / eff_bw_bps) * 1e6) + float(prof["tpu_launch_us"])
            return max((stage_flops / cpu_peak) * 1e6, (stage_bytes / eff_bw_bps) * 1e6) + float(prof["cpu_launch_us"])

        def _stage_sol_us(dev: str, stage_flops: float, stage_bytes: float) -> float:
            peak = gpu_peak if dev == "GPU_VK" else (tpu_peak if dev == "TPU" else cpu_peak)
            return max((stage_flops / peak) * 1e6, (stage_bytes / bw_bps) * 1e6)

        handoffs = (1 if prefill_dev != attn_dev else 0) + (1 if attn_dev != decode_dev else 0)
        sol_floor_us = (
            _stage_sol_us(prefill_dev, prefill_flops, prefill_bytes)
            + _stage_sol_us(attn_dev, attn_flops, attn_bytes)
            + _stage_sol_us(decode_dev, decode_flops, decode_bytes)
        )
        twin_pred_us = (
            _stage_us(prefill_dev, prefill_flops, prefill_bytes)
            + _stage_us(attn_dev, attn_flops, attn_bytes)
            + _stage_us(decode_dev, decode_flops, decode_bytes)
            + handoffs * float(prof["handoff_us"])
        )
        est_hw_us = twin_pred_us

    effective_us = measured_hw_us if (measured_hw_us is not None and measured_hw_us > 0.0) else twin_pred_us
    pct_sol = min(1.0, max(0.0, sol_floor_us / effective_us)) if effective_us > 0.0 else 0.0
    twin_error_pct: Optional[float] = None
    if measured_hw_us is not None and measured_hw_us > 0.0:
        twin_error_pct = abs(twin_pred_us - measured_hw_us) / measured_hw_us

    diagnostics: List[Dict[str, Any]] = []
    fault_reason: Optional[str] = None
    reward = 0.0

    numerical_ok = (
        (not math.isnan(nmse))
        and (not math.isinf(nmse))
        and (not math.isnan(max_abs_err))
        and (not math.isinf(max_abs_err))
        and nmse <= 1e-4
    )

    checks = {
        "zero_copy_dmabuf": fallback_copies == 0,
        "ownership_invariant": not ownership_race,
        "sram_fit": sram_bytes <= sram_limit,
        "numerical_parity": numerical_ok,
        "prefill_dev": prefill_dev,
        "attn_dev": attn_dev,
        "decode_dev": decode_dev,
        "zero_copy_handoffs": handoffs if fallback_copies == 0 else 0,
        "fallback_copies": fallback_copies,
        "degraded": (not gpu_online) or (not tpu_online),
    }

    if ownership_race:
        fault_reason = "DMABUF_OWNERSHIP_RACE: double-owner or use-after-handoff detected"
        reward = -0.95
        diagnostics.append(
            {
                "code": "DMABUF_OWNERSHIP_RACE",
                "severity": "error",
                "offset": None,
                "field": "owner_dev",
                "message": fault_reason,
            }
        )
    elif require_zero_copy and fallback_copies > 0:
        fault_reason = f"ZERO_COPY_VIOLATION: {fallback_copies} host fallback copies"
        reward = -0.80
        diagnostics.append(
            {
                "code": "ZERO_COPY_VIOLATION",
                "severity": "error",
                "offset": None,
                "field": "fallback_copies",
                "message": fault_reason,
            }
        )
    elif sram_bytes > sram_limit:
        fault_reason = f"SRAM_OVERFLOW: {sram_bytes} B > {sram_limit} B"
        reward = -0.75
        diagnostics.append(
            {
                "code": "SRAM_OVERFLOW",
                "severity": "error",
                "offset": None,
                "field": "sram_bytes",
                "message": fault_reason,
            }
        )
    elif not numerical_ok:
        fault_reason = f"PARITY_FAIL: nmse={nmse} max_abs_err={max_abs_err}"
        reward = -0.85
        diagnostics.append(
            {
                "code": "PARITY_FAIL",
                "severity": "error",
                "offset": None,
                "field": "nmse",
                "message": fault_reason,
            }
        )

    passed = fault_reason is None
    if passed:
        accuracy_score = max(0.0, min(1.0, 1.0 - nmse * 1e4))
        sram_eff = max(0.0, min(1.0, 1.0 - (sram_bytes / float(sram_limit)) * 0.5))
        latency_score = max(0.0, min(1.0, 1.0 / (1.0 + est_hw_us / 500.0)))
        reward = round(min(1.0, max(0.05, 0.50 + 0.25 * accuracy_score + 0.15 * sram_eff + 0.10 * latency_score)), 6)
        breakdown = {
            "gate_pass": 1.0,
            "accuracy_score": round(accuracy_score, 6),
            "sram_efficiency": round(sram_eff, 6),
            "latency_score": round(latency_score, 6),
        }
    else:
        breakdown = {
            "gate_pass": 0.0,
            "accuracy_score": 0.0,
            "sram_efficiency": 0.0,
            "latency_score": 0.0,
        }

    eval_ms = round((time.perf_counter() - t0) * 1000.0, 4)
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": "freesched",
        "target": schema_target,
        "sdk_target": sdk_target,
        "workload": workload,
        "pass": passed,
        "reward": reward,
        "nmse": 0.0 if (math.isnan(nmse) or math.isinf(nmse)) else nmse,
        "max_abs_err": 0.0 if (math.isnan(max_abs_err) or math.isinf(max_abs_err)) else max_abs_err,
        "sram_bytes": sram_bytes,
        "sram_limit_bytes": sram_limit,
        "est_hw_us": round(est_hw_us, 3),
        "bytes_moved": bytes_moved_int,
        "flops": flops_int,
        "sol_floor_us": round(sol_floor_us, 3),
        "twin_pred_us": round(twin_pred_us, 3),
        "measured_hw_us": round(measured_hw_us, 3) if measured_hw_us is not None else None,
        "pct_sol": round(pct_sol, 6),
        "twin_error_pct": round(twin_error_pct, 6) if twin_error_pct is not None else None,
        "eval_ms": eval_ms,
        "fault_reason": fault_reason,
        "diagnostics": diagnostics,
        "checks": checks,
        "reward_breakdown": breakdown,
    }


def _verdict_to_receipt(verdict: Dict[str, Any], requested_target: str) -> SolarReceipt:
    """Convert an internal freesilicon.twin.v1 verdict dict to a SolarReceipt."""
    raw_pct_sol = verdict.get("pct_sol")
    pct_sol = float(raw_pct_sol) if raw_pct_sol is not None else 0.0
    raw_lat = verdict.get("twin_pred_us")
    if raw_lat is None:
        raw_lat = verdict.get("est_hw_us")
    latency_us = float(raw_lat) if raw_lat is not None else 0.0
    raw_floor = verdict.get("sol_floor_us")
    sol_floor_us = float(raw_floor) if raw_floor is not None else 0.0

    return SolarReceipt(
        latency_us=latency_us,
        sol_pct=round(pct_sol * 100.0, 4),
        valid=bool(verdict["pass"]),
        reward=float(verdict["reward"]),
        diagnostics=list(verdict.get("diagnostics") or []),
        target=str(verdict.get("sdk_target", requested_target)),
        domain=str(verdict["domain"]),
        workload=str(verdict["workload"]),
        sol_floor_us=sol_floor_us,
        pct_sol=pct_sol,
        bytes_moved=verdict.get("bytes_moved"),
        flops=verdict.get("flops"),
        sram_bytes=verdict.get("sram_bytes"),
        sram_limit_bytes=verdict.get("sram_limit_bytes"),
        nmse=verdict.get("nmse"),
        max_abs_err=verdict.get("max_abs_err"),
        measured_hw_us=verdict.get("measured_hw_us"),
        twin_error_pct=verdict.get("twin_error_pct"),
        fault_reason=verdict.get("fault_reason"),
        checks=dict(verdict.get("checks") or {}),
        reward_breakdown=dict(verdict.get("reward_breakdown") or {}),
        eval_ms=float(verdict.get("eval_ms", 0.0)),
        schema_target=str(verdict["target"]),
    )


def score_kernel(spec: Any, target: str = "tensor_g5") -> SolarReceipt:
    """Score a kernel or schedule specification against a calibrated SOLAR target profile.

    Never raises on malformed or invalid specs; fails closed with ``valid=False`` and a
    negative bounded reward in ``[-1.0, -0.1]``.
    """
    t0 = time.perf_counter()
    parsed, err = parse_spec(spec)
    if err is not None or parsed is None:
        verdict = _fail_verdict(
            domain="freesched",
            target=target if isinstance(target, str) else "tensor_g5",
            workload="malformed-spec",
            code="INVALID_SPEC",
            message=err or "INVALID_SPEC",
            field="spec",
            reward=-1.0,
            t0=t0,
        )
        return _verdict_to_receipt(verdict, target if isinstance(target, str) else "tensor_g5")

    domain = parsed.get("domain", "freesched")
    if not isinstance(domain, str) or domain not in VALID_DOMAINS:
        verdict = _fail_verdict(
            domain="freesched",
            target=target if isinstance(target, str) else "tensor_g5",
            workload=str(parsed.get("workload") or "invalid-domain"),
            code="INVALID_DOMAIN",
            message=f"Unsupported domain {domain!r}",
            field="domain",
            reward=-1.0,
            t0=t0,
            checks={"domain_valid": False},
        )
        return _verdict_to_receipt(verdict, target if isinstance(target, str) else "tensor_g5")

    effective_target = parsed.get("target", target)
    req_target_str = effective_target if isinstance(effective_target, str) else str(target)

    if domain == "freesched":
        verdict = eval_freesched_item(parsed, default_target=target)
        return _verdict_to_receipt(verdict, req_target_str)

    if domain in ("freecs", "freeusc"):
        item = dict(parsed)
        item.setdefault("target", target)
        hook = get_registered_adapter(domain)
        if hook is not None:
            verdict = hook(item)
        else:
            freepvr_dir = find_freepvr_dir()
            if freepvr_dir is None:
                verdict = engine_missing_verdict(
                    domain,
                    str(item.get("target", "volcanic-dxt-48-1536")),
                    str(item.get("workload", "unknown")),
                    "FREEPVR_TWIN_UNAVAILABLE: set FREEPVR_TWIN_DIR or install freepvr-twin",
                )
            else:
                verdict = eval_freepvr_item(item, freepvr_dir=freepvr_dir)
        return _verdict_to_receipt(verdict, req_target_str)

    # domain in ("freetile", "dgc0")
    item = dict(parsed)
    item.setdefault("target", target)
    verdict = eval_freetile_item(item, freetile_bin=find_freetile_bin())
    return _verdict_to_receipt(verdict, req_target_str)


def score_completion(
    completions: Iterable[Any],
    target: str = "tensor_g5",
    **kwargs: Any,
) -> List[float]:
    """Drop-in GRPO / TRL / NeMo-RL reward function mapping completions -> list[float] in [-1.0, 1.0].

    Accepts raw JSON strings, markdown-fenced JSON, dict specs, or TRL conversational
    message lists. Extra keyword arguments (e.g., ``prompts``, ``completion_ids``) are
    accepted and ignored for seamless compatibility with ``trl.GRPOTrainer``.
    """
    _ = kwargs
    if isinstance(completions, (str, bytes, dict)):
        items: List[Any] = [completions]
    else:
        items = list(completions)
    return [score_kernel(item, target=target).reward for item in items]


def verify_sol_receipt(receipt_path: Optional[Path] = None) -> Dict[str, Any]:
    """Re-evaluate a Speed-of-Light calibration receipt and verify SOL floor and silicon parity."""
    resolved = Path(receipt_path) if receipt_path is not None else DEFAULT_TENSOR_G5_RECEIPT
    doc = json.loads(resolved.read_text(encoding="utf-8"))
    verdict_a = eval_freesched_item(doc["schedule_a_baseline"]["candidate"])
    verdict_b = eval_freesched_item(doc["schedule_b_chained"]["candidate"])
    errors: List[str] = []

    for label, v in (("schedule_a_baseline", verdict_a), ("schedule_b_chained", verdict_b)):
        errs = validate_verdict(v)
        if errs:
            errors.extend(f"{label}: {e}" for e in errs)
        if not v["pass"]:
            errors.append(f"{label}: expected pass=True")

    exp_a = doc["schedule_a_baseline"].get("expected", {})
    exp_b = doc["schedule_b_chained"].get("expected", {})
    if "max_twin_error_pct" in exp_a and verdict_a["twin_error_pct"] is not None:
        if verdict_a["twin_error_pct"] > exp_a["max_twin_error_pct"]:
            errors.append(f"schedule_a twin_error_pct {verdict_a['twin_error_pct']} > {exp_a['max_twin_error_pct']}")
    if "max_twin_error_pct" in exp_b and verdict_b["twin_error_pct"] is not None:
        if verdict_b["twin_error_pct"] > exp_b["max_twin_error_pct"]:
            errors.append(f"schedule_b twin_error_pct {verdict_b['twin_error_pct']} > {exp_b['max_twin_error_pct']}")
    if "min_pct_sol" in exp_b and verdict_b["pct_sol"] < exp_b["min_pct_sol"]:
        errors.append(f"schedule_b pct_sol {verdict_b['pct_sol']} < {exp_b['min_pct_sol']}")

    if not (verdict_b["twin_pred_us"] < verdict_a["twin_pred_us"] and verdict_b["pct_sol"] > verdict_a["pct_sol"]):
        errors.append("1-change A/B schedule_b did not improve latency and pct_sol over schedule_a")

    cross_results: List[Dict[str, Any]] = []
    for item in doc.get("cross_shape_validation", []):
        cv = eval_freesched_item(item["candidate"])
        cerrs = validate_verdict(cv)
        if cerrs:
            errors.extend(f"{item['name']}: {e}" for e in cerrs)
        cexp = item.get("expected", {})
        if "max_twin_error_pct" in cexp and cv["twin_error_pct"] is not None:
            if cv["twin_error_pct"] > cexp["max_twin_error_pct"]:
                errors.append(f"{item['name']} twin_error_pct {cv['twin_error_pct']} > {cexp['max_twin_error_pct']}")
        if "min_pct_sol" in cexp and cv["pct_sol"] < cexp["min_pct_sol"]:
            errors.append(f"{item['name']} pct_sol {cv['pct_sol']} < {cexp['min_pct_sol']}")
        cross_results.append(
            {
                "name": item["name"],
                "sol_floor_us": cv["sol_floor_us"],
                "twin_pred_us": cv["twin_pred_us"],
                "measured_hw_us": cv["measured_hw_us"],
                "pct_sol": cv["pct_sol"],
                "twin_error_pct": cv["twin_error_pct"],
            }
        )

    return {
        "receipt": resolved.name,
        "valid": len(errors) == 0,
        "errors": errors,
        "schedule_a": {
            "workload": verdict_a["workload"],
            "bytes_moved": verdict_a["bytes_moved"],
            "flops": verdict_a["flops"],
            "sol_floor_us": verdict_a["sol_floor_us"],
            "twin_pred_us": verdict_a["twin_pred_us"],
            "measured_hw_us": verdict_a["measured_hw_us"],
            "pct_sol": verdict_a["pct_sol"],
            "twin_error_pct": verdict_a["twin_error_pct"],
            "reward": verdict_a["reward"],
        },
        "schedule_b": {
            "workload": verdict_b["workload"],
            "bytes_moved": verdict_b["bytes_moved"],
            "flops": verdict_b["flops"],
            "sol_floor_us": verdict_b["sol_floor_us"],
            "twin_pred_us": verdict_b["twin_pred_us"],
            "measured_hw_us": verdict_b["measured_hw_us"],
            "pct_sol": verdict_b["pct_sol"],
            "twin_error_pct": verdict_b["twin_error_pct"],
            "reward": verdict_b["reward"],
        },
        "ab_delta": {
            "twin_speedup": round(verdict_a["twin_pred_us"] / verdict_b["twin_pred_us"], 3),
            "pct_sol_gain": round(verdict_b["pct_sol"] - verdict_a["pct_sol"], 4),
            "bytes_saved": verdict_a["bytes_moved"] - verdict_b["bytes_moved"],
        },
        "cross_shape_validation": cross_results,
    }
