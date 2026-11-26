# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FreeSilicon Contributors
"""Optional sibling twin adapter discovery and in-process registration hooks.

Supports:
  - freetile-twin (freesilicon/freetpu) via Python module, registered hook, or CLI binary
  - freepvr-twin  (freesilicon/freepvr) via Python package, FREEPVR_TWIN_DIR, or sibling repo
  - freesched     native bench_heterogeneous binary when built
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .schema import SCHEMA_VERSION, VALID_DOMAINS, VALID_TARGETS

AdapterFn = Callable[[Dict[str, Any]], Dict[str, Any]]

_CUSTOM_ADAPTERS: Dict[str, AdapterFn] = {}


def _repo_root_hint() -> Optional[Path]:
    """Return repo root when running from a source checkout, else None."""
    here = Path(__file__).resolve()
    # python/freesilicon_twin/adapters.py -> parents[2] is repo root
    if len(here.parents) >= 3:
        candidate = here.parents[2]
        if (candidate / "twin").is_dir():
            return candidate
    return None


def register_adapter(domain: str, evaluator: Optional[AdapterFn]) -> None:
    """Register or unregister an in-process evaluator hook for a domain.

    Pass ``evaluator=None`` to remove a previously registered custom hook.
    """
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unsupported adapter domain {domain!r}; expected one of {sorted(VALID_DOMAINS)}")
    if evaluator is None:
        _CUSTOM_ADAPTERS.pop(domain, None)
    else:
        _CUSTOM_ADAPTERS[domain] = evaluator


def get_registered_adapter(domain: str) -> Optional[AdapterFn]:
    """Return a registered in-process adapter for ``domain``, if any."""
    return _CUSTOM_ADAPTERS.get(domain)


def find_freepvr_dir(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate FreePVR twin directory via explicit path, FREEPVR_TWIN_DIR, or sibling paths."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("FREEPVR_TWIN_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    spec = importlib.util.find_spec("freepvr_twin")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().parent.parent)

    repo_root = _repo_root_hint()
    if repo_root is not None:
        twin_dir = repo_root / "twin"
        candidates.extend(
            (
                twin_dir / "freepvr-twin",
                repo_root.parent / "freepvr" / "twin",
                repo_root.parent / "freepvr" / "freepvr-twin",
                repo_root.parent / "freepvr",
            )
        )

    for c in candidates:
        if (c / "freecs-validate").is_file() and (c / "freeusc-sim").is_file():
            return c.resolve()
    return None


def find_freetile_bin(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate freetile-twin binary via explicit path, FREETILE_TWIN_BIN, PATH, or sibling paths."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_bin = os.environ.get("FREETILE_TWIN_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    which_bin = shutil.which("freetile-twin")
    if which_bin:
        candidates.append(Path(which_bin))

    repo_root = _repo_root_hint()
    if repo_root is not None:
        twin_dir = repo_root / "twin"
        candidates.extend(
            (
                twin_dir / "bin" / "freetile-twin",
                twin_dir / "freetile-twin",
                repo_root.parent / "freetpu" / "build" / "freetile-twin",
                repo_root.parent / "freetpu" / "freetile" / "build" / "freetile-twin",
            )
        )

    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    return None


def find_freesched_bench_bin(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate compiled freesched bench_heterogeneous binary if built."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_bin = os.environ.get("FREESCHED_BENCH_BIN")
    if env_bin:
        candidates.append(Path(env_bin))

    repo_root = _repo_root_hint()
    if repo_root is not None:
        candidates.extend(
            (
                repo_root / "build" / "bench_heterogeneous",
                repo_root / "build-cmake" / "bench_heterogeneous",
            )
        )

    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    return None


def discover_adapters() -> Dict[str, Any]:
    """Report availability of built-in and optional sibling digital twin bridges."""
    freepvr_dir = find_freepvr_dir()
    freetile_bin = find_freetile_bin()
    bench_bin = find_freesched_bench_bin()
    freepvr_pkg = importlib.util.find_spec("freepvr_twin") is not None
    freetile_pkg = importlib.util.find_spec("freetile_twin") is not None
    return {
        "freesched": True,
        "freesched_native_bench": str(bench_bin) if bench_bin else None,
        "freepvr_twin": str(freepvr_dir) if freepvr_dir else (
            "python:freepvr_twin" if freepvr_pkg else None
        ),
        "freetile_twin": str(freetile_bin) if freetile_bin else (
            "python:freetile_twin" if freetile_pkg else None
        ),
        "registered_hooks": sorted(_CUSTOM_ADAPTERS.keys()),
    }


def engine_missing_verdict(domain: str, target: str, workload: str, msg: str) -> Dict[str, Any]:
    """Return a fail-closed freesilicon.twin.v1 verdict when an external twin is absent."""
    safe_domain = domain if domain in VALID_DOMAINS else "freesched"
    safe_target = target if target in VALID_TARGETS else "laguna"
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": safe_domain,
        "target": safe_target,
        "workload": workload or "unknown",
        "pass": False,
        "reward": -1.0,
        "nmse": None,
        "max_abs_err": None,
        "sram_bytes": None,
        "sram_limit_bytes": None,
        "est_hw_us": None,
        "eval_ms": 0.0,
        "fault_reason": msg,
        "diagnostics": [
            {
                "code": "ENGINE_UNAVAILABLE",
                "severity": "error",
                "offset": None,
                "field": "domain",
                "message": msg,
            }
        ],
        "checks": {},
    }


def eval_freepvr_item(item: Dict[str, Any], freepvr_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Evaluate a freecs or freeusc candidate via FreePVR twin package."""
    hook = get_registered_adapter(str(item.get("domain", "")))
    if hook is not None:
        return hook(item)

    if freepvr_dir is not None and str(freepvr_dir) not in sys.path:
        sys.path.insert(0, str(freepvr_dir))

    domain = item["domain"]
    target = item.get("target", "volcanic-dxt-48-1536")
    workload = item.get("workload", f"{domain}-candidate")

    if domain == "freecs":
        from freepvr_twin.fixtures import build_valid_freecs_bundle
        from freepvr_twin.freecs import validate_bundle

        kind = item.get("kind", "bundle")
        if "payload" in item:
            payload = item["payload"]
        elif item.get("fixture") == "valid_bundle":
            payload = build_valid_freecs_bundle()
        elif "input" in item:
            p = Path(item["input"])
            payload = json.loads(p.read_text(encoding="utf-8")) if kind == "bundle" else {kind: p.read_bytes()}
        else:
            payload = build_valid_freecs_bundle()
        return validate_bundle(payload, workload=workload, target=target)

    if domain == "freeusc":
        from freepvr_twin.fixtures import (
            DEFAULT_SSBO_VA,
            build_clean_atomic_kernel,
            build_clean_bda_fma_kernel,
            build_clean_bitwise_loop_barrier_kernel,
            build_clean_subgroup_ballot_add_shuffle_kernel,
            run_mesa_fma32,
            run_mesa_iadd_sg,
        )
        from freepvr_twin.freeusc_isa import bytes_to_words
        from freepvr_twin.freeusc_sim import (
            SUBGROUP_SIZE,
            GlobalMemory,
            VolcanicUSCState,
            evaluate_compute_shader,
        )

        fixture = item.get("fixture")
        if fixture == "fma32":
            v = run_mesa_fma32()
            v["target"] = target
            return v
        if fixture == "iadd_sg":
            v = run_mesa_iadd_sg()
            v["target"] = target
            return v
        if fixture == "clean_bda_fma":
            fma_w, _ = build_clean_bda_fma_kernel()
            mem = GlobalMemory()
            in_va, out_va = DEFAULT_SSBO_VA, DEFAULT_SSBO_VA + 0x1000
            mem.map_region(in_va, 4096, struct.pack(f"<{SUBGROUP_SIZE}f", *[float(i) for i in range(SUBGROUP_SIZE)]))
            mem.map_region(out_va, 4096)
            st = VolcanicUSCState(
                sh_init={
                    0: in_va & 0xFFFFFFFF,
                    1: (in_va >> 32) & 0xFFFFFFFF,
                    2: out_va & 0xFFFFFFFF,
                    3: (out_va >> 32) & 0xFFFFFFFF,
                },
                memory=mem,
            )
            return evaluate_compute_shader(
                fma_w,
                workload=item.get("workload", "clean-bda-fma"),
                target=target,
                state=st,
                ssbo_va=out_va,
                expected_f32=[float(i) * 2.0 + 1.0 for i in range(SUBGROUP_SIZE)],
            )
        if fixture == "clean_bitwise_loop":
            blb_w = build_clean_bitwise_loop_barrier_kernel()
            mem = GlobalMemory()
            mem.map_region(DEFAULT_SSBO_VA, 4096)
            st = VolcanicUSCState(
                sh_init={0: DEFAULT_SSBO_VA & 0xFFFFFFFF, 1: (DEFAULT_SSBO_VA >> 32) & 0xFFFFFFFF},
                memory=mem,
            )
            return evaluate_compute_shader(
                blb_w,
                workload=item.get("workload", "clean-bitwise-loop-barrier"),
                target=target,
                state=st,
                ssbo_va=DEFAULT_SSBO_VA,
                expected_u32=[((4 * i) ^ 0xFF) >> 1 for i in range(SUBGROUP_SIZE)],
            )
        if fixture == "clean_subgroup":
            sg_w = build_clean_subgroup_ballot_add_shuffle_kernel()
            mem = GlobalMemory()
            mem.map_region(DEFAULT_SSBO_VA, 4096)
            st = VolcanicUSCState(
                sh_init={0: DEFAULT_SSBO_VA & 0xFFFFFFFF, 1: (DEFAULT_SSBO_VA >> 32) & 0xFFFFFFFF},
                memory=mem,
            )
            exp: List[int] = []
            for lane in range(SUBGROUP_SIZE):
                exp.extend([1042, 8128, 127 - lane, 0x55555555])
            return evaluate_compute_shader(
                sg_w,
                workload=item.get("workload", "clean-subgroup-128wide"),
                target=target,
                state=st,
                ssbo_va=DEFAULT_SSBO_VA,
                expected_u32=exp,
            )
        if fixture == "clean_atomic":
            at_w = build_clean_atomic_kernel()
            mem = GlobalMemory()
            mem.map_region(DEFAULT_SSBO_VA, 64)
            st = VolcanicUSCState(
                sh_init={0: DEFAULT_SSBO_VA & 0xFFFFFFFF, 1: (DEFAULT_SSBO_VA >> 32) & 0xFFFFFFFF},
                memory=mem,
            )
            return evaluate_compute_shader(
                at_w,
                workload=item.get("workload", "clean-bda-atomic-add"),
                target=target,
                state=st,
                ssbo_va=DEFAULT_SSBO_VA,
                expected_u32=[8128],
            )
        if "hex" in item:
            dwords = [int(tok, 16) for tok in str(item["hex"]).replace(",", " ").split()]
            return evaluate_compute_shader(
                dwords,
                workload=workload,
                target=target,
                expected_u32=item.get("expected_u32"),
                expected_f32=item.get("expected_f32"),
            )
        if "bin" in item:
            dwords = bytes_to_words(Path(item["bin"]).read_bytes())
            return evaluate_compute_shader(
                dwords,
                workload=workload,
                target=target,
                expected_u32=item.get("expected_u32"),
                expected_f32=item.get("expected_f32"),
            )
        raise ValueError(f"unsupported freeusc candidate spec: {item!r}")

    raise ValueError(f"unsupported FreePVR domain: {domain}")


def eval_freetile_item(item: Dict[str, Any], freetile_bin: Optional[Path] = None) -> Dict[str, Any]:
    """Evaluate a single freetile/dgc0 candidate via hook, python module, or freetile-twin binary."""
    domain = str(item.get("domain", "freetile"))
    hook = get_registered_adapter(domain)
    if hook is not None:
        return hook(item)

    if importlib.util.find_spec("freetile_twin") is not None:
        mod = importlib.import_module("freetile_twin")
        if hasattr(mod, "eval_candidate"):
            return mod.eval_candidate(item)

    resolved_bin = freetile_bin or find_freetile_bin()
    if resolved_bin is None:
        return engine_missing_verdict(
            domain,
            str(item.get("target", "buenos-a0")),
            str(item.get("workload", "unknown")),
            "FREETILE_TWIN_UNAVAILABLE: set FREETILE_TWIN_BIN or install freetile-twin",
        )

    req = dict(item)
    if domain == "dgc0" and "workload" not in req:
        req["workload"] = "dgc0"
    proc = subprocess.run(
        [str(resolved_bin), "--batch-jsonl"],
        input=(json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    out_lines = [ln for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]
    if len(out_lines) != 1:
        raise RuntimeError(f"freetile-twin --batch-jsonl returned {len(out_lines)} lines for 1 request")
    return json.loads(out_lines[0])
