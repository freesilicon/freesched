# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FreeSilicon Contributors
"""freesilicon-twin CLI entry point (both repo wrapper and pip-installed console script)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adapters import (
    engine_missing_verdict,
    eval_freepvr_item,
    find_freepvr_dir,
    find_freesched_bench_bin,
    find_freetile_bin,
    get_registered_adapter,
)
from .profiles import list_targets
from .roofline import DEFAULT_TENSOR_G5_RECEIPT, eval_freesched_item, score_kernel, verify_sol_receipt
from .schema import SCHEMA_VERSION, load_reward_schema, validate_verdict


def _run_passthrough(cmd: List[str]) -> int:
    return subprocess.run(cmd).returncode


def _eval_native_freesched_bench(bench_bin: Path, target: str) -> Dict[str, Any]:
    """Run compiled bench_heterogeneous (FREESCHED_MOCK=1) and convert telemetry into a verdict."""
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["FREESCHED_MOCK"] = "1"
    proc = subprocess.run(
        [str(bench_bin)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    text = proc.stdout.decode("utf-8", errors="replace")
    section_re = re.compile(
        rf"=== freesched profile: {re.escape(target)}.*?telemetry \(8 iters\): "
        r"gpu_us=(\d+) tpu_us=(\d+) cpu_us=(\d+) handoff_us=(\d+) "
        r"zero_copy=(\d+) fallback_copies=(\d+) est_watts=([0-9.]+) W",
        re.DOTALL,
    )
    m = section_re.search(text)
    if not m:
        raise RuntimeError(f"Could not parse native bench_heterogeneous output for target={target}")
    gpu_us, tpu_us, cpu_us, handoff_us, zero_copy, fallback_copies = (int(m.group(i)) for i in range(1, 7))
    est_watts = float(m.group(7))
    verdict = eval_freesched_item(
        {
            "domain": "freesched",
            "target": target,
            "workload": f"native-bench-{target}",
            "fallback_copies": fallback_copies,
        }
    )
    verdict["eval_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
    verdict["checks"].update(
        {
            "native_binary_verified": True,
            "gpu_us": gpu_us,
            "tpu_us": tpu_us,
            "cpu_us": cpu_us,
            "handoff_us": handoff_us,
            "zero_copy_handoffs": zero_copy,
            "fallback_copies": fallback_copies,
            "est_watts": est_watts,
        }
    )
    return verdict


def eval_mixed_batch(
    items: List[Dict[str, Any]],
    freepvr_dir: Optional[Path],
    freetile_bin: Optional[Path],
    strict_schema: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """Evaluate a mixed batch of FreeSched, FreePVR, and FreeTile/dgc0 candidates preserving order."""
    results: List[Optional[Dict[str, Any]]] = [None] * len(items)
    freetile_indices: List[int] = []
    freetile_lines: List[str] = []

    for idx, item in enumerate(items):
        domain = str(item.get("domain", "freesched"))
        workload = str(item.get("workload", "unknown"))
        if domain == "freesched":
            results[idx] = eval_freesched_item(item)
        elif domain in ("freecs", "freeusc"):
            hook = get_registered_adapter(domain)
            if hook is not None:
                results[idx] = hook(item)
            elif freepvr_dir is None:
                results[idx] = engine_missing_verdict(
                    domain,
                    str(item.get("target", "volcanic-dxt-48-1536")),
                    workload,
                    "FREEPVR_TWIN_UNAVAILABLE: set FREEPVR_TWIN_DIR or --freepvr-dir",
                )
            else:
                results[idx] = eval_freepvr_item(item, freepvr_dir=freepvr_dir)
        elif domain in ("freetile", "dgc0"):
            hook = get_registered_adapter(domain)
            if hook is not None:
                results[idx] = hook(item)
            else:
                req = dict(item)
                if domain == "dgc0" and "workload" not in req:
                    req["workload"] = "dgc0"
                freetile_indices.append(idx)
                freetile_lines.append(json.dumps(req, separators=(",", ":")))
        else:
            results[idx] = engine_missing_verdict(
                "freesched",
                "laguna",
                workload,
                f"UNSUPPORTED_DOMAIN: {domain!r}",
            )

    if freetile_indices:
        if freetile_bin is None:
            for idx in freetile_indices:
                it = items[idx]
                results[idx] = engine_missing_verdict(
                    str(it.get("domain", "freetile")),
                    str(it.get("target", "buenos-a0")),
                    str(it.get("workload", "unknown")),
                    "FREETILE_TWIN_UNAVAILABLE: set FREETILE_TWIN_BIN or --freetile-bin",
                )
        else:
            proc = subprocess.run(
                [str(freetile_bin), "--batch-jsonl"],
                input=("\n".join(freetile_lines) + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            out_lines = [ln for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]
            if len(out_lines) != len(freetile_indices):
                raise RuntimeError(
                    f"freetile-twin --batch-jsonl returned {len(out_lines)} lines for {len(freetile_indices)} requests"
                )
            for idx, raw_line in zip(freetile_indices, out_lines):
                results[idx] = json.loads(raw_line)

    schema_errors = 0
    final: List[Dict[str, Any]] = []
    for v in results:
        assert v is not None
        if strict_schema:
            errs = validate_verdict(v)
            if errs:
                schema_errors += 1
                v = dict(v)
                v["_schema_errors"] = errs
        final.append(v)
    return final, schema_errors


def cmd_self_test(
    freepvr_dir: Optional[Path],
    freetile_bin: Optional[Path],
    bench_bin: Optional[Path],
) -> int:
    t0 = time.perf_counter()
    schema_doc = load_reward_schema()
    if schema_doc.get("properties", {}).get("schema_version", {}).get("const") != SCHEMA_VERSION:
        print(json.dumps({"self_test": "FAIL", "reason": "REWARD-SCHEMA.json mismatch"}))
        return 1

    mixed_items: List[Dict[str, Any]] = [
        {"domain": "freesched", "target": "laguna", "workload": "prefill-decode-attn", "m_prefill": 512, "m_decode": 1},
        {"domain": "freesched", "target": "malibu", "workload": "prefill-decode-sme2", "m_prefill": 512, "m_decode": 1},
        {"domain": "freesched", "target": "laguna", "workload": "degraded-no-gpu", "gpu_online": False},
        {"domain": "freesched", "target": "laguna", "workload": "intentional-fallback-copy", "fallback_copies": 2},
        {"domain": "freesched", "target": "malibu", "workload": "intentional-ownership-race", "ownership_race": True},
    ]
    expected_fails = 2

    freecs_status = "skipped"
    freeusc_status = "skipped"
    mesa_fma32_pass: Optional[bool] = None
    mesa_iadd_bug_caught: Optional[bool] = None

    if freepvr_dir is not None:
        freecs_proc = subprocess.run(
            [sys.executable, str(freepvr_dir / "freecs-validate"), "--self-test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        freecs_res = json.loads(freecs_proc.stdout.decode("utf-8"))
        assert freecs_res["self_test"] == "PASS"
        assert not validate_verdict(freecs_res["sample_valid_verdict"])
        freecs_status = freecs_res["self_test"]

        freeusc_proc = subprocess.run(
            [sys.executable, str(freepvr_dir / "freeusc-sim"), "--self-test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        freeusc_res = json.loads(freeusc_proc.stdout.decode("utf-8"))
        assert freeusc_res["self_test"] == "PASS"
        freeusc_status = freeusc_res["self_test"]

        mesa_proc = subprocess.run(
            [sys.executable, str(freepvr_dir / "freeusc-sim"), "--mesa-proof"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        mesa_res = json.loads(mesa_proc.stdout.decode("utf-8"))["mesa_stage_g3b_proof"]
        assert mesa_res["fma32"]["pass"] is True
        assert mesa_res["iadd_sg"]["pass"] is False
        assert not validate_verdict(mesa_res["fma32"])
        assert not validate_verdict(mesa_res["iadd_sg"])
        mesa_fma32_pass = bool(mesa_res["fma32"]["pass"])
        mesa_iadd_bug_caught = not bool(mesa_res["iadd_sg"]["pass"])

        mixed_items.extend(
            [
                {"domain": "freecs", "fixture": "valid_bundle", "workload": "smoke-freecs-bundle", "target": "volcanic-dxt-48-1536"},
                {"domain": "freeusc", "fixture": "fma32", "target": "volcanic-dxt-48-1536"},
                {"domain": "freeusc", "fixture": "iadd_sg", "target": "volcanic-dxt-48-1536"},
                {"domain": "freeusc", "fixture": "clean_subgroup", "target": "volcanic-cxtp-48-1536"},
            ]
        )
        expected_fails += 1

    if freetile_bin is not None:
        mixed_items.extend(
            [
                {"domain": "freetile", "target": "buenos-a0", "workload": "gemm", "m": 16, "k": 64, "n": 64},
                {"domain": "freetile", "target": "buenos-a0", "workload": "swiglu", "swiglu_11chunk": True, "m": 1, "k": 576, "h": 512, "n": 576},
                {"domain": "freetile", "target": "buenos-a0", "workload": "q4_k", "m": 8, "k": 256, "n": 64},
                {"domain": "freetile", "target": "malibu", "workload": "flash_attn", "fa_phase": "prefill", "head_dim": 128, "seq_len": 256},
                {"domain": "freetile", "target": "malibu", "workload": "iq4_xs", "m": 4, "k": 256, "n": 64},
                {"domain": "freetile", "target": "buenos-a0", "workload": "flash_attn", "fa_phase": "prefill", "head_dim": 128, "br": 64, "bc": 1024},
            ]
        )
        expected_fails += 1

    verdicts, schema_errs = eval_mixed_batch(
        mixed_items,
        freepvr_dir=freepvr_dir,
        freetile_bin=freetile_bin,
        strict_schema=True,
    )
    assert schema_errs == 0, f"schema validation failed on {schema_errs} verdicts"

    native_verdicts: List[Dict[str, Any]] = []
    if bench_bin is not None:
        for tgt in ("laguna", "malibu"):
            nv = _eval_native_freesched_bench(bench_bin, tgt)
            assert not validate_verdict(nv), f"native verdict schema error for {tgt}"
            assert nv["pass"] is True
            native_verdicts.append(nv)

    sol_receipt_res: Optional[Dict[str, Any]] = None
    if DEFAULT_TENSOR_G5_RECEIPT.is_file():
        sol_receipt_res = verify_sol_receipt(DEFAULT_TENSOR_G5_RECEIPT)
        assert sol_receipt_res["valid"], f"SOL calibration receipt failed: {sol_receipt_res['errors']}"

    pass_count = sum(1 for v in verdicts if v["pass"])
    fail_count = sum(1 for v in verdicts if not v["pass"])
    assert fail_count == expected_fails, f"expected {expected_fails} intentional fails, got {fail_count}"

    total_ms = (time.perf_counter() - t0) * 1000.0
    summary = {
        "self_test": "PASS",
        "schema_version": SCHEMA_VERSION,
        "schema_errors": schema_errs,
        "engines_available": {
            "freesched": True,
            "freesched_native_bench": bench_bin is not None,
            "freepvr_twin": freepvr_dir is not None,
            "freetile_twin": freetile_bin is not None,
        },
        "sol_receipt_verified": sol_receipt_res is not None and sol_receipt_res["valid"],
        "freecs_self_test": freecs_status,
        "freeusc_self_test": freeusc_status,
        "mesa_fma32_pass": mesa_fma32_pass,
        "mesa_iadd_sg_bug_caught": mesa_iadd_bug_caught,
        "mixed_batch_total": len(verdicts),
        "mixed_batch_passed": pass_count,
        "mixed_batch_intentional_fails": fail_count,
        "native_bench_verified": len(native_verdicts),
        "total_ms": round(total_ms, 3),
        "verdict_rewards": [
            {
                "domain": v["domain"],
                "target": v["target"],
                "workload": v["workload"],
                "pass": v["pass"],
                "reward": v["reward"],
            }
            for v in verdicts
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_bench(
    freepvr_dir: Optional[Path],
    freetile_bin: Optional[Path],
    n_freesched: int,
    n_freecs: int,
    n_freeusc: int,
    n_freetile: int,
) -> int:
    t0 = time.perf_counter()
    sample_laguna = {"domain": "freesched", "target": "laguna", "workload": "bench-laguna", "m_prefill": 512, "m_decode": 1}
    sample_malibu = {"domain": "freesched", "target": "malibu", "workload": "bench-malibu", "m_prefill": 512, "m_decode": 1}
    for i in range(n_freesched):
        eval_freesched_item(sample_laguna if (i & 1) == 0 else sample_malibu)
    elapsed_s = max(time.perf_counter() - t0, 1e-9)
    freesched_b = {
        "evals": n_freesched,
        "elapsed_ms": round(elapsed_s * 1000.0, 3),
        "ms_per_eval": round((elapsed_s * 1000.0) / n_freesched, 5),
        "evals_per_min": round((n_freesched / elapsed_s) * 60.0, 1),
    }

    engines: Dict[str, Any] = {"freesched-twin": freesched_b}
    gates: Dict[str, bool] = {
        "freesched_ge_1000_per_min": freesched_b["evals_per_min"] >= 1000.0 and freesched_b["ms_per_eval"] < 5.0
    }

    if freepvr_dir is not None:
        freecs_p = subprocess.run(
            [sys.executable, str(freepvr_dir / "freecs-validate"), "--bench", str(n_freecs)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        freecs_b = json.loads(freecs_p.stdout.decode("utf-8"))
        engines["freecs-validate"] = freecs_b
        gates["freecs_ge_1000_per_min"] = freecs_b["evals_per_min"] >= 1000.0 and freecs_b["ms_per_eval"] < 5.0

        freeusc_p = subprocess.run(
            [sys.executable, str(freepvr_dir / "freeusc-sim"), "--bench", str(n_freeusc)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        freeusc_b = json.loads(freeusc_p.stdout.decode("utf-8"))
        engines["freeusc-sim"] = freeusc_b
        gates["freeusc_ge_1000_per_min"] = freeusc_b["evals_per_min"] >= 1000.0

    if freetile_bin is not None:
        ft_p = subprocess.run(
            [str(freetile_bin), "--bench", str(n_freetile)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        freetile_b = json.loads(ft_p.stdout.decode("utf-8"))
        engines["freetile-twin"] = freetile_b
        gates["freetile_ge_1000_per_min"] = freetile_b["evals_per_min"] >= 1000.0

    out = {
        "schema_version": SCHEMA_VERSION,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "engines": engines,
    }
    print(json.dumps(out, indent=2))
    return 0 if out["all_gates_pass"] else 1


def main(argv: Optional[List[str]] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)

    explicit_pvr: Optional[str] = None
    explicit_ft: Optional[str] = None
    explicit_fs: Optional[str] = None
    filtered: List[str] = []
    i = 0
    while i < len(args_list):
        tok = args_list[i]
        if tok == "--freepvr-dir" and i + 1 < len(args_list):
            explicit_pvr = args_list[i + 1]
            i += 2
        elif tok.startswith("--freepvr-dir="):
            explicit_pvr = tok.split("=", 1)[1]
            i += 1
        elif tok == "--freetile-bin" and i + 1 < len(args_list):
            explicit_ft = args_list[i + 1]
            i += 2
        elif tok.startswith("--freetile-bin="):
            explicit_ft = tok.split("=", 1)[1]
            i += 1
        elif tok == "--freesched-bin" and i + 1 < len(args_list):
            explicit_fs = args_list[i + 1]
            i += 2
        elif tok.startswith("--freesched-bin="):
            explicit_fs = tok.split("=", 1)[1]
            i += 1
        else:
            filtered.append(tok)
            i += 1

    if not filtered or filtered[0] in ("-h", "--help"):
        print(
            "freesilicon-twin — FreeSilicon Computer 2 Digital Twin CLI (freesilicon.twin.v1)\n\n"
            "Usage:\n"
            "  freesilicon-twin [--freepvr-dir DIR] [--freetile-bin PATH] [--freesched-bin PATH] <command> [args...]\n\n"
            "Commands:\n"
            "  sched [--target tensor_g5|laguna|malibu] [--native]  Evaluate Layer 2 freesched schedule\n"
            "  score <JSON | FILE> [--target tensor_g5]             Score a kernel spec via SolarReceipt\n"
            "  targets                                              List supported hardware target profiles\n"
            "  freecs [args...]                                     Run FreePVR command-stream/FW-ABI validator\n"
            "  freeusc [args...]                                    Run FreePVR 128-lane Volcanic USC simulator\n"
            "  freetile [args...]                                   Run FreeTPU / FreeTile twin simulator\n"
            "  dgc0 <file.dgc0> [args...]                           Validate a raw .dgc0 wire payload via freetile-twin\n"
            "  eval [--spec FILE | --jsonl] [--pretty]              Evaluate mixed GPU+TPU+Scheduler candidates\n"
            "  validate-verdict <FILE | ->                          Validate JSON/JSONL verdict(s) against REWARD-SCHEMA.json\n"
            "  verify-receipt [FILE]                                Verify Speed-of-Light silicon calibration receipt\n"
            "  self-test                                            Run offline verification across all available engines\n"
            "  bench [--freesched N] [--freecs N] [--freeusc N] [--freetile N]\n"
            "                                                       Run combined throughput benchmark (>= 1,000 evals/min)\n"
        )
        return 0

    cmd = filtered[0]
    rest = filtered[1:]
    freepvr_dir = find_freepvr_dir(explicit_pvr)
    freetile_bin = find_freetile_bin(explicit_ft)
    bench_bin = find_freesched_bench_bin(explicit_fs)

    if cmd == "targets":
        print(json.dumps(list_targets(), indent=2))
        return 0

    if cmd == "score":
        p = argparse.ArgumentParser(prog="freesilicon-twin score")
        p.add_argument("spec", help="JSON string or path to a JSON kernel spec file")
        p.add_argument("--target", default="tensor_g5")
        p.add_argument("--pretty", action="store_true")
        ns = p.parse_args(rest)
        spec_arg: Any = ns.spec
        cand_path = Path(ns.spec)
        if cand_path.is_file():
            spec_arg = cand_path.read_text(encoding="utf-8")
        receipt = score_kernel(spec_arg, target=ns.target)
        print(json.dumps(receipt.to_dict(), indent=2 if ns.pretty else None))
        return 0 if receipt.valid else 1

    if cmd == "sched":
        p = argparse.ArgumentParser(prog="freesilicon-twin sched")
        p.add_argument("--target", default="laguna", choices=["tensor_g5", "laguna", "malibu", "host-mock"])
        p.add_argument("--workload", default="prefill-decode-attn")
        p.add_argument("--m-prefill", type=int, default=512)
        p.add_argument("--m-decode", type=int, default=1)
        p.add_argument("--native", action="store_true", help="Run compiled bench_heterogeneous binary")
        p.add_argument("--pretty", action="store_true")
        ns = p.parse_args(rest)
        if ns.native:
            if bench_bin is None:
                print(json.dumps({"error": "bench_heterogeneous not built; run `make all` or set --freesched-bin"}), file=sys.stderr)
                return 2
            verdict = _eval_native_freesched_bench(bench_bin, ns.target)
        else:
            verdict = eval_freesched_item(
                {
                    "domain": "freesched",
                    "target": ns.target,
                    "workload": ns.workload,
                    "m_prefill": ns.m_prefill,
                    "m_decode": ns.m_decode,
                }
            )
        print(json.dumps(verdict, indent=2 if ns.pretty else None))
        return 0 if verdict["pass"] else 1

    if cmd == "freecs":
        if freepvr_dir is None:
            print(json.dumps({"error": "FreePVR twin not found; set FREEPVR_TWIN_DIR or --freepvr-dir"}), file=sys.stderr)
            return 2
        return _run_passthrough([sys.executable, str(freepvr_dir / "freecs-validate"), *rest])

    if cmd == "freeusc":
        if freepvr_dir is None:
            print(json.dumps({"error": "FreePVR twin not found; set FREEPVR_TWIN_DIR or --freepvr-dir"}), file=sys.stderr)
            return 2
        return _run_passthrough([sys.executable, str(freepvr_dir / "freeusc-sim"), *rest])

    if cmd == "freetile":
        if freetile_bin is None:
            print(json.dumps({"error": "freetile-twin binary not found; set FREETILE_TWIN_BIN or --freetile-bin"}), file=sys.stderr)
            return 2
        return _run_passthrough([str(freetile_bin), *rest])

    if cmd == "dgc0":
        if freetile_bin is None:
            print(json.dumps({"error": "freetile-twin binary not found; set FREETILE_TWIN_BIN or --freetile-bin"}), file=sys.stderr)
            return 2
        if not rest:
            print("Usage: freesilicon-twin dgc0 <payload.dgc0> [--target buenos-a0|malibu]", file=sys.stderr)
            return 2
        return _run_passthrough([str(freetile_bin), "--dgc0", rest[0], *rest[1:]])

    if cmd == "eval":
        p = argparse.ArgumentParser(prog="freesilicon-twin eval")
        p.add_argument("--spec", help="JSON file containing one candidate object or a list of candidates")
        p.add_argument("--jsonl", action="store_true", help="Read candidate JSONL from stdin and emit JSONL")
        p.add_argument("--pretty", action="store_true", help="Pretty-print single/list JSON output")
        ns = p.parse_args(rest)
        if ns.jsonl:
            items = [json.loads(ln) for ln in sys.stdin if ln.strip()]
            verdicts, errs = eval_mixed_batch(
                items, freepvr_dir=freepvr_dir, freetile_bin=freetile_bin, strict_schema=True
            )
            for v in verdicts:
                print(json.dumps(v, separators=(",", ":")))
            return 0 if errs == 0 else 1
        if ns.spec:
            raw = json.loads(Path(ns.spec).read_text(encoding="utf-8"))
            is_list = isinstance(raw, list)
            items = raw if is_list else [raw]
            verdicts, errs = eval_mixed_batch(
                items, freepvr_dir=freepvr_dir, freetile_bin=freetile_bin, strict_schema=True
            )
            out_obj: Any = verdicts if is_list else verdicts[0]
            print(json.dumps(out_obj, indent=2 if ns.pretty else None))
            return 0 if errs == 0 else 1
        p.print_help()
        return 2

    if cmd == "validate-verdict":
        if not rest:
            print("Usage: freesilicon-twin validate-verdict <file.json | ->", file=sys.stderr)
            return 2
        src = (sys.stdin.read() if rest[0] == "-" else Path(rest[0]).read_text(encoding="utf-8")).strip()
        records: List[Any] = []
        if src.startswith("["):
            records = json.loads(src)
        else:
            try:
                records = [json.loads(src)]
            except json.JSONDecodeError:
                records = [json.loads(ln) for ln in src.splitlines() if ln.strip()]
        bad = []
        for idx, rec in enumerate(records):
            errs = validate_verdict(rec)
            if errs:
                bad.append({"index": idx, "errors": errs})
        print(json.dumps({"valid": len(bad) == 0, "checked": len(records), "failures": bad}, indent=2))
        return 0 if not bad else 1

    if cmd == "verify-receipt":
        rpath = Path(rest[0]) if rest else DEFAULT_TENSOR_G5_RECEIPT
        res = verify_sol_receipt(rpath)
        print(json.dumps(res, indent=2))
        return 0 if res["valid"] else 1

    if cmd == "self-test":
        return cmd_self_test(freepvr_dir=freepvr_dir, freetile_bin=freetile_bin, bench_bin=bench_bin)

    if cmd == "bench":
        p = argparse.ArgumentParser(prog="freesilicon-twin bench")
        p.add_argument("--freesched", type=int, default=5000)
        p.add_argument("--freecs", type=int, default=5000)
        p.add_argument("--freeusc", type=int, default=500)
        p.add_argument("--freetile", type=int, default=1200)
        ns = p.parse_args(rest)
        return cmd_bench(
            freepvr_dir=freepvr_dir,
            freetile_bin=freetile_bin,
            n_freesched=ns.freesched,
            n_freecs=ns.freecs,
            n_freeusc=ns.freeusc,
            n_freetile=ns.freetile,
        )

    print(f"Unknown subcommand: {cmd}", file=sys.stderr)
    return 2
