"""Integration tests for the unified freesilicon-twin CLI and REWARD-SCHEMA.json contract."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TWIN_DIR = REPO_ROOT / "twin"
CLI = TWIN_DIR / "freesilicon-twin"
SCHEMA = TWIN_DIR / "REWARD-SCHEMA.json"


class TestFreeSiliconTwin(unittest.TestCase):
    def test_schema_file_valid_json(self) -> None:
        doc = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(doc["properties"]["schema_version"]["const"], "freesilicon.twin.v1")
        for req in (
            "schema_version",
            "domain",
            "target",
            "workload",
            "pass",
            "reward",
            "fault_reason",
            "eval_ms",
            "checks",
        ):
            self.assertIn(req, doc["required"])
        domains = set(doc["properties"]["domain"]["enum"])
        self.assertTrue({"freecs", "freeusc", "freetile", "dgc0", "freesched"}.issubset(domains))
        for sol_field in (
            "bytes_moved",
            "flops",
            "sol_floor_us",
            "twin_pred_us",
            "measured_hw_us",
            "pct_sol",
            "twin_error_pct",
        ):
            self.assertIn(sol_field, doc["properties"])

    def test_unified_self_test(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(CLI), "self-test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        res = json.loads(proc.stdout.decode("utf-8"))
        self.assertEqual(res["self_test"], "PASS")
        self.assertEqual(res["schema_version"], "freesilicon.twin.v1")
        self.assertEqual(res["schema_errors"], 0)
        self.assertTrue(res["engines_available"]["freesched"])
        self.assertGreaterEqual(res["mixed_batch_passed"], 3)
        self.assertGreaterEqual(res["mixed_batch_intentional_fails"], 2)

    def test_freesched_jsonl_rl_batch_and_schema_validation(self) -> None:
        batch = [
            {"domain": "freesched", "target": "laguna", "workload": "rl-laguna-ok", "m_prefill": 512, "m_decode": 1},
            {"domain": "freesched", "target": "malibu", "workload": "rl-malibu-ok", "m_prefill": 512, "m_decode": 1},
            {"domain": "freesched", "target": "laguna", "workload": "rl-degraded-no-tpu", "tpu_online": False},
            {"domain": "freesched", "target": "laguna", "workload": "rl-copy-violation", "fallback_copies": 3},
            {"domain": "freesched", "target": "malibu", "workload": "rl-ownership-race", "ownership_race": True},
            {"domain": "freesched", "target": "laguna", "workload": "rl-sram-overflow", "sram_bytes": 8 * 1024 * 1024},
        ]
        stdin_data = "\n".join(json.dumps(x) for x in batch) + "\n"
        proc = subprocess.run(
            [sys.executable, str(CLI), "eval", "--jsonl"],
            input=stdin_data.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        lines = [ln for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), len(batch))

        val_proc = subprocess.run(
            [sys.executable, str(CLI), "validate-verdict", "-"],
            input=proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        val_res = json.loads(val_proc.stdout.decode("utf-8"))
        self.assertTrue(val_res["valid"])
        self.assertEqual(val_res["checked"], len(batch))

        verdicts = [json.loads(ln) for ln in lines]
        self.assertTrue(verdicts[0]["pass"])
        self.assertGreater(verdicts[0]["reward"], 0.0)
        self.assertEqual(verdicts[0]["checks"]["prefill_dev"], "GPU_VK")
        self.assertEqual(verdicts[0]["checks"]["attn_dev"], "CPU_NEON")
        self.assertEqual(verdicts[0]["checks"]["decode_dev"], "TPU")

        self.assertTrue(verdicts[1]["pass"])
        self.assertEqual(verdicts[1]["checks"]["attn_dev"], "CPU_SME2")

        self.assertTrue(verdicts[2]["pass"])
        self.assertTrue(verdicts[2]["checks"]["degraded"])

        self.assertFalse(verdicts[3]["pass"])
        self.assertLess(verdicts[3]["reward"], 0.0)
        self.assertIn("ZERO_COPY_VIOLATION", verdicts[3]["fault_reason"])

        self.assertFalse(verdicts[4]["pass"])
        self.assertLess(verdicts[4]["reward"], 0.0)
        self.assertIn("DMABUF_OWNERSHIP_RACE", verdicts[4]["fault_reason"])

        self.assertFalse(verdicts[5]["pass"])
        self.assertLess(verdicts[5]["reward"], 0.0)
        self.assertIn("SRAM_OVERFLOW", verdicts[5]["fault_reason"])

    def test_validate_verdict_multi_domain_and_rejection(self) -> None:
        valid_records = [
            {
                "schema_version": "freesilicon.twin.v1",
                "domain": "freecs",
                "target": "volcanic-dxt-48-1536",
                "workload": "cdm-smoke",
                "pass": True,
                "reward": 0.99,
                "fault_reason": None,
                "eval_ms": 0.02,
                "checks": {"cccb_header_bytes": 16},
            },
            {
                "schema_version": "freesilicon.twin.v1",
                "domain": "freeusc",
                "target": "volcanic-dxt-48-1536",
                "workload": "fma32",
                "pass": True,
                "reward": 0.997,
                "nmse": 0.0,
                "max_abs_err": 0.0,
                "fault_reason": None,
                "eval_ms": 3.1,
                "checks": {"lanes_matched": 128},
            },
            {
                "schema_version": "freesilicon.twin.v1",
                "domain": "freetile",
                "target": "buenos-a0",
                "workload": "swiglu",
                "pass": True,
                "reward": 0.94,
                "sram_bytes": 512000,
                "sram_limit_bytes": 786432,
                "fault_reason": None,
                "eval_ms": 1.2,
                "checks": {"chunks": 11},
            },
            {
                "schema_version": "freesilicon.twin.v1",
                "domain": "dgc0",
                "target": "malibu",
                "workload": "dgc0-integrity",
                "pass": False,
                "reward": -0.9,
                "fault_reason": "BAD_MAGIC",
                "eval_ms": 0.1,
                "checks": {"magic_ok": False},
            },
        ]
        proc = subprocess.run(
            [sys.executable, str(CLI), "validate-verdict", "-"],
            input=json.dumps(valid_records).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        res = json.loads(proc.stdout.decode("utf-8"))
        self.assertTrue(res["valid"])
        self.assertEqual(res["checked"], 4)

        bad_record = dict(valid_records[0])
        bad_record["reward"] = 2.5  # out of [-1.0, 1.0]
        bad_proc = subprocess.run(
            [sys.executable, str(CLI), "validate-verdict", "-"],
            input=json.dumps(bad_record).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(bad_proc.returncode, 0)
        bad_res = json.loads(bad_proc.stdout.decode("utf-8"))
        self.assertFalse(bad_res["valid"])

    def test_native_sched_if_built(self) -> None:
        bench_bin = REPO_ROOT / "build" / "bench_heterogeneous"
        if not bench_bin.is_file():
            bench_bin = REPO_ROOT / "build-cmake" / "bench_heterogeneous"
        if not bench_bin.is_file():
            self.skipTest("bench_heterogeneous not built")
        proc = subprocess.run(
            [sys.executable, str(CLI), "--freesched-bin", str(bench_bin), "sched", "--target", "malibu", "--native"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        verdict = json.loads(proc.stdout.decode("utf-8"))
        self.assertTrue(verdict["pass"])
        self.assertTrue(verdict["checks"]["native_binary_verified"])
        self.assertEqual(verdict["checks"]["fallback_copies"], 0)
        self.assertGreater(verdict["checks"]["zero_copy_handoffs"], 0)

    def test_sol_calibration_receipt_swiglu_b1(self) -> None:
        receipt = TWIN_DIR / "receipts" / "tensor-g5-swiglu-b1.json"
        self.assertTrue(receipt.is_file())
        proc = subprocess.run(
            [sys.executable, str(CLI), "verify-receipt", str(receipt)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        res = json.loads(proc.stdout.decode("utf-8"))
        self.assertTrue(res["valid"], f"receipt errors: {res['errors']}")
        a = res["schedule_a"]
        b = res["schedule_b"]
        self.assertEqual(b["flops"], 5314560)
        self.assertEqual(b["bytes_moved"], 2669952)
        self.assertAlmostEqual(b["sol_floor_us"], 55.624, places=2)
        self.assertEqual(b["measured_hw_us"], 114.0)
        self.assertGreater(b["pct_sol"], a["pct_sol"])
        self.assertLess(b["twin_pred_us"], a["twin_pred_us"])
        self.assertLess(b["twin_error_pct"], 0.05)
        self.assertGreater(res["ab_delta"]["twin_speedup"], 1.4)
        self.assertEqual(len(res["cross_shape_validation"]), 1)
        self.assertLess(res["cross_shape_validation"][0]["twin_error_pct"], 0.05)

    def test_sol_schema_rejects_out_of_range_pct_sol(self) -> None:
        bad = {
            "schema_version": "freesilicon.twin.v1",
            "domain": "freesched",
            "target": "laguna",
            "workload": "bad-sol",
            "pass": True,
            "reward": 0.9,
            "fault_reason": None,
            "eval_ms": 0.01,
            "pct_sol": 1.25,
            "checks": {},
        }
        proc = subprocess.run(
            [sys.executable, str(CLI), "validate-verdict", "-"],
            input=json.dumps(bad).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_bench_throughput_gate(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(CLI), "bench", "--freesched", "2000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        res = json.loads(proc.stdout.decode("utf-8"))
        self.assertTrue(res["all_gates_pass"])
        self.assertGreaterEqual(res["engines"]["freesched-twin"]["evals_per_min"], 1000.0)


if __name__ == "__main__":
    unittest.main()
