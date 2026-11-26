"""Unit, determinism, fail-closed, receipt-accuracy, and throughput tests for freesilicon-twin."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import freesilicon_twin as fst  # noqa: E402
from freesilicon_twin import (  # noqa: E402
    SolarReceipt,
    discover_adapters,
    list_targets,
    load_profile,
    register_adapter,
    score_completion,
    score_kernel,
    validate_verdict,
    verify_sol_receipt,
)


class TestFreeSiliconTwinSDK(unittest.TestCase):
    def test_public_exports_and_targets(self) -> None:
        targets = list_targets()
        self.assertIn("tensor_g5", targets)
        self.assertIn("laguna", targets)
        self.assertIn("malibu", targets)
        self.assertIn("buenos-a0", targets)

        prof = load_profile("tensor_g5")
        self.assertEqual(prof["soc"], "laguna")
        self.assertEqual(prof["tpu_tops"], 12.0)
        self.assertEqual(prof["mem_bw_gbps"], 48.0)
        self.assertEqual(prof["sram_limit_bytes"], 2097152)
        # Returned profile must be an isolated copy
        prof["tpu_tops"] = 999.0
        self.assertEqual(load_profile("tensor_g5")["tpu_tops"], 12.0)

        with self.assertRaises(ValueError):
            load_profile("nonexistent_chip")

    def test_score_kernel_receipt_structure_and_schema(self) -> None:
        res = score_kernel(
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
        self.assertIsInstance(res, SolarReceipt)
        self.assertTrue(res.valid)
        self.assertGreater(res.latency_us, 0.0)
        self.assertGreater(res.sol_pct, 45.0)
        self.assertLessEqual(res.sol_pct, 100.0)
        self.assertGreater(res.reward, 0.0)
        self.assertLessEqual(res.reward, 1.0)
        self.assertEqual(res.diagnostics, [])
        self.assertEqual(validate_verdict(res.to_verdict()), [])

    def test_tensor_g5_sub_1pct_calibration_accuracy(self) -> None:
        receipt_path = REPO_ROOT / "twin" / "receipts" / "tensor-g5-swiglu-b1.json"
        doc = json.loads(receipt_path.read_text(encoding="utf-8"))

        # Schedule A (unfused 3-VII SwiGLU): <1% error vs 176.0 us real silicon
        cand_a = dict(doc["schedule_a_baseline"]["candidate"])
        res_a = score_kernel(cand_a, target="tensor_g5")
        self.assertTrue(res_a.valid)
        self.assertIsNotNone(res_a.twin_error_pct)
        self.assertLess(res_a.twin_error_pct, 0.01)

        # Schedule B (1-change chained_in_sram=True): improves latency, SOL%, and reward
        cand_b = dict(doc["schedule_b_chained"]["candidate"])
        res_b = score_kernel(cand_b, target="tensor_g5")
        self.assertTrue(res_b.valid)
        self.assertLess(res_b.latency_us, res_a.latency_us)
        self.assertGreater(res_b.sol_pct, res_a.sol_pct)
        self.assertGreater(res_b.reward, res_a.reward)

        # Cross-shape Qwen2.5-1.5B SwiGLU (1x1536x8960): <1% error vs 1688.0 us silicon
        cross = doc["cross_shape_validation"][0]["candidate"]
        res_cross = score_kernel(cross, target="tensor_g5")
        self.assertTrue(res_cross.valid)
        self.assertIsNotNone(res_cross.twin_error_pct)
        self.assertLess(res_cross.twin_error_pct, 0.01)

        # Built-in receipt verifier matches
        summary = verify_sol_receipt()
        self.assertTrue(summary["valid"], summary["errors"])

    def test_fail_closed_invalid_and_malformed_specs(self) -> None:
        bad_inputs = [
            None,
            "",
            "   ",
            "not json at all",
            "{bad json",
            "[]",
            "{}",
            {"unrelated": 42},
            {"op": "SWIGLU", "m": -1, "k": 576, "h": 1536, "n": 576},
            {"op": "SWIGLU", "m": True, "k": 576, "h": 1536, "n": 576},
            {"op": "GEMM", "m": 1, "k": 0, "n": 64},
            {"op": "GEMM", "m": 1, "k": 64, "n": 64, "dtype": "UNKNOWN_DTYPE"},
            {"op": "GEMM", "m": 1, "k": 64, "n": 64, "measured_hw_us": -5.0},
            {"op": "GEMM", "m": 1, "k": 64, "n": 64, "chained_in_sram": "yes"},
            {"domain": "unknown_domain", "workload": "x"},
            {"workload": "bad-target", "target": "nonexistent_chip"},
            {"workload": "sram-overflow", "sram_bytes": 8 * 1024 * 1024},
            {"workload": "copy-violation", "fallback_copies": 2},
            {"workload": "race", "ownership_race": True},
            {"workload": "parity-fail", "nmse": 0.5},
        ]
        for spec in bad_inputs:
            res = score_kernel(spec, target="tensor_g5")
            self.assertFalse(res.valid, f"expected fail-closed for {spec!r}")
            self.assertGreaterEqual(res.reward, -1.0)
            self.assertLess(res.reward, 0.0)
            self.assertTrue(res.diagnostics, f"expected diagnostics for {spec!r}")
            self.assertEqual(validate_verdict(res.to_verdict()), [])

        # Also fail-closed on invalid target kwarg
        res_bad_tgt = score_kernel({"op": "GEMM", "m": 1, "k": 64, "n": 64}, target="no_such_chip")
        self.assertFalse(res_bad_tgt.valid)
        self.assertEqual(res_bad_tgt.reward, -1.0)

    def test_score_completion_formats_and_trl_kwargs(self) -> None:
        valid_dict = {"op": "SWIGLU", "m": 1, "k": 576, "h": 1536, "n": 576, "chained_in_sram": True}
        completions = [
            valid_dict,
            json.dumps(valid_dict),
            f"Here is the kernel:\n```json\n{json.dumps(valid_dict)}\n```",
            [{"role": "assistant", "content": f"<think>plan</think>\n{json.dumps(valid_dict)}"}],
            "garbage completion",
        ]
        rewards = score_completion(
            completions,
            target="tensor_g5",
            prompts=["p"] * len(completions),
            completion_ids=[[1, 2]] * len(completions),
        )
        self.assertEqual(len(rewards), len(completions))
        for r in rewards:
            self.assertIsInstance(r, float)
            self.assertGreaterEqual(r, -1.0)
            self.assertLessEqual(r, 1.0)
        # First 4 parse to the exact same kernel spec -> identical reward
        self.assertEqual(rewards[0], rewards[1])
        self.assertEqual(rewards[1], rewards[2])
        self.assertEqual(rewards[2], rewards[3])
        # Malformed completion gets negative penalty
        self.assertLess(rewards[4], 0.0)

    def test_determinism(self) -> None:
        batch = [
            json.dumps({"op": "SWIGLU", "m": 1, "k": 576, "h": 1536, "n": 576, "chained_in_sram": True}),
            json.dumps({"op": "SWIGLU", "m": 1, "k": 576, "h": 1536, "n": 576, "chained_in_sram": False}),
            json.dumps({"workload": "prefill-decode-attn", "m_prefill": 512, "m_decode": 1}),
            json.dumps({"workload": "sram-overflow", "sram_bytes": 4 * 1024 * 1024}),
            "invalid",
        ]
        r1 = score_completion(batch, target="tensor_g5")
        r2 = score_completion(batch, target="tensor_g5")
        self.assertEqual(r1, r2)

    def test_throughput_ge_10k_evals_per_min(self) -> None:
        batch = [
            json.dumps({"op": "SWIGLU", "m": 1, "k": 576, "h": 1536, "n": 576, "chained_in_sram": True}),
            json.dumps({"workload": "prefill-decode-attn", "m_prefill": 512, "m_decode": 1}),
        ] * 500  # 1,000 completions
        t0 = time.perf_counter()
        rewards = score_completion(batch, target="tensor_g5")
        elapsed_s = max(time.perf_counter() - t0, 1e-9)
        evals_per_min = (len(rewards) / elapsed_s) * 60.0
        self.assertEqual(len(rewards), 1000)
        self.assertGreaterEqual(evals_per_min, 10000.0)

    def test_optional_adapter_hooks(self) -> None:
        adapters = discover_adapters()
        self.assertTrue(adapters["freesched"])

        def _mock_freetile(item: dict) -> dict:
            return {
                "schema_version": "freesilicon.twin.v1",
                "domain": "freetile",
                "target": "buenos-a0",
                "workload": item.get("workload", "gemm"),
                "pass": True,
                "reward": 0.92,
                "est_hw_us": 42.0,
                "twin_pred_us": 42.0,
                "sol_floor_us": 21.0,
                "pct_sol": 0.5,
                "fault_reason": None,
                "eval_ms": 0.1,
                "diagnostics": [],
                "checks": {"mock_adapter": True},
            }

        try:
            register_adapter("freetile", _mock_freetile)
            res = score_kernel({"domain": "freetile", "workload": "gemm"}, target="buenos-a0")
            self.assertTrue(res.valid)
            self.assertEqual(res.reward, 0.92)
            self.assertEqual(res.sol_pct, 50.0)
            self.assertTrue(res.checks["mock_adapter"])
        finally:
            register_adapter("freetile", None)

    def test_package_data_synced_with_twin_dir(self) -> None:
        pkg_data = Path(fst.__file__).resolve().parent / "data"
        self.assertEqual(
            (pkg_data / "REWARD-SCHEMA.json").read_text(encoding="utf-8"),
            (REPO_ROOT / "twin" / "REWARD-SCHEMA.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            (pkg_data / "receipts" / "tensor-g5-swiglu-b1.json").read_text(encoding="utf-8"),
            (REPO_ROOT / "twin" / "receipts" / "tensor-g5-swiglu-b1.json").read_text(encoding="utf-8"),
        )

    def test_examples_run_standalone(self) -> None:
        env = {"PYTHONPATH": str(PY_ROOT)}
        for script in ("trl_grpo_reward.py", "nemo_rl_reward.py"):
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "examples" / script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertIn("reward", proc.stdout.decode("utf-8").lower())


if __name__ == "__main__":
    unittest.main()
