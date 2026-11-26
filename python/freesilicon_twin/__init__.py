# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FreeSilicon Contributors
"""freesilicon-twin: Pure-Python SOLAR Digital Twin SDK & GRPO RL reward oracle."""

from __future__ import annotations

from .adapters import discover_adapters, register_adapter
from .profiles import list_targets, load_profile
from .receipt import SolarReceipt
from .roofline import score_completion, score_kernel, verify_sol_receipt
from .schema import SCHEMA_VERSION, validate_verdict

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "SolarReceipt",
    "score_kernel",
    "score_completion",
    "list_targets",
    "load_profile",
    "validate_verdict",
    "verify_sol_receipt",
    "discover_adapters",
    "register_adapter",
]
