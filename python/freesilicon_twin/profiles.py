# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FreeSilicon Contributors
"""Bundled hardware roofline profiles and target discovery."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any, Dict, List

_PROFILES_DIR = Path(__file__).resolve().parent / "data" / "profiles"

CANONICAL_TARGETS = (
    "tensor_g5",
    "laguna",
    "malibu",
    "buenos-a0",
    "host-mock",
    "volcanic-dxt-48-1536",
    "volcanic-cxtp-48-1536",
)

_TARGET_FILES: Dict[str, str] = {
    "tensor_g5": "tensor_g5.json",
    "tensor-g5": "tensor_g5.json",
    "pixel10": "tensor_g5.json",
    "laguna": "laguna.json",
    "malibu": "malibu.json",
    "tensor_g6": "malibu.json",
    "tensor-g6": "malibu.json",
    "pixel11": "malibu.json",
    "buenos-a0": "buenos_a0.json",
    "buenos_a0": "buenos_a0.json",
    "host-mock": "host_mock.json",
    "host_mock": "host_mock.json",
    "volcanic-dxt-48-1536": "volcanic_dxt_48_1536.json",
    "volcanic-cxtp-48-1536": "volcanic_cxtp_48_1536.json",
}


def list_targets() -> List[str]:
    """Return the supported hardware target profile names."""
    return list(CANONICAL_TARGETS)


@functools.lru_cache(maxsize=32)
def _load_profile_cached(target: str) -> Dict[str, Any]:
    key = target.strip().lower()
    filename = _TARGET_FILES.get(key)
    if filename is None:
        supported = ", ".join(CANONICAL_TARGETS)
        raise ValueError(f"Unsupported target {target!r}; supported targets: {supported}")
    path = _PROFILES_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_profile_or_none(target: Any) -> Dict[str, Any] | None:
    """Fast cached profile lookup returning None if target is unsupported."""
    if not isinstance(target, str) or not target.strip():
        return None
    key = target.strip().lower()
    if key not in _TARGET_FILES:
        return None
    return _load_profile_cached(key)


def load_profile(target: str = "tensor_g5") -> Dict[str, Any]:
    """Load a bundled SOLAR hardware profile by target name (returns a deep copy)."""
    if not isinstance(target, str):
        raise ValueError(f"target must be a string, got {type(target).__name__}")
    return copy.deepcopy(_load_profile_cached(target))
