#!/usr/bin/env bash
# Clean-room & OSS boundary audit for freesilicon/freesched.
#
# Copyright 2026 FreeSilicon Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 - <<'PY'
import os
import re
import subprocess
import sys
from pathlib import Path

def list_files():
    if (Path(".git")).exists():
        out = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        if out:
            return out
    files = []
    for root, dirs, filenames in os.walk("."):
        dirs[:] = [d for d in dirs if d not in (".git", "build") and not d.startswith("build-")]
        for fn in filenames:
            p = Path(root) / fn
            files.append(p.as_posix().lstrip("./"))
    return sorted(files)

files = list_files()
bad_exts = (".so", ".ko", ".img", ".apk", ".gguf", ".tflite", ".dgc0", ".dmg0", ".bin", ".a", ".o", ".dylib")
errors = []

forbidden_pats = [
    (r"/" + r"Users/", "private macOS home path"),
    (r"/" + r"home/", "private Linux home path"),
    (r"saari" + r"-co", "private org reference"),
    (r"ro\.boot\.hw\." + r"soc\.id", "device hardware serial property"),
    (r"file" + r"://", "local file URI"),
    (r"/ven" + r"dor/", "proprietary vendor path"),
    (r"lib(tach" + r"yon|darw" + r"inn|gxp)", "proprietary vendor library"),
    (r"Co-Authored" + r"-By:", "AI commit trailer"),
    (r"[A-Za-z0-9._%+-]+@(gm" + r"ail|goo" + r"gle|smoky" + r"product)\.[A-Za-z]{2,}", "private email address"),
]

for rel in files:
    p = Path(rel)
    if not p.exists():
        continue
    if p.name.endswith(bad_exts) or ".so." in p.name:
        errors.append(f"Forbidden binary file: {rel}")
        continue
    data = p.read_bytes()
    if data[:4] in (b"\x7fELF", b"TFL3", b"GGUF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf") or \
       data[4:8] in (b"DGC0", b"DWS0", b"DGD0", b"DMG0"):
        errors.append(f"Forbidden binary magic in: {rel}")
        continue
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        text = data.decode("utf-8", errors="ignore")
        for pat, desc in forbidden_pats:
            if re.search(pat, text):
                errors.append(f"Forbidden {desc} ({pat!r}) in {rel}")

if errors:
    for e in errors:
        print("ERROR:", e)
    sys.exit(1)

print(f"Clean-room audit PASSED across {len(files)} files (0 errors).")
PY
