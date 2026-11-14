# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Put `reference/` and `extensions/` on the import path.

The repository is not a package — the modules are meant to be lifted into your
own tree, not installed from here. Tests import them by adding the two source
directories to `sys.path`, which keeps the modules free of any packaging
scaffolding you would only have to strip out later.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _sub in ("reference", "extensions"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
