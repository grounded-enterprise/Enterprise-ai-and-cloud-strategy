# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Load a printed listing as a module, by its path.

The chapter listings are filed under the names the book prints
(`listing-4.1-redactor-cache-tool-gate.py`), which are not importable names.
`load()` imports one by path so a test can execute it exactly as shipped.

Two listings import a framework at module scope: 3.1 imports FastAPI and 6.3
imports `redis.asyncio`. When the real package is absent a minimal stand-in is
registered so the listing still imports and its logic can run. A stand-in
replaces the framework only, never the code under test, and `HAVE` records
which packages were real so a test can say what it did not exercise.
"""
import importlib.util
import sys
import types
from pathlib import Path

import _paths  # noqa: F401  (import for the sys.path side effect)

ROOT = Path(__file__).resolve().parent.parent
HAVE: dict = {}


def _have(name: str) -> bool:
    if name not in HAVE:
        try:
            __import__(name)
            HAVE[name] = True
        except ImportError:
            HAVE[name] = False
    return HAVE[name]


def _stub_fastapi() -> None:
    fastapi = types.ModuleType("fastapi")

    class _State:
        pass

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.state = _State()

        def post(self, *args, **kwargs):
            return lambda fn: fn

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = "", **kwargs):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:
        pass

    class StreamingResponse:
        def __init__(self, content, **kwargs):
            self.content = content
            self.kwargs = kwargs

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    fastapi.Request = Request
    responses = types.ModuleType("fastapi.responses")
    responses.StreamingResponse = StreamingResponse
    fastapi.responses = responses
    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("fastapi.responses", responses)


def _stub_redis() -> None:
    redis = types.ModuleType("redis")
    redis_asyncio = types.ModuleType("redis.asyncio")

    class Redis:  # the type the listing annotates against; tests pass their own client
        pass

    redis_asyncio.Redis = Redis
    redis.asyncio = redis_asyncio
    sys.modules.setdefault("redis", redis)
    sys.modules.setdefault("redis.asyncio", redis_asyncio)


def load(relative_path: str):
    """Import `relative_path` (from the repository root) as a module."""
    if not _have("fastapi"):
        _stub_fastapi()
    if not _have("redis"):
        _stub_redis()
    path = ROOT / relative_path
    name = "listing_" + "".join(c if c.isalnum() else "_" for c in path.stem)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def have(name: str) -> bool:
    """True when the real third-party package `name` is installed."""
    return _have(name)
