# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 10.2 — streaming_core.py : async SSE proxy with sliding-window PII redaction (FastAPI).
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

"""Streaming SSE transformation & PII filter core (FastAPI, async).

Proxies a provider SSE stream to the client, normalizing frames and redacting
structured PII across chunk boundaries with a bounded hold-back. Nothing on the
token path is buffered: every safe frame is flushed before the next is awaited,
so Time-to-First-Token is preserved (Chapter 3) and the chunk-boundary leak is
closed (Chapter 4).
"""
from __future__ import annotations

import json
import re
import time
from enum import Enum
from typing import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Multi-Cloud Abstraction Gateway")

SSE_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",          # defeat reverse-proxy buffering
}


# --------------------------------------------------------------------------- #
#  PII redaction (structured, deterministic — Chapter 4 tier 1)
# --------------------------------------------------------------------------- #
class PIIKind(str, Enum):
    SSN = "ssn"
    PAN = "pan"
    SECRET = "api_key"


# Compiled once at import. Structured PII only; unstructured -> async NER.
_PATTERNS: list[tuple[PIIKind, re.Pattern]] = [
    (PIIKind.SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (PIIKind.PAN, re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    (PIIKind.SECRET, re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")),
]
_LONGEST_PATTERN = 24
_SAFETY_MARGIN = 8
_HOLDBACK = _LONGEST_PATTERN + _SAFETY_MARGIN   # chars withheld across a boundary


def _luhn_valid(candidate: str) -> bool:
    """Validate a payment-card number with the Luhn checksum to suppress
    false positives on arbitrary 13-19 digit runs."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def redact(text: str) -> str:
    """Replace structured PII spans with typed labels. PAN spans are redacted
    only when they pass the Luhn check, so ordinary long digit runs survive."""
    for kind, pattern in _PATTERNS:
        def _replace(match: re.Match, _kind: PIIKind = kind) -> str:
            span = match.group()
            if _kind is PIIKind.PAN and not _luhn_valid(span):
                return span
            return f"[REDACTED-{_kind.value.upper()}]"
        text = pattern.sub(_replace, text)
    return text


# --------------------------------------------------------------------------- #
#  Provider delta extraction (response-side normalization)
# --------------------------------------------------------------------------- #
def extract_delta(obj: dict, provider: str) -> str:
    """Pull the incremental text from one provider SSE frame. Returns '' when
    the frame carries no text (e.g. role or tool-call deltas)."""
    if provider == "azure" or provider == "oci":
        choices = obj.get("choices") or [{}]
        return choices[0].get("delta", {}).get("content") or ""
    if provider == "aws":
        return obj.get("delta", {}).get("text", "")
    if provider == "gcp":
        cands = obj.get("candidates") or [{}]
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if "text" in p)
    return obj.get("text", "")


# --------------------------------------------------------------------------- #
#  Streaming transform with sliding-window redaction
# --------------------------------------------------------------------------- #
async def transform_and_sanitize(
    upstream: AsyncIterator[bytes],
    provider: str,
) -> AsyncIterator[bytes]:
    """Re-frame a provider SSE stream into unified frames and redact structured
    PII across chunk boundaries. Holds back at most `_HOLDBACK` characters; never
    buffers the full completion, so TTFT is preserved.
    """
    parse_buffer = ""       # accumulates raw bytes until a full SSE event
    carry = ""              # unresolved suffix carried across output frames

    async for raw in upstream:
        parse_buffer += raw.decode("utf-8", errors="ignore")

        while "\n\n" in parse_buffer:                 # one complete SSE event
            event, parse_buffer = parse_buffer.split("\n\n", 1)
            line = event.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()

            if payload == "[DONE]":
                if carry:                             # flush the final tail
                    yield f"data: {json.dumps({'delta': redact(carry)})}\n\n".encode()
                    carry = ""
                yield b"data: [DONE]\n\n"
                return

            try:
                delta = extract_delta(json.loads(payload), provider)
            except json.JSONDecodeError:
                continue                              # skip malformed frame
            if not delta:
                continue

            window = carry + delta
            safe, carry = window[:-_HOLDBACK], window[-_HOLDBACK:]
            if safe:
                out = json.dumps({"delta": redact(safe)})
                yield f"data: {out}\n\n".encode()     # flush immediately

    if carry:                                         # stream ended without [DONE]
        yield f"data: {json.dumps({'delta': redact(carry)})}\n\n".encode()


# --------------------------------------------------------------------------- #
#  FastAPI streaming endpoint
# --------------------------------------------------------------------------- #
OpenStream = Callable[[dict], AsyncIterator[bytes]]


async def _resolve_upstream(body: dict) -> tuple[str, AsyncIterator[bytes]]:
    """Resolve the request to a provider and an open upstream byte stream.
    In production this consults the router (Chapter 3); here it is injected."""
    endpoint = app.state.router.select(body["model_role"])   # raises if none
    provider = endpoint.binding.provider
    upstream = endpoint.adapter.open_stream(body, endpoint.binding)
    return provider, upstream


@app.post("/v1/inference:stream")
async def stream_inference(request: Request) -> StreamingResponse:
    """Stream a provider-neutral inference response as sanitized SSE frames."""
    try:
        body = await request.json()
    except Exception as exc:                          # malformed request body
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if "model_role" not in body:
        raise HTTPException(status_code=422, detail="model_role is required")

    try:
        provider, upstream = await _resolve_upstream(body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown model_role") from exc
    except RuntimeError as exc:                        # all endpoints unavailable
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def generator() -> AsyncIterator[bytes]:
        started = time.monotonic()
        first = True
        try:
            async for frame in transform_and_sanitize(upstream, provider):
                if first:
                    app.state.metrics.record_ttft(provider, time.monotonic() - started)
                    first = False
                yield frame
        except Exception:                              # upstream failure mid-stream
            yield b'data: {"error": "upstream_interrupted"}\n\n'
            yield b"data: [DONE]\n\n"

    return StreamingResponse(generator(), headers=SSE_HEADERS)
