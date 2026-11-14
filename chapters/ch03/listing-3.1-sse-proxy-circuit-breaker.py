# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 3.1 — Async SSE streaming proxy & circuit-breaker engine (FastAPI).
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class EndpointCircuitBreaker:
    failure_threshold: int = 5
    cooldown_seconds: float = 10.0
    half_open_probe_limit: int = 2

    _state: BreakerState = BreakerState.CLOSED
    _failure_count: int = 0
    _opened_timestamp: float = 0.0
    _active_probes: int = 0

    def can_execute(self) -> bool:
        now = time.monotonic()
        if self._state == BreakerState.OPEN:
            if now - self._opened_timestamp >= self.cooldown_seconds:
                self._state = BreakerState.HALF_OPEN
                self._active_probes = 0
            else:
                return False

        if self._state == BreakerState.HALF_OPEN:
            if self._active_probes >= self.half_open_probe_limit:
                return False
            self._active_probes += 1

        return True

    def record_success(self) -> None:
        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._active_probes = 0

    def record_failure(self, error_type: str) -> None:
        # Ignore moderation blocks and rate-limiting from trip metrics
        if error_type in ("content_filter", "rate_limit"):
            return

        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_timestamp = time.monotonic()

@dataclass
class ManagedEndpoint:
    endpoint_id: str
    provider: str
    model_id: str
    cost_per_1k_tokens: float
    circuit_breaker: EndpointCircuitBreaker = field(default_factory=EndpointCircuitBreaker)
    ewma_ttft_seconds: float = 0.35
    alpha: float = 0.2

    def update_ttft(self, sample_seconds: float) -> None:
        self.ewma_ttft_seconds = (self.alpha * sample_seconds) + ((1.0 - self.alpha) * self.ewma_ttft_seconds)

class LatencyAwareRouter:
    def __init__(self, endpoint_registry: Dict[str, List[ManagedEndpoint]], latency_weight: float = 0.6):
        self._registry = endpoint_registry
        self.latency_weight = latency_weight
        self.cost_weight = 1.0 - latency_weight

    def _score_endpoint(self, ep: ManagedEndpoint) -> float:
        health_penalty = 0.0 if ep.circuit_breaker._state == BreakerState.CLOSED else 100.0
        return (self.latency_weight * ep.ewma_ttft_seconds) + (self.cost_weight * ep.cost_per_1k_tokens) + health_penalty

    def select_best_endpoint(self, role: str) -> ManagedEndpoint:
        candidates = self._registry.get(role, [])
        available = [ep for ep in candidates if ep.circuit_breaker.can_execute()]

        if not available:
            raise HTTPException(status_code=503, detail=f"All upstream endpoints for role '{role}' are unavailable.")

        return min(available, key=self._score_endpoint)

app = FastAPI()

async def transform_provider_stream(upstream_bytes: AsyncIterator[bytes], provider: str) -> AsyncIterator[bytes]:
    """Processes SSE chunks incrementally to prevent buffering and maintain low TTFT."""
    buffer = ""
    async for chunk in upstream_bytes:
        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n\n" in buffer:
            event_block, buffer = buffer.split("\n\n", 1)
            lines = event_block.strip().split("\n")

            for line in lines:
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    yield b"data: [DONE]\n\n"
                    return

                try:
                    parsed = json.loads(payload)
                    delta_text = ""
                    if provider == "azure":
                        delta_text = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    elif provider == "aws":
                        delta_text = parsed.get("delta", {}).get("text", "")
                    elif provider == "gcp":
                        parts = parsed.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        delta_text = "".join([p.get("text", "") for p in parts])

                    if delta_text:
                        out_frame = json.dumps({"delta": delta_text})
                        yield f"data: {out_frame}\n\n".encode("utf-8")
                except json.JSONDecodeError:
                    continue

@app.post("/v1/inference/stream")
async def stream_inference(request: Request):
    body = await request.json()
    role = body.get("model_role", "default")

    selected_ep = router.select_best_endpoint(role)

    # Mocking upstream provider stream acquisition
    async def mock_upstream_stream():
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Enterprise "}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"AI "}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Architecture."}}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        for c in chunks:
            await asyncio.sleep(0.05)
            yield c

    async def response_generator():
        start_time = time.monotonic()
        first_token_recorded = False

        try:
            async for frame in transform_provider_stream(mock_upstream_stream(), selected_ep.provider):
                if not first_token_recorded:
                    ttft = time.monotonic() - start_time
                    selected_ep.update_ttft(ttft)
                    selected_ep.circuit_breaker.record_success()
                    first_token_recorded = True
                yield frame
        except Exception as exc:
            selected_ep.circuit_breaker.record_failure("connectivity")
            raise exc

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
