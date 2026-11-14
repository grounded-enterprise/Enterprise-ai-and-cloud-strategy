# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 10.4b — otel_instrumentation.py : GenAI semantic-convention + COGS telemetry.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

"""OpenTelemetry instrumentation for the gateway (Datadog / Prometheus).

Emits the CNCF-standard GenAI semantic-convention signals plus the platform's
unit-economics metrics: prompt/completion tokens, Time-to-First-Token, Tokens-
Per-Second, and attributed tenant COGS. Because the vocabulary is standard, one
dashboard compares AWS, Azure, GCP, and OCI without per-vendor glue.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from opentelemetry import metrics, trace

tracer = trace.get_tracer("ai.gateway")
meter = metrics.get_meter("ai.gateway")

# --- Standard GenAI + unit-economics metric instruments ------------------- #
_token_usage = meter.create_histogram(
    name="gen_ai.client.token.usage",
    unit="{token}",
    description="Token consumption, split by gen_ai.token.type.",
)
_ttft = meter.create_histogram(
    name="gen_ai.server.time_to_first_token",
    unit="s",
    description="Time from request to first streamed token.",
)
_tps = meter.create_histogram(
    name="gen_ai.server.tokens_per_second",
    unit="{token}/s",
    description="Sustained output token throughput.",
)
_cogs = meter.create_histogram(
    name="ai.gateway.request.cost_usd",
    unit="USD",
    description="Attributed cost of goods sold per request.",
)


@dataclass(frozen=True)
class InferenceObservation:
    """One completed inference, ready to be recorded as telemetry."""
    provider: str
    request_model: str
    response_model: str
    operation: str                 # "chat" | "embeddings" | ...
    tenant: str
    feature: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    ttft_s: float
    duration_s: float
    cost_usd: Decimal
    finish_reason: str

    @property
    def tokens_per_second(self) -> float:
        """Sustained decode rate; guards against a zero-duration divide."""
        decode_window = max(self.duration_s - self.ttft_s, 1e-6)
        return self.output_tokens / decode_window


def _base_attributes(obs: InferenceObservation) -> dict[str, str]:
    """Low-cardinality attribute set shared by every metric (OTel GenAI keys)."""
    return {
        "gen_ai.provider.name": obs.provider,
        "gen_ai.request.model": obs.request_model,
        "gen_ai.operation.name": obs.operation,
        "tenant": obs.tenant,
        "feature": obs.feature,
    }


def record_inference(obs: InferenceObservation) -> None:
    """Record one inference as an OTel span plus GenAI + COGS metrics.

    Safe to call on the hot path: metric recording is non-blocking, and any
    exporter failure is swallowed so telemetry never degrades the request.
    """
    attrs = _base_attributes(obs)
    span_attrs = {
        **attrs,
        "gen_ai.response.model": obs.response_model,
        "gen_ai.response.finish_reasons": [obs.finish_reason],
    }
    with tracer.start_as_current_span(obs.operation, attributes=span_attrs):
        try:
            _token_usage.record(obs.input_tokens, {**attrs, "gen_ai.token.type": "input"})
            _token_usage.record(obs.output_tokens, {**attrs, "gen_ai.token.type": "output"})
            if obs.reasoning_tokens:
                _token_usage.record(
                    obs.reasoning_tokens, {**attrs, "gen_ai.token.type": "reasoning"})
            _ttft.record(obs.ttft_s, attrs)
            _tps.record(obs.tokens_per_second, attrs)
            _cogs.record(float(obs.cost_usd), attrs)
        except Exception:                     # never let telemetry fail a request
            trace.get_current_span().add_event("telemetry_record_failed")
