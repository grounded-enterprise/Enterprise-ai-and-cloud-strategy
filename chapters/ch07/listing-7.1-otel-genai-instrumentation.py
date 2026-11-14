# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 7.1 — OpenTelemetry GenAI instrumentation: standard gen_ai.* span attributes and token/TTFT metrics.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
from dataclasses import dataclass, field
from opentelemetry import trace, metrics

tracer = trace.get_tracer("ai.gateway")
meter = metrics.get_meter("ai.gateway")

# --- OTel GenAI metrics (standard semantic conventions) ---
token_usage = meter.create_histogram(
    "gen_ai.client.token.usage", unit="{token}",
    description="Token consumption, split by gen_ai.token.type")
ttft = meter.create_histogram(
    "gen_ai.server.time_to_first_token", unit="s",
    description="Time to first streamed token")
op_duration = meter.create_histogram(
    "gen_ai.client.operation.duration", unit="s")


@dataclass
class GenAISpanAttributes:
    """OTel GenAI semantic-convention attribute keys (gen_ai.*)."""
    provider_name: str          # gen_ai.provider.name
    request_model: str          # gen_ai.request.model
    operation_name: str         # gen_ai.operation.name  (chat|embeddings|...)
    response_model: str = ""    # gen_ai.response.model  (as actually served)
    finish_reasons: tuple[str, ...] = ()   # gen_ai.response.finish_reasons

    def as_otel(self) -> dict[str, object]:
        return {
            "gen_ai.provider.name": self.provider_name,
            "gen_ai.request.model": self.request_model,
            "gen_ai.operation.name": self.operation_name,
            "gen_ai.response.model": self.response_model or self.request_model,
            "gen_ai.response.finish_reasons": list(self.finish_reasons),
        }


def record_inference(attrs: GenAISpanAttributes, in_tok: int, out_tok: int,
                     ttft_s: float, dur_s: float) -> None:
    with tracer.start_as_current_span("chat", attributes=attrs.as_otel()):
        base = {"gen_ai.provider.name": attrs.provider_name,
                "gen_ai.request.model": attrs.request_model}
        token_usage.record(in_tok,  {**base, "gen_ai.token.type": "input"})
        token_usage.record(out_tok, {**base, "gen_ai.token.type": "output"})
        ttft.record(ttft_s, base)
        op_duration.record(dur_s, base)
