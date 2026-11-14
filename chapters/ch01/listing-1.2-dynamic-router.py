# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 1.2 — Dynamic router implementation (Pydantic v2).
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class Provider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    OCI = "oci"

class RoutingConstraints(BaseModel):
    residency: str = "US"
    zero_data_retention: bool = True
    max_p95_latency_ms: int = 1500
    cost_weight: float = Field(0.5, ge=0.0, le=1.0)

class Message(BaseModel):
    role: str
    content: str

class UnifiedInferenceRequest(BaseModel):
    model_role: str
    messages: List[Message]
    max_output_tokens: int = Field(1024, ge=1)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    stream: bool = False
    routing_constraints: RoutingConstraints = RoutingConstraints()

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, value: List[Message]) -> List[Message]:
        if not value:
            raise ValueError("Inference request requires at least one input message.")
        return value

class ModelBinding(BaseModel):
    provider: Provider
    model_id: str
    region: str
    usd_per_1m_input: float
    usd_per_1m_output: float
    measured_p95_ms: int
    supports_zdr: bool
    healthy: bool = True
    quota_remaining: int = 1_000_000

class ModelRole(BaseModel):
    role: str
    bindings: List[ModelBinding]

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    computed_cost_usd: float

class ResolvedBinding(BaseModel):
    provider: Provider
    model_id: str
    region: str

class UnifiedInferenceResponse(BaseModel):
    content: str
    finish_reason: str
    usage: Usage
    resolved_binding: ResolvedBinding

class ProviderAdapter(Protocol):
    provider: Provider
    async def invoke(self, req: UnifiedInferenceRequest, binding: ModelBinding) -> UnifiedInferenceResponse: ...

class RoutingException(Exception):
    """Base exception for dynamic routing errors."""
    pass

class PolicyRouter:
    """Ranks and selects candidate provider bindings based on cost, latency, and health."""
    def __init__(self, registry: Dict[str, ModelRole], adapters: Dict[Provider, ProviderAdapter]):
        self._registry = registry
        self._adapters = adapters

    def _calculate_score(self, binding: ModelBinding, constraints: RoutingConstraints) -> float:
        total_cost_per_1m = binding.usd_per_1m_input + binding.usd_per_1m_output
        normalized_cost = total_cost_per_1m * constraints.cost_weight
        normalized_latency = (binding.measured_p95_ms / 1000.0) * (1.0 - constraints.cost_weight)
        return normalized_cost + normalized_latency

    def resolve_candidates(self, req: UnifiedInferenceRequest) -> List[ModelBinding]:
        role = self._registry.get(req.model_role)
        if not role:
            raise RoutingException(f"Unrecognized model role: {req.model_role}")

        c = req.routing_constraints
        qualified = [
            b for b in role.bindings
            if b.healthy
            and b.quota_remaining > 0
            and b.measured_p95_ms <= c.max_p95_latency_ms
            and (b.supports_zdr or not c.zero_data_retention)
        ]

        if not qualified:
            raise RoutingException(f"No providers satisfy active routing policies for role: {req.model_role}")

        return sorted(qualified, key=lambda b: self._calculate_score(b, c))

    async def execute_route(self, req: UnifiedInferenceRequest) -> UnifiedInferenceResponse:
        candidates = self.resolve_candidates(req)
        last_exception: Optional[Exception] = None

        for candidate in candidates:
            adapter = self._adapters.get(candidate.provider)
            if not adapter:
                continue
            try:
                return await adapter.invoke(req, candidate)
            except Exception as exc:
                last_exception = exc
                candidate.healthy = False  # Mark temporarily unhealthy on failure
                continue

        raise RoutingException("All target providers failed execution.") from last_exception
