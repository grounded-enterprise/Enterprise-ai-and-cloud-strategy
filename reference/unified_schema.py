# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 10.1 — unified_schema.py : provider-neutral request/response models (Pydantic v2).
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

"""Unified multi-cloud payload schema (Pydantic v2).

The single request/response vocabulary the gateway speaks to its callers.
Callers name a model ROLE and an intent; adapters (Chapter 3) translate this
schema into the native dialects of AWS Bedrock, Azure OpenAI, GCP Vertex AI,
and OCI Generative AI. Nothing above the adapter boundary references a provider.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
#  Enumerations
# --------------------------------------------------------------------------- #
class Provider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    OCI = "oci"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """One vocabulary. TRUNCATED is first-class and never folded into STOP."""
    STOP = "stop"
    TRUNCATED = "truncated"           # hit max output tokens (provider 'length')
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    ERROR = "error"


class ReasoningEffort(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --------------------------------------------------------------------------- #
#  Content & messages
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    """A single conversational turn. System instructions use role=system and
    are relocated by each adapter to the provider's required position."""
    model_config = {"extra": "forbid"}

    role: Role
    content: str = Field(..., description="UTF-8 message text.")
    name: Optional[str] = Field(default=None, description="Tool/function name for role=tool.")

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if v is None:
            raise ValueError("content must not be None")
        return v


# --------------------------------------------------------------------------- #
#  Tool calling (function invocation)
# --------------------------------------------------------------------------- #
class ToolParameter(BaseModel):
    """JSON-Schema fragment describing one tool argument."""
    model_config = {"extra": "forbid"}

    type: Literal["string", "number", "integer", "boolean", "object", "array"]
    description: Optional[str] = None
    enum: Optional[list[Any]] = None


class ToolSpec(BaseModel):
    """A callable tool the model may propose. The parameters block is a strict
    JSON Schema; the gateway validates any proposed call against it (Chapter 4)
    before invocation."""
    model_config = {"extra": "forbid"}

    name: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]{1,64}$")
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_subset_of_params(self) -> "ToolSpec":
        unknown = set(self.required) - set(self.parameters)
        if unknown:
            raise ValueError(f"required names not in parameters: {sorted(unknown)}")
        return self


class ToolChoice(BaseModel):
    """How the model should treat tools: auto, none, or a forced tool."""
    model_config = {"extra": "forbid"}

    mode: Literal["auto", "none", "required"] = "auto"
    forced_tool: Optional[str] = None

    @model_validator(mode="after")
    def forced_requires_required(self) -> "ToolChoice":
        if self.forced_tool and self.mode != "required":
            raise ValueError("forced_tool requires mode='required'")
        return self


class ToolCall(BaseModel):
    """A tool invocation proposed by the model (response side). Arguments are
    always a parsed object, never a raw JSON string, regardless of provider."""
    model_config = {"extra": "forbid"}

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Structured output & reasoning
# --------------------------------------------------------------------------- #
class ResponseFormat(BaseModel):
    """Requests structured output. When kind='json_schema' the model is
    constrained to emit JSON validating against `json_schema`."""
    model_config = {"extra": "forbid"}

    kind: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def schema_present_when_needed(self) -> "ResponseFormat":
        if self.kind == "json_schema" and not self.json_schema:
            raise ValueError("kind='json_schema' requires json_schema")
        return self


class ReasoningConfig(BaseModel):
    """Neutral extended-thinking control. Each adapter maps effort onto the
    provider mechanism (token budget, effort enum, or thinkingConfig)."""
    model_config = {"extra": "forbid"}

    effort: ReasoningEffort = ReasoningEffort.NONE
    max_reasoning_tokens: Optional[int] = Field(default=None, ge=1)


# --------------------------------------------------------------------------- #
#  Routing constraints
# --------------------------------------------------------------------------- #
class RoutingConstraints(BaseModel):
    """Policy attached to a request. residency and zero_data_retention are
    enforced by removing non-compliant endpoints from the router set."""
    model_config = {"extra": "forbid"}

    residency: str = "US"
    zero_data_retention: bool = True
    max_p95_latency_ms: int = Field(default=1500, ge=1)
    cost_weight: float = Field(default=0.5, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
#  Unified request
# --------------------------------------------------------------------------- #
class UnifiedRequest(BaseModel):
    """The provider-neutral request. Callers specify a model_role, never a
    provider; binding to a concrete endpoint is resolved at runtime."""
    model_config = {"extra": "forbid"}

    model_role: str = Field(..., min_length=1, description="Logical capability, e.g. 'reasoning-tier'.")
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: ToolChoice = Field(default_factory=ToolChoice)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    max_output_tokens: int = Field(default=1024, ge=1, le=200_000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stop: list[str] = Field(default_factory=list)
    stream: bool = False
    routing_constraints: RoutingConstraints = Field(default_factory=RoutingConstraints)
    idempotency_key: Optional[str] = Field(default=None, description="Stable key for exactly-once side effects.")

    @field_validator("messages")
    @classmethod
    def non_empty_messages(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        return v

    @model_validator(mode="after")
    def forced_tool_is_declared(self) -> "UnifiedRequest":
        if self.tool_choice.forced_tool:
            names = {t.name for t in self.tools}
            if self.tool_choice.forced_tool not in names:
                raise ValueError("forced_tool must be a declared tool")
        return self


# --------------------------------------------------------------------------- #
#  Unified response
# --------------------------------------------------------------------------- #
class TokenUsage(BaseModel):
    """Normalized token accounting across providers."""
    model_config = {"extra": "forbid"}

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def billable_output(self) -> int:
        return self.output_tokens + self.reasoning_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


class ResolvedBinding(BaseModel):
    """The concrete endpoint that served the request (provenance)."""
    model_config = {"extra": "forbid"}

    provider: Provider
    model_id: str
    region: str


class UnifiedResponse(BaseModel):
    """The provider-neutral response. Application code observes identical
    semantics regardless of which provider served the request."""
    model_config = {"extra": "forbid"}

    content: str
    finish_reason: FinishReason
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    resolved_binding: ResolvedBinding

    @property
    def is_truncated(self) -> bool:
        return self.finish_reason is FinishReason.TRUNCATED
