# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 2.1 — Multi-provider response normalizer (Pydantic v2).
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class FinishReason(str, Enum):
    STOP = "stop"
    TRUNCATED = "truncated"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    ERROR = "error"

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_billable_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]

class NormalizedResponse(BaseModel):
    provider: str
    model_id: str
    content: str
    finish_reason: FinishReason
    tool_calls: List[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_payload: Dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def was_truncated(self) -> bool:
        return self.finish_reason == FinishReason.TRUNCATED

class ResponseNormalizer:
    """Translates provider-specific JSON responses into standardized NormalizedResponse models."""

    @staticmethod
    def normalize_aws_bedrock(raw: Dict[str, Any], model_id: str) -> NormalizedResponse:
        output_msg = raw.get("output", {}).get("message", {})
        content_blocks = output_msg.get("content", [])
        text_content = "".join([b.get("text", "") for b in content_blocks if "text" in b])

        tool_calls = [
            ToolCall(
                id=b["toolUse"]["toolUseId"],
                name=b["toolUse"]["name"],
                arguments=b["toolUse"].get("input", {})
            )
            for b in content_blocks if "toolUse" in b
        ]

        usage_raw = raw.get("usage", {})
        stop_reason = raw.get("stopReason", "end_turn")

        reason_map = {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.TRUNCATED,
            "tool_use": FinishReason.TOOL_CALL,
            "content_filtered": FinishReason.CONTENT_FILTER
        }

        return NormalizedResponse(
            provider="aws",
            model_id=model_id,
            content=text_content,
            finish_reason=reason_map.get(stop_reason, FinishReason.STOP),
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=usage_raw.get("inputTokens", 0),
                output_tokens=usage_raw.get("outputTokens", 0)
            ),
            raw_payload=raw
        )

    @staticmethod
    def normalize_azure_openai(raw: Dict[str, Any], model_id: str) -> NormalizedResponse:
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        import json
        tool_calls = []
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "{}")
            args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args_dict))

        usage_raw = raw.get("usage", {})
        details = usage_raw.get("completion_tokens_details", {}) or {}
        prompt_details = usage_raw.get("prompt_tokens_details", {}) or {}

        reason_map = {
            "stop": FinishReason.STOP,
            "length": FinishReason.TRUNCATED,
            "tool_calls": FinishReason.TOOL_CALL,
            "content_filter": FinishReason.CONTENT_FILTER
        }

        return NormalizedResponse(
            provider="azure",
            model_id=model_id,
            content=content,
            finish_reason=reason_map.get(choice.get("finish_reason"), FinishReason.STOP),
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=usage_raw.get("prompt_tokens", 0),
                output_tokens=usage_raw.get("completion_tokens", 0),
                reasoning_tokens=details.get("reasoning_tokens", 0),
                cached_input_tokens=prompt_details.get("cached_tokens", 0)
            ),
            raw_payload=raw
        )

    @staticmethod
    def normalize_gcp_vertex(raw: Dict[str, Any], model_id: str) -> NormalizedResponse:
        candidate = (raw.get("candidates") or [{}])[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        text_content = "".join([p.get("text", "") for p in content_parts if "text" in p])

        tool_calls = [
            ToolCall(
                id=p["functionCall"].get("name", ""),
                name=p["functionCall"].get("name", ""),
                arguments=p["functionCall"].get("args", {})
            )
            for p in content_parts if "functionCall" in p
        ]

        usage_raw = raw.get("usageMetadata", {})
        finish_reason_raw = candidate.get("finishReason", "STOP")

        reason_map = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.TRUNCATED,
            "SAFETY": FinishReason.CONTENT_FILTER,
            "RECITATION": FinishReason.CONTENT_FILTER
        }

        mapped_reason = reason_map.get(finish_reason_raw, FinishReason.STOP)
        if tool_calls and mapped_reason == FinishReason.STOP:
            mapped_reason = FinishReason.TOOL_CALL

        return NormalizedResponse(
            provider="gcp",
            model_id=model_id,
            content=text_content,
            finish_reason=mapped_reason,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=usage_raw.get("promptTokenCount", 0),
                output_tokens=usage_raw.get("candidatesTokenCount", 0),
                reasoning_tokens=usage_raw.get("thoughtsTokenCount", 0)
            ),
            raw_payload=raw
        )
