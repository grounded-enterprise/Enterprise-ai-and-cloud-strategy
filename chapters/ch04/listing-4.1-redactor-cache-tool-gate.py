# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 4.1 — Sliding redactor, semantic cache, and tool-gate engine.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
import re
import hashlib
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any, Type
from pydantic import BaseModel, Field

# --- MODULE 1: SLIDING-WINDOW PII REDACTION ---

class PIIKind(str, Enum):
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    API_KEY = "api_key"

_PATTERNS = [
    (PIIKind.SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (PIIKind.CREDIT_CARD, re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    (PIIKind.API_KEY, re.compile(r"\b(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")),
]

_HOLDBACK_WINDOW = 32

def verify_luhn_checksum(card_num: str) -> bool:
    digits = [int(c) for c in card_num if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for idx, digit in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0

def apply_redaction(text: str) -> str:
    for kind, pattern in _PATTERNS:
        def replace_match(match: re.Match) -> str:
            val = match.group()
            if kind == PIIKind.CREDIT_CARD and not verify_luhn_checksum(val):
                return val
            return f"[REDACTED-{kind.value.upper()}]"
        text = pattern.sub(replace_match, text)
    return text

async def redact_stream_pipeline(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    trailing_buffer = ""
    async for chunk in chunks:
        combined = trailing_buffer + chunk
        if len(combined) > _HOLDBACK_WINDOW:
            safe_segment = combined[:-_HOLDBACK_WINDOW]
            trailing_buffer = combined[-_HOLDBACK_WINDOW:]
            yield apply_redaction(safe_segment)
        else:
            trailing_buffer = combined

    if trailing_buffer:
        yield apply_redaction(trailing_buffer)

# --- MODULE 2: SEMANTIC CACHE LAYER ---

class CacheScopeKey(BaseModel):
    tenant_id: str
    model_role: str
    residency: str = "US"

    def compute_exact_hash(self, prompt: str) -> str:
        normalized = " ".join(prompt.lower().split())
        raw_key = f"{self.tenant_id}:{self.model_role}:{self.residency}:{normalized}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

class MockVectorCacheStore:
    def __init__(self):
        self._exact_store: Dict[str, str] = {}

    def fetch_exact(self, hash_key: str) -> Optional[str]:
        return self._exact_store.get(hash_key)

    def save_exact(self, hash_key: str, completion: str) -> None:
        self._exact_store[hash_key] = completion

class ProxySemanticCache:
    def __init__(self, store: MockVectorCacheStore, similarity_threshold: float = 0.92):
        self.store = store
        self.threshold = similarity_threshold

    def resolve_cache(self, key: CacheScopeKey, prompt: str) -> Optional[str]:
        exact_hash = key.compute_exact_hash(prompt)
        exact_match = self.store.fetch_exact(exact_hash)
        if exact_match:
            return exact_match
        return None

    def store_completion(self, key: CacheScopeKey, prompt: str, completion: str) -> None:
        exact_hash = key.compute_exact_hash(prompt)
        self.store.save_exact(exact_hash, completion)

# --- MODULE 3: BOUNDED AUTONOMY TOOL GATE ---

class ToolRiskTier(str, Enum):
    T0_READ = "t0"
    T1_IDEMPOTENT = "t1"
    T2_SENSITIVE = "t2"
    T3_CRITICAL = "t3"

class ValidatedToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class ToolDefinitionSpec(BaseModel):
    name: str
    risk_tier: ToolRiskTier
    schema_model: Type[BaseModel]

class ToolExecutionGate:
    def __init__(self, registry: Dict[str, ToolDefinitionSpec]):
        self.registry = registry

    def evaluate_and_authorize(self, call: ValidatedToolCall, user_approvals: List[str]) -> bool:
        spec = self.registry.get(call.tool_name)
        if not spec:
            raise PermissionError(f"Tool '{call.tool_name}' is not authorized on this proxy.")

        # Strict Pydantic Schema Validation
        spec.schema_model.model_validate(call.parameters)

        if spec.risk_tier in (ToolRiskTier.T0_READ, ToolRiskTier.T1_IDEMPOTENT):
            return True

        if spec.risk_tier == ToolRiskTier.T2_SENSITIVE and len(user_approvals) >= 1:
            return True

        if spec.risk_tier == ToolRiskTier.T3_CRITICAL and len(user_approvals) >= 2:
            return True

        return False
