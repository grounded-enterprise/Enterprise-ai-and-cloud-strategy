# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 6.1 — Unified Telemetry Event (Pydantic): normalized measures, full attribution key, cost priced at construction.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import BaseModel, Field, computed_field


class PriceBook(BaseModel):
    version: str
    usd_per_m_input: Decimal
    usd_per_m_output: Decimal
    usd_per_m_cached: Decimal = Decimal("0")
    usd_per_m_reasoning: Decimal = Decimal("0")
    usd_per_retrieval_op: Decimal = Decimal("0")
    usd_per_compute_s: Decimal = Decimal("0")


class TelemetryEvent(BaseModel):
    model_config = {"extra": "forbid"}          # strict: no stray fields
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # --- dimensions (attribution key) ---
    tenant: str
    department: str
    feature: str
    user: str
    provider: str
    model: str
    region: str
    # --- measures (normalized across providers) ---
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    retrieval_ops: int = 0
    compute_ms: int = 0
    # --- pricing provenance ---
    price_book_version: str

    @computed_field
    @property
    def cost_usd(self) -> Decimal:
        return self._cost

    _cost: Decimal = Decimal("0")

    def price(self, book: PriceBook) -> "TelemetryEvent":
        M = Decimal(1_000_000)
        self._cost = (
            Decimal(self.prompt_tokens)     / M * book.usd_per_m_input
            + Decimal(self.completion_tokens) / M * book.usd_per_m_output
            + Decimal(self.cached_tokens)     / M * book.usd_per_m_cached
            + Decimal(self.reasoning_tokens)  / M * book.usd_per_m_reasoning
            + Decimal(self.retrieval_ops)         * book.usd_per_retrieval_op
            + Decimal(self.compute_ms) / Decimal(1000) * book.usd_per_compute_s
        )
        self.price_book_version = book.version
        return self
