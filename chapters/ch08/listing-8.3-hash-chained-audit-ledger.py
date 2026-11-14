# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 8.3 — Tamper-evident hash-chained audit ledger (Pydantic v2, HMAC-signed).
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
import hmac
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class AuditRecordPayload(BaseModel):
    event_id: str
    tenant_id: str
    model_role: str
    resolved_provider: str
    resolved_model_id: str
    resolved_region: str
    zero_data_retention: bool
    input_hash: str   # SHA-256 of normalized input prompt
    output_hash: str  # SHA-256 of normalized output completion
    input_tokens: int
    output_tokens: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChainEntry(BaseModel):
    sequence_number: int
    timestamp: str
    payload: AuditRecordPayload
    previous_block_hash: str
    block_hash: str
    hmac_signature: str


class CryptographicAuditLedger:
    """Tamper-evident, hash-chained audit ledger with authenticated integrity for AI inference."""

    def __init__(self, hmac_key: bytes, genesis_seed: str = "GENESIS_BLOCK_SEED_2026"):
        self._hmac_key = hmac_key
        self._last_block_hash = hashlib.sha256(genesis_seed.encode("utf-8")).hexdigest()
        self._sequence_counter = 0

    def _compute_payload_hash(self, payload: AuditRecordPayload) -> str:
        serialized = payload.model_dump_json(by_alias=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def append_event(self, payload: AuditRecordPayload) -> ChainEntry:
        self._sequence_counter += 1
        now_str = datetime.now(timezone.utc).isoformat()

        payload_hash = self._compute_payload_hash(payload)
        block_contents = f"{self._sequence_counter}|{now_str}|{self._last_block_hash}|{payload_hash}"
        block_hash = hashlib.sha256(block_contents.encode("utf-8")).hexdigest()

        # Sign block hash with HSM-derived HMAC key
        signature = hmac.new(self._hmac_key, block_hash.encode("utf-8"), hashlib.sha256).hexdigest()

        entry = ChainEntry(
            sequence_number=self._sequence_counter,
            timestamp=now_str,
            payload=payload,
            previous_block_hash=self._last_block_hash,
            block_hash=block_hash,
            hmac_signature=signature,
        )

        # Advance chain tip
        self._last_block_hash = block_hash
        return entry

    @staticmethod
    def verify_entry(entry: ChainEntry, previous_entry_hash: str, hmac_key: bytes) -> bool:
        """Cryptographically validates a block entry against its predecessor and signature."""
        payload_hash = hashlib.sha256(
            entry.payload.model_dump_json(by_alias=True).encode("utf-8")).hexdigest()
        expected_block_contents = (
            f"{entry.sequence_number}|{entry.timestamp}|{previous_entry_hash}|{payload_hash}")
        expected_block_hash = hashlib.sha256(
            expected_block_contents.encode("utf-8")).hexdigest()

        if expected_block_hash != entry.block_hash:
            return False

        expected_sig = hmac.new(
            hmac_key, entry.block_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected_sig, entry.hmac_signature)
