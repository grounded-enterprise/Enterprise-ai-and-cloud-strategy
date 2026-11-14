# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 8.2 — Tamper-evident audit log: hash-chained, HMAC-signed, content-free; verify() detects any alteration.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
import hashlib, hmac, json, os, time
from dataclasses import dataclass, asdict


def _sha256(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def salted_hash(text: str, salt: bytes) -> str:
    # store a salted hash of content, never the content itself
    return _sha256(salt, text.encode("utf-8"))


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    ts: float
    input_hash: str            # salted hash of the prompt
    output_hash: str           # salted hash of the completion
    policy_results: dict       # which guardrails fired / gates passed
    user: str
    tenant: str
    prev_hash: str             # hash of the previous entry -> chain
    entry_hash: str = ""       # hash of this entry's canonical body
    hmac_sig: str = ""         # HMAC over entry_hash (HSM key in prod)

    def body(self) -> bytes:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("entry_hash", "hmac_sig")}
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


class AuditLog:
    """Append-only, hash-chained, HMAC-signed. Stores no raw PII/PHI."""

    GENESIS = "0" * 64

    def __init__(self, hmac_key: bytes) -> None:
        self._key = hmac_key            # in prod: KMS GENERATE_VERIFY_MAC
        self._chain: list[AuditEntry] = []

    def _last_hash(self) -> str:
        return self._chain[-1].entry_hash if self._chain else self.GENESIS

    def append(self, input_hash: str, output_hash: str,
               policy_results: dict, user: str, tenant: str) -> AuditEntry:
        draft = AuditEntry(
            seq=len(self._chain), ts=time.time(),
            input_hash=input_hash, output_hash=output_hash,
            policy_results=policy_results, user=user, tenant=tenant,
            prev_hash=self._last_hash(),
        )
        entry_hash = _sha256(draft.body())
        sig = hmac.new(self._key, entry_hash.encode(), hashlib.sha256).hexdigest()
        entry = AuditEntry(**{**asdict(draft),
                              "entry_hash": entry_hash, "hmac_sig": sig})
        self._chain.append(entry)
        return entry

    def verify(self) -> bool:
        prev = self.GENESIS
        for e in self._chain:
            if e.prev_hash != prev:
                return False                         # chain broken
            if _sha256(e.body()) != e.entry_hash:
                return False                         # entry altered
            expect = hmac.new(self._key, e.entry_hash.encode(),
                              hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expect, e.hmac_sig):
                return False                         # signature forged
            prev = e.entry_hash
        return True


if __name__ == "__main__":
    log = AuditLog(hmac_key=os.urandom(32))
    salt = os.urandom(16)
    for i in range(3):
        log.append(salted_hash(f"prompt {i}", salt),
                   salted_hash(f"answer {i}", salt),
                   {"pii_redacted": True, "hitl": False, "residency": "EU"},
                   user="u-42", tenant="acme")
    print("intact chain verifies :", log.verify())      # True
    # simulate tampering with a past entry's stored hash:
    log._chain[1].policy_results["residency"] = "US"
    print("after tamper verifies  :", log.verify())      # False
