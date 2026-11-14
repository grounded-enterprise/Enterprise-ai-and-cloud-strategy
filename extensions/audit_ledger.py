# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Hash-chained audit ledger with a pluggable signer.

This is the deployable form of the ledger taught as Listing 8.3. The printed
listing signs with HMAC and hard-codes it; this module makes the signer an
interface, so the choice between authenticated integrity and non-repudiation
is a constructor argument rather than a rewrite.

Three differences from the printed listing, each deliberate:

1.  **The signer is injected.** `AuditLedger(signer=...)` accepts anything
    implementing `Signer`. `HMACSigner` reproduces the printed behaviour.
    `Ed25519Signer` gives non-repudiation. The entry records which algorithm
    signed it, so a ledger that changes signers mid-life stays verifiable.

2.  **Serialization is canonical and pinned here.** The printed listing hashes
    `payload.model_dump_json()`, which makes the chain depend on Pydantic's
    field ordering and serialization behaviour. A library upgrade could then
    invalidate every historical entry. Records that must verify years from now
    should not have their hash preimage defined by a dependency, so this module
    defines it explicitly in `canonical_bytes()` and takes no dependencies.

3.  **Verification walks the chain from genesis.** `verify_entry()` in the
    printed listing checks one entry against a predecessor hash the caller
    supplies — which assumes the caller already knows the correct predecessor.
    `verify_chain()` recomputes the whole chain from the genesis seed, which is
    the check that actually detects a deleted or reordered block.

On what a signature here proves
-------------------------------
`HMACSigner` gives tamper evidence and authenticated integrity. It does not
give non-repudiation: the key is symmetric, so any party able to verify a tag
is equally able to produce one. It cannot settle a dispute about who wrote a
record, because the verifier is a suspect.

`Ed25519Signer` gives non-repudiation, provided the private key is generated
in, and never leaves, an HSM or KMS. Hold the private key in application memory
and you are back to tamper evidence with extra steps.

Choose HMAC when the ledger answers "has this been altered since we wrote it".
Choose Ed25519 when it must answer "prove to a third party that you wrote it" —
regulatory attestation, contractual dispute, anything an auditor outside your
trust boundary has to accept.

Requires only the standard library. `Ed25519Signer` additionally needs
`cryptography`; it is imported lazily, so HMAC use never pulls the dependency.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

__all__ = [
    "AuditRecord",
    "ChainEntry",
    "Signer",
    "Verifier",
    "HMACSigner",
    "Ed25519Signer",
    "Ed25519Verifier",
    "AuditLedger",
    "verify_chain",
    "canonical_bytes",
    "DEFAULT_GENESIS_SEED",
    "SIGNING_DOMAIN",
]

DEFAULT_GENESIS_SEED = "GENESIS_BLOCK_SEED_2026"

# Domain separation. Signatures are computed over this prefix plus the block
# hash, so a signature produced by this ledger can never be replayed as a valid
# signature in another protocol that happens to share the key. The version
# suffix is what lets the preimage change later without ambiguity.
SIGNING_DOMAIN = b"tge-audit-ledger-v1|"


# --------------------------------------------------------------------------- #
#  Canonical serialization
# --------------------------------------------------------------------------- #
def canonical_bytes(obj: Any) -> bytes:
    """Serialize to the pinned canonical form used for every hash preimage.

    Keys sorted, no insignificant whitespace, UTF-8, non-ASCII preserved rather
    than escaped. This function is a wire format: changing it invalidates every
    previously written entry. If it must change, add a new signing domain
    version rather than editing this in place.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
#  Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuditRecord:
    """One inference event, as it is committed to the ledger.

    Prompts and completions are never stored — only their SHA-256 digests. The
    ledger proves what was sent and returned without becoming a second copy of
    the data the zero-data-retention posture exists to avoid holding.
    """

    event_id: str
    tenant_id: str
    model_role: str
    resolved_provider: str
    resolved_model_id: str
    resolved_region: str
    zero_data_retention: bool
    input_hash: str                       # SHA-256 of the normalized prompt
    output_hash: str                      # SHA-256 of the normalized completion
    input_tokens: int
    output_tokens: int
    timestamp: str = field(default_factory=_utc_now_iso)

    def payload_hash(self) -> str:
        return _sha256_hex(canonical_bytes(asdict(self)))


@dataclass(frozen=True)
class ChainEntry:
    """A committed block. `signature` and `algorithm` replace the printed
    listing's `hmac_signature`, so the entry is not tied to one primitive."""

    sequence_number: int
    timestamp: str
    payload: AuditRecord
    previous_block_hash: str
    block_hash: str
    signature: str
    algorithm: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Signer interface
# --------------------------------------------------------------------------- #
@runtime_checkable
class Signer(Protocol):
    """Produces a signature over a block hash."""

    algorithm: str

    def sign(self, message: bytes) -> bytes: ...


@runtime_checkable
class Verifier(Protocol):
    """Checks a signature over a block hash."""

    algorithm: str

    def verify(self, message: bytes, signature: bytes) -> bool: ...


class HMACSigner:
    """Symmetric signer. Tamper evidence and authenticated integrity.

    Reproduces the printed listing's behaviour. Verification requires the same
    secret used to sign, which is precisely why this cannot establish
    authorship to anyone who does not already hold the key.
    """

    algorithm = "hmac-sha256"

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("key must be bytes")
        if len(key) < 32:
            raise ValueError("key must be at least 32 bytes; derive it from an HSM or KMS")
        self._key = bytes(key)

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def verify(self, message: bytes, signature: bytes) -> bool:
        return hmac.compare_digest(self.sign(message), signature)


class Ed25519Signer:
    """Asymmetric signer. Non-repudiation, if the private key stays in the HSM.

    Construct from a KMS/HSM-held key where possible. Accepting raw private key
    bytes is supported for local testing and key import, and is the case in
    which the non-repudiation property is lost.
    """

    algorithm = "ed25519"

    def __init__(self, private_key: Any) -> None:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        if isinstance(private_key, (bytes, bytearray)):
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(private_key))
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            raise TypeError("private_key must be Ed25519PrivateKey or 32 raw bytes")
        self._key = private_key

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)

    def public_key_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization

        return self._key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def verifier(self) -> "Ed25519Verifier":
        return Ed25519Verifier(self.public_key_bytes())


class Ed25519Verifier:
    """Verifies Ed25519 signatures from the public key alone.

    This is the asymmetry that matters: an auditor can be given this object and
    check every entry without ever being able to forge one.
    """

    algorithm = "ed25519"

    def __init__(self, public_key: Any) -> None:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        if isinstance(public_key, (bytes, bytearray)):
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes(public_key))
        if not isinstance(public_key, ed25519.Ed25519PublicKey):
            raise TypeError("public_key must be Ed25519PublicKey or 32 raw bytes")
        self._key = public_key

    def verify(self, message: bytes, signature: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature

        try:
            self._key.verify(signature, message)
            return True
        except InvalidSignature:
            return False


# --------------------------------------------------------------------------- #
#  Ledger
# --------------------------------------------------------------------------- #
class AuditLedger:
    """Append-only, hash-chained ledger of inference events.

    Each block commits to its predecessor's hash, so altering any historical
    record invalidates every block after it. The signature binds the chain to a
    key, so an attacker who can rewrite the store still cannot re-sign it.
    """

    def __init__(self, signer: Signer, genesis_seed: str = DEFAULT_GENESIS_SEED) -> None:
        if not hasattr(signer, "sign") or not hasattr(signer, "algorithm"):
            raise TypeError("signer must implement Signer (sign() and .algorithm)")
        self._signer = signer
        self._genesis_seed = genesis_seed
        self._last_block_hash = genesis_hash(genesis_seed)
        self._sequence_counter = 0

    @property
    def head(self) -> str:
        """Current chain tip. Publish or counter-sign this to timestamp the
        whole history in one value."""
        return self._last_block_hash

    @property
    def sequence(self) -> int:
        return self._sequence_counter

    def append(self, record: AuditRecord) -> ChainEntry:
        self._sequence_counter += 1
        now = _utc_now_iso()

        block_hash = compute_block_hash(
            sequence_number=self._sequence_counter,
            timestamp=now,
            previous_block_hash=self._last_block_hash,
            payload_hash=record.payload_hash(),
        )
        signature = self._signer.sign(SIGNING_DOMAIN + block_hash.encode("ascii"))

        entry = ChainEntry(
            sequence_number=self._sequence_counter,
            timestamp=now,
            payload=record,
            previous_block_hash=self._last_block_hash,
            block_hash=block_hash,
            signature=signature.hex(),
            algorithm=self._signer.algorithm,
        )
        self._last_block_hash = block_hash
        return entry


def genesis_hash(genesis_seed: str = DEFAULT_GENESIS_SEED) -> str:
    return _sha256_hex(genesis_seed.encode("utf-8"))


def compute_block_hash(
    sequence_number: int,
    timestamp: str,
    previous_block_hash: str,
    payload_hash: str,
) -> str:
    """The block preimage. Pipe-delimited and field-ordered; see the note in
    `canonical_bytes` about treating this as a wire format."""
    contents = f"{sequence_number}|{timestamp}|{previous_block_hash}|{payload_hash}"
    return _sha256_hex(contents.encode("utf-8"))


def verify_entry(entry: ChainEntry, previous_block_hash: str, verifier: Verifier) -> bool:
    """Check one block: its hash recomputes, and its signature holds."""
    if entry.previous_block_hash != previous_block_hash:
        return False
    if entry.algorithm != verifier.algorithm:
        return False

    expected = compute_block_hash(
        sequence_number=entry.sequence_number,
        timestamp=entry.timestamp,
        previous_block_hash=previous_block_hash,
        payload_hash=entry.payload.payload_hash(),
    )
    if not hmac.compare_digest(expected, entry.block_hash):
        return False

    try:
        signature = bytes.fromhex(entry.signature)
    except ValueError:
        return False
    return verifier.verify(SIGNING_DOMAIN + entry.block_hash.encode("ascii"), signature)


def verify_chain(
    entries: Iterable[ChainEntry],
    verifier: Verifier,
    genesis_seed: str = DEFAULT_GENESIS_SEED,
) -> tuple[bool, Optional[int]]:
    """Walk the chain from genesis.

    Returns `(True, None)` if every block verifies and the sequence is
    unbroken, otherwise `(False, sequence_number)` naming the first bad block.
    Catches deletion and reordering, which per-entry verification cannot: a
    deleted block leaves a gap in the sequence and breaks the hash link.
    """
    previous = genesis_hash(genesis_seed)
    expected_sequence = 0

    for entry in entries:
        expected_sequence += 1
        if entry.sequence_number != expected_sequence:
            return False, entry.sequence_number
        if not verify_entry(entry, previous, verifier):
            return False, entry.sequence_number
        previous = entry.block_hash

    return True, None


# --------------------------------------------------------------------------- #
#  Demonstration
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    def _record(event_id: str, tokens: int) -> AuditRecord:
        return AuditRecord(
            event_id=event_id,
            tenant_id="tenant-acme",
            model_role="reasoning",
            resolved_provider="aws",
            resolved_model_id="anthropic.claude-sonnet-4-5",
            resolved_region="us-east-1",
            zero_data_retention=True,
            input_hash=_sha256_hex(b"prompt"),
            output_hash=_sha256_hex(b"completion"),
            input_tokens=tokens,
            output_tokens=tokens // 2,
        )

    print("HMAC — tamper evidence")
    hmac_signer = HMACSigner(os.urandom(32))
    ledger = AuditLedger(hmac_signer)
    entries = [ledger.append(_record(f"evt-{i}", 100 * i)) for i in range(1, 4)]
    print(f"  appended {len(entries)}, head {ledger.head[:16]}…")
    print(f"  verify_chain -> {verify_chain(entries, hmac_signer)}")

    tampered = list(entries)
    tampered[1] = ChainEntry(
        **{**entries[1].to_dict(), "payload": AuditRecord(
            **{**asdict(entries[1].payload), "input_tokens": 999_999})}
    )
    print(f"  after altering block 2 -> {verify_chain(tampered, hmac_signer)}")

    print("\nEd25519 — non-repudiation")
    try:
        signer = Ed25519Signer(os.urandom(32))
        ledger = AuditLedger(signer)
        entries = [ledger.append(_record(f"evt-{i}", 100 * i)) for i in range(1, 4)]
        auditor = signer.verifier()          # public key only
        print(f"  auditor holds a {len(signer.public_key_bytes())}-byte public key")
        print(f"  verify_chain -> {verify_chain(entries, auditor)}")
        print("  the auditor can check every block and forge none")
    except ImportError:
        print("  skipped: pip install cryptography")
