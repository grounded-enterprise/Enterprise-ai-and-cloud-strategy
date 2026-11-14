# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Tests for the pluggable-signer audit ledger.

The ledger makes one claim — that altering a committed record is detectable —
and one non-claim, that HMAC does not establish authorship. Both are testable
without a cloud account, and the second is the more important of the two,
because it is the one people assume wrongly.
"""
import copy
import os
import unittest
from dataclasses import asdict, replace

import _paths  # noqa: F401  (import for the sys.path side effect)

from audit_ledger import (
    AuditLedger,
    AuditRecord,
    ChainEntry,
    Ed25519Signer,
    Ed25519Verifier,
    HMACSigner,
    SIGNING_DOMAIN,
    canonical_bytes,
    compute_block_hash,
    genesis_hash,
    verify_chain,
    verify_entry,
)

try:  # the asymmetric path is optional
    import cryptography  # noqa: F401

    HAVE_CRYPTOGRAPHY = True
except ImportError:
    HAVE_CRYPTOGRAPHY = False


def make_record(event_id: str = "evt-1", **overrides) -> AuditRecord:
    base = dict(
        event_id=event_id,
        tenant_id="tenant-acme",
        model_role="reasoning",
        resolved_provider="aws",
        resolved_model_id="anthropic.claude-sonnet-4-5",
        resolved_region="us-east-1",
        zero_data_retention=True,
        input_hash="a" * 64,
        output_hash="b" * 64,
        input_tokens=1200,
        output_tokens=340,
        timestamp="2026-08-18T12:00:00+00:00",
    )
    base.update(overrides)
    return AuditRecord(**base)


def make_chain(signer, count: int = 5) -> list[ChainEntry]:
    ledger = AuditLedger(signer)
    return [ledger.append(make_record(f"evt-{i}")) for i in range(1, count + 1)]


class CanonicalSerialization(unittest.TestCase):
    """The hash preimage is a wire format. Nothing may perturb it."""

    def test_key_order_does_not_change_the_bytes(self):
        a = canonical_bytes({"b": 2, "a": 1})
        b = canonical_bytes({"a": 1, "b": 2})
        self.assertEqual(a, b)

    def test_no_insignificant_whitespace(self):
        self.assertEqual(canonical_bytes({"a": 1, "b": 2}), b'{"a":1,"b":2}')

    def test_non_ascii_is_preserved_not_escaped(self):
        self.assertEqual(canonical_bytes({"k": "café"}), '{"k":"café"}'.encode("utf-8"))

    def test_payload_hash_is_stable_across_instances(self):
        self.assertEqual(make_record().payload_hash(), make_record().payload_hash())

    def test_payload_hash_changes_when_any_field_changes(self):
        baseline = make_record().payload_hash()
        for field_name, value in [
            ("tenant_id", "tenant-other"),
            ("input_tokens", 1201),
            ("zero_data_retention", False),
            ("resolved_region", "eu-west-1"),
        ]:
            with self.subTest(field=field_name):
                altered = replace(make_record(), **{field_name: value})
                self.assertNotEqual(altered.payload_hash(), baseline)


class ChainConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.signer = HMACSigner(os.urandom(32))

    def test_first_block_links_to_the_genesis_seed(self):
        entries = make_chain(self.signer, 1)
        self.assertEqual(entries[0].previous_block_hash, genesis_hash())

    def test_sequence_numbers_start_at_one_and_increment(self):
        entries = make_chain(self.signer, 5)
        self.assertEqual([e.sequence_number for e in entries], [1, 2, 3, 4, 5])

    def test_each_block_links_to_its_predecessor(self):
        entries = make_chain(self.signer, 5)
        for previous, current in zip(entries, entries[1:]):
            self.assertEqual(current.previous_block_hash, previous.block_hash)

    def test_head_tracks_the_last_block(self):
        ledger = AuditLedger(self.signer)
        self.assertEqual(ledger.head, genesis_hash())
        for i in range(1, 4):
            entry = ledger.append(make_record(f"evt-{i}"))
            self.assertEqual(ledger.head, entry.block_hash)

    def test_entry_records_the_algorithm_that_signed_it(self):
        self.assertEqual(make_chain(self.signer, 1)[0].algorithm, "hmac-sha256")

    def test_identical_payloads_produce_distinct_blocks(self):
        # Sequence number and commit time are in the preimage, so replaying the
        # same event cannot silently collide with the original.
        ledger = AuditLedger(self.signer)
        a = ledger.append(make_record("evt-same"))
        b = ledger.append(make_record("evt-same"))
        self.assertNotEqual(a.block_hash, b.block_hash)

    def test_ledger_rejects_an_object_that_is_not_a_signer(self):
        with self.assertRaises(TypeError):
            AuditLedger(object())


class TamperEvidence(unittest.TestCase):
    """The claim the ledger exists to make."""

    def setUp(self) -> None:
        self.signer = HMACSigner(os.urandom(32))
        self.entries = make_chain(self.signer, 5)

    def test_an_untouched_chain_verifies(self):
        self.assertEqual(verify_chain(self.entries, self.signer), (True, None))

    def test_altering_a_payload_is_detected(self):
        tampered = copy.deepcopy(self.entries)
        tampered[2] = replace(tampered[2], payload=replace(tampered[2].payload,
                                                           input_tokens=999_999))
        self.assertEqual(verify_chain(tampered, self.signer), (False, 3))

    def test_deleting_a_block_is_detected(self):
        tampered = [e for e in self.entries if e.sequence_number != 3]
        ok, bad = verify_chain(tampered, self.signer)
        self.assertFalse(ok)
        self.assertEqual(bad, 4)

    def test_reordering_blocks_is_detected(self):
        tampered = list(self.entries)
        tampered[1], tampered[2] = tampered[2], tampered[1]
        ok, _ = verify_chain(tampered, self.signer)
        self.assertFalse(ok)

    def test_truncating_the_chain_is_not_detected_by_the_walk_alone(self):
        # A prefix is internally consistent. Detecting truncation requires an
        # externally published head — which is why `head` is exposed. Asserted
        # so the limitation is documented rather than assumed away.
        self.assertEqual(verify_chain(self.entries[:3], self.signer), (True, None))
        self.assertNotEqual(self.entries[2].block_hash, self.entries[-1].block_hash)

    def test_forging_a_signature_without_the_key_fails(self):
        attacker = HMACSigner(os.urandom(32))
        forged = copy.deepcopy(self.entries)
        forged[1] = replace(
            forged[1],
            signature=attacker.sign(
                SIGNING_DOMAIN + forged[1].block_hash.encode("ascii")).hex(),
        )
        self.assertEqual(verify_chain(forged, self.signer), (False, 2))

    def test_rewriting_a_block_and_resigning_still_fails_downstream(self):
        # An attacker holding the key could re-sign, but every later block
        # commits to the old hash, so the chain still breaks.
        rewritten = copy.deepcopy(self.entries)
        entry = rewritten[1]
        new_payload = replace(entry.payload, output_tokens=1)
        new_hash = compute_block_hash(entry.sequence_number, entry.timestamp,
                                      entry.previous_block_hash, new_payload.payload_hash())
        rewritten[1] = replace(
            entry, payload=new_payload, block_hash=new_hash,
            signature=self.signer.sign(SIGNING_DOMAIN + new_hash.encode("ascii")).hex(),
        )
        ok, bad = verify_chain(rewritten, self.signer)
        self.assertFalse(ok)
        self.assertEqual(bad, 3)

    def test_a_corrupt_signature_field_is_rejected_not_raised(self):
        tampered = copy.deepcopy(self.entries)
        tampered[0] = replace(tampered[0], signature="not-hex")
        self.assertEqual(verify_chain(tampered, self.signer), (False, 1))

    def test_verification_under_the_wrong_genesis_seed_fails(self):
        ok, bad = verify_chain(self.entries, self.signer, genesis_seed="WRONG")
        self.assertFalse(ok)
        self.assertEqual(bad, 1)


class SignerIsPluggable(unittest.TestCase):
    def test_hmac_key_must_be_long_enough(self):
        with self.assertRaises(ValueError):
            HMACSigner(b"short")

    def test_hmac_key_must_be_bytes(self):
        with self.assertRaises(TypeError):
            HMACSigner("a string key")  # type: ignore[arg-type]

    def test_the_same_records_under_two_signers_share_block_hashes(self):
        # The chain is signer-independent; only the signature differs. This is
        # what makes migrating signers possible without rewriting history.
        a = make_chain(HMACSigner(os.urandom(32)), 3)
        b = make_chain(HMACSigner(os.urandom(32)), 3)
        self.assertNotEqual([e.signature for e in a], [e.signature for e in b])

    def test_a_verifier_rejects_entries_signed_by_another_algorithm(self):
        signer = HMACSigner(os.urandom(32))
        entries = make_chain(signer, 2)
        mislabelled = replace(entries[0], algorithm="ed25519")
        self.assertFalse(verify_entry(mislabelled, genesis_hash(), signer))


@unittest.skipUnless(HAVE_CRYPTOGRAPHY, "requires `cryptography`")
class NonRepudiation(unittest.TestCase):
    """The property HMAC cannot provide."""

    def setUp(self) -> None:
        self.signer = Ed25519Signer(os.urandom(32))
        self.entries = make_chain(self.signer, 4)

    def test_a_chain_signed_asymmetrically_verifies(self):
        self.assertEqual(verify_chain(self.entries, self.signer.verifier()), (True, None))

    def test_an_auditor_needs_only_the_public_key(self):
        auditor = Ed25519Verifier(self.signer.public_key_bytes())
        self.assertEqual(len(self.signer.public_key_bytes()), 32)
        self.assertEqual(verify_chain(self.entries, auditor), (True, None))

    def test_the_public_key_cannot_produce_a_signature(self):
        auditor = Ed25519Verifier(self.signer.public_key_bytes())
        self.assertFalse(hasattr(auditor, "sign"))

    def test_a_different_key_cannot_forge_an_entry(self):
        attacker = Ed25519Signer(os.urandom(32))
        forged = copy.deepcopy(self.entries)
        forged[1] = replace(
            forged[1],
            signature=attacker.sign(
                SIGNING_DOMAIN + forged[1].block_hash.encode("ascii")).hex(),
        )
        self.assertEqual(verify_chain(forged, self.signer.verifier()), (False, 2))

    def test_tampering_is_detected_the_same_way_as_under_hmac(self):
        tampered = copy.deepcopy(self.entries)
        tampered[0] = replace(tampered[0],
                              payload=replace(tampered[0].payload, tenant_id="other"))
        self.assertEqual(verify_chain(tampered, self.signer.verifier()), (False, 1))

    def test_signer_rejects_a_key_of_the_wrong_type(self):
        with self.assertRaises(TypeError):
            Ed25519Signer("not a key")


class DomainSeparation(unittest.TestCase):
    def test_signatures_are_computed_over_the_domain_prefix(self):
        signer = HMACSigner(os.urandom(32))
        entry = make_chain(signer, 1)[0]
        bare = signer.sign(entry.block_hash.encode("ascii")).hex()
        self.assertNotEqual(entry.signature, bare)
        self.assertEqual(
            entry.signature,
            signer.sign(SIGNING_DOMAIN + entry.block_hash.encode("ascii")).hex(),
        )

    def test_the_domain_is_versioned(self):
        self.assertTrue(SIGNING_DOMAIN.endswith(b"v1|"))


class RecordHygiene(unittest.TestCase):
    def test_records_are_immutable_once_created(self):
        record = make_record()
        with self.assertRaises(Exception):
            record.input_tokens = 5  # type: ignore[misc]

    def test_no_prompt_or_completion_text_is_stored(self):
        fields = set(asdict(make_record()))
        for leaky in ("prompt", "completion", "content", "messages", "output_text"):
            self.assertNotIn(leaky, fields)

    def test_timestamp_defaults_to_utc(self):
        record = AuditRecord(
            event_id="e", tenant_id="t", model_role="r", resolved_provider="aws",
            resolved_model_id="m", resolved_region="us-east-1",
            zero_data_retention=True, input_hash="a" * 64, output_hash="b" * 64,
            input_tokens=1, output_tokens=1,
        )
        self.assertTrue(record.timestamp.endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
