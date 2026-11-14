# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""The redaction window, under every chunking.

`test_streaming_core.py` checks that a value split across two frames is caught.
It places every value at the end of its stream, inside the hold-back, where it
is held until the final flush. That leaves one path untested: a value followed
by more than a hold-back of text, which has to pass through the point where the
window is cut.

Two things are pinned here.

1.  The printed sliding window (Listings 4.1 and 10.2) splits a value at that
    cut and emits it in clear. Marked `expectedFailure`: the loop is printed in
    the book, so the fix is a manuscript change. When these report *unexpected
    success* the fix has landed and the decorators come off.

2.  `extensions/streaming_redaction.py` does not. For any way of cutting a text
    into chunks, it emits exactly what redacting the whole text returns.

A third class records two limits of the printed card-number pattern itself.
"""
import asyncio
import json
import random
import unittest

import _listings
from streaming_redaction import (
    HOLDBACK,
    SECRET_LABEL,
    StreamRedactor,
    redact_stream,
    redact_text,
)
import test_streaming_core  # noqa: F401  (registers the FastAPI stand-in if needed)
from streaming_core import redact as printed_redact, transform_and_sanitize

SSN = "123-45-6789"
PAN = "4111 1111 1111 1111"            # Luhn-valid test card
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
KEY_23 = "sk-" + "a1B2c3D4e5F6g7H8i9J0"
KEY_43 = "sk-" + "A" * 20 + "b" * 20
KEY_200 = "sk-" + "Zz9" * 65 + "Qq"
VALUES = {"SSN": SSN, "PAN": PAN, "AWS key": AWS_KEY, "23-char key": KEY_23,
          "43-char key": KEY_43, "200-char key": KEY_200}

SENTENCE = ("Customer SSN is 123-45-6789. The account review is complete "
            "and the file was closed today.")


def _fixed(text: str, size: int) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)]


def _random_cuts(text: str, rng: random.Random) -> list:
    count = rng.randint(1, max(1, min(12, len(text) - 1)))
    cuts = sorted(rng.sample(range(1, len(text)), count)) if len(text) > 1 else []
    return [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)], strict=True)]


def _stream(chunks) -> str:
    redactor = StreamRedactor()
    return "".join(redactor.feed(c) for c in chunks) + redactor.flush()


def _printed_10_2(chunks) -> str:
    """Listing 10.2's transform, driven with one provider frame per chunk."""

    async def upstream():
        for chunk in chunks:
            yield f'data: {json.dumps({"delta": {"text": chunk}})}\n\n'.encode()
        yield b"data: [DONE]\n\n"

    async def run():
        out = []
        async for frame in transform_and_sanitize(upstream(), "aws"):
            payload = frame.decode()[len("data: "):].strip()
            if payload != "[DONE]":
                out.append(json.loads(payload)["delta"])
        return "".join(out)

    return asyncio.run(run())


def _printed_4_1(chunks) -> str:
    listing = _listings.load("chapters/ch04/listing-4.1-redactor-cache-tool-gate.py")

    async def upstream():
        for chunk in chunks:
            yield chunk

    async def run():
        return "".join([part async for part in listing.redact_stream_pipeline(upstream())])

    return asyncio.run(run())


class PrintedWindowLeak(unittest.TestCase):
    """KNOWN DEFECT, found 2026-11-13. The cut between the emitted text and the
    carry lands inside a value and the value goes out in two unredacted pieces.
    The existing tests never put enough text after the value to move it through
    the cut. Both listings share the loop; the fix belongs in the manuscript."""

    def test_the_whole_text_in_one_frame_is_redacted(self):
        # Not the defect: with one frame there is one cut, well past the SSN.
        self.assertNotIn(SSN, _printed_10_2([SENTENCE]))

    @unittest.expectedFailure
    def test_listing_10_2_redacts_an_ssn_in_ten_character_frames(self):
        self.assertNotIn(SSN, _printed_10_2(_fixed(SENTENCE, 10)))

    @unittest.expectedFailure
    @unittest.skipUnless(_listings.have("pydantic"), "Listing 4.1 imports pydantic")
    def test_listing_4_1_redacts_an_ssn_in_ten_character_frames(self):
        self.assertNotIn(SSN, _printed_4_1(_fixed(SENTENCE, 10)))

    @unittest.expectedFailure
    def test_listing_10_2_redacts_a_card_number_in_small_frames(self):
        text = f"Card on file {PAN} was charged for the annual plan renewal this morning."
        self.assertNotIn(PAN, _printed_10_2(_fixed(text, 7)))


class PrintedPatternLimits(unittest.TestCase):
    """KNOWN LIMITS of the card-number pattern, found 2026-11-13. These are in
    the pattern, not the window, so they hold for a whole text in one piece.
    `extensions/streaming_redaction.py` uses the printed patterns unchanged and
    shares both."""

    @unittest.expectedFailure
    def test_a_card_number_that_follows_another_number_is_redacted(self):
        # The pattern takes up to 19 digits across single spaces or hyphens, so
        # it starts at the earlier number, runs into the card, fails the Luhn
        # check on the combined digits and moves on past them. The card is
        # never tried on its own.
        for text in ("Order 12 4111 1111 1111 1111 shipped.", "ref 2026 4111-1111-1111-1111."):
            self.assertIn("[REDACTED-PAN]", printed_redact(text), text)

    @unittest.expectedFailure
    def test_the_space_after_a_card_number_is_kept(self):
        # `[ -]?` after the last digit takes the following space into the match.
        self.assertEqual(printed_redact(f"card {PAN} expires soon"),
                         "card [REDACTED-PAN] expires soon")


class WholeTextRedaction(unittest.TestCase):
    """`redact_text` is the reference the stream is held to."""

    def test_each_kind_is_replaced_by_its_label(self):
        self.assertEqual(redact_text(f"a {SSN} b"), "a [REDACTED-SSN] b")
        self.assertEqual(redact_text(f"a {PAN}. b"), "a [REDACTED-PAN]. b")
        self.assertEqual(redact_text(f"a {AWS_KEY} b"), f"a {SECRET_LABEL} b")
        self.assertEqual(redact_text(f"a {KEY_200} b"), f"a {SECRET_LABEL} b")

    def test_a_digit_run_failing_luhn_survives(self):
        self.assertEqual(redact_text("order 4111111111111112 shipped"),
                         "order 4111111111111112 shipped")

    def test_it_agrees_with_the_printed_redact_on_whole_texts(self):
        # Same patterns, same order, same labels as Listing 10.2.
        rng = random.Random(20261113)
        for _ in range(300):
            text = _random_text(rng)
            self.assertEqual(redact_text(text), printed_redact(text), text)


def _random_text(rng: random.Random) -> str:
    """Prose mixed with values and with near-misses that must not match."""
    words = ["the", "router", "selects", "an", "endpoint", "cost", "latency", "ok.",
             "id", "ref:", "call", "+1", "4111", "0000", "12", "sk-short", "AKIA", "x-",
             "2026-11-13", "555-01-00", "9" * 25, "4111111111111112", "sk-" + "q" * 19]
    parts = []
    for _ in range(rng.randint(3, 40)):
        roll = rng.random()
        if roll < 0.18:
            parts.append(rng.choice(list(VALUES.values())))
        elif roll < 0.24:
            parts.append("sk-" + "".join(rng.choice("abcXYZ0189") for _ in range(rng.randint(15, 90))))
        elif roll < 0.30:
            parts.append("".join(rng.choice("0123456789 -") for _ in range(rng.randint(8, 30))))
        else:
            parts.append(rng.choice(words))
    return rng.choice(["", " "]).join(parts) if rng.random() < 0.1 else " ".join(parts)


class StreamEqualsWholeText(unittest.TestCase):
    """The invariant: chunking never changes what the client receives."""

    def test_the_sentence_that_leaks_from_the_printed_window(self):
        out = _stream(_fixed(SENTENCE, 10))
        self.assertNotIn(SSN, out)
        self.assertEqual(out, redact_text(SENTENCE))

    def test_every_value_at_every_alignment_and_frame_size(self):
        for name, value in VALUES.items():
            for lead in range(0, 2 * HOLDBACK + 3):
                text = "x" * lead + " " + value + " " + "y" * (2 * HOLDBACK)
                expected = redact_text(text)
                for size in (1, 2, 3, 5, 7, 10, 16, 31, 32, 33, HOLDBACK, 64, 500):
                    with self.subTest(value=name, lead=lead, size=size):
                        out = _stream(_fixed(text, size))
                        self.assertNotIn(value, out)
                        self.assertEqual(out, expected)

    def test_random_texts_under_random_chunkings(self):
        rng = random.Random(4414)
        for _ in range(400):
            text = _random_text(rng)
            expected = redact_text(text)
            for chunks in (_fixed(text, 1), _random_cuts(text, rng), _random_cuts(text, rng)):
                self.assertEqual(_stream(chunks), expected, (text, [len(c) for c in chunks]))

    def test_two_values_back_to_back(self):
        text = f"{SSN} {PAN} {KEY_43} {AWS_KEY} {SSN}" + " tail" * 30
        for size in (1, 4, 9, 20, 47):
            self.assertEqual(_stream(_fixed(text, size)), redact_text(text))

    def test_clean_text_passes_through_intact(self):
        text = "The router selects an endpoint using live latency and cost. " * 5
        for size in (1, 13, 200):
            self.assertEqual(_stream(_fixed(text, size)), text)

    def test_the_async_form_matches(self):
        async def upstream():
            for chunk in _fixed(SENTENCE, 10):
                yield chunk

        async def run():
            return "".join([part async for part in redact_stream(upstream())])

        self.assertEqual(asyncio.run(run()), redact_text(SENTENCE))


class BoundedHoldback(unittest.TestCase):
    """Redaction must not be bought by buffering the completion."""

    def test_ordinary_text_is_held_no_longer_than_the_holdback(self):
        redactor = StreamRedactor()
        for chunk in _fixed("plain words with nothing to hide in them at all. " * 20, 7):
            redactor.feed(chunk)
            self.assertLessEqual(redactor.pending, HOLDBACK)

    def test_output_begins_before_the_stream_ends(self):
        redactor = StreamRedactor()
        emitted = "".join(redactor.feed(c) for c in _fixed("a" * (HOLDBACK * 4), 10))
        self.assertGreaterEqual(len(emitted), HOLDBACK * 3 - 10)

    def test_the_holdback_is_small_and_fixed(self):
        self.assertLess(HOLDBACK, 128)
        self.assertGreater(HOLDBACK, len(PAN))

    def test_a_holdback_too_short_to_cover_a_card_is_refused(self):
        with self.assertRaises(ValueError):
            StreamRedactor(holdback=32)

    def test_a_key_still_arriving_is_held_whole(self):
        redactor = StreamRedactor()
        out = redactor.feed("token ")
        for chunk in _fixed(KEY_200, 9):
            out += redactor.feed(chunk)
            self.assertNotIn("sk-", out)
        out += redactor.feed(" and the rest of the answer follows here." * 3) + redactor.flush()
        self.assertNotIn(KEY_200, out)
        self.assertIn(SECRET_LABEL, out)

    def test_an_endless_key_is_redacted_and_memory_stays_bounded(self):
        redactor = StreamRedactor(max_open_secret=100)
        out = redactor.feed("leak: sk-")
        for _ in range(400):
            out += redactor.feed("A" * 25)
            self.assertLessEqual(redactor.pending, 100 + 25)
        out += redactor.feed(" done.") + redactor.flush()
        self.assertEqual(out, f"leak: {SECRET_LABEL} done.")

    def test_flush_leaves_the_redactor_reusable_and_empty(self):
        redactor = StreamRedactor()
        redactor.feed(f"id {SSN}")
        self.assertEqual(redactor.flush(), "id [REDACTED-SSN]")
        self.assertEqual(redactor.flush(), "")
        self.assertEqual(redactor.pending, 0)


if __name__ == "__main__":
    unittest.main()
