# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Streaming and redaction tests.

The invariant under test is the one that is easy to break and expensive to
discover in production: redaction must survive a value split across two SSE
frames, and the gateway must emit before the stream ends. Buffer the whole body
and redaction gets easy while time-to-first-token — the metric users actually
judge — is destroyed.

`streaming_core` imports FastAPI at module scope for its endpoint, but every
function tested here is pure. When FastAPI is not installed a minimal stub is
registered so these tests still run; the stub stands in for the web framework
only, never for the logic being tested. With FastAPI installed the real package
is used and the stub is never touched.
"""
import asyncio
import json
import re
import sys
import types
import unittest

import _paths  # noqa: F401  (import for the sys.path side effect)


def _install_fastapi_stub() -> None:
    """Register a no-op FastAPI just sufficient for module import."""
    fastapi = types.ModuleType("fastapi")

    class _State:
        pass

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.state = _State()

        def post(self, *args, **kwargs):
            return lambda fn: fn

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = "", **kwargs):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:  # pragma: no cover - never instantiated in these tests
        pass

    class StreamingResponse:  # pragma: no cover - never instantiated here
        def __init__(self, content, **kwargs):
            self.content = content

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    fastapi.Request = Request

    responses = types.ModuleType("fastapi.responses")
    responses.StreamingResponse = StreamingResponse
    fastapi.responses = responses

    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("fastapi.responses", responses)


try:
    import fastapi  # noqa: F401

    USING_REAL_FASTAPI = True
except ImportError:
    _install_fastapi_stub()
    USING_REAL_FASTAPI = False

from streaming_core import (  # noqa: E402
    _HOLDBACK,
    _LONGEST_PATTERN,
    _PATTERNS,
    PIIKind,
    _luhn_valid,
    extract_delta,
    redact,
    transform_and_sanitize,
)

# A Luhn-valid test PAN from the standard test-card set, and one that is not.
VALID_PAN = "4111111111111111"
INVALID_PAN = "4111111111111112"


async def _agen(chunks):
    for chunk in chunks:
        yield chunk


def _drain(chunks) -> list[str]:
    """Run the transform over a list of raw byte chunks, return output frames."""

    async def run():
        return [frame.decode() async for frame in transform_and_sanitize(_agen(chunks), "azure")]

    return asyncio.run(run())


def _sse(text: str) -> bytes:
    return f'data: {json.dumps({"choices": [{"delta": {"content": text}}]})}\n\n'.encode()


def _deltas(frames: list[str]) -> str:
    """Concatenate the delta text out of the emitted unified frames."""
    out = []
    for frame in frames:
        payload = frame[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload).get("delta", ""))
    return "".join(out)


class Redaction(unittest.TestCase):
    def test_ssn_is_redacted(self):
        self.assertEqual(redact("my SSN is 123-45-6789 ok"),
                         "my SSN is [REDACTED-SSN] ok")

    def test_luhn_valid_pan_is_redacted(self):
        self.assertIn("[REDACTED-PAN]", redact(f"card {VALID_PAN} on file"))

    def test_digit_run_failing_luhn_survives(self):
        # An order number is not a card number. False positives on long digit
        # runs would silently corrupt legitimate output.
        self.assertIn(INVALID_PAN, redact(f"order {INVALID_PAN}"))

    def test_api_key_shapes_are_redacted(self):
        # PIIKind.SECRET carries the value "api_key", so the label reads
        # [REDACTED-API_KEY]. Derived from the enum rather than hardcoded.
        label = f"[REDACTED-{PIIKind.SECRET.value.upper()}]"
        self.assertIn(label, redact("key sk-" + "a" * 24))
        self.assertIn(label, redact("key AKIA" + "A1B2C3D4E5F6G7H8"))

    def test_clean_text_is_returned_unchanged(self):
        clean = "The gateway resolves a role to an endpoint at request time."
        self.assertEqual(redact(clean), clean)

    def test_luhn_checksum(self):
        self.assertTrue(_luhn_valid(VALID_PAN))
        self.assertFalse(_luhn_valid(INVALID_PAN))
        self.assertFalse(_luhn_valid("123"))          # too short to be a PAN


class DeltaExtraction(unittest.TestCase):
    """Each provider puts the incremental text somewhere different. This is the
    executable form of the response-shape table in Chapter 2."""

    def test_azure_and_oci_share_the_openai_shape(self):
        frame = {"choices": [{"delta": {"content": "hello"}}]}
        self.assertEqual(extract_delta(frame, "azure"), "hello")
        self.assertEqual(extract_delta(frame, "oci"), "hello")

    def test_aws_uses_a_top_level_delta(self):
        self.assertEqual(extract_delta({"delta": {"text": "hello"}}, "aws"), "hello")

    def test_gcp_joins_candidate_parts(self):
        frame = {"candidates": [{"content": {"parts": [{"text": "he"}, {"text": "llo"}]}}]}
        self.assertEqual(extract_delta(frame, "gcp"), "hello")

    def test_frames_carrying_no_text_yield_empty_string(self):
        for provider, frame in [
            ("azure", {"choices": [{"delta": {"role": "assistant"}}]}),
            ("aws", {"delta": {}}),
            ("gcp", {"candidates": [{"content": {"parts": []}}]}),
        ]:
            with self.subTest(provider=provider):
                self.assertEqual(extract_delta(frame, provider), "")

    def test_empty_choices_do_not_raise(self):
        self.assertEqual(extract_delta({"choices": []}, "azure"), "")
        self.assertEqual(extract_delta({"candidates": []}, "gcp"), "")

    def test_a_null_content_is_treated_as_empty(self):
        self.assertEqual(extract_delta({"choices": [{"delta": {"content": None}}]}, "azure"), "")


class SlidingWindowRedaction(unittest.TestCase):
    """The case that makes per-frame redaction wrong."""

    def test_value_split_across_two_frames_is_still_caught(self):
        frames = _drain([_sse("my SSN is 123-45"), _sse("-6789 thanks"), b"data: [DONE]\n\n"])
        text = _deltas(frames)
        self.assertIn("[REDACTED-SSN]", text)
        self.assertNotIn("123-45-6789", text)

    def test_value_split_across_many_frames_is_still_caught(self):
        chunks = [_sse(c) for c in "my SSN is 123-45-6789 thanks"]
        text = _deltas(_drain(chunks + [b"data: [DONE]\n\n"]))
        self.assertIn("[REDACTED-SSN]", text)
        self.assertNotIn("123-45-6789", text)

    def test_the_tail_is_flushed_on_done(self):
        # Text shorter than the holdback would otherwise never be emitted.
        text = _deltas(_drain([_sse("short tail"), b"data: [DONE]\n\n"]))
        self.assertEqual(text, "short tail")

    def test_the_tail_is_flushed_when_the_stream_ends_without_done(self):
        text = _deltas(_drain([_sse("no done marker")]))
        self.assertEqual(text, "no done marker")

    def test_clean_text_passes_through_intact(self):
        original = "The router selects an endpoint using live latency and cost. " * 3
        self.assertEqual(_deltas(_drain([_sse(original), b"data: [DONE]\n\n"])), original)


class TimeToFirstToken(unittest.TestCase):
    """The reason redaction is windowed rather than whole-body."""

    def test_output_begins_before_the_stream_ends(self):
        # Enough text to clear the holdback, then a long silence before [DONE].
        long_text = "a" * (_HOLDBACK * 4)

        async def run():
            emitted_before_done = 0

            async def upstream():
                yield _sse(long_text)
                for _ in range(50):                   # stream continues
                    await asyncio.sleep(0)
                    yield _sse("b" * 10)
                yield b"data: [DONE]\n\n"

            async for frame in transform_and_sanitize(upstream(), "azure"):
                if b"[DONE]" in frame:
                    break
                emitted_before_done += 1
                if emitted_before_done >= 1:
                    return emitted_before_done
            return emitted_before_done

        self.assertGreaterEqual(asyncio.run(run()), 1)

    def test_holdback_is_bounded_and_small(self):
        # Whatever the window is, it must not scale with completion length.
        self.assertLess(_HOLDBACK, 512)
        self.assertGreater(_HOLDBACK, 12)            # must exceed an SSN


class HoldbackSizing(unittest.TestCase):
    """The holdback must be at least as long as the longest match a pattern can
    produce, or a value straddling a frame boundary escapes redaction.

    `_LONGEST_PATTERN` is 24, but the API-key pattern `sk-[A-Za-z0-9]{20,}` is
    unbounded above. Any key longer than the holdback is redacted when it lands
    in a single frame and passes through in the clear when it is split — the
    exact failure the sliding window exists to prevent, reappearing for the one
    pattern whose length is not capped.

    The fix is to bound the pattern and size the holdback from that bound;
    raising the constant alone leaves the same hole further out.
    """

    @unittest.expectedFailure
    def test_every_pattern_has_a_bounded_maximum_match(self):
        # KNOWN DEFECT, same root cause as below: `{20,}` has no upper bound,
        # so no fixed holdback can be proven sufficient. `{13,19}` is fine.
        open_ended = re.compile(r"\{\d+,\}|(?<!\\)[+*]")
        unbounded = [kind.name for kind, pattern in _PATTERNS
                     if open_ended.search(pattern.pattern)]
        self.assertEqual(unbounded, [], f"unbounded quantifier in {unbounded}")

    def test_ssn_fits_within_the_holdback(self):
        self.assertLessEqual(len("123-45-6789"), _HOLDBACK)

    def test_pan_fits_within_the_holdback(self):
        self.assertLessEqual(19 + 18, _LONGEST_PATTERN + _HOLDBACK)

    @unittest.expectedFailure
    def test_a_long_api_key_split_across_frames_is_redacted(self):
        # KNOWN DEFECT. Reported 2026-08-18; the constant is printed in the
        # book, so the fix is a manuscript change, not a repo change.
        # When this starts reporting "unexpected success", the fix has landed
        # and the decorator should be removed.
        key = "sk-" + "A" * 40                       # 43 chars, holdback is 32
        self.assertIn("[REDACTED-API_KEY]", redact(f"token {key} end"))
        chunks = [_sse(c) for c in f"token {key} end"] + [b"data: [DONE]\n\n"]
        self.assertNotIn(key, _deltas(_drain(chunks)))


class MalformedInput(unittest.TestCase):
    def test_a_malformed_frame_is_skipped_not_fatal(self):
        chunks = [b"data: {not json}\n\n", _sse("recovered"), b"data: [DONE]\n\n"]
        self.assertEqual(_deltas(_drain(chunks)), "recovered")

    def test_non_data_lines_are_ignored(self):
        chunks = [b": keep-alive comment\n\n", _sse("payload"), b"data: [DONE]\n\n"]
        self.assertEqual(_deltas(_drain(chunks)), "payload")

    def test_an_event_split_across_chunk_boundaries_is_reassembled(self):
        raw = _sse("split event")
        half = len(raw) // 2
        self.assertEqual(_deltas(_drain([raw[:half], raw[half:], b"data: [DONE]\n\n"])),
                         "split event")

    def test_done_terminates_the_stream(self):
        frames = _drain([_sse("before"), b"data: [DONE]\n\n", _sse("after")])
        self.assertNotIn("after", _deltas(frames))
        self.assertTrue(frames[-1].strip().endswith("[DONE]"))


if __name__ == "__main__":
    unittest.main()
