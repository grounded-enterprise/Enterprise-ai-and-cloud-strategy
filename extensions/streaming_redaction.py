# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Sliding-window PII redaction that holds for every chunking of a stream.

This is the deployable form of the redactor taught in Listing 4.1 and used by
Listing 10.2 (`reference/streaming_core.py`). It keeps their design, a bounded
hold-back and no buffering of the completion, and closes a leak in the way they
cut the window.

The leak
--------
The printed loop keeps the last 32 characters and emits the rest:

    window = carry + delta
    safe, carry = window[:-_HOLDBACK], window[-_HOLDBACK:]
    yield redact(safe)

The cut between `safe` and `carry` falls wherever the arithmetic puts it. When
it falls inside a value, the head of the value is emitted in `safe`, where the
pattern cannot match half a value, and the tail stays in `carry`, where it
cannot match either. The value reaches the client whole, in two pieces. The
hold-back protects a value that is still arriving; it does nothing for a value
that has arrived and is now passing through the cut. Any SSN, card number or key
followed by more than a hold-back of text is exposed to this, for most frame
sizes. `tests/test_redaction_window.py` reproduces it against both listings.

The repair
----------
Three rules, applied to the same window:

1.  **Match first, cut second.** Complete values are located on the whole window,
    and the cut is moved back to the start of any value it would split. A digit
    run that looked like a card and failed the Luhn check is treated the same
    way, because where it ends decides where the search for the next card starts.

2.  **A decision is emitted only once later text cannot change it.** Everything
    within the hold-back of the window's end stays in `carry` as raw text and is
    matched again when more arrives. The hold-back is sized from the longest
    value a bounded pattern can produce, and that is asserted at import.

3.  **The one unbounded pattern is held by its own token.** `sk-…` has no upper
    length, so no fixed hold-back can cover it (the defect logged in ERRATA on
    2026-08-18). A key still arriving at the window's end is held from its
    first character until it ends. Ordinary text is never delayed by this rule;
    only a token that starts like a key is. `max_open_secret` caps the hold: past
    it the token is redacted and the rest of it swallowed, so memory stays
    bounded and the failure is closed.

The property this buys, and the tests pin: for any way of cutting a text into
chunks, the stream emits exactly what `redact_text` returns for the whole text.

Patterns, labels and the Luhn check are those of Listing 10.2, so this is a
drop-in: construct one `StreamRedactor` per response, replace the three printed
lines with `safe = redactor.feed(delta)`, and emit `redactor.flush()` where the
listing flushes `carry`. Standard library only.
"""
from __future__ import annotations

import re
from typing import AsyncIterator

__all__ = ["StreamRedactor", "redact_text", "redact_stream", "HOLDBACK"]

SSN_LABEL = "[REDACTED-SSN]"
PAN_LABEL = "[REDACTED-PAN]"
SECRET_LABEL = "[REDACTED-API_KEY]"

# The patterns of Listing 10.2, in its order. Order matters: each pattern sees
# the text with the earlier patterns' matches already taken out.
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PAN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")

# A key that has started and not yet ended, at the end of the text seen so far.
_OPEN_SECRET = re.compile(r"\bsk-[A-Za-z0-9]*\Z")
_SECRET_CHAR = re.compile(r"[A-Za-z0-9]")

# Longest value each bounded pattern can match: an SSN is 11 characters, a PAN is
# 19 digits each followed by an optional separator, an AWS key id is 20.
_LONGEST_BOUNDED = max(11, 19 * 2, 20)
_SAFETY_MARGIN = 10
HOLDBACK = _LONGEST_BOUNDED + _SAFETY_MARGIN
assert HOLDBACK > _LONGEST_BOUNDED, "hold-back must exceed the longest bounded value"

_MASK = "\x00"          # stands in for a matched value while later patterns run


def _luhn_valid(candidate: str) -> bool:
    """Luhn checksum, as in Listing 10.2: keeps ordinary digit runs out."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _spans(text: str, start: int = 0) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    """Locate every value in `text` that begins at or after `start`.

    Characters before `start` are left context only: they are seen by the word
    boundaries in the patterns and can never begin a match themselves.

    Returns `(spans, atoms)`. `spans` are the values to redact, as
    (start, end, label), in order and non-overlapping. `atoms` are the stretches
    the window must not be cut inside: every span, plus every card-number
    candidate that failed the Luhn check. A rejected candidate still decides
    where the scan for the next card resumes, so cutting inside one would let a
    stream and a whole text disagree about a card that follows it.

    The patterns run in sequence, as the printed `redact` runs them. A matched
    value is masked in place, same length, so positions stay true and a later
    pattern cannot match across it.
    """
    found: list[tuple[int, int, str]] = []
    atoms: list[tuple[int, int]] = []
    work = text
    for pattern, label in ((_SSN, SSN_LABEL), (_PAN, PAN_LABEL), (_SECRET, SECRET_LABEL)):
        hits = []
        for match in pattern.finditer(work, start):
            atoms.append((match.start(), match.end()))
            if pattern is _PAN and not _luhn_valid(match.group()):
                continue
            hits.append((match.start(), match.end(), label))
        if hits:
            pieces, last = [], 0
            for hit_start, hit_end, _ in hits:
                pieces.append(work[last:hit_start])
                pieces.append(_MASK * (hit_end - hit_start))
                last = hit_end
            pieces.append(work[last:])
            work = "".join(pieces)
            found.extend(hits)
    found.sort()
    atoms.sort()
    return found, atoms


def _render(text: str, spans: list[tuple[int, int, str]], start: int, end: int) -> str:
    """`text[start:end]` with every span inside it replaced by its label."""
    out, pos = [], start
    for s, e, label in spans:
        if s < start or e > end:
            continue
        out.append(text[pos:s])
        out.append(label)
        pos = e
    out.append(text[pos:end])
    return "".join(out)


def redact_text(text: str) -> str:
    """Redact a complete text. The reference the stream must agree with."""
    return _render(text, _spans(text)[0], 0, len(text))


class StreamRedactor:
    """Redacts a text that arrives in pieces. One instance per response.

        redactor = StreamRedactor()
        for delta in deltas:
            safe = redactor.feed(delta)      # may be "", emit when non-empty
        tail = redactor.flush()              # once, when the stream ends
    """

    def __init__(self, holdback: int = HOLDBACK, max_open_secret: int = 512) -> None:
        if holdback <= _LONGEST_BOUNDED:
            raise ValueError(
                f"holdback {holdback} does not cover the longest bounded value "
                f"({_LONGEST_BOUNDED}); a value could be split at the cut")
        self._holdback = holdback
        self._max_open = max_open_secret
        self._context = ""      # the last character already emitted: left context only
        self._carry = ""        # raw text not yet emitted
        self._swallowing = False

    @property
    def pending(self) -> int:
        """Characters currently held back."""
        return len(self._carry)

    def feed(self, delta: str) -> str:
        """Take the next piece. Return what is now safe to emit, possibly ""."""
        if self._swallowing:
            # The rest of a key already redacted for exceeding max_open_secret.
            kept = 0
            while kept < len(delta) and _SECRET_CHAR.match(delta[kept]):
                kept += 1
            delta = delta[kept:]
            if not delta:
                return ""
            self._swallowing = False

        window = self._context + self._carry + delta
        base = len(self._context)
        spans, atoms = _spans(window, base)

        cut = max(base, len(window) - self._holdback)
        moved = True
        while moved:                             # rule 1: never split a value
            moved = False
            for start, end in atoms:
                if start < cut < end:
                    cut, moved = start, True

        opened = _OPEN_SECRET.search(window, base)   # rule 3: a key still arriving
        overflow = ""
        if opened and opened.start() < cut:
            cut = opened.start()
        if opened and len(window) - opened.start() > self._max_open:
            # Fail closed: redact what has arrived and swallow the remainder.
            cut = opened.start()
            overflow = SECRET_LABEL
            self._swallowing = True

        safe = _render(window, spans, base, cut) + overflow
        if overflow:
            self._context, self._carry = "A", ""     # a key character: no word boundary
        else:
            if cut > 0:
                self._context = window[cut - 1]
            self._carry = window[cut:]
        return safe

    def flush(self) -> str:
        """The stream has ended: redact and return everything still held."""
        window = self._context + self._carry
        base = len(self._context)
        spans, _ = _spans(window, base)
        tail = _render(window, spans, base, len(window))
        self._context, self._carry, self._swallowing = "", "", False
        return tail


async def redact_stream(chunks: AsyncIterator[str], holdback: int = HOLDBACK) -> AsyncIterator[str]:
    """Async form, with the signature of Listing 4.1's `redact_stream_pipeline`."""
    redactor = StreamRedactor(holdback=holdback)
    async for chunk in chunks:
        safe = redactor.feed(chunk)
        if safe:
            yield safe
    tail = redactor.flush()
    if tail:
        yield tail


if __name__ == "__main__":
    text = ("Customer SSN is 123-45-6789. The account review is complete "
            "and the file was closed today.")
    frames = [text[i:i + 10] for i in range(0, len(text), 10)]
    redactor = StreamRedactor()
    out = "".join(redactor.feed(f) for f in frames) + redactor.flush()
    print("in  :", text)
    print("out :", out)
    print("same as redacting the whole text:", out == redact_text(text))
