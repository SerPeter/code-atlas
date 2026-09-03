"""Length-based text chunking.

Its own module because both sides of the pipeline need it and neither may import
the other: ``parsing`` splits oversized doc sections at parse time, ``search``
splits oversized embed texts at embed time, and ``search.embeddings`` pulls in
litellm — a dependency a parser has no business acquiring.

Pure functions only. The unit being measured (characters, tokens) is the caller's
choice, passed in as *measure*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Length-based chunking
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN_FALLBACK: int = 3
"""Chars-per-token assumed when the tokenizer for a model is unavailable.

Deliberately pessimistic. Prose runs nearer 4, but code -- punctuation-dense and
full of identifiers no vocabulary has -- runs nearer 3, and the two failure modes
are not symmetric: over-estimating splits a node one chunk sooner than it had to,
while under-estimating hands the provider an over-length input and loses the whole
batch it travelled in.
"""

_BORDER_LADDER: tuple[str, ...] = ("\n\n\n", "\n\n", "\n", ". ", ", ", " ")
"""Separators tried in order, coarsest first.

Each rung is a weaker claim about meaning: a blank line separates thoughts, a
newline separates statements, a space separates nothing at all. Descending only as
far as needed keeps chunk boundaries on the strongest border that fits.
"""


# Headroom the `_pack` fast path leaves before trusting its upper bound. Measured worst
# case is +1 token, on an empty separator only; four is generous without meaningfully
# reducing how densely chunks pack, because the band merely falls back to an exact
# measurement rather than cutting early.
_SEAM_SLACK = 4


def _pack(pieces: list[str], sep: str, limit: int, measure: Callable[[str], int]) -> list[str]:
    """Greedily re-join *pieces* into the fewest groups that each fit *limit*.

    Splitting on a separator and embedding the pieces one by one would produce a
    chunk per line; the point of the ladder is to cut at a border, not at every
    border.

    Measures each *piece* once and the accumulator only when it might not fit.

    The obvious version measures the whole growing candidate on every piece, which is
    quadratic in the input: splitting a 21 KB function into four chunks tokenized
    1,297,656 characters — 61x the input — because each of 400 lines re-tokenized
    everything accumulated before it.

    The saving rests on ``measure(a) + measure(sep) + measure(b)`` being a near-upper
    bound on ``measure(a + sep + b)``: joining two strings usually merges tokens at the
    seam and never adds many.

    It is **not** a strict upper bound, and assuming it was is a bug this nearly shipped.
    A seam can re-split a token rather than merge one: measured over 48,000 random pairs
    on cl100k_base, joining ADDED a token in the empty-separator case. The violation was
    bounded at +1, and was exactly 0 for every whitespace separator (36,000 pairs) —
    whitespace forces a token boundary, so nothing can cross it.

    So the fast path takes ``_SEAM_SLACK`` tokens of headroom before trusting the bound.
    Inside that band the exact measurement still runs, which is what the naive version did
    anyway, so the chunks produced are identical. `tests/unit/test_chunking_pack_equivalence.py`
    holds both halves: that the violation stays within the slack, and that the grouping
    matches the naive algorithm byte for byte.
    """
    out: list[str] = []
    current = ""
    current_tokens = 0
    sep_tokens = measure(sep) if sep else 0
    for piece in pieces:
        piece_tokens = measure(piece)
        if not current:
            current, current_tokens = piece, piece_tokens
            continue

        upper = current_tokens + sep_tokens + piece_tokens
        if upper + _SEAM_SLACK <= limit:
            # Far enough under that a seam re-split cannot push it over, so the exact
            # measurement is skipped. The stored count stays the bound rather than the
            # true one, which is safe because it only ever feeds a larger bound.
            current = current + sep + piece
            current_tokens = upper
            continue

        candidate = current + sep + piece
        exact = measure(candidate)
        if exact > limit:
            out.append(current)
            current, current_tokens = piece, piece_tokens
        else:
            current, current_tokens = candidate, exact
    if current:
        out.append(current)
    return out


def _hard_split(text: str, limit: int, measure: Callable[[str], int]) -> list[str]:
    """Cut *text* mid-border, by characters, when no separator got it under *limit*.

    Reached by a single unbroken run longer than the model accepts -- a minified
    bundle, a base64 blob, a one-line SQL dump. The first guess is proportional
    (characters times limit over measured units) and then shrinks until it fits, so
    a text whose tokens are unusually dense converges instead of looping.
    """
    out: list[str] = []
    rest = text
    while rest:
        if measure(rest) <= limit:
            out.append(rest)
            break
        units = max(measure(rest), 1)
        take = max(1, int(len(rest) * limit / units))
        while take > 1 and measure(rest[:take]) > limit:
            take = int(take * 0.9)
        out.append(rest[:take])
        rest = rest[take:]
    return out


class SplitResult(NamedTuple):
    """What :func:`split_embed_text` did, including what it could not keep."""

    chunks: list[str]
    """The pieces, each measuring at most the requested limit."""

    hard_split: bool
    """True when a cut landed mid-border because the ladder ran out."""

    dropped: int
    """Measured units discarded past ``max_chunks``, 0 when nothing was lost.

    A count rather than a flag, and returned rather than logged here, because this
    module is pure: only the caller knows which node lost the text and can say so.
    Before this existed the tail vanished and ``len(chunks)`` was the only signal --
    indistinguishable from a text that genuinely needed exactly that many.
    """


def split_embed_text(
    text: str,
    *,
    limit: int,
    measure: Callable[[str], int],
    max_chunks: int = 8,
) -> SplitResult:
    """Split *text* into chunks that each measure at most *limit*.

    A *limit* of 0 or less means "no known limit" and returns the text unsplit: that
    is the state a model absent from litellm's registry is in, and guessing a limit
    for it would be worse than the provider's own error.

    At most *max_chunks* are returned. The cap bounds what one pathological node can
    cost in provider calls and index entries, and :attr:`SplitResult.dropped` says
    what that cost the text.

    Chunks do **not** concatenate back to the input: the separator a cut lands on is
    consumed, so a boundary loses one blank line, newline or space. That is a
    whitespace character at a place the text was already being divided, and is left
    alone -- but it means ``"".join(chunks) == text`` is not an invariant to test
    against, and ``dropped`` is measured over the discarded chunks rather than as a
    difference from the input's own length.
    """
    if not text:
        return SplitResult([], False, 0)
    if limit <= 0 or measure(text) <= limit:
        return SplitResult([text], False, 0)

    chunks = [text]
    for sep in _BORDER_LADDER:
        nxt: list[str] = []
        for chunk in chunks:
            if measure(chunk) <= limit:
                nxt.append(chunk)
            else:
                nxt.extend(_pack(chunk.split(sep), sep, limit, measure))
        chunks = nxt
        if all(measure(c) <= limit for c in chunks):
            break

    hard_split = False
    final: list[str] = []
    for chunk in chunks:
        if measure(chunk) <= limit:
            final.append(chunk)
        else:
            hard_split = True
            final.extend(_hard_split(chunk, limit, measure))

    dropped = sum(measure(c) for c in final[max_chunks:])
    return SplitResult(final[:max_chunks], hard_split, dropped)


_FENCE_MARKERS = ("```", "~~~")


def _fence_marker(line: str) -> str | None:
    """The fence marker opening or closing on *line*, if any."""
    stripped = line.lstrip()
    return next((m for m in _FENCE_MARKERS if stripped.startswith(m)), None)


def repair_fences(parts: list[str]) -> list[str]:
    """Re-open and close code fences so each part stands alone as valid markdown.

    The border ladder cuts on blank lines, and blank lines occur *inside* fenced
    blocks -- so a long example splits into a part that opens a fence and never closes
    it, followed by parts of bare code with no fence and no language tag. Measured on
    one section: part 2 came back with zero fence markers, starting mid-code.

    A post-pass rather than a rung on the ladder, because the ladder is shared with
    code embed text, where a fence means nothing. The caller that knows its text is
    markdown asks for this.

    Re-opening repeats the original opening line, so the language tag survives into
    every continuation part. That adds a few characters to a part, which can push it
    marginally over the limit; that is preferred to handing a retrieval index an
    unlabelled code fragment.
    """
    out: list[str] = []
    carry: str | None = None
    for part in parts:
        opener = carry
        for line in part.splitlines():
            marker = _fence_marker(line)
            if marker is None:
                continue
            opener = line.rstrip() if opener is None else None
        body = (carry + "\n" if carry else "") + part
        if opener is not None:
            body += "\n" + (_fence_marker(opener) or "```")
        out.append(body)
        carry = opener
    return out
