"""Length-based text chunking.

Its own module because both sides of the pipeline need it and neither may import
the other: ``parsing`` splits oversized doc sections at parse time, ``search``
splits oversized embed texts at embed time, and ``search.embeddings`` pulls in
litellm — a dependency a parser has no business acquiring.

Pure functions only. The unit being measured (characters, tokens) is the caller's
choice, passed in as *measure*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def _pack(pieces: list[str], sep: str, limit: int, measure: Callable[[str], int]) -> list[str]:
    """Greedily re-join *pieces* into the fewest groups that each fit *limit*.

    Splitting on a separator and embedding the pieces one by one would produce a
    chunk per line; the point of the ladder is to cut at a border, not at every
    border.
    """
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + sep + piece
        if current and measure(candidate) > limit:
            out.append(current)
            current = piece
        else:
            current = candidate
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


def split_embed_text(
    text: str,
    *,
    limit: int,
    measure: Callable[[str], int],
    max_chunks: int = 8,
) -> tuple[list[str], bool]:
    """Split *text* into chunks that each measure at most *limit*.

    Returns ``(chunks, hard_split)``. *hard_split* is True when at least one cut
    landed mid-border because :data:`_BORDER_LADDER` ran out -- the caller uses it
    to say *why* a node was split, not whether.

    A *limit* of 0 or less means "no known limit" and returns the text unsplit:
    that is the state a model absent from litellm's registry is in, and guessing a
    limit for it would be worse than the provider's own error.

    At most *max_chunks* are returned. The cap bounds what one pathological node
    can cost, in provider calls and in index entries; the tail past it is dropped
    and the caller is told by the length it gets back.
    """
    if not text:
        return [], False
    if limit <= 0 or measure(text) <= limit:
        return [text], False

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

    return final[:max_chunks], hard_split
