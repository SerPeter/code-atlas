"""`_pack`'s fast path must produce byte-identical output to the naive algorithm.

`_pack` used to measure the whole growing candidate on every piece, which is quadratic:
splitting a 21 KB function tokenized 1,297,656 characters, 61x the input. It now measures
each piece once and consults the accumulator exactly only when an upper bound says the
candidate might not fit.

The first version of that fast path assumed token counts are strictly **subadditive**
under concatenation — that joining two strings can merge tokens at the seam but never add
any. **That is false, and this test is what found it**: on cl100k_base with an empty
separator, joining added a token (74 -> 75). A seam can re-split a token as readily as
merge one.

The violation is bounded, so `_pack` leaves `_SEAM_SLACK` tokens of headroom rather than
trusting the bound outright. Both halves are asserted here: that a seam stays inside that
headroom, and that the grouping still matches the naive algorithm byte for byte.

Without the first assertion the optimisation would keep chunks fractionally over the
limit, and a provider would reject a whole batch somewhere far from this code.
"""

from __future__ import annotations

import random

import pytest

from code_atlas.chunking import _SEAM_SLACK, _pack

tiktoken = pytest.importorskip("tiktoken")
_ENC = tiktoken.get_encoding("cl100k_base")


def _measure(text: str) -> int:
    return len(_ENC.encode(text))


def _pack_naive(pieces: list[str], sep: str, limit: int, measure) -> list[str]:
    """The implementation this replaced, kept as the oracle."""
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


def _corpora(seed: int) -> list[list[str]]:
    """Piece lists with the shapes the ladder actually produces.

    Includes single oversized pieces and empty strings, because those are the cases where
    an upper-bound shortcut is most likely to diverge from an exact measurement.
    """
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma_value", "compute", "x", "return", "self.method", "###", ""]
    out: list[list[str]] = [
        [],
        [""],
        ["one"],
        ["a", "b", "c"],
        ["x" * 5000],
        ["short", "y" * 4000, "short"],
    ]
    out.extend(
        [" ".join(rng.choices(words, k=rng.randint(1, 30))) for _ in range(rng.randint(1, 40))] for _ in range(24)
    )
    return out


class TestSeamBehaviour:
    """The property the fast path is safe under, checked against a real tokenizer."""

    @pytest.mark.parametrize("sep", ["\n\n", "\n", " ", ""])
    def test_a_seam_never_adds_more_than_the_slack(self, sep):
        """The bound, not the absolute claim.

        The first version asserted `joined <= apart` and failed: with an empty separator
        a seam re-splits a token and the join costs one more. What has to hold is that
        the addition stays inside the headroom `_pack` reserves.
        """
        rng = random.Random(11)
        alphabet = "abcdefghij klmnop\n\t_.(){}=+#"
        worst = 0
        for _ in range(400):
            a = "".join(rng.choices(alphabet, k=rng.randint(0, 160)))
            b = "".join(rng.choices(alphabet, k=rng.randint(0, 160)))
            violation = _measure(a + sep + b) - (_measure(a) + _measure(sep) + _measure(b))
            worst = max(worst, violation)

        assert worst <= _SEAM_SLACK, (
            f"a seam added {worst} tokens with sep={sep!r}, past the {_SEAM_SLACK}-token headroom "
            "`_pack` reserves — the fast path would keep chunks over the model's limit"
        )

    def test_whitespace_separators_do_not_move_the_seam_at_all(self):
        """Why the real separators are the safe case.

        Whitespace forces a token boundary, so nothing merges or splits across it. The
        ladder only ever uses whitespace, which is why the slack is headroom rather than
        something the pipeline leans on.
        """
        rng = random.Random(3)
        alphabet = "abcdefghij klmnop\n\t_.(){}=+#"
        for sep in ("\n\n", "\n", " "):
            for _ in range(300):
                a = "".join(rng.choices(alphabet, k=rng.randint(0, 160)))
                b = "".join(rng.choices(alphabet, k=rng.randint(0, 160)))
                joined = _measure(a + sep + b)
                apart = _measure(a) + _measure(sep) + _measure(b)
                assert joined <= apart, f"whitespace seam moved a token boundary: sep={sep!r}"


class TestPackEquivalence:
    @pytest.mark.parametrize("sep", ["\n\n", "\n", " "])
    @pytest.mark.parametrize("limit", [8, 32, 200])
    def test_identical_to_the_naive_algorithm(self, sep, limit):
        """Byte-identical grouping, not merely 'chunks that fit'.

        A weaker assertion would pass for an implementation that cuts at different
        borders, which changes what every downstream vector represents.
        """
        for pieces in _corpora(seed=limit):
            fast = _pack(pieces, sep, limit, _measure)
            naive = _pack_naive(pieces, sep, limit, _measure)
            assert fast == naive, f"diverged on {pieces[:3]}... sep={sep!r} limit={limit}"

    def test_the_fast_path_actually_saves_work(self):
        """Non-vacuity: if it measured the same characters, there was no optimisation.

        Counts characters rather than calls — tokenizer cost tracks text length, and the
        new version deliberately makes *more* calls on *less* text.
        """
        pieces = [f"    step_{i} = compute({i}) + offset_{i}" for i in range(400)]

        fast_chars = 0
        naive_chars = 0

        def measure_fast(text: str) -> int:
            nonlocal fast_chars
            fast_chars += len(text)
            return _measure(text)

        def measure_naive(text: str) -> int:
            nonlocal naive_chars
            naive_chars += len(text)
            return _measure(text)

        assert _pack(pieces, "\n", 2000, measure_fast) == _pack_naive(pieces, "\n", 2000, measure_naive)
        assert fast_chars * 3 < naive_chars, (
            f"the fast path tokenized {fast_chars:,} characters against the naive {naive_chars:,} — "
            "less than a 3x saving means the quadratic behaviour is still there"
        )


class TestTokenCache:
    """`count_tokens` memoises, because the splitter asks the same question repeatedly.

    Descending the ladder, `split_embed_text` measures a chunk to decide whether it fits,
    measures it again to decide whether to keep descending, and measures it once more in
    the final pass. Identical strings, three encodes.

    Measured on a 21 KB function through a real client: 270,461 characters encoded
    without the cache, 84,584 with it — same chunks either way.
    """

    @staticmethod
    def _client():
        from code_atlas.search.embeddings import EmbedClient
        from code_atlas.settings import EmbeddingSettings

        return EmbedClient(
            EmbeddingSettings(provider="litellm", model="text-embedding-3-small", dimension=32, max_input_tokens=2000)
        )

    def test_a_repeated_text_is_encoded_once(self):
        import litellm

        client = self._client()
        real = litellm.encode
        calls = {"n": 0}

        def counting(**kwargs):
            calls["n"] += 1
            return real(**kwargs)

        litellm.encode = counting
        try:
            first = client.count_tokens("def helper(x):\n    return x + 1\n")
            for _ in range(9):
                client.count_tokens("def helper(x):\n    return x + 1\n")
        finally:
            litellm.encode = real

        assert calls["n"] == 1, f"ten identical measurements cost {calls['n']} encodes"
        assert first > 0

    def test_the_cache_is_bounded(self):
        """Chunk texts run to kilobytes, so an unbounded cache is a slow leak in a daemon
        that indexes for hours."""
        from code_atlas.search.embeddings import _TOKEN_CACHE_SIZE

        client = self._client()
        for i in range(_TOKEN_CACHE_SIZE + 50):
            client.count_tokens(f"unique text number {i}")

        assert len(client._token_cache) <= _TOKEN_CACHE_SIZE, (
            f"cache grew to {len(client._token_cache)} past its {_TOKEN_CACHE_SIZE} bound"
        )

    def test_caching_does_not_change_the_answer(self):
        """A cache that returned a different number would silently move every chunk
        boundary."""
        import litellm

        client = self._client()
        texts = ["short", "a" * 4000, "def f():\n    pass\n", ""]
        cached = [client.count_tokens(t) for t in texts]
        direct = [len(litellm.encode(model=client._model, text=t)) for t in texts]

        assert cached == direct
