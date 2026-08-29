"""The length-preserving source shims, as properties rather than examples.

Apex and dbt are both parsed by borrowing another language's grammar behind a rewrite of
the source. The rewrite's whole contract is one invariant: it may change bytes, but never
their count and never the position of a newline. Break either and every line number the
parser reports is silently wrong -- entities land on the wrong line, and nothing fails.

That is a property, not a set of cases: it has to hold for input nobody thought to write
down, which is exactly what example-based tests cannot reach.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from code_atlas.parsing.languages.apex import _shim
from code_atlas.parsing.languages.sql import _neutralize_jinja

# Deliberately not arbitrary bytes: both shims run on decoded source, and the adversarial
# input that matters is unbalanced delimiters, not invalid UTF-8.
_CHARS = "{}%[]()<>=_.,;:`abcXYZ019 \t\r\n" + chr(34) + chr(39)
_TOKENS = ["SELECT", "FROM", "ref", "source", "trigger", "{{", "}}", "{%", "%}"]
_SOURCE = st.lists(st.one_of(st.sampled_from(list(_CHARS)), st.sampled_from(_TOKENS)), max_size=120).map("".join)


def _newline_offsets(data: bytes) -> list[int]:
    return [i for i, b in enumerate(data) if b == 0x0A]


@given(_SOURCE)
@settings(max_examples=300, deadline=None)
def test_jinja_neutralisation_preserves_length_and_lines(source: str) -> None:
    raw = source.encode()
    out = _neutralize_jinja(raw)

    assert len(out) == len(raw), "byte count changed -- every later line number is now wrong"
    assert _newline_offsets(out) == _newline_offsets(raw), (
        "newlines moved, so dbt entities would be reported on the wrong lines"
    )


@given(_SOURCE)
@settings(max_examples=300, deadline=None)
def test_apex_shim_preserves_length_and_lines(source: str) -> None:
    raw = source.encode()
    out, _facts = _shim(raw, allow_trigger=True)

    assert len(out) == len(raw), "byte count changed -- every later line number is now wrong"
    assert _newline_offsets(out) == _newline_offsets(raw), (
        "newlines moved, so the Java grammar would report Apex entities on the wrong lines"
    )


@given(_SOURCE)
@settings(max_examples=200, deadline=None)
def test_neither_shim_raises_on_unbalanced_delimiters(source: str) -> None:
    """An unterminated `{{`, a stray `%}`, a `[` with no `]`.

    Real files are saved mid-keystroke and the watcher parses them anyway, so a shim that
    raises on half-written input takes the whole batch down with it.
    """
    raw = source.encode()
    _neutralize_jinja(raw)
    _shim(raw, allow_trigger=True)
