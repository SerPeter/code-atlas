"""Per-language extraction-coverage floors, measured against real code.

Synthetic snippets passed in every language while TypeScript dropped nine calls
in ten (ATL-096) — the shapes that break a walker are the ones nobody writes by
hand. So each fixture directory holds trimmed files from a real open-source
repo, and this asserts that the walker still finds what it found the day the
floor was recorded.

To add a language, create ``tests/fixtures/langcov/<name>/`` containing a
``floor.json`` and the vendored sources. Nothing else needs editing — that is
deliberate, so language work can proceed on separate branches without
colliding on a shared table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.langcov import LANGS, measure

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "langcov"


def _fixture_dirs() -> list[Path]:
    if not FIXTURES.is_dir():
        return []
    return sorted(p for p in FIXTURES.iterdir() if p.is_dir() and (p / "floor.json").is_file())


@pytest.mark.parametrize("fixture", _fixture_dirs(), ids=lambda p: p.name)
def test_extraction_coverage_holds_its_floor(fixture: Path) -> None:
    floor = json.loads((fixture / "floor.json").read_text(encoding="utf-8"))
    lang = floor["lang"]
    assert lang in LANGS, f"{fixture.name}: unknown language {lang!r}"

    cov = measure(fixture, lang)
    assert cov.files > 0, f"{fixture.name}: no {lang} files found — fixture is empty or extensions are wrong"
    assert cov.failed == 0, f"{fixture.name}: {cov.failed} file(s) failed to parse"

    assert cov.named_funcs >= floor["named_funcs"], (
        f"{fixture.name}: named-function capture regressed to {cov.named_funcs:.3f}, "
        f"floor is {floor['named_funcs']:.3f}. Run "
        f"`python -m tests.support.langcov tests/fixtures/langcov/{fixture.name} {lang}` to see which forms."
    )
    assert cov.calls >= floor["calls"], (
        f"{fixture.name}: call extraction regressed to {cov.calls:.3f}, floor is {floor['calls']:.3f}. "
        f"{cov.calls_in_missed} call(s) sit inside a function that produced no entity."
    )

    # A uid is the graph's identity. Two definitions emitting the same one merge
    # into a single node with an arbitrary winner's source and the union of both
    # edge sets — a confident wrong answer, which is worse than the silence of a
    # missing entity. Ceilings are declared per language and ratchet downward;
    # absent means zero.
    ceiling = floor.get("max_duplicate_uids", 0)
    assert cov.duplicate_uids <= ceiling, (
        f"{fixture.name}: {cov.duplicate_uids} colliding uid(s), ceiling is {ceiling}. "
        f"Worst: {cov.worst_collisions[:3]}. Two definitions are merging into one graph node."
    )


@pytest.mark.parametrize("fixture", _fixture_dirs(), ids=lambda p: p.name)
def test_floor_records_its_provenance(fixture: Path) -> None:
    """A floor is only justifiable if you can tell where the code came from."""
    floor = json.loads((fixture / "floor.json").read_text(encoding="utf-8"))
    for key in ("lang", "named_funcs", "calls", "source_repo", "source_commit", "license", "rationale"):
        assert key in floor, f"{fixture.name}: floor.json is missing {key!r}"
    assert 0.0 <= floor["named_funcs"] <= 1.0
    assert 0.0 <= floor["calls"] <= 1.0


def test_the_corpus_is_not_empty() -> None:
    """Guard the parametrisation itself.

    Every assertion above is parametrised over discovered directories, so an
    empty or mislocated fixture root turns the whole module into a silent pass.
    """
    assert _fixture_dirs(), f"no language fixtures found under {FIXTURES}"
