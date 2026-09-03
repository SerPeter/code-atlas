"""Per-language parser benchmark over real source, with an explicit coverage ledger.

Parsing is the best benchmark target in this repository and had the least effort spent
on it: one synthetic Python corpus, asserted against `files_per_sec > 100` — a floor two
orders of magnitude below what the parser actually does, on one of twenty-two registered
languages, in a unit that changes when the machine does.

It is also the only stage that is **entirely pure CPU**. The single `open` in the whole
parsing package is unreachable from `parse_file`, which is always handed its source, so
every language here is instruction-countable and therefore comparable across machines.

**The ledger is the load-bearing part.** Every registered language must appear in exactly
one of `LANGUAGE_CORPUS` or `UNCOVERED`, and `test_every_language_is_accounted_for` fails
if one appears in neither. A language cannot be registered without a deliberate decision
about whether it is measured — the same shape `tests/integration/backends/test_conformance.py`
uses for protocol methods, and for the same reason: a silent exclusion list is
indistinguishable from an oversight.

Corpora are real source, never generated. A generator writes the language and shape its
author had in mind, which is the root cause behind every regression this epic exists to
catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.parsing.ast import parse_file

if TYPE_CHECKING:
    from collections.abc import Iterable

pytestmark = [pytest.mark.bench, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Real files already in this repository, verified to route to the expected language and
# to parse to a non-zero entity count. Directories are expanded to every file the parser
# claims; explicit paths are used where the corpus is one good file rather than a tree.
LANGUAGE_CORPUS: dict[str, tuple[str, ...]] = {
    # Vendored third-party sources, already here for the extraction coverage floors.
    "cpp": ("tests/fixtures/langcov/cpp",),
    "csharp": ("tests/fixtures/langcov/csharp",),
    "go": ("tests/fixtures/langcov/go",),
    "java": ("tests/fixtures/langcov/java",),
    "php": ("tests/fixtures/langcov/php",),
    "python": ("tests/fixtures/langcov/python",),
    "ruby": ("tests/fixtures/langcov/ruby",),
    "rust": ("tests/fixtures/langcov/rust",),
    "typescript": ("tests/fixtures/langcov/typescript",),
    # This repository's own files. Real by definition, and the ones whose parse cost the
    # maintainer actually pays.
    "c": ("tests/fixtures/langcov/cpp/test/c-test.c",),
    "containerfile": ("Dockerfile",),
    "javascript": (
        "src/code_atlas/server/web/static/map.js",
        "src/code_atlas/server/web/static/app.js",
    ),
    "json": ("tests/fixtures/langcov/cpp/floor.json",),
    "markdown": ("CHANGELOG.md", "wiki/SCHEMA.md"),
    "toml": ("pyproject.toml", "atlas.toml"),
    "yaml": ("docker-compose.yml", ".pre-commit-config.yaml"),
}

# Registered, parseable, and with **no real source anywhere in this repository**.
#
# This is a finding rather than an omission: the grammar is installed and the handler
# runs, but nothing here exercises it, so its parse cost is unmeasured and a regression
# in it would be invisible. Closing this means vendoring third-party sources the way
# `langcov` already does — deliberately out of scope for the story that added this file,
# and tracked rather than silently tolerated.
#
# Verified 2026-09-03 by extension search across the repo, excluding .venv/node_modules/.git.
UNCOVERED: dict[str, str] = {
    "apex": "no .cls file in the repo; apex tests use inline synthetic source",
    "hcl": "no .tf/.tfvars/.hcl file in the repo",
    "shell": "no .sh/.bash/.zsh file in the repo",
    "sql": "no .sql file in the repo",
    "tsx": "no .tsx/.jsx file in the repo",
    "xml": "no .xml file in the repo",
}


def _registered_languages() -> set[str]:
    from code_atlas.parsing.ast import _LANGUAGES
    from code_atlas.parsing.languages import discover_plugins

    discover_plugins()
    return {config.name for config in _LANGUAGES.values()}


def _corpus_files(language: str) -> list[tuple[str, bytes]]:
    """Every file in *language*'s corpus that the parser actually claims.

    Filtered by `get_language_for_file` rather than by extension, so a directory of
    vendored sources contributes only the files this language owns — `langcov/cpp`
    carries a `.json` and a `LICENSE` that belong to other languages or to none.
    """
    from code_atlas.parsing.ast import get_language_for_file

    out: list[tuple[str, bytes]] = []
    for entry in LANGUAGE_CORPUS.get(language, ()):
        path = _REPO_ROOT / entry
        candidates: Iterable[Path] = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in candidates:
            if not file.is_file():
                continue
            rel = file.relative_to(_REPO_ROOT).as_posix()
            source = file.read_bytes()
            config = get_language_for_file(rel, source)
            if config is not None and config.name == language:
                out.append((rel, source))
    return out


class TestCoverageLedger:
    def test_every_language_is_accounted_for(self):
        """A new language cannot be registered without deciding whether it is measured."""
        registered = _registered_languages()
        classified = set(LANGUAGE_CORPUS) | set(UNCOVERED)

        unclassified = registered - classified
        assert not unclassified, (
            f"registered but in neither LANGUAGE_CORPUS nor UNCOVERED: {sorted(unclassified)} — "
            "add a corpus or record why there is none"
        )

        stale = classified - registered
        assert not stale, f"classified but no longer registered: {sorted(stale)}"

    def test_uncovered_languages_carry_a_reason(self):
        """A bare exclusion list is indistinguishable from an oversight."""
        assert all(reason.strip() for reason in UNCOVERED.values())

    def test_the_covered_set_is_the_majority(self):
        """A tripwire on the ledger itself.

        Not a quality bar — it exists so that quietly moving languages into UNCOVERED to
        make a red benchmark green has a floor it will hit.
        """
        registered = _registered_languages()
        assert len(LANGUAGE_CORPUS) >= len(registered) // 2


class TestEveryCoveredLanguageParses:
    @pytest.mark.parametrize("language", sorted(LANGUAGE_CORPUS))
    def test_the_corpus_yields_entities(self, language: str):
        """A corpus that stopped producing entities reads exactly like a fast parser.

        Asserted per language rather than in aggregate: one rich language can carry a
        total while another silently contributes nothing.
        """
        files = _corpus_files(language)
        assert files, f"{language}: corpus resolved to no files — the paths in LANGUAGE_CORPUS are stale"

        entities = 0
        for rel, source in files:
            parsed = parse_file(rel, source, project_name="bench")
            if parsed is not None:
                entities += len(parsed.entities)

        assert entities > 0, f"{language}: parsed {len(files)} files and produced no entities"


class TestPerLanguageThroughput:
    def test_report_bytes_and_entities_for_every_covered_language(self):
        """The table a person reads. Reports, and asserts only non-vacuity.

        Wall-clock per language is printed but never asserted: it is what the existing
        `files_per_sec > 100` floor does, and that floor is why the current parser
        benchmark cannot detect anything. The gate is the instruction count below.
        """
        import json
        import time

        rows: list[dict[str, Any]] = []
        for language in sorted(LANGUAGE_CORPUS):
            files = _corpus_files(language)
            byte_total = sum(len(s) for _, s in files)
            started = time.perf_counter()
            entities = sum(
                len(parsed.entities)
                for rel, source in files
                if (parsed := parse_file(rel, source, project_name="bench")) is not None
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "language": language,
                    "files": len(files),
                    "kib": round(byte_total / 1024, 1),
                    "entities": entities,
                    "elapsed_s": round(elapsed, 4),
                    "kib_per_s": round(byte_total / 1024 / elapsed, 1) if elapsed > 0 else 0,
                }
            )

        print(f"\n{json.dumps({'languages': rows, 'uncovered': sorted(UNCOVERED)}, indent=2)}")
        assert all(int(r["entities"]) > 0 for r in rows)


class TestInstructionCounts:
    """The hardware-independent half, and the only part that can gate anything.

    Without `--codspeed` each of these is one ordinary pass over its corpus, so they cost
    a normal run only the parse they were already doing.
    """

    def test_parse_every_covered_language(self, benchmark):
        """One count across the whole multi-language corpus.

        Per-language counts would be more precise and are deliberately not done here: a
        `benchmark` fixture can only be called once per test, and twenty-two
        near-identical parametrized tests would swamp the report for a signal that a
        single total already carries — a regression in any handler moves this.
        """
        corpus = [(rel, src) for language in sorted(LANGUAGE_CORPUS) for rel, src in _corpus_files(language)]
        assert corpus, "no corpus resolved at all"

        def parse_all() -> int:
            return sum(
                len(parsed.entities)
                for rel, source in corpus
                if (parsed := parse_file(rel, source, project_name="bench")) is not None
            )

        # Reads are hoisted out of the measured callable on purpose: they are I/O, and
        # including them would make the count depend on the page cache.
        entities = benchmark(parse_all)
        assert entities > 0, "measured a parse that produced nothing"


class TestTheHotPythonPasses:
    """Instruction counts for the three passes that actually dominate the handler.

    Profiling put these at 0.887s of cumtime together, against 0.431s for the main tree
    walk. A benchmark aimed at "the tree walk" therefore measures the smaller half, which
    is why they get their own counts rather than being folded into a whole-parse total.

    They are called directly on a parsed tree. That is a deliberate coupling to private
    functions: measuring them through `parse_file` would bury each one under the other
    two and under everything else the handler does, which is the thing being avoided.
    """

    @staticmethod
    def _python_trees() -> list[tuple[str, bytes, Any, list[Any]]]:
        """Parsed trees plus the entities a real parse produced for each.

        The entities are not optional. `_extract_constant_reads` reads
        `entities[0].qualified_name` on its first line, so handing it an empty list
        raises IndexError rather than measuring an empty walk — the shortcut tried
        first here.
        """
        from tree_sitter import Parser

        from code_atlas.parsing.languages.python import _PY_LANGUAGE

        parser = Parser(_PY_LANGUAGE)
        out = []
        for rel, src in _corpus_files("python"):
            parsed = parse_file(rel, src, project_name="bench")
            if parsed is None or not parsed.entities:
                continue
            out.append((rel, src, parser.parse(src).root_node, list(parsed.entities)))
        return out

    def test_extract_text_blocks(self, benchmark):
        from code_atlas.parsing.languages.python import _extract_text_blocks

        trees = self._python_trees()
        assert trees, "no python corpus to measure"

        def run() -> int:
            total = 0
            for rel, _src, root, _ents in trees:
                entities: list[Any] = []
                rels: list[Any] = []
                _extract_text_blocks(root, rel, "bench", "m", entities, rels)
                total += len(entities)
            return total

        benchmark(run)

    def test_extract_constant_reads(self, benchmark):
        from code_atlas.parsing.languages.python import _extract_constant_reads

        trees = self._python_trees()
        assert trees, "no python corpus to measure"

        def run() -> int:
            total = 0
            for _rel, _src, root, ents in trees:
                rels: list[Any] = []
                _extract_constant_reads(root, ents, rels)
                total += len(rels)
            return total

        benchmark(run)

    def test_extract_config_refs(self, benchmark):
        from code_atlas.parsing.languages.python import _extract_config_refs

        trees = self._python_trees()
        assert trees, "no python corpus to measure"

        def run() -> int:
            total = 0
            for _rel, src, root, ents in trees:
                rels: list[Any] = []
                _extract_config_refs(root, src, ents, rels)
                total += len(rels)
            return total

        benchmark(run)
