"""ATL-191 P3-P5: resolving and reading an external package's public entrypoints.

Built on a synthetic package written to `tmp_path` and put on `sys.path`, not on whatever
this venv happens to hold. Asserting against real installed packages would make the suite
a description of the machine, and the rules are what matter -- the same reason
`TestDependencyManifests` builds a synthetic metadata table.
"""

from __future__ import annotations

import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from code_atlas.indexing.stubs import (
    extract_stub,
    read_package_stubs,
    resolve_stub_target,
)
from code_atlas.settings import LibrarySettings

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def site(tmp_path: Path, monkeypatch):
    """An importable directory, cleaned out of the module cache afterwards."""
    root = tmp_path / "site"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    yield root
    for name in [n for n in sys.modules if n.split(".")[0].startswith("synth")]:
        del sys.modules[name]


class TestResolution:
    def test_a_name_from_another_ecosystem_never_resolves(self):
        """`ghcr.io/acme/api`, `actions/checkout` and `rack/protection` are on the graph as
        ExternalPackages and are not Python at all. They keep their provenance weight and
        get no stub, which is a coverage gap rather than an error."""
        for name in ("ghcr.io/acme/api", "actions/checkout", "rack/protection", "memgraph/memgraph-mage"):
            assert resolve_stub_target(name) is None, name

    def test_an_uninstalled_distribution_resolves_to_nothing(self):
        assert resolve_stub_target("definitely_not_a_real_distribution_xyz") is None

    def test_a_bundled_pyi_wins_over_the_source_beside_it(self, site: Path):
        """A hand-written stub is signature-only and authoritative; the source is the
        fallback. Resolution order is the one thing the original design specified."""
        _write(site, "synthpkg/__init__.py", "def real(): ...")
        _write(site, "synthpkg/__init__.pyi", "def stubbed() -> int: ...")

        target = resolve_stub_target("synthpkg")

        assert target is not None
        assert target.kind == "bundled-pyi"
        assert target.entry.suffix == ".pyi"

    def test_a_stubs_distribution_is_found_beside_the_package(self, site: Path):
        _write(site, "synthonly/__init__.py", "def real(): ...")
        _write(site, "synthonly-stubs/__init__.pyi", "def real() -> int: ...")

        target = resolve_stub_target("synthonly")

        assert target is not None
        assert target.kind == "stubs-package"

    def test_a_frozen_stdlib_module_still_resolves_to_its_source(self):
        """`os`, `io`, `abc` and `codecs` have been frozen since 3.11 -- `find_spec`
        reports `origin == "frozen"` and no path. `os` alone is one of the most-imported
        names on any graph, so treating frozen as unresolvable lost it entirely."""
        target = resolve_stub_target("os")

        assert target is not None, "a frozen stdlib module must still find its Lib/ source"
        assert target.kind == "frozen-stdlib"
        assert target.entry.name == "os.py"

    def test_a_stdlib_module_is_versioned_by_the_interpreter(self):
        """It has no distribution, so without this it carries no version and is re-read on
        every index. The interpreter version is exactly what changes when it changes."""
        target = resolve_stub_target("json")

        assert target is not None
        assert target.version.startswith("python-")


class TestExtraction:
    def test_all_is_the_public_surface_and_local_definitions_carry_signatures(self, site: Path):
        """A name in `__all__` with no reachable definition is still recorded. Knowing
        that `pydantic.BaseModel` exists is most of the value even without its signature."""
        _write(
            site,
            "synthall/__init__.py",
            '''
            __all__ = ["visible", "Widget", "reexported"]

            def visible(a: int, b: str = "x") -> bool: ...

            class Widget:
                """A widget."""

            def _private(): ...

            def not_exported(): ...
            ''',
        )

        target = resolve_stub_target("synthall")
        assert target is not None
        result = extract_stub(target)

        by_name = {s.name: s for s in result.symbols}
        assert set(by_name) == {"visible", "Widget", "reexported"}
        assert "a: int" in by_name["visible"].signature
        assert by_name["Widget"].docstring == "A widget."
        assert by_name["reexported"].signature == "", "nothing defines it, so there is no signature to invent"
        assert by_name["reexported"].resolved is False
        assert "_private" not in by_name
        assert "not_exported" not in by_name, "__all__ is the package saying what it exports"

    def test_a_package_with_no_all_exposes_what_it_re_exports(self, site: Path):
        """`jinja2`, `typer` and `tiktoken` define nothing in `__init__.py` and declare no
        `__all__` -- their whole surface is `from .x import Y`. Reading only local
        definitions gave each of them **zero** entrypoints."""
        _write(site, "synthre/__init__.py", "from .core import Engine, run\nfrom typing import Any\n")
        _write(
            site,
            "synthre/core.py",
            """
            class Engine: ...

            def run(times: int) -> None: ...
            """,
        )

        target = resolve_stub_target("synthre")
        assert target is not None
        result = extract_stub(target)

        by_name = {s.name: s for s in result.symbols}
        assert set(by_name) == {"Engine", "run"}
        assert "times: int" in by_name["run"].signature, "the re-export hop must reach the definition"
        assert "Any" not in by_name, "an absolute import is a dependency, not an entrypoint"

    def test_a_package_that_re_exports_from_itself_by_absolute_path(self, site: Path):
        """`pathlib/__init__.py` writes `from pathlib._local import Path`. Reading that as
        a foreign dependency cost pathlib every signature it had."""
        _write(site, "synthself/__init__.py", '__all__ = ["Thing"]\nfrom synthself._impl import Thing\n')
        _write(site, "synthself/_impl.py", "class Thing:\n    pass\n")

        target = resolve_stub_target("synthself")
        assert target is not None
        result = extract_stub(target)

        assert [s.name for s in result.symbols] == ["Thing"]
        assert result.symbols[0].resolved is True

    def test_a_star_only_entrypoint_still_has_a_surface(self, site: Path):
        """`asyncio`, `sqlite3` and `urllib` are nothing but `from .x import *`, and their
        `__all__` is *computed* from the submodules' -- which no literal read evaluates.
        All three yielded zero entrypoints before the star modules were scanned."""
        _write(site, "synthstar/__init__.py", "from .alpha import *\nfrom .beta import *\n")
        _write(
            site,
            "synthstar/alpha.py",
            '__all__ = ["exported"]\n\ndef exported(n: int) -> int: ...\n\ndef hidden(): ...\n',
        )
        _write(site, "synthstar/beta.py", "class Beta: ...\n")

        target = resolve_stub_target("synthstar")
        assert target is not None
        result = extract_stub(target)

        by_name = {s.name: s for s in result.symbols}
        assert "exported" in by_name
        assert "Beta" in by_name
        assert "hidden" not in by_name, "a star import honours the target module's own __all__"

    def test_full_index_reads_the_whole_tree(self, site: Path):
        """The entrypoint gives the surface; `full_index` gives the internals, for a
        library where understanding the implementation matters."""
        _write(site, "synthdeep/__init__.py", '__all__ = ["Front"]\n\nclass Front: ...\n')
        _write(site, "synthdeep/inner/__init__.py", "")
        _write(site, "synthdeep/inner/engine.py", "def buried(x: float) -> float: ...\n")

        target = resolve_stub_target("synthdeep")
        assert target is not None

        surface = {s.name for s in extract_stub(target).symbols}
        internals = {s.name for s in extract_stub(target, full=True).symbols}

        assert surface == {"Front"}
        assert "buried" in internals, "full_index must reach a module the entrypoint never names"


class TestConfiguration:
    """`[libraries]` was documented, declared and read by nothing at all before ATL-191."""

    def test_stubs_false_reads_nothing(self, site: Path):
        _write(site, "synthoff/__init__.py", "def thing(): ...\n")

        assert read_package_stubs(["synthoff"], LibrarySettings(stubs=False)) == {}

    def test_stub_index_restricts_and_empty_means_everything(self, site: Path):
        _write(site, "syntha/__init__.py", "def a(): ...\n")
        _write(site, "synthb/__init__.py", "def b(): ...\n")

        everything = read_package_stubs(["syntha", "synthb"], LibrarySettings())
        restricted = read_package_stubs(["syntha", "synthb"], LibrarySettings(stub_index=["syntha"]))

        assert set(everything) == {"syntha", "synthb"}
        assert set(restricted) == {"syntha"}

    def test_full_index_implies_stubbing_even_against_stub_index(self, site: Path):
        """Asking for a library's internals is asking for its surface. Without this a
        `full_index` entry missing from `stub_index` would be silently ignored."""
        _write(site, "synthc/__init__.py", "def c(): ...\n")
        _write(site, "synthd/__init__.py", "def d(): ...\n")

        results = read_package_stubs(
            ["synthc", "synthd"], LibrarySettings(stub_index=["synthc"], full_index=["synthd"])
        )

        assert set(results) == {"synthc", "synthd"}

    def test_an_unchanged_version_is_skipped_before_anything_is_read(self):
        """The order is the point. Resolving is a `find_spec` and a `stat`; extracting is
        181 files and 6s. Checking afterwards -- which is how this was first written --
        did all the work and then threw it away, so a re-index whose dependencies had not
        moved still paid in full."""
        first = read_package_stubs(["json"], LibrarySettings())
        assert "json" in first
        version = first["json"].target.version
        assert version

        again = read_package_stubs(["json"], LibrarySettings(), {"json": version})
        assert again == {}, "an unchanged package must not be read at all"

        moved = read_package_stubs(["json"], LibrarySettings(), {"json": "python-0.0.0"})
        assert "json" in moved, "a version that moved must be re-read"
        assert moved["json"].files_read > 0

    def test_introspection_is_off_by_default(self, site: Path, monkeypatch):
        """It runs the package's import-time code inside the indexer. The default path
        must never import anything, so a package that explodes on import is still safe to
        resolve and read statically."""
        _write(site, "synthboom/__init__.py", '__all__ = ["thing"]\n\nraise RuntimeError("import-time boom")\n')

        assert LibrarySettings().introspect is False

        target = resolve_stub_target("synthboom")
        assert target is not None
        result = extract_stub(target)  # would raise if anything imported it
        assert [s.name for s in result.symbols] == ["thing"]

    def test_introspection_survives_a_package_that_cannot_be_imported(self, site: Path):
        """An arbitrary third-party import can raise anything at all. A failure to enrich
        keeps the static read rather than failing the index."""
        _write(site, "synthboom2/__init__.py", '__all__ = ["thing"]\n\nraise RuntimeError("import-time boom")\n')

        target = resolve_stub_target("synthboom2")
        assert target is not None
        result = extract_stub(target, introspect=True)

        assert [s.name for s in result.symbols] == ["thing"]

    def test_introspection_reaches_what_a_static_read_cannot(self, site: Path):
        """A name produced at runtime has no source to parse. This is the whole reason the
        option exists -- compiled extensions and metaclass-generated classes."""
        _write(
            site,
            "synthdyn/__init__.py",
            """
            __all__ = ["Generated"]

            Generated = type("Generated", (), {"__doc__": "Built at import time."})
            """,
        )

        target = resolve_stub_target("synthdyn")
        assert target is not None

        # Statically this is a module-level assignment and nothing more: no `class
        # Generated` exists in the source, so there is no kind, no signature, no docstring.
        static = {s.name: s for s in extract_stub(target).symbols}
        assert static["Generated"].kind == "variable"
        assert static["Generated"].signature == ""
        assert static["Generated"].docstring == ""

        introspected = {s.name: s for s in extract_stub(target, introspect=True).symbols}
        assert introspected["Generated"].kind == "class", "only the live object knows it is a class"
        assert introspected["Generated"].docstring == "Built at import time."
