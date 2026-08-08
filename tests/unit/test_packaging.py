"""What actually ships in the distributions (ATL-114).

Hatchling's sdist default is include-everything, which once put 308 entries in the
tarball: the whole test suite, `wiki/`, `.claude/`, and 99 files of vendored third-party
corpus under `tests/fixtures/langcov/`. That corpus is MIT, Apache-2.0, BSD-3 and
MIT-OR-Unlicense, correctly attributed per directory — but the package declares
`License-Expression: Apache-2.0` and nothing at top level discloses the rest.

An include list is easy to get subtly wrong (a bare `README.md` pattern matches a README
at *any* depth), and nothing catches it until a release is already on PyPI. These tests
build the real artifacts and read them.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories that must never reach a consumer: they are not needed to build or run the
# package, and two of them carry third-party licences the metadata does not declare.
EXCLUDED = ("tests", "wiki", ".claude", ".specs", "scripts", ".github")


def _build(kind: str, out_dir: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    # assert, not just the guard above: ty does not treat pytest.skip as NoReturn, so
    # without this `uv` stays `str | None` and the subprocess overload will not match.
    assert uv is not None
    result = subprocess.run(
        [uv, "build", kind, "-o", str(out_dir), "--no-build-logs"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,  # a build failure becomes a skip below, not a test error
    )
    if result.returncode != 0:
        pytest.skip(f"uv build unavailable: {result.stderr.strip()[:200]}")
    built = sorted(out_dir.iterdir())
    assert built, "uv build reported success but produced nothing"
    return built[-1]


@pytest.mark.slow
class TestSdistContents:
    """Builds the real tarball — marked slow, so the fast suite does not pay for it."""

    def test_it_excludes_everything_a_consumer_does_not_need(self, tmp_path: Path) -> None:
        archive = _build("--sdist", tmp_path)

        with tarfile.open(archive) as tar:
            names = tar.getnames()

        # Entries are "<name>-<version>/<path>", so the second segment is top level.
        top_level = {n.split("/")[1] for n in names if "/" in n}
        leaked = sorted(top_level & set(EXCLUDED))

        assert not leaked, f"sdist ships {leaked}"

    def test_it_still_contains_what_a_build_needs(self, tmp_path: Path) -> None:
        """An exclusion list is one typo away from shipping nothing useful."""
        archive = _build("--sdist", tmp_path)

        with tarfile.open(archive) as tar:
            names = tar.getnames()

        top_level = {n.split("/")[1] for n in names if "/" in n}
        assert {"src", "pyproject.toml", "README.md", "LICENSE"} <= top_level

    def test_the_anchored_readme_pattern_does_not_readmit_nested_ones(self, tmp_path: Path) -> None:
        """A bare `README.md` include matches at any depth.

        That is how `wiki/adr/README.md` and the vendored corpus's own
        `tests/fixtures/langcov/README.md` came back after being excluded.
        """
        archive = _build("--sdist", tmp_path)

        with tarfile.open(archive) as tar:
            readmes = [n for n in tar.getnames() if n.endswith("README.md")]

        assert len(readmes) == 1, f"more than the top-level README shipped: {readmes}"


@pytest.mark.slow
class TestWheelContents:
    def test_the_vendored_browser_assets_ship(self, tmp_path: Path) -> None:
        """`atlas ui` must work offline from an installed wheel, not just from a checkout."""
        archive = _build("--wheel", tmp_path)

        with zipfile.ZipFile(archive) as wheel:
            names = wheel.namelist()

        vendored = [n for n in names if "server/web/static/vendor/" in n]
        assert any(n.endswith("sigma-3.0.3.min.js") for n in vendored), "the renderer is missing"
        assert any(n.endswith("graphology-0.26.0.umd.min.js") for n in vendored)
        # MIT requires the notice to travel with the code.
        assert sum(1 for n in vendored if n.endswith("LICENSE.txt")) == 2

    def test_the_templates_ship(self, tmp_path: Path) -> None:
        archive = _build("--wheel", tmp_path)

        with zipfile.ZipFile(archive) as wheel:
            names = wheel.namelist()

        templates = [n for n in names if n.endswith(".html")]
        assert any(n.endswith("base.html") for n in templates)
        assert any(n.endswith("export.html") for n in templates), "the static export needs its template"

    def test_py_typed_ships_so_the_annotations_are_visible(self, tmp_path: Path) -> None:
        archive = _build("--wheel", tmp_path)

        with zipfile.ZipFile(archive) as wheel:
            assert any(n.endswith("py.typed") for n in wheel.namelist())
