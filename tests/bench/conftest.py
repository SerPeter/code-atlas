"""Shared fixtures for performance benchmarks.

Provides a deterministic synthetic Python codebase generator with three
size presets (small, medium, large) and session-scoped fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic codebase generator
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, int]] = {
    "small": {"files": 100, "classes_per_file": 2, "methods_per_class": 5},
    "medium": {"files": 1000, "classes_per_file": 2, "methods_per_class": 5},
    "large": {"files": 5000, "classes_per_file": 1, "methods_per_class": 6},
}


def generate_synthetic_codebase(root: Path, preset: str = "small") -> list[str]:
    """Generate a deterministic Python project and return relative file paths.

    Each file contains classes with methods, module-level functions, cross-file
    imports, and docstrings — exercising all parser paths.

    **Call sites must resolve to project callables.** Until 2026-08-07 every
    generated body called only ``len()`` and ``sum()``, so the corpus had full
    node scale and no resolvable calls at all: `_resolve_one_call` found neither
    name in the project lookup, the CALLS write received an empty set, and the
    query never ran. That is why the ``large`` preset — 5,000 files, ~35,000
    callables — still failed to expose a CALLS write that timed out on a real
    72-file C++ project. A benchmark with node scale but no edge density measures
    the wrong half of indexing.
    """
    cfg = _PRESETS[preset]
    n_files = cfg["files"]
    classes = cfg["classes_per_file"]
    methods = cfg["methods_per_class"]

    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")

    rel_paths: list[str] = ["src/__init__.py"]

    for i in range(n_files):
        # Distribute files across sub-packages to exercise package detection
        pkg_idx = i % 10
        pkg_dir = src_dir / f"pkg_{pkg_idx}"
        pkg_dir.mkdir(exist_ok=True)
        init_path = pkg_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")
            rel_paths.append(f"src/pkg_{pkg_idx}/__init__.py")

        fname = f"mod_{i:04d}.py"
        fpath = pkg_dir / fname
        rel_path = f"src/pkg_{pkg_idx}/{fname}"

        lines: list[str] = []

        # Import from a "previous" module to create cross-file references
        if i > 0:
            prev_pkg = (i - 1) % 10
            prev_mod = f"mod_{i - 1:04d}"
            lines.append(f"from src.pkg_{prev_pkg}.{prev_mod} import Class0_{i - 1}")
        lines.append("")

        # Module-level constant
        lines.append(f'MODULE_CONST_{i} = "value_{i}"')
        lines.append("")

        # Classes with methods
        for c in range(classes):
            cname = f"Class{c}_{i}"
            lines.append(f"class {cname}:")
            lines.append(f'    """Docstring for {cname}."""')
            lines.append("")
            for m in range(methods):
                mname = f"method_{m}"
                lines.append(f"    def {mname}(self, x: int, y: str = 'default') -> bool:")
                lines.append(f'        """Compute {mname} for {cname}."""')
                # Calls a function defined in THIS module, so it resolves to a
                # project Callable. `len(y)` alone left the corpus with call
                # sites whose only targets were builtins — see below.
                lines.append(f"        return module_func_{i}([x]) > len(y) + {m}")
                lines.append("")

        # Module-level function. When there is a previous module, instantiate its
        # class and call a method on it: that is the cross-file, receiver-typed
        # shape the resolver actually has to work for.
        lines.append(f"def module_func_{i}(data: list[int]) -> int:")
        lines.append(f'    """Process data in module {i}."""')
        if i > 0:
            lines.append(f"    helper = Class0_{i - 1}()")
            lines.append('    if helper.method_0(len(data), "x"):')
            lines.append("        return sum(data) + 1")
        lines.append("    return sum(data)")
        lines.append("")

        fpath.write_text("\n".join(lines), encoding="utf-8")
        rel_paths.append(rel_path)

    rel_paths.sort()
    return rel_paths


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bench_small(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    """Small synthetic codebase (~100 files, ~1300 entities)."""
    root = tmp_path_factory.mktemp("bench_small")
    paths = generate_synthetic_codebase(root, "small")
    return root, paths


@pytest.fixture(scope="session")
def bench_medium(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    """Medium synthetic codebase (~1000 files, ~13000 entities)."""
    root = tmp_path_factory.mktemp("bench_medium")
    paths = generate_synthetic_codebase(root, "medium")
    return root, paths


@pytest.fixture(scope="session")
def bench_large(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    """Large synthetic codebase (~5000 files, ~35000 entities)."""
    root = tmp_path_factory.mktemp("bench_large")
    paths = generate_synthetic_codebase(root, "large")
    return root, paths


def write_bench_result(name: str, data: dict, tmp_path_factory: pytest.TempPathFactory | None = None) -> None:
    """Write benchmark result as JSON for later aggregation."""
    out_dir = Path("tests/bench/.results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
