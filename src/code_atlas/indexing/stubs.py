"""Read the public API surface of an installed external package (ATL-191 P3/P4).

An `ExternalPackage` is otherwise a bare name. This module answers "what can I call on
it" by reading the package's **entrypoint** -- the single file its public names come
from -- rather than its whole tree. On this venv that is the difference between 1 MB and
37 MB, and `litellm` alone accounts for 2,068 files of the larger number.

A new module rather than more of `orchestrator.py` (which is already 2,900 lines):
resolution, extraction and introspection are one subsystem with one entry point, and
nothing else in indexing needs any of its internals.

Four things worth knowing before changing any of it.

**Signatures and docstrings come from the existing Python parser**, not from a second
implementation. `parse_file` already formats a signature, extracts a docstring, decides
`visibility` and maps to a `NodeLabel`, and `.pyi` is a registered Python extension, so a
type stub parses with no special case. This module only adds what `parse_file` does not
expose: the `__all__` list and the re-export map, which need import-and-assignment
analysis specific to Python. That read is 40 lines of `ast` and is deliberately not a
second parser.

**Static reading cannot see everything, and the gap is structural.** Measured across 21
packages here: 100% of public entrypoint *names*, 26% of them with a signature. The
misses are `from .x import *` (reachable with more work), lazy `__getattr__` (undecidable
-- the name-to-module map is computed at runtime), compiled extensions (no Python source
exists) and runtime-generated APIs (the class does not exist until a metaclass runs).
`[libraries] introspect` trades safety for the last three.

**Only atlas's own environment is visible.** `importlib.util.find_spec` searches this
process's path. Indexing somebody else's repo, most imports resolve to nothing and keep
the provenance weight they already had. A coverage gap, not an error, and the same
limitation `_distribution_import_names` already documents.

**Nothing is imported unless `introspect` is on.** `find_spec` locates a module without
executing it; `import_module` runs its import-time code, and in the wild that means
network calls, CUDA initialisation and thread spawning. The default path never does it.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.metadata
import importlib.util
import inspect
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from code_atlas.parsing.ast import parse_file
from code_atlas.schema import NodeLabel, Visibility

if TYPE_CHECKING:
    from code_atlas.settings import LibrarySettings

# A package's whole tree is only read when it is named in `full_index`; even then, stop
# somewhere. 25 MB of litellm is not an accident somebody wants to discover at index time.
_MAX_FULL_INDEX_FILES = 2_000

# The re-export hop adds *signatures* only -- every public name is already recorded from
# the entrypoint file -- so truncating it costs detail, never coverage. It needs a cap
# because a package with no `__all__` and a long list of relative imports has no natural
# one: litellm's entrypoint re-exports from 231 modules, which turned a 1 MB read into
# 6.6 MB and 10s on its own. Modules are visited most-wanted-first, so the cap keeps the
# highest-yield ones.
_MAX_HOP_FILES = 40

# Stands in for a distribution version on stdlib modules, which have none.
_STDLIB_VERSION = f"python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# Entities worth recording as an entrypoint. A module-level constant is part of a public
# API (`httpx.codes`), a nested one is an implementation detail.
_STUB_LABELS = frozenset({NodeLabel.CALLABLE, NodeLabel.TYPE_DEF, NodeLabel.VALUE})

# `find_spec` imports a name's *parent* packages, so a broken or exotic one raises rather
# than returning None -- and which exception depends on how it is broken.
_LOOKUP_ERRORS = (ImportError, ValueError, AttributeError, TypeError)

# A stub is read from whatever the environment happens to hold, including a file written
# for a newer Python than this one, and a deeply nested type expression can exhaust the
# parser's stack. None of that is worth failing an index over.
_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError)


@dataclass(frozen=True)
class StubTarget:
    """Where a package's API surface lives on disk, and which shape it is."""

    import_name: str
    kind: str  # bundled-pyi | stubs-package | frozen-stdlib | source
    entry: Path
    root: Path | None  # the package directory, or None for a single-module package
    distribution: str = ""
    version: str = ""


@dataclass(frozen=True)
class StubSymbol:
    """One public entrypoint of an external package."""

    name: str
    kind: str
    label: NodeLabel
    signature: str = ""
    docstring: str = ""
    # False when the name is exported but its definition was not reachable -- the caller
    # still learns the name exists, which is most of the value.
    resolved: bool = False


@dataclass
class StubResult:
    """What one package yielded, and what it cost."""

    target: StubTarget
    symbols: list[StubSymbol] = field(default_factory=list)
    files_read: int = 0
    bytes_read: int = 0

    @property
    def with_signature(self) -> int:
        return sum(1 for s in self.symbols if s.signature)


# ---------------------------------------------------------------------------
# P3 — resolution
# ---------------------------------------------------------------------------


@functools.cache
def _distribution_versions() -> dict[str, str]:
    """Import name -> installed version, for stub invalidation.

    The version is what makes a stub re-readable: the source is in site-packages, not the
    repo, so no `file_hash` gate covers it and a dependency upgrade would otherwise leave
    the old API surface in the graph forever.
    """
    versions: dict[str, str] = {}
    for import_name, distributions in importlib.metadata.packages_distributions().items():
        for distribution in distributions:
            try:
                versions.setdefault(import_name, importlib.metadata.version(distribution))
            except importlib.metadata.PackageNotFoundError:  # pragma: no cover - racy uninstall
                continue
    return versions


def _spec_origin(import_name: str) -> str | None:
    """`spec.origin` for a top-level import name, without importing it.

    None for anything this process cannot locate: a name from another ecosystem
    (`ghcr.io/acme/api`, `actions/checkout`, `rack/protection` are not identifiers), an
    uninstalled distribution, or a namespace package with no single origin.
    """
    if not import_name.isidentifier():
        return None
    try:
        spec = importlib.util.find_spec(import_name)
    except _LOOKUP_ERRORS:
        return None
    return spec.origin if spec is not None else None


def _frozen_stdlib_target(import_name: str) -> StubTarget | None:
    """A frozen stdlib module reports no path, but its source is still shipped in `Lib/`.

    `os`, `io`, `abc` and `codecs` have all been frozen since 3.11 — `find_spec` returns
    `origin == "frozen"` for them — and `os` alone is one of the most-imported names on
    any graph. A true built-in (`sys`, `_socket`) has no source anywhere and stays
    unresolved.
    """
    source = Path(sysconfig.get_paths()["stdlib"]) / f"{import_name}.py"
    return StubTarget(import_name, "frozen-stdlib", source, None, version=_STDLIB_VERSION) if source.is_file() else None


def resolve_stub_target(import_name: str) -> StubTarget | None:
    """Locate *import_name*'s API surface, in the order the original design specified.

    ``.pyi`` first (a bundled stub or a ``-stubs`` distribution, both hand-written and
    signature-only), then the source, whether or not it carries a ``py.typed`` marker.

    `py.typed` deliberately does **not** get its own branch. It advertises that the
    *source* is annotated, which is a statement about type checkers and not a different
    file to read -- treating it as one produced a resolution kind that behaved identically
    to `source` and suggested a stub existed where none did.

    Returns None for anything not importable from this process: a name from another
    ecosystem, a docker image, an uninstalled distribution, a namespace package with no
    single origin, or a built-in with no file at all.
    """
    located = _spec_origin(import_name)
    if located is None:
        return None
    if located in ("built-in", "frozen"):
        return _frozen_stdlib_target(import_name)

    origin = Path(located)
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    root = origin.parent if origin.name.startswith("__init__.") else None
    # A stdlib module has no distribution, so without this it would carry no version and
    # be re-read on every single index -- measured at 848 of 3,272 symbols and most of the
    # 9s a skipped pass still cost. The interpreter version is the right key: it is exactly
    # what changes when the standard library does.
    version = _distribution_versions().get(import_name, "")
    if not version and stdlib in origin.parents:
        version = _STDLIB_VERSION

    bundled = origin.with_suffix(".pyi")
    if bundled.is_file():
        return StubTarget(import_name, "bundled-pyi", bundled, root, version=version)

    if root is not None:
        stubs = root.parent / f"{import_name}-stubs"
        entry = stubs / "__init__.pyi"
        if entry.is_file():
            return StubTarget(import_name, "stubs-package", entry, stubs, version=version)

    return StubTarget(import_name, "source", origin, root, version=version)


# ---------------------------------------------------------------------------
# P4 — extraction
# ---------------------------------------------------------------------------


def _toplevel(tree: ast.Module):
    """Module-level statements, descending into `if`/`try` bodies.

    `if TYPE_CHECKING:` and `try: from _x import y / except ImportError:` are both
    ordinary ways to write a re-export, and both put the import one level down. Measured,
    walking them lifted signature coverage from 23% to 26%.
    """
    for node in tree.body:
        yield node
        if isinstance(node, ast.If):
            yield from node.body
            yield from node.orelse
        elif isinstance(node, ast.Try):
            yield from node.body


def _read_exports(source: str, package: str = "") -> tuple[set[str], dict[str, str], list[str]]:
    """`(__all__ names, public name -> relative module, star-imported modules)`.

    The one thing `parse_file` does not expose and this module genuinely needs. A
    package's public API is what `__all__` says it is; where that name is *defined* is
    what the import statements say.

    The second element separates the two kinds of import deliberately. A **relative**
    `from .main import Typer` is a re-export -- part of this package's surface, and the
    module name says where to look for the signature. An **absolute** `from typing import
    Any` is a dependency this module happens to use, and maps to `""` so it can never be
    mistaken for one of the package's own entrypoints.
    """
    try:
        tree = ast.parse(source)
    except _PARSE_ERRORS:
        return set(), {}, []

    exported: set[str] = set()
    origin: dict[str, str] = {}
    stars: list[str] = []
    self_prefix = f"{package}." if package else None
    for node in _toplevel(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # A relative import is a re-export by construction. An absolute one usually is
            # not -- except when it names this very package: `pathlib/__init__.py` says
            # `from pathlib._local import Path`, and reading that as a foreign dependency
            # cost pathlib every one of its signatures.
            if node.level:
                submodule = node.module
            elif self_prefix and node.module.startswith(self_prefix):
                submodule = node.module[len(self_prefix) :]
            else:
                submodule = ""
            for alias in node.names:
                if alias.name == "*":
                    if submodule:
                        stars.append(submodule)
                else:
                    origin[alias.asname or alias.name] = submodule
        elif isinstance(node, ast.Assign | ast.AugAssign) and isinstance(node.value, ast.List | ast.Tuple):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                exported |= {
                    e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
    return exported, origin, stars


def _entities(path: Path, project_name: str) -> tuple[dict[str, StubSymbol], str, int]:
    """Public top-level entities of one file, via the ordinary Python parser.

    Returns `(name -> symbol, source text, bytes read)`. The parser gives the signature,
    the docstring, the visibility rule and the label mapping; re-deriving any of that here
    would be a second Python parser that drifts.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.debug("Stub read failed for {}: {}", path, exc)
        return {}, "", 0

    parsed = parse_file(str(path), raw, project_name, max_source_chars=0)
    text = raw.decode("utf-8", errors="replace")
    if parsed is None:
        return {}, text, len(raw)

    symbols: dict[str, StubSymbol] = {}
    for entity in parsed.entities:
        if entity.label not in _STUB_LABELS or entity.visibility != Visibility.PUBLIC:
            continue
        symbols.setdefault(
            entity.name,
            StubSymbol(
                name=entity.name,
                kind=entity.kind,
                label=entity.label,
                signature=entity.signature or "",
                docstring=(entity.docstring or "")[:2000],
                resolved=True,
            ),
        )
    return symbols, text, len(raw)


def _module_file(root: Path, dotted: str) -> Path | None:
    """The file backing a relative module name, `.pyi` preferred over `.py`."""
    parts = dotted.split(".")
    for candidate in (
        root.joinpath(*parts).with_suffix(".pyi"),
        root.joinpath(*parts).with_suffix(".py"),
        root.joinpath(*parts, "__init__.pyi"),
        root.joinpath(*parts, "__init__.py"),
    ):
        if candidate.is_file():
            return candidate
    return None


def extract_stub(target: StubTarget, *, introspect: bool = False, full: bool = False) -> StubResult:
    """Read *target*'s public entrypoints.

    Three passes, each strictly additive, so a later one can only fill in a name an
    earlier one left unresolved:

    1. **The entrypoint file.** Its own public definitions, plus every name in `__all__`.
       A name in `__all__` with no local definition is still recorded -- knowing that
       `pydantic.BaseModel` exists is most of the value even without its signature.
    2. **One re-export hop.** For each unresolved name the entrypoint imported relatively,
       the module it came from. Bounded by the entrypoint's own import list, and
       `from .x import *` contributes its module too.
    3. **Introspection**, only when asked. See the module docstring.

    *full* replaces pass 2 with the package's whole tree, for `[libraries] full_index`.
    """
    result = StubResult(target=target)
    project = f"stub-{target.import_name}"

    local, text, size = _entities(target.entry, project)
    result.files_read, result.bytes_read = 1, size
    exported, origin, stars = _read_exports(text, target.import_name)

    public = exported or _implicit_surface(local, origin)
    symbols: dict[str, StubSymbol] = {n: local[n] for n in public if n in local}
    unresolved = public - set(symbols)

    if full and target.root is not None:
        symbols |= _scan_tree(target, project, result, skip=set(symbols))
    elif target.root is not None:
        symbols |= _scan_hop(target, project, result, unresolved, origin, stars)
        # `asyncio`, `sqlite3` and `urllib` are entrypoints that consist of nothing but
        # `from .x import *`, and their `__all__` is *computed* (`base_events.__all__ +
        # coroutines.__all__ + ...`), which no literal read can evaluate. Every one of
        # them yielded zero entrypoints. When the entrypoint declares no surface of its
        # own, the star targets' public definitions are the surface.
        if not public and stars:
            symbols |= _scan_stars(target, project, result, stars)

    for name in public - set(symbols):
        symbols[name] = StubSymbol(name=name, kind="symbol", label=NodeLabel.EXTERNAL_SYMBOL)

    if introspect:
        symbols = _introspect(target, symbols)

    result.symbols = sorted(symbols.values(), key=lambda s: s.name)
    return result


def _implicit_surface(local: dict[str, StubSymbol], origin: dict[str, str]) -> set[str]:
    """The public surface of a package that declares no `__all__`.

    Its own public definitions, plus every name it re-exported from a sibling module.
    Without the second half, `jinja2`, `typer` and `tiktoken` each yielded **zero**
    entrypoints -- their `__init__.py` defines nothing and is one long list of
    `from .x import Y`, which is exactly the surface somebody wants to see.

    Absolute imports are excluded by construction: `_read_exports` maps them to `""`, so
    `from typing import Any` cannot make `Any` an entrypoint of the package importing it.
    """
    return {n for n in local if not n.startswith("_")} | {
        n for n, module in origin.items() if module and not n.startswith("_")
    }


def _scan_hop(
    target: StubTarget,
    project: str,
    result: StubResult,
    unresolved: set[str],
    origin: dict[str, str],
    stars: list[str],
) -> dict[str, StubSymbol]:
    """Pass 2: the modules the entrypoint re-exported unresolved names from."""
    assert target.root is not None
    wanted: dict[str, set[str]] = {}
    for name in unresolved:
        if module := origin.get(name):
            wanted.setdefault(module, set()).add(name)
    for module in stars:
        wanted.setdefault(module, set()).update(unresolved)

    found: dict[str, StubSymbol] = {}
    ranked = sorted(wanted.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for module, names in ranked[:_MAX_HOP_FILES]:
        path = _module_file(target.root, module)
        if path is None:
            continue
        defined, _, size = _entities(path, project)
        result.files_read += 1
        result.bytes_read += size
        for name in names & set(defined):
            found.setdefault(name, defined[name])
    if len(ranked) > _MAX_HOP_FILES:
        logger.debug(
            "'{}' re-exports from {} modules; read the {} with the most names",
            target.import_name,
            len(ranked),
            _MAX_HOP_FILES,
        )
    return found


def _scan_stars(target: StubTarget, project: str, result: StubResult, stars: list[str]) -> dict[str, StubSymbol]:
    """The public definitions of every module the entrypoint star-imported.

    Only reached when the entrypoint declared no surface of its own -- a literal `__all__`
    or a local definition always wins, because it is the package saying what it exports
    rather than this guessing from what it happened to pull in.
    """
    assert target.root is not None
    found: dict[str, StubSymbol] = {}
    for module in stars[:_MAX_HOP_FILES]:
        path = _module_file(target.root, module)
        if path is None:
            continue
        defined, text, size = _entities(path, project)
        result.files_read += 1
        result.bytes_read += size
        # A star import honours the target's own `__all__` when it has one.
        exported, _, _ = _read_exports(text, target.import_name)
        for name, symbol in defined.items():
            if not exported or name in exported:
                found.setdefault(name, symbol)
    return found


def _scan_tree(target: StubTarget, project: str, result: StubResult, *, skip: set[str]) -> dict[str, StubSymbol]:
    """`full_index`: every module in the package, not just the entrypoint's neighbours."""
    assert target.root is not None
    found: dict[str, StubSymbol] = {}
    files = sorted(p for p in target.root.rglob("*") if p.suffix in {".py", ".pyi"} and p.is_file())
    if len(files) > _MAX_FULL_INDEX_FILES:
        logger.warning(
            "full_index for '{}' covers {:,} files; reading the first {:,}",
            target.import_name,
            len(files),
            _MAX_FULL_INDEX_FILES,
        )
        files = files[:_MAX_FULL_INDEX_FILES]
    for path in files:
        defined, _, size = _entities(path, project)
        result.files_read += 1
        result.bytes_read += size
        for name, symbol in defined.items():
            if name not in skip:
                found.setdefault(name, symbol)
    return found


def _introspect(target: StubTarget, symbols: dict[str, StubSymbol]) -> dict[str, StubSymbol]:
    """Pass 3: import the package and fill in what no static read can reach.

    Compiled extensions and runtime-generated classes have no source to parse; `inspect`
    reads `__text_signature__` off the compiled object and the real class off the module.
    The cost is that this executes the package's import-time code inside the indexer,
    which is why it is opt-in.

    A package that raises on import is skipped whole and keeps its static result. Every
    exception is caught deliberately: an arbitrary third-party import can raise anything
    at all, including SystemExit, and a failure to enrich must never fail an index.
    """
    try:
        module = importlib.import_module(target.import_name)
    except BaseException as exc:
        logger.warning("introspect: '{}' failed to import ({}); keeping the static read", target.import_name, exc)
        return symbols

    enriched = dict(symbols)
    for name, symbol in symbols.items():
        if symbol.signature:
            continue
        try:
            obj = getattr(module, name)
            signature = str(inspect.signature(obj))
            doc = (inspect.getdoc(obj) or "")[:2000]
        except BaseException:
            continue
        label = NodeLabel.TYPE_DEF if inspect.isclass(obj) else NodeLabel.CALLABLE
        enriched[name] = StubSymbol(
            name=name,
            kind="class" if inspect.isclass(obj) else "function",
            label=label,
            signature=f"{name}{signature}",
            docstring=doc or symbol.docstring,
            resolved=True,
        )
    return enriched


# ---------------------------------------------------------------------------
# The one entry point the orchestrator uses
# ---------------------------------------------------------------------------


def read_package_stubs(
    import_names: list[str],
    settings: LibrarySettings,
    known_versions: dict[str, str] | None = None,
) -> dict[str, StubResult]:
    """Resolve and read every package *settings* admits, keyed by import name.

    `stub_index` restricts; empty means every name that resolves. `full_index` widens one
    package from its entrypoint to its whole tree, and implies it is stubbed even when
    `stub_index` does not list it -- asking for the internals is asking for the surface.

    **The version check happens between resolution and extraction, and the order is the
    point.** Resolving is a `find_spec` and a `stat`; extracting is 181 files and 6s.
    Checking afterwards -- which is how this was first written -- did all the work and
    then threw it away, so a re-index whose dependencies had not moved still paid in full.
    """
    if not settings.stubs:
        return {}
    allowed = set(settings.stub_index)
    full = set(settings.full_index)
    known = known_versions or {}
    results: dict[str, StubResult] = {}
    skipped = 0
    for name in import_names:
        if allowed and name not in allowed and name not in full:
            continue
        target = resolve_stub_target(name)
        if target is None:
            continue
        if target.version and known.get(name) == target.version:
            skipped += 1
            continue
        results[name] = extract_stub(target, introspect=settings.introspect, full=name in full)
    if skipped:
        logger.debug("Stub pass: {} package(s) unchanged since the last index", skipped)
    return results
