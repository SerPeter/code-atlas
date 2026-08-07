"""Language plugin discovery for Code Atlas.

Built-in languages (Python, Markdown) are registered at import time.
External languages can be added via entry points::

    # In your package's pyproject.toml:
    [project.entry-points."code_atlas.languages"]
    rust = "code_atlas_rust:register"

The entry point must be a callable that takes no arguments and calls
``register_language()`` (and optionally ``register_detector()``) when invoked.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_log = logging.getLogger(__name__)

_discovered = False

# Built-in language modules, imported for their module-level register_language()
# side effects. Each is imported independently so one module's failure does not
# prevent the others (including this list's own remaining entries) from loading.
_BUILTIN_LANGUAGE_MODULES: tuple[str, ...] = (
    "code_atlas.parsing.languages.apex",
    "code_atlas.parsing.languages.config",
    "code_atlas.parsing.languages.containerfile",
    "code_atlas.parsing.languages.cpp",
    "code_atlas.parsing.languages.go",
    "code_atlas.parsing.languages.hcl",
    "code_atlas.parsing.languages.jvm",
    "code_atlas.parsing.languages.markdown",
    "code_atlas.parsing.languages.php",
    "code_atlas.parsing.languages.python",
    "code_atlas.parsing.languages.ruby",
    "code_atlas.parsing.languages.rust",
    # No register_language() of its own — it is imported by config.py, which owns
    # the .xml registration and offers every XML document to it first.
    "code_atlas.parsing.languages.salesforce",
    "code_atlas.parsing.languages.shell",
    "code_atlas.parsing.languages.sql",
    "code_atlas.parsing.languages.typescript",
)


def discover_plugins() -> None:
    """Import built-in languages and load external entry-point plugins.

    Safe to call multiple times — subsequent calls are no-ops. A failure
    importing one built-in language module is logged and does not prevent
    the remaining language modules from being imported.
    """
    global _discovered  # noqa: PLW0603
    if _discovered:
        return

    for module_name in _BUILTIN_LANGUAGE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            _log.warning("Failed to load built-in language module %r", module_name, exc_info=True)

    # External plugins via entry points
    for ep in importlib.metadata.entry_points(group="code_atlas.languages"):
        try:
            register_func = ep.load()
            register_func()
        except Exception:
            _log.warning("Failed to load language plugin %r", ep.name, exc_info=True)

    _discovered = True


# ---------------------------------------------------------------------------
# Optional-grammar reporting
# ---------------------------------------------------------------------------
#
# Only `tree-sitter-python` and `tree-sitter-markdown` are base dependencies; every
# other grammar lives behind an extra (see [project.optional-dependencies]). A missing
# grammar is not an error — it is a deliberate install choice — but it must never be
# SILENT. Each language module swallows its own ImportError so one absent wheel cannot
# take the others down, `_DEFAULT_INCLUDE` still lists the extensions, and `parse_file`
# then returns None with no log at any level. The result was `Done - 4823 files, 0
# entities`, exit 0, on a TypeScript repo (ATL-110).
#
# Keyed by extension rather than by language name because that is what a scan has in
# hand, and mapped to the extra rather than the wheel because the extra is what a user
# types.
_EXTENSION_EXTRAS: dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".mjs": "typescript",
    ".cjs": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".c": "cpp",
    ".h": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".hcl": "terraform",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".toml": "config",
    ".xml": "config",
}

_ALL_LANGUAGES_EXTRA = "all-languages"


def missing_grammar_extras(extensions: Iterable[str]) -> dict[str, str]:
    """Map each extension in *extensions* with no registered language to its extra.

    An extension absent from ``_EXTENSION_EXTRAS`` is omitted rather than guessed at:
    a file type this project simply does not support is not an install problem, and
    telling a user to install something that would not help is worse than silence.
    """
    # Local: ast.py imports discover_plugins from this module (ast.py:176), so a
    # top-level import here would close the cycle. Same reason, same direction.
    from code_atlas.parsing.ast import get_language_for_file  # noqa: PLC0415

    missing: dict[str, str] = {}
    for ext in extensions:
        extra = _EXTENSION_EXTRAS.get(ext.lower())
        if extra is not None and get_language_for_file(f"probe{ext}") is None:
            missing[ext.lower()] = extra
    return missing


def install_hint(extras: Iterable[str]) -> str:
    """The `pip install` suffix that would add *extras*, collapsed when it is most of them."""
    wanted = sorted(set(extras))
    if len(wanted) >= 4:
        return _ALL_LANGUAGES_EXTRA
    return ",".join(wanted)
