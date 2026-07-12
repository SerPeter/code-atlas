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

_log = logging.getLogger(__name__)

_discovered = False

# Built-in language modules, imported for their module-level register_language()
# side effects. Each is imported independently so one module's failure does not
# prevent the others (including this list's own remaining entries) from loading.
_BUILTIN_LANGUAGE_MODULES: tuple[str, ...] = (
    "code_atlas.parsing.languages.cpp",
    "code_atlas.parsing.languages.go",
    "code_atlas.parsing.languages.jvm",
    "code_atlas.parsing.languages.markdown",
    "code_atlas.parsing.languages.php",
    "code_atlas.parsing.languages.python",
    "code_atlas.parsing.languages.ruby",
    "code_atlas.parsing.languages.rust",
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
