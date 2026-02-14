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

import importlib.metadata
import logging

_log = logging.getLogger(__name__)

_discovered = False


def discover_plugins() -> None:
    """Import built-in languages and load external entry-point plugins.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _discovered  # noqa: PLW0603
    if _discovered:
        return
    _discovered = True

    # Built-in languages
    import code_atlas.parsing.languages.markdown  # noqa: PLC0415
    import code_atlas.parsing.languages.python  # noqa: PLC0415, F401

    # External plugins via entry points
    for ep in importlib.metadata.entry_points(group="code_atlas.languages"):
        try:
            register_func = ep.load()
            register_func()
        except Exception:
            _log.warning("Failed to load language plugin %r", ep.name, exc_info=True)
