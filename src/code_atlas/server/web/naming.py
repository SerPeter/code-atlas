"""Entity naming — the IDE breadcrumb (ATL-124).

`conftest` appears four times in this project's module map and `test_client` twice. A
bare basename does not identify anything, and neither does a fully-qualified dotted name
once it is long enough to truncate.

The convention here is the one VS Code and JetBrains put above the editor:

    graph/client.py  >  GraphClient  >  resolve_calls  (rendered with U+203A)

Truncation drops **leading** parts, never the symbol. The symbol is the answer to "what
am I looking at"; the path is context for it, and context is what you give up first.
"""

from __future__ import annotations

from dataclasses import dataclass

# The breadcrumb separator. Ruff flags U+203A as visually ambiguous with `>` — it is,
# and that resemblance is exactly why it reads as a path separator rather than markup.
SEPARATOR = " › "  # noqa: RUF001  # U+203A is the separator, not a stray ">"

# Package roots that add depth without adding meaning: every path in the project starts
# with them, so they are the first thing worth dropping.
_NOISE_PREFIXES = ("src/", "./")


@dataclass(frozen=True)
class Breadcrumb:
    """An entity's name, split into the parts a reader scans in order."""

    path: str
    owner: str
    symbol: str

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(p for p in (self.path, self.owner, self.symbol) if p)

    @property
    def full(self) -> str:
        return SEPARATOR.join(self.parts)

    @property
    def short(self) -> str:
        """For a graph label, where horizontal room is measured in pixels.

        Owner and symbol when there is an owner, otherwise file and symbol — the two
        parts that distinguish it from its namesakes.
        """
        if self.owner and self.symbol:
            return f"{self.owner}{SEPARATOR}{self.symbol}"
        tail = self.symbol or self.owner
        leaf = self.path.rsplit("/", 1)[-1]
        if not (tail and leaf):
            return tail or leaf
        # A module names itself, so "client.py > client" says one thing twice.
        if leaf.rsplit(".", 1)[0] == tail:
            return leaf
        return f"{leaf}{SEPARATOR}{tail}"

    def truncated(self, limit: int) -> str:
        """``full``, shortened from the left to fit *limit* characters.

        Never drops the symbol, even when the symbol alone exceeds the limit — a label
        reading `…` identifies nothing at all.
        """
        if len(self.full) <= limit:
            return self.full
        parts = list(self.parts)
        while len(parts) > 1:
            parts.pop(0)
            candidate = "… " + SEPARATOR.join(parts)
            if len(candidate) <= limit:
                return candidate
        return parts[-1] if parts else ""


def breadcrumb(*, qualified_name: str, file_path: str = "", kind: str = "", label: str = "") -> Breadcrumb:
    """Split an entity into ``path > owner > symbol``.

    *owner* is the class (or other enclosing scope) when the qualified name has one, and
    empty for a module-level function — inventing a container to fill the slot would be
    worse than leaving it out.
    """
    path = _tidy_path(file_path)
    segments = [s for s in qualified_name.split(".") if s]

    # A module names itself; there is no symbol inside it to point at.
    if label == "Module" or kind in {"module", "package"}:
        return Breadcrumb(path=path, owner="", symbol=segments[-1] if segments else path)

    if not segments:
        return Breadcrumb(path=path, owner="", symbol="")

    symbol = segments[-1]
    owner = ""
    if len(segments) >= 2:
        candidate = segments[-2]
        # A class is capitalised by convention in every language this indexes; a lowercase
        # parent is the module, which the path already shows.
        if candidate[:1].isupper():
            owner = candidate
    return Breadcrumb(path=path, owner=owner, symbol=symbol)


def _tidy_path(file_path: str) -> str:
    """Normalise separators and drop prefixes shared by every path in the project."""
    path = file_path.replace("\\", "/")
    for prefix in _NOISE_PREFIXES:
        path = path.removeprefix(prefix)
    return path
