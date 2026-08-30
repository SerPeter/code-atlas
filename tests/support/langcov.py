"""Extraction-coverage measurement, shared by the floor test and manual runs.

Compares the raw tree-sitter AST against what a language's walker actually
emitted, so a gap shows up as a number instead of as a plausible-looking but
short answer from a graph tool.

Two ratios are reported, because two different things go wrong:

``named_funcs``
    Of the function forms that carry a name, how many became Callable
    entities? Anonymous forms are excluded by design — the project's rule is
    that they get no entity and their calls attribute to the nearest named
    enclosing scope (ADR-0031). A miss here means the walker never visited
    the node.

``calls``
    Of the call nodes in the AST, how many became CALLS relationships? This
    one is form-agnostic and catches the anonymous case: a callback whose
    body is skipped drops every call inside it.

``retrievability``
    Of the file's bytes, how many reach the search index as some entity's
    searchable content? The other two measure the *graph*: whether the walker
    found the forms it should and whether calls survived. Neither notices a file
    that is perfectly parsed and completely unsearchable. Measured: a 907-line
    TypeScript test file scored named_funcs 1.000 and calls 0.996 while 0.2% of
    it was reachable, because every callback was declined by design and nothing
    carried their text (ATL-139).

Run it against a real checkout to see *where* the misses sit::

    uv run --no-sync python -m tests.support.langcov <dir> <lang>
    uv run --no-sync python -m tests.support.langcov <dir> <lang> --census
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tree_sitter import Node, Parser

from code_atlas.parsing.ast import get_language_for_file, parse_file
from code_atlas.parsing.languages import discover_plugins
from code_atlas.schema import NodeLabel, RelType
from code_atlas.settings import IndexSettings

if TYPE_CHECKING:
    from collections.abc import Mapping


_PRODUCTION_SOURCE_CHARS = IndexSettings().max_source_chars
"""Read from ``IndexSettings`` rather than mirrored as a literal, so the measurement
sees the same text the index does and cannot drift from the default it claims to
track -- it was a hardcoded 2000 while the default moved to 48,000."""


@dataclass(frozen=True)
class LangSpec:
    """What counts as a function, a call, and a deliberate non-entity."""

    exts: tuple[str, ...]
    named: tuple[str, ...]
    """Function forms carrying a name — these must become Callable entities."""
    anon: tuple[str, ...] = ()
    """Function forms with no name of their own. No entity by design; their
    calls must still reach the graph, attributed to the enclosing named scope."""
    decl_only: tuple[str, ...] = ()
    """Signatures with no body. Nothing to walk, so excluded from both ratios."""
    calls: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()

    named_requires_ancestor: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """A form in ``named`` that only carries a referable name under some ancestor.

    TypeScript's ``method_definition`` is the case: inside a ``class_body`` it is
    a class method and must be an entity, but in an inline object literal passed
    as an argument there is no name a developer could use to reach it, so
    ADR-0031 makes it category 3. Both spell the same grammar node. Asserting
    capture on the second kind measures the opposite of the decision — in `ky`,
    179 of the 240 nodes in the named bucket are exactly the ones the walker is
    supposed to decline.

    Forms failing the constraint drop out of ``named_funcs`` entirely rather
    than moving to ``anon``: capture is neither required nor forbidden, and a
    bound object literal's method is still emitted.
    """

    calls_require_parent: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """A node type that is only a call in statement position.

    Ruby needs this: a bare ``identifier`` alone on a line is an implicit
    ``self`` call (``content_type``, ``pass``), and the walker rightly emits one
    — but every parameter and variable read is the same node type, so the form
    can only be counted where it stands as a statement. Without it the
    denominator omits calls the numerator contains and the ratio exceeds 1.0.
    """

    named_blocked_by_scope: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """A form whose nearest enclosing scope may be anonymous, making it unqualifiable.

    The mirror of ``named_requires_ancestor``: that one names where a form DOES
    carry a name, this one names where it stops. Ruby needs it — a ``def`` inside
    a ``do_block`` cannot be qualified, because a block has no name and eight
    sibling blocks each defining ``call`` would claim one uid (ADR-0032).

    Resolved against ``named_scope_anchors``, not by "has such an ancestor
    anywhere": a named class or module inside a block re-anchors the chain, and
    its methods are genuinely qualifiable. Ruby resolves constants against
    lexical nesting and a block is transparent to it, so 33 classes and 20
    modules under blocks in sinatra hold real methods that must stay asserted.
    """

    named_scope_anchors: tuple[str, ...] = ()
    """Ancestor types that re-anchor a qualified name, ending the search above."""

    @property
    def funcs(self) -> frozenset[str]:
        return frozenset(self.named + self.anon + self.decl_only)

    def is_qualifiable(self, node: Node) -> bool:
        """False when the nearest scope-defining ancestor is an anonymous one."""
        blockers = self.named_blocked_by_scope.get(node.type)
        if not blockers:
            return True
        cur = node.parent
        while cur is not None:
            if cur.type in blockers:
                return False
            if cur.type in self.named_scope_anchors:
                return True
            cur = cur.parent
        return True

    def counts_as_call(self, node: Node) -> bool:
        required = self.calls_require_parent.get(node.type)
        if required is None:
            return True
        return node.parent is not None and node.parent.type in required


# Node types verified with --census against the corpus repo named in each
# fixture's floor.json, not assumed from grammar documentation.
LANGS: dict[str, LangSpec] = {
    "typescript": LangSpec(
        exts=(".ts", ".tsx", ".mts", ".cts"),
        named=("function_declaration", "generator_function_declaration", "method_definition"),
        anon=("arrow_function", "function_expression", "generator_function"),
        decl_only=("function_signature", "method_signature"),
        calls=("call_expression", "new_expression"),
        skip=("node_modules", "/dist/", ".d.ts"),
        named_requires_ancestor={"method_definition": ("class_body",)},
    ),
    "javascript": LangSpec(
        exts=(".js", ".jsx", ".mjs", ".cjs"),
        named=("function_declaration", "generator_function_declaration", "method_definition"),
        anon=("arrow_function", "function_expression", "generator_function"),
        calls=("call_expression", "new_expression"),
        skip=("node_modules", "/dist/"),
    ),
    "rust": LangSpec(
        exts=(".rs",),
        named=("function_item", "function_signature_item"),
        anon=("closure_expression",),
        calls=("call_expression", "macro_invocation"),
        skip=("/target/",),
    ),
    "cpp": LangSpec(
        exts=(".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".c"),
        named=("function_definition",),
        anon=("lambda_expression",),
        calls=("call_expression",),
        skip=("/build/",),
    ),
    "ruby": LangSpec(
        exts=(".rb",),
        named=("method", "singleton_method"),
        anon=("do_block", "block", "lambda"),
        calls=("call", "identifier"),
        skip=("/vendor/",),
        named_blocked_by_scope={
            "method": ("do_block", "block"),
            "singleton_method": ("do_block", "block"),
        },
        named_scope_anchors=("class", "module"),
        calls_require_parent={
            "identifier": ("body_statement", "then", "else", "do", "block_body", "program", "begin_block"),
        },
    ),
    "go": LangSpec(
        exts=(".go",),
        named=("function_declaration", "method_declaration"),
        anon=("func_literal",),
        calls=("call_expression",),
        skip=("/vendor/",),
    ),
    "java": LangSpec(
        exts=(".java",),
        named=("method_declaration", "constructor_declaration"),
        anon=("lambda_expression",),
        calls=("method_invocation", "object_creation_expression"),
        skip=("/build/",),
    ),
    "csharp": LangSpec(
        exts=(".cs",),
        named=("method_declaration", "constructor_declaration", "local_function_statement"),
        anon=("lambda_expression", "anonymous_method_expression"),
        calls=("invocation_expression", "object_creation_expression"),
        skip=("/obj/", "/bin/"),
    ),
    "php": LangSpec(
        exts=(".php",),
        named=("function_definition", "method_declaration"),
        anon=("anonymous_function", "anonymous_function_creation_expression", "arrow_function"),
        calls=(
            "function_call_expression",
            "member_call_expression",
            "scoped_call_expression",
            "object_creation_expression",
        ),
        skip=("/vendor/",),
    ),
    "python": LangSpec(
        exts=(".py",),
        named=("function_definition",),
        anon=("lambda",),
        calls=("call",),
        skip=("/.venv/", "/site-packages/"),
    ),
}


@dataclass(frozen=True)
class FuncSite:
    form: str
    file: str
    line: int
    captured: bool
    parent: str
    chain: str
    name_bearing: bool = True
    """False when the form is in ``named`` but failed its ancestor constraint —
    same grammar node, no referable name, so capture is not asserted."""


@dataclass
class Coverage:
    lang: str
    files: int = 0
    failed: int = 0
    callables: int = 0
    ast_calls: int = 0
    calls_emitted: int = 0
    calls_in_captured: int = 0
    calls_in_missed: int = 0
    calls_at_module: int = 0
    file_bytes: int = 0
    content_bytes: int = 0
    sites: list[FuncSite] = field(default_factory=list)
    uids: Counter[str] = field(default_factory=Counter)

    @property
    def duplicate_uids(self) -> int:
        """Callable entities sharing a qualified name with another.

        A uid is the graph's identity: two definitions emitting the same one
        upsert into a single node carrying an arbitrary winner's source and the
        union of both edge sets. That is worse than a missing entity, because a
        missing entity is silence and a merged one is a confident wrong answer.

        Counted as occurrences beyond the first, so N definitions sharing a name
        contribute N-1.
        """
        return sum(n - 1 for n in self.uids.values() if n > 1)

    @property
    def worst_collisions(self) -> list[tuple[str, int]]:
        return [(uid, n) for uid, n in self.uids.most_common(10) if n > 1]

    @property
    def named_sites(self) -> list[FuncSite]:
        spec = LANGS[self.lang]
        return [s for s in self.sites if s.form in spec.named and s.name_bearing]

    @property
    def named_funcs(self) -> float:
        sites = self.named_sites
        return sum(s.captured for s in sites) / len(sites) if sites else 1.0

    @property
    def calls(self) -> float:
        return self.calls_emitted / self.ast_calls if self.ast_calls else 1.0

    @property
    def retrievability(self) -> float:
        """Searchable content bytes over file bytes.

        The numerator is each entity's own content -- docstring, source, signature --
        and deliberately NOT its embed text. Embed text carries a breadcrumb header
        (``Module: x`` / ``Class: Y``) that grows with the number of entities, so a
        walker emitting more nodes would raise the ratio without making one more byte
        of the file findable. Content is what an agent searches for.

        Not a fraction, so not capped at 1.0: a container's ``source`` contains its
        members', and Python and Java measure above 1.0 for that reason. It is a floor
        against regression, not a target to reach.
        """
        return self.content_bytes / self.file_bytes if self.file_bytes else 1.0


def _walk(node: Node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _has_ancestor(node: Node, required: tuple[str, ...] | None) -> bool:
    """True when *node* sits under one of *required*, or nothing is required."""
    if not required:
        return True
    cur = node.parent
    while cur is not None:
        if cur.type in required:
            return True
        cur = cur.parent
    return False


def _enclosing(node: Node, funcs: frozenset[str]) -> Node | None:
    cur = node.parent
    while cur is not None:
        if cur.type in funcs:
            return cur
        cur = cur.parent
    return None


def _chain(node: Node, depth: int = 3) -> str:
    parts = []
    cur = node.parent
    while cur is not None and len(parts) < depth:
        parts.append(cur.type)
        cur = cur.parent
    return " < ".join(parts) or "<root>"


def _captured_by(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    """Did some Callable entity claim this AST function node?

    Tolerant at the head — a decorator, an ``export``, or a ``const foo =``
    all make the entity start earlier than the function node — and by one line
    at the tail for a trailing semicolon. Otherwise the spans must coincide,
    so an enclosing function is never credited with a nested one.
    """
    return any(s <= start and e >= end and (start - s) <= 4 and (e - end) <= 1 for s, e in spans)


def measure(root: Path, lang: str) -> Coverage:
    """Parse every file of *lang* under *root* and compare AST against output."""
    discover_plugins()
    spec = LANGS[lang]
    funcs = spec.funcs
    calls = frozenset(spec.calls)
    cov = Coverage(lang=lang)
    parsers: dict[str, Parser] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in spec.exts:
            continue
        posix = path.as_posix()
        if ".git/" in posix or any(s in posix for s in spec.skip):
            continue
        try:
            src = path.read_bytes()
        except OSError:
            continue

        config = get_language_for_file(str(path))
        # The production cap, not 0: `retrievability` asks what reaches the real index,
        # and source is most of it. Neither of the other two ratios is affected --
        # max_source_chars only truncates a field in the post-parse pass, after
        # content_hash is computed, and changes no entity or relationship.
        parsed = parse_file(str(path), src, "cov", max_source_chars=_PRODUCTION_SOURCE_CHARS)
        if config is None or parsed is None:
            cov.failed += 1
            continue
        if config.name not in parsers:
            parsers[config.name] = Parser(config.language)
        tree = parsers[config.name].parse(src)
        cov.files += 1

        cov.file_bytes += len(src)
        cov.content_bytes += sum(
            len(e.docstring or "") + len(e.source or "") + len(e.signature or "") for e in parsed.entities
        )

        callables = [e for e in parsed.entities if e.label == NodeLabel.CALLABLE]
        spans = [(e.line_start, e.line_end) for e in callables]
        cov.callables += len(spans)
        cov.uids.update(e.qualified_name for e in callables)
        cov.calls_emitted += sum(1 for r in parsed.relationships if r.rel_type == RelType.CALLS)
        rel = path.relative_to(root).as_posix()

        for node in _walk(tree.root_node):
            if node.type in funcs:
                start, end = node.start_point[0] + 1, node.end_point[0] + 1
                cov.sites.append(
                    FuncSite(
                        form=node.type,
                        file=rel,
                        line=start,
                        captured=_captured_by(spans, start, end),
                        parent=node.parent.type if node.parent else "-",
                        chain=_chain(node),
                        name_bearing=(
                            _has_ancestor(node, spec.named_requires_ancestor.get(node.type))
                            and spec.is_qualifiable(node)
                        ),
                    )
                )
            elif node.type in calls and spec.counts_as_call(node):
                cov.ast_calls += 1
                owner = _enclosing(node, funcs)
                if owner is None:
                    cov.calls_at_module += 1
                elif _captured_by(spans, owner.start_point[0] + 1, owner.end_point[0] + 1):
                    cov.calls_in_captured += 1
                else:
                    cov.calls_in_missed += 1

    return cov


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:5.1f}%" if b else "    -"


def _report(cov: Coverage) -> None:
    spec = LANGS[cov.lang]
    print(f"files: {cov.files} parsed, {cov.failed} failed   Callables emitted: {cov.callables}")
    print(
        f"named_funcs: {cov.named_funcs:.3f}    calls: {cov.calls:.3f}    "
        f"retrievability: {cov.retrievability:.3f}    duplicate_uids: {cov.duplicate_uids}"
    )
    if cov.worst_collisions:
        print("\ncolliding uids (two definitions merging into one graph node):")
        for uid, n in cov.worst_collisions:
            print(f"  {n:4d}x  {uid.rsplit(':', 1)[-1]}")
    print("\nper form:")
    total: Counter[str] = Counter(s.form for s in cov.sites)
    ok: Counter[str] = Counter(s.form for s in cov.sites if s.captured)
    asserted: Counter[str] = Counter(s.form for s in cov.named_sites)
    for form, n in total.most_common():
        bucket = "named" if form in spec.named else ("anon" if form in spec.anon else "decl")
        note = ""
        if form in spec.named_requires_ancestor and asserted[form] != n:
            note = f"  ({asserted[form]} asserted, {n - asserted[form]} not name-bearing)"
        print(f"  {form:40s} {ok[form]:6d} / {n:6d}  {_pct(ok[form], n)}  [{bucket}]{note}")

    missed = [s for s in cov.sites if not s.captured and s.form in spec.named and s.name_bearing]
    if missed:
        print("\nmissed NAMED forms, by immediate parent:")
        for (form, parent), n in Counter((s.form, s.parent) for s in missed).most_common(12):
            ex = next(s for s in missed if s.form == form and s.parent == parent)
            print(f"  {n:6d}  {form} <- {parent}\n            e.g. {ex.file}:{ex.line}  ({ex.chain})")

    anon_missed = [s for s in cov.sites if not s.captured and s.form in spec.anon]
    if anon_missed:
        print(f"\nanon forms with no entity (by design): {len(anon_missed)}")

    print(
        f"\ncall nodes: {cov.ast_calls}   CALLS emitted: {cov.calls_emitted}  {_pct(cov.calls_emitted, cov.ast_calls)}"
    )
    print(f"  inside a captured function: {cov.calls_in_captured:6d}  {_pct(cov.calls_in_captured, cov.ast_calls)}")
    print(f"  inside a MISSED function:   {cov.calls_in_missed:6d}  {_pct(cov.calls_in_missed, cov.ast_calls)}")
    print(f"  at module scope:            {cov.calls_at_module:6d}  {_pct(cov.calls_at_module, cov.ast_calls)}")


def _census(root: Path, lang: str) -> None:
    """Raw node-type histogram — use to verify a LangSpec against real code."""
    discover_plugins()
    spec = LANGS[lang]
    counts: Counter[str] = Counter()
    parsers: dict[str, Parser] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in spec.exts:
            continue
        posix = path.as_posix()
        if ".git/" in posix or any(s in posix for s in spec.skip):
            continue
        config = get_language_for_file(str(path))
        if config is None:
            continue
        if config.name not in parsers:
            parsers[config.name] = Parser(config.language)
        for node in _walk(parsers[config.name].parse(path.read_bytes()).root_node):
            counts[node.type] += 1
    for node_type, n in counts.most_common(60):
        print(f"{n:8d}  {node_type}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("lang", choices=sorted(LANGS))
    ap.add_argument("--census", action="store_true", help="raw node-type histogram, then exit")
    args = ap.parse_args()

    if args.census:
        _census(args.root, args.lang)
    else:
        _report(measure(args.root, args.lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
