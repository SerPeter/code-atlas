"""C and C++ language support — tree-sitter parser for C/C++ source files."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    call_receiver_props,
    node_text,
    normalize_type_text,
    register_language,
)
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Node

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grammar imports (optional — may not be installed)
# ---------------------------------------------------------------------------

_C_AVAILABLE = False
_CPP_AVAILABLE = False

try:
    import tree_sitter_c as ts_c
    from tree_sitter import Language, Parser, Query

    _C_LANGUAGE = Language(ts_c.language())
    _C_QUERY = Query(_C_LANGUAGE, "(translation_unit) @root")
    _C_AVAILABLE = True
except ImportError:
    _log.debug("tree-sitter-c not installed — C language support disabled")

try:
    import tree_sitter_cpp as ts_cpp
    from tree_sitter import Language, Parser, Query

    _CPP_LANGUAGE = Language(ts_cpp.language())
    _CPP_QUERY = Query(_CPP_LANGUAGE, "(translation_unit) @root")
    _CPP_AVAILABLE = True
except ImportError:
    _log.debug("tree-sitter-cpp not installed — C++ language support disabled")


# ---------------------------------------------------------------------------
# C/C++ file extensions
# ---------------------------------------------------------------------------

_C_EXTENSIONS = frozenset({".c", ".h"})
_CPP_EXTENSIONS = frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".hh"})

# Node types that represent type definitions with a body
_TYPE_DEF_NODES = frozenset({"struct_specifier", "enum_specifier", "union_specifier", "class_specifier"})

# `static_cast<T>(x)` parses as a call_expression whose function is a
# template_function, but a named cast is a keyword, not a callable — an edge to
# one could never resolve to anything.
_NAMED_CASTS = frozenset({"static_cast", "reinterpret_cast", "const_cast", "dynamic_cast"})

# googletest declares a test case with a function-like macro followed by a
# block, which the grammar cannot tell from a function definition — so every
# case in a file arrives named after the macro. gtest itself identifies a case
# as `Suite.Case` (that is what `--gtest_filter` takes and what the runner
# prints), so that is the name used here.
_GTEST_CASE_MACROS = frozenset({"TEST", "TEST_F", "TEST_P", "TYPED_TEST", "TYPED_TEST_P"})

# Tags derived from storage class / qualifier specifiers
_TAG_KEYWORDS = frozenset({"virtual", "override", "static", "const", "inline", "extern"})

# Nodes whose children belong to the enclosing scope rather than to a scope of
# their own — recurse through them without changing namespace, class or
# visibility.
#
# The preprocessor conditionals dominate.  tree-sitter has no preprocessor, so
# an ``#ifdef``-guarded declaration is not lifted to file scope: it stays nested
# under a ``preproc_ifdef``/``preproc_if`` node, along with every declaration
# after it up to the ``#endif``.  In a header that opens with an include guard
# that is the *entire file*.  ``#ifndef`` also produces ``preproc_ifdef``.
#
# Both arms of an ``#if``/``#else`` are walked.  Without a preprocessor there is
# no way to know which one the build selects, and indexing the arm that happens
# to be listed first would be a guess; C++ overloading already means a qualified
# name is not unique within a file, so the arms collide no worse than overloads
# already do.
_TRANSPARENT_CONTAINERS = frozenset(
    {
        "preproc_ifdef",
        "preproc_ifndef",
        "preproc_if",
        "preproc_else",
        "preproc_elif",
        "linkage_specification",  # extern "C" { ... }
        "declaration_list",  # body of the above
        "template_declaration",  # the declared class/function is a child
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert file path to a module-style qualified name.

    ``src/server.c`` -> ``src.server``
    ``include/utils.h`` -> ``include.utils``
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts:
        filename = parts[-1]
        # Strip all C/C++ extensions
        dot = filename.rfind(".")
        if dot > 0:
            parts[-1] = filename[:dot]
    return ".".join(parts)


# Cheap pre-filter. If none of these byte sequences occurs anywhere then no
# marker below can match either, so the file skips the comment strip entirely.
# A pure C header — the case that must stay fast and must not change behaviour
# — takes this exit.
_CPP_HINT_BYTES = (
    b"namespace",
    b"template",
    b"class",
    b"public",
    b"private",
    b"protected",
    b"virtual",
    b"typename",
    b"operator",
    b"::",
    b'"C++"',
)

# Constructs that cannot appear in valid C. Each demands syntactic context
# rather than a bare keyword, because C does not reserve any of these words:
# `class`, `template` and `namespace` are legal C identifiers, so `int class;`
# and `struct template *t;` must not flip a file. `extern "C"` is deliberately
# absent — it is a *C header* idiom (216 of CPython's 283 headers use it); only
# `extern "C++"` is C++-only.
_CPP_MARKERS = re.compile(
    r"""
      \bnamespace(?:\s+\w|\s*\{)                            # namespace foo / namespace {
    | \btemplate\s*<                                        # template <
    | \bclass\s+\w                                          # class Foo
    | \b(?:public|private|protected)\s*:                    # access specifier
    | \bvirtual\s+\w                                        # virtual void
    | \btypename\s+\w                                       # typename T
    | \boperator\s*(?:[(\[]|\bnew\b|\bdelete\b|[-+*/%^&|~!=<>])  # operator overload
    | ::                                                    # scope resolution
    | \bextern\s*"C\+\+"                                    # extern "C++"
    """,
    re.VERBOSE,
)


# Alternation order is the whole trick: whichever construct opens first wins the
# scan, so a quote inside a `//` comment cannot open a literal and a `//` inside
# a literal cannot open a comment. An unterminated block comment runs to EOF
# rather than being left in place.
_COMMENT_OR_LITERAL = re.compile(
    r"""
      //[^\n]*                      # line comment
    | /\*.*?(?:\*/|\Z)              # block comment, unterminated tolerated
    | "(?:\\.|[^"\\\n])*"           # string literal
    | '(?:\\.|[^'\\\n])*'           # character literal
    """,
    re.VERBOSE | re.DOTALL,
)


def _strip_comments_and_literals(text: str) -> str:
    """Blank out comments and string/char literals, preserving length.

    Without this a C header whose *prose* mentions "class" or "namespace"
    routes itself to C++. That is not hypothetical: across CPython's headers,
    every occurrence of those words outside the genuinely dual-language ones is
    inside a comment.
    """
    return _COMMENT_OR_LITERAL.sub(lambda m: " " * len(m.group(0)), text)


def sniff_header_is_cpp(source: bytes) -> bool:
    """Does this ``.h`` file contain C++-only constructs?

    ``.h`` is the standard C header extension *and* the extension most C++
    projects use for headers, and nothing in the name says which. Sending it to
    C unconditionally is what left 23 of fmt's 25 headers unparseable; sending
    it to C++ unconditionally puts the risk on C users, who are the status quo.
    So sniff, and resolve every ambiguity to C.
    """
    if not any(hint in source for hint in _CPP_HINT_BYTES):
        return False
    text = _strip_comments_and_literals(source.decode("utf-8", errors="replace"))
    return _CPP_MARKERS.search(text) is not None


def _is_cpp_file(path: str, source: bytes | None = None) -> bool:
    """Return True if this file should be treated as C++ rather than C.

    ``.h`` is claimed by both languages, so for that extension alone the answer
    comes from the content. Every other extension is unambiguous.
    """
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    if suffix in _CPP_EXTENSIONS:
        return True
    if suffix == ".h" and source is not None:
        return sniff_header_is_cpp(source)
    return False


def _extract_doxygen_comment(node: Node, source: bytes) -> str | None:
    """Extract Doxygen-style doc comment immediately before a declaration.

    Recognizes ``///`` line comments and ``/** ... */`` block comments.
    """
    prev = node.prev_named_sibling
    if prev is None or prev.type != "comment":
        return None
    # Check the comment is immediately before this node (no gap > 1 line)
    if node.start_point[0] - prev.end_point[0] > 1:
        return None
    text = source[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")

    # Collect consecutive comment lines above (for multi-line /// style)
    comment_lines = [text]
    cursor = prev.prev_named_sibling
    while cursor is not None and cursor.type == "comment":
        if prev.start_point[0] - cursor.end_point[0] <= 1:
            line_text = source[cursor.start_byte : cursor.end_byte].decode("utf-8", errors="replace")
            comment_lines.insert(0, line_text)
            prev = cursor
            cursor = cursor.prev_named_sibling
        else:
            break

    return _clean_doc_comment("\n".join(comment_lines))


def _clean_doc_comment(text: str) -> str | None:
    """Strip comment delimiters from a doc comment string."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # /** ... */ block comment
        if stripped.startswith("/**"):
            stripped = stripped[3:]
        elif stripped.startswith("/*"):
            stripped = stripped[2:]
        stripped = stripped.removesuffix("*/")
        # /// line comment
        if stripped.startswith("///"):
            stripped = stripped[3:]
        elif stripped.startswith("//"):
            # Not a doxygen comment, skip
            continue
        # Leading * in block comment lines
        stripped = stripped.lstrip("*").lstrip()
        if stripped:
            cleaned.append(stripped)
    result = " ".join(cleaned).strip()
    return result or None


def _extract_tags(node: Node) -> list[str]:
    """Extract tag keywords (virtual, static, const, etc.) from a declaration node.

    Checks both direct children of the node and children of the declarator,
    since ``override`` appears as a ``virtual_specifier`` inside ``function_declarator``.
    """
    tags: list[str] = []
    _collect_tag_keywords(node, tags)
    # Also check inside the declarator (override/final are virtual_specifier children)
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        _collect_tag_keywords(declarator, tags)
    return tags


def _collect_tag_keywords(node: Node, tags: list[str]) -> None:
    """Collect tag keywords from direct children of a node."""
    for child in node.children:
        if child.type in (
            "storage_class_specifier",
            "type_qualifier",
            "virtual_function_specifier",
            "virtual",
            "override",
            "virtual_specifier",
            "function_specifier",
        ):
            kw = node_text(child).strip()
            if kw in _TAG_KEYWORDS:
                tags.append(kw)
            else:
                # virtual_specifier may contain override/final as inner nodes
                for inner in child.children:
                    inner_kw = node_text(inner).strip()
                    if inner_kw in _TAG_KEYWORDS:
                        tags.append(inner_kw)


def _has_storage_class(node: Node, keyword: str) -> bool:
    """Check if a declaration has a given storage class specifier (e.g. 'static', 'extern')."""
    for child in node.children:
        if child.type == "storage_class_specifier" and node_text(child).strip() == keyword:
            return True
    return False


def _extract_function_signature(node: Node, source: bytes) -> str | None:
    """Extract function/method signature — the declaration line without the body."""
    body = node.child_by_field_name("body")
    if body is None:
        # No body (forward declaration) — use entire node
        sig = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()
        return sig or None
    # Everything up to the body
    sig_bytes = source[node.start_byte : body.start_byte]
    sig = sig_bytes.decode("utf-8", errors="replace").strip()
    # Remove trailing opening brace if present
    if sig.endswith("{"):
        sig = sig[:-1].strip()
    return sig or None


_LEAF_DECLARATOR_TYPES = frozenset(
    {
        "identifier",
        "type_identifier",
        "field_identifier",
        "primitive_type",  # tree-sitter-c treats some typedef names (e.g. size_t) as primitive_type
        "destructor_name",  # ~ClassName
        "operator_name",  # operator+, operator==, etc. — no 'declarator' field, no named children
    }
)


def _conversion_operator_name(declarator: Node) -> str | None:
    """Name a conversion operator — ``operator bool``, ``operator T``.

    ``operator_cast`` names itself with its target *type*, not with an
    identifier, and its ``declarator`` field points at an
    ``abstract_function_declarator`` (the parameter list, which by definition
    carries no name).  The generic descent therefore walks straight past the
    name and returns None, dropping every conversion operator in the file.
    """
    type_node = declarator.child_by_field_name("type")
    if type_node is None:
        return None
    text = node_text(type_node).strip()
    return f"operator {text}" if text else None


def _get_declarator_name(declarator: Node) -> str | None:
    """Recursively extract the identifier name from a declarator tree.

    Handles function_declarator, pointer_declarator, array_declarator, etc.
    """
    if declarator.type in _LEAF_DECLARATOR_TYPES:
        return node_text(declarator)

    if declarator.type == "operator_cast":
        return _conversion_operator_name(declarator)

    # function_declarator, pointer_declarator, etc.: has a `declarator` child
    inner = declarator.child_by_field_name("declarator")
    if inner is not None:
        return _get_declarator_name(inner)

    # Fallback: search named children
    for child in declarator.named_children:
        if child.type in _LEAF_DECLARATOR_TYPES:
            return node_text(child)
        result = _get_declarator_name(child)
        if result is not None:
            return result
    return None


def _template_wrapper(node: Node) -> Node | None:
    """Return the enclosing template_declaration wrapper, if any."""
    parent = node.parent
    return parent if parent is not None and parent.type == "template_declaration" else None


def _prototype_declarator(declarator: Node) -> Node | None:
    """Return the function_declarator of a function prototype, unwrapping pointer/reference returns.

    Returns None for non-function declarators and for function-pointer
    declarators like ``int (*cb)(int)`` whose inner declarator is parenthesized.
    """
    node = declarator
    while node.type in ("pointer_declarator", "reference_declarator"):
        # reference_declarator carries its inner declarator as an unnamed field
        inner = node.child_by_field_name("declarator") or next(
            (c for c in node.named_children if c.type.endswith("declarator")), None
        )
        if inner is None:
            return None
        node = inner
    if node.type != "function_declarator":
        return None
    inner = node.child_by_field_name("declarator")
    if inner is None or inner.type == "parenthesized_declarator":
        return None
    return node


def _get_qualified_declarator_name(declarator: Node) -> tuple[list[str], str | None]:
    """Extract name from declarator, returning (scope_parts, name) for qualified names.

    For ``Outer::Inner::method_name``, returns ``(["Outer", "Inner"], "method_name")``
    — tree-sitter-cpp parses this as ``qualified_identifier(scope: 'Outer', name:
    qualified_identifier('Inner::method_name'))``, so nested scopes are unwrapped
    recursively rather than taking the inner qualified_identifier's text verbatim
    (which would otherwise leak a '::'-joined name like 'Inner::method_name').
    For ``ClassName::method_name``, returns ``(["ClassName"], "method_name")``.
    For ``plain_func``, returns ``([], "plain_func")``.
    """
    if declarator.type == "qualified_identifier":
        scope_node = declarator.child_by_field_name("scope")
        name_node = declarator.child_by_field_name("name")
        if scope_node is not None and scope_node.type == "template_type":
            # Box<T>::method — strip template arguments from the scope
            scope_node = scope_node.child_by_field_name("name") or scope_node
        scope = node_text(scope_node) if scope_node is not None else None
        scope_parts = [scope] if scope else []
        if name_node is not None and name_node.type == "qualified_identifier":
            inner_scope_parts, name = _get_qualified_declarator_name(name_node)
            return (scope_parts + inner_scope_parts, name)
        if name_node is not None and name_node.type == "operator_cast":
            # `Widget::operator bool() const` — node_text would take the whole
            # thing, parameter list and trailing qualifiers included.
            return (scope_parts, _conversion_operator_name(name_node))
        name = node_text(name_node) if name_node is not None else None
        return (scope_parts, name)

    # function_declarator wrapping a qualified_identifier.  operator_cast is
    # excluded: its `declarator` field is the nameless parameter list, so
    # following it loses the name — _get_declarator_name reads it directly.
    inner = None if declarator.type == "operator_cast" else declarator.child_by_field_name("declarator")
    if inner is not None:
        if inner.type == "qualified_identifier":
            return _get_qualified_declarator_name(inner)
        return ([], _get_declarator_name(inner))

    return ([], _get_declarator_name(declarator))


def _is_destructor_name(name: str) -> bool:
    """Check if a name looks like a destructor (starts with ~)."""
    return name.startswith("~")


def _is_constructor(name: str, class_name: str | None) -> bool:
    """Check if a function name matches the current class name (constructor)."""
    if class_name is None:
        return False
    return name == class_name


def _method_callable_kind(name: str, class_name: str, tags: list[str]) -> str:
    """Determine the callable kind for a method inside a class."""
    if _is_destructor_name(name):
        return CallableKind.DESTRUCTOR
    if _is_constructor(name, class_name):
        return CallableKind.CONSTRUCTOR
    if "static" in tags:
        return CallableKind.STATIC_METHOD
    return CallableKind.METHOD


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------


def _extract_cpp(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
    is_cpp: bool,
) -> ParsedFile:
    """Extract entities and relationships from one C or C++ parse tree.

    Split out of :func:`_parse_cpp` so the shim can run it against a second tree and the
    two results be compared. Takes *is_cpp* rather than sniffing, so both runs agree.
    """
    module_qn = _module_qualified_name(path)
    lang = "cpp" if is_cpp else "c"

    entities: list[ParsedEntity] = []
    relationships: list[ParsedRelationship] = []

    # Module entity
    entities.append(
        ParsedEntity(
            name=module_qn.rsplit(".", 1)[-1] if "." in module_qn else module_qn,
            qualified_name=f"{project_name}:{module_qn}",
            label=NodeLabel.MODULE,
            kind="module",
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=path,
        )
    )

    # Walk the top-level translation_unit
    _walk_translation_unit(
        root,
        path=path,
        source=source,
        project_name=project_name,
        module_qn=module_qn,
        is_cpp=is_cpp,
        namespace_parts=[],
        class_stack=[],
        current_visibility=Visibility.PUBLIC,
        entities=entities,
        relationships=relationships,
    )

    return ParsedFile(
        file_path=path,
        language=lang,
        entities=entities,
        relationships=relationships,
    )


# ---------------------------------------------------------------------------
# Preprocessor shim (ATL-143)
# ---------------------------------------------------------------------------

_MACRO_TOKEN = re.compile(rb"\b[A-Z][A-Z0-9_]{3,}\b")
"""An ALL-CAPS identifier of at least four characters.

Four, not two: `IN`, `OUT` and `MAX` are ordinary identifiers in real code, and the
shim's job is the declaration decorators and namespace openers that macro-heavy headers
put where the grammar expects a keyword.
"""

_PROTECTED_NODES = frozenset({"comment", "string_literal", "raw_string_literal", "char_literal", "system_lib_string"})
"""Never blanked. A doxygen comment naming FMT_API is documentation, and a string
literal reading "ERROR" is data -- both reach the index as text, and mangling them to
help the parser would trade a parse problem for a retrieval one."""


def _protected_ranges(root: Node) -> list[tuple[int, int]]:
    """Byte ranges of comments and literals, from the ORIGINAL parse.

    Taken from the failed parse on purpose: tree-sitter still tokenises comments and
    strings inside an ERROR region, so even a file the grammar could not assemble tells
    us where its prose is.
    """
    ranges: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in _PROTECTED_NODES:
            ranges.append((node.start_byte, node.end_byte))
            continue
        stack.extend(node.children)
    return sorted(ranges)


def _shim_macros(source: bytes, root: Node) -> bytes:
    """Blank macro-shaped tokens, preserving every byte offset and newline.

    Length-preserving is the whole trick, and the same one ``apex.py`` uses on the Java
    grammar and ``sql.py`` on Jinja: byte offsets, and therefore line numbers, stay true,
    so an entity found in the shimmed tree points at the real file.

    A token followed by ``(`` is left alone. ``FMT_ASSERT(x)`` is an expression the
    grammar parses fine; a bare ``FMT_BEGIN_NAMESPACE`` is a namespace opener standing
    where a keyword belongs, and that is what collapses a file into one ERROR node.
    """
    protected = _protected_ranges(root)
    out = bytearray(source)
    for match in _MACRO_TOKEN.finditer(source):
        start, end = match.span()
        if source[end : end + 1] == b"(":
            continue
        if any(lo <= start < hi for lo, hi in protected):
            continue
        for i in range(start, end):
            out[i] = ord(" ")
    return bytes(out)


def _error_line_count(root: Node) -> int:
    """Lines the grammar could not assemble. The shim's acceptance test measures this."""
    lines: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            lines.update(range(node.start_point[0], node.end_point[0] + 1))
        stack.extend(node.children)
    return len(lines)


def _reslice_sources(entities: list[ParsedEntity], source: bytes) -> None:
    """Replace each entity's ``source`` with the original text for its lines.

    The shimmed tree is used for STRUCTURE only. Its text has macro names blanked, and
    an agent reading that source would see gaps where the code says FMT_API. Because the
    shim preserves length, an entity's line span means the same thing in both, so the
    real text is one slice away.

    ``signature`` is not re-sliced -- it is assembled from several nodes rather than one
    span -- so a recovered entity's signature can still be missing a macro token. Small,
    and only on files that produced nothing at all before.
    """
    lines = source.decode("utf-8", errors="replace").splitlines()
    for index, entity in enumerate(entities):
        if not entity.source:
            continue
        start = max(entity.line_start - 1, 0)
        end = min(entity.line_end, len(lines))
        if start < end:
            entities[index] = replace(entity, source="\n".join(lines[start:end]))


def _shimmed_parse(path: str, source: bytes, root: Node, project_name: str, is_cpp: bool) -> ParsedFile | None:
    """Re-parse with macros blanked, and return it only if it is strictly better.

    THE GUARD IS THE DESIGN, not an optimisation on it. Measured on fmtlib, shimming
    unconditionally is a large net win that nonetheless DESTROYS entities in particular
    files -- test/base-test.cc falls from 140 entities to 49, and include/fmt/os.h goes
    from 25 ERROR lines to 98. Blanking cannot invent tokens, but it changes parse
    structure, and the effect on entities is not one-directional.

    Parsing twice and keeping the shimmed result only when it has strictly fewer ERROR
    lines AND no fewer entities makes the heuristic's quality a performance question
    rather than a correctness one. On the 16-file fixture it is better than the
    unconditional shim on both axes -- 300 ERROR lines against 446, 772 entities against
    677 -- because it never accepts a regression, and it leaves 10 of those 16 files
    untouched.
    """
    # Flags, not a None check: the Language names only exist when their grammar wheel
    # is installed, so referencing one unguarded is a NameError, not a None.
    if is_cpp and not _CPP_AVAILABLE:
        return None
    if not is_cpp and not _C_AVAILABLE:
        return None
    language = _CPP_LANGUAGE if is_cpp else _C_LANGUAGE
    shimmed_source = _shim_macros(source, root)
    if shimmed_source == source:
        return None

    shimmed_root = Parser(language).parse(shimmed_source).root_node
    if _error_line_count(shimmed_root) >= _error_line_count(root):
        return None

    candidate = _drop_colliding(_extract_cpp(path, shimmed_source, shimmed_root, project_name, is_cpp))
    _reslice_sources(candidate.entities, source)
    return candidate


def _calls_in(parsed: ParsedFile) -> int:
    return sum(1 for r in parsed.relationships if r.rel_type is RelType.CALLS)


def _uid_collisions(parsed: ParsedFile) -> int:
    """Entities claiming a qualified_name another already claimed."""
    counts = Counter(e.qualified_name for e in parsed.entities)
    return sum(n - 1 for n in counts.values() if n > 1)


def _drop_colliding(parsed: ParsedFile) -> ParsedFile:
    """Remove every entity whose qualified_name identifies more than one definition.

    Recovering a scope can also recover an AMBIGUITY: a macro-hidden overload set
    arrives as several definitions of one name, and two nodes merging into one is a
    confident wrong answer, worse than the silence of a missing entity (ADR-0032).

    Dropping the colliding few rather than the whole recovery is what makes that rule
    affordable here. On ``color.h`` the choice is between 3 ambiguous entities and 236
    good ones, or nothing at all -- and ADR-0032's own remedy for an unqualifiable
    definition is to emit no entity for it, not to abandon its neighbours.

    Relationships *from* a dropped entity go with it. Edges *to* one are left: they are
    resolved by name post-batch and simply find nothing, which is the same outcome as
    never having been written.
    """
    counts = Counter(e.qualified_name for e in parsed.entities)
    colliding = {qn for qn, n in counts.items() if n > 1}
    if not colliding:
        return parsed
    return ParsedFile(
        file_path=parsed.file_path,
        language=parsed.language,
        entities=[e for e in parsed.entities if e.qualified_name not in colliding],
        relationships=[r for r in parsed.relationships if r.from_qualified_name not in colliding],
    )


def _parse_cpp(
    path: str,
    source: bytes,
    root: Node,
    project_name: str,
) -> ParsedFile:
    """Extract entities and relationships from a C or C++ file.

    Parses once. If the grammar hit no ERROR the result stands unchanged -- a clean file
    never pays for the shim, and never risks it. Otherwise the preprocessor is very
    likely the reason (tree-sitter has none), and :func:`_shimmed_parse` gets a second
    opinion.
    """
    # Sniffed here as well as in the router, so the walker's C++ branches and
    # the recorded language agree however this handler was reached. The sniff
    # short-circuits on a pure C header, so the repeat is nearly free.
    is_cpp = _is_cpp_file(path, source)
    parsed = _extract_cpp(path, source, root, project_name, is_cpp)
    if not root.has_error:
        return parsed

    recovered = _shimmed_parse(path, source, root, project_name, is_cpp)
    # A net gain in graph facts, not a gain on every axis independently.
    #
    # Entities alone is too weak: blanking a token can cost a call edge the grammar had
    # managed, and entities-only acceptance measured calls 0.924 -> 0.905 on fmtlib.
    # But "no fewer calls" is far too strong -- it rejects color.h, where 5 lost call
    # edges buy 235 entities including every type and method in the file. Requiring an
    # improvement on all four axes at once left retrievability at 0.610, below where it
    # started.
    #
    # An entity and a call edge are both one fact, so compare them as such: accept when
    # more facts arrive than leave. Collisions are handled by dropping the colliding
    # entities (see _drop_colliding), not by refusing the file.
    if recovered is not None and (
        len(recovered.entities) - len(parsed.entities) > _calls_in(parsed) - _calls_in(recovered)
    ):
        _log.debug(
            "cpp: macro shim recovered %d -> %d entities in %s",
            len(parsed.entities),
            len(recovered.entities),
            path,
        )
        return recovered
    return parsed


def _build_qn(module_qn: str, namespace_parts: list[str], class_stack: list[str], name: str) -> str:
    """Build a qualified name from namespace, class stack, and entity name."""
    parts = [module_qn, *namespace_parts, *class_stack, name]
    return ".".join(parts)


def _parent_qn(project_name: str, module_qn: str, namespace_parts: list[str], class_stack: list[str]) -> str:
    """Build the parent qualified name (module + namespaces + classes)."""
    parts = [module_qn, *namespace_parts, *class_stack]
    return f"{project_name}:{'.'.join(parts)}"


def _walk_translation_unit(  # noqa: PLR0912
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    overloaded: frozenset[str] | None = None,
) -> None:
    """Recursively walk the AST and extract entities/relationships.

    This is the *structural* traversal: it visits declaration-level constructs
    and never enters a function body — ``_process_function`` hands bodies to
    ``_extract_calls`` instead.  That split is what lets ``_extract_calls`` be
    fully transparent without double-counting anything.

    *overloaded* is the scope's set of ambiguous qualified names (ADR-0032).
    ``None`` means *node* opens a new naming scope, so the set is computed here;
    a recursion that only steps through a scope-transparent wrapper passes the
    set it already has, because an ``#ifdef`` arm is not a scope of its own and
    its members must be weighed against their siblings outside the arm.
    """
    if overloaded is None:
        overloaded = _overloaded_qns(
            node, module_qn=module_qn, namespace_parts=namespace_parts, class_stack=class_stack
        )

    # Calls that are not inside any function belong to the module (ADR-0031).
    # A class or namespace is not a callable, so the fallback never narrows
    # below the module as the walker descends.
    module_scope_qn = f"{project_name}:{module_qn}"

    for child in node.children:
        # ----- #include -----
        if child.type == "preproc_include":
            _process_include(child, project_name, module_qn, namespace_parts, class_stack, relationships)
            continue

        # ----- scope-transparent wrappers (#ifdef, extern "C", template) -----
        if child.type in _TRANSPARENT_CONTAINERS:
            _walk_translation_unit(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
                overloaded=overloaded,
            )
            continue

        # ----- friend declaration (C++ only) -----
        if is_cpp and child.type == "friend_declaration":
            # A friend function defined inside a class belongs to the enclosing
            # namespace, not to the class — it is found by ADL, and
            # `Class::friend_fn` does not name it.  So drop one level of class
            # stack before recursing.
            _walk_translation_unit(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack[:-1],
                current_visibility=Visibility.PUBLIC,
                entities=entities,
                relationships=relationships,
                overloaded=overloaded,
            )
            continue

        # ----- namespace (C++ only) -----
        if is_cpp and child.type == "namespace_definition":
            _process_namespace(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- Access specifiers (C++ only) -----
        if is_cpp and child.type == "access_specifier":
            spec_text = node_text(child).strip().rstrip(":")
            if spec_text == "public":
                current_visibility = Visibility.PUBLIC
            elif spec_text == "private":
                current_visibility = Visibility.PRIVATE
            elif spec_text == "protected":
                current_visibility = Visibility.PROTECTED
            continue

        # ----- struct / class / enum / union -----
        if child.type in _TYPE_DEF_NODES:
            _process_type_def(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- type_definition (typedef) -----
        if child.type == "type_definition":
            _process_typedef(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            continue

        # ----- function_definition -----
        if child.type == "function_definition":
            _process_function(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
                overloaded=overloaded,
            )
            continue

        # ----- declaration (global variables, forward declarations) -----
        if child.type == "declaration":
            claimed_by_type_def = _process_declaration(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            if not claimed_by_type_def:
                # `static auto x = make();` — the initializer runs, so the call
                # is real.  Skipped when the declaration was really an inline
                # type definition, because that subtree was walked structurally
                # and its method bodies already extracted their own calls.
                _extract_calls(child, module_scope_qn, relationships)
            continue

        # ----- field_declaration (struct/class fields) -----
        if child.type == "field_declaration":
            claimed_by_type_def = _process_field_declaration(
                child,
                path=path,
                source=source,
                project_name=project_name,
                is_cpp=is_cpp,
                module_qn=module_qn,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
                overloaded=overloaded,
            )
            if not claimed_by_type_def:
                # Default member initializer: `std::string s = build();`
                _extract_calls(child, module_scope_qn, relationships)
            continue

        # ----- anything else -----
        # Static initializers, expression statements at file scope, and the
        # debris tree-sitter leaves where an unexpandable macro defeated it.
        # None of these define an entity, but the calls inside them are real
        # and belong to the module (ADR-0031).
        _extract_calls(child, module_scope_qn, relationships)


def _include_path_text(path_node: Node) -> str:
    """Strip delimiters from a #include path node's text.

    ``system_lib_string`` (``<vector>``) and ``string_literal`` (``"util.h"``)
    both include their delimiters in ``node.text`` — strip them so the emitted
    IMPORTS name is a bare path (``vector``, ``util.h``) instead of garbage
    like ``<vector>`` or ``"util.h"`` (which corrupts ExternalPackage naming
    downstream in ``resolve_imports``).
    """
    text = node_text(path_node)
    if len(text) >= 2 and text[0] in '<"' and text[-1] in '>"':
        return text[1:-1]
    return text


def _process_include(
    node: Node,
    project_name: str,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a #include directive and emit an IMPORTS relationship."""
    path_node = node.child_by_field_name("path")
    if path_node is None:
        return
    include_path = _include_path_text(path_node)
    if not include_path:
        return

    from_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=from_qn,
            rel_type=RelType.IMPORTS,
            to_name=include_path,
        )
    )


def _process_namespace(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a namespace_definition and recurse into its body."""
    name_node = node.child_by_field_name("name")
    ns_name = node_text(name_node) if name_node is not None else ""
    if not ns_name:
        # Anonymous namespace — treat body as same scope
        body = node.child_by_field_name("body")
        if body is not None:
            _walk_translation_unit(
                body,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=Visibility.PRIVATE,
                entities=entities,
                relationships=relationships,
            )
        return

    new_ns = [*namespace_parts, ns_name]
    body = node.child_by_field_name("body")
    if body is not None:
        _walk_translation_unit(
            body,
            path=path,
            source=source,
            project_name=project_name,
            module_qn=module_qn,
            is_cpp=is_cpp,
            namespace_parts=new_ns,
            class_stack=class_stack,
            current_visibility=Visibility.PUBLIC,
            entities=entities,
            relationships=relationships,
        )


def _process_type_def(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process struct_specifier, class_specifier, enum_specifier, or union_specifier."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        # Anonymous struct/enum/union — skip
        return
    name = node_text(name_node)
    if not name:
        return

    # Determine kind
    kind_map = {
        "struct_specifier": TypeDefKind.STRUCT,
        "class_specifier": TypeDefKind.CLASS,
        "enum_specifier": TypeDefKind.ENUM,
        "union_specifier": TypeDefKind.UNION,
    }
    kind = kind_map.get(node.type, TypeDefKind.STRUCT)

    template_parent = _template_wrapper(node)
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    # Doc comments for templates sit above the template_declaration wrapper
    docstring = _extract_doxygen_comment(template_parent or node, source)
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)

    # Visibility: at file scope (no class stack), use current_visibility or PUBLIC
    # If inside class, use current_visibility (set by access specifiers)
    visibility = current_visibility
    if not class_stack and not _has_storage_class(node, "static"):
        visibility = Visibility.PUBLIC

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=visibility,
            tags=["template"] if template_parent is not None else [],
        )
    )

    # DEFINES from parent
    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )

    # C++ class/struct inheritance: `: public Base, private Other`
    if is_cpp:
        _extract_base_classes(node, project_name, qn, relationships)

    # Process body contents (fields, methods, enum values)
    body = node.child_by_field_name("body")
    if body is not None:
        # Default visibility for class body
        body_visibility = Visibility.PRIVATE if node.type == "class_specifier" else Visibility.PUBLIC

        new_class_stack = [*class_stack, name]

        if node.type == "enum_specifier":
            # Enum body contains enumerator nodes
            _process_enum_body(
                body,
                path=path,
                project_name=project_name,
                module_qn=module_qn,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                enum_name=name,
                entities=entities,
                relationships=relationships,
            )
        else:
            _walk_translation_unit(
                body,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=new_class_stack,
                current_visibility=body_visibility,
                entities=entities,
                relationships=relationships,
            )


def _extract_base_classes(
    node: Node,
    project_name: str,
    class_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Extract base class specifiers and emit INHERITS relationships."""
    for child in node.children:
        if child.type == "base_class_clause":
            for base in child.children:
                if base.type in ("type_identifier", "qualified_identifier"):
                    base_name = node_text(base)
                    if base_name:
                        relationships.append(
                            ParsedRelationship(
                                from_qualified_name=f"{project_name}:{class_qn}",
                                rel_type=RelType.INHERITS,
                                to_name=base_name,
                            )
                        )
                elif base.type == "base_class_specifier":
                    # Some tree-sitter-cpp versions wrap bases in specifier nodes
                    for inner in base.children:
                        if inner.type in ("type_identifier", "qualified_identifier"):
                            base_name = node_text(inner)
                            if base_name:
                                relationships.append(
                                    ParsedRelationship(
                                        from_qualified_name=f"{project_name}:{class_qn}",
                                        rel_type=RelType.INHERITS,
                                        to_name=base_name,
                                    )
                                )


def _process_enum_body(
    body: Node,
    *,
    path: str,
    project_name: str,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    enum_name: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process enum body (enumerator_list) to extract enum members."""
    for child in body.children:
        if child.type == "enumerator":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = node_text(name_node)
            if not name:
                continue

            enum_stack = [*class_stack, enum_name]
            qn = _build_qn(module_qn, namespace_parts, enum_stack, name)
            line_start = child.start_point[0] + 1
            line_end = child.end_point[0] + 1

            entities.append(
                ParsedEntity(
                    name=name,
                    qualified_name=f"{project_name}:{qn}",
                    label=NodeLabel.VALUE,
                    kind=ValueKind.ENUM_MEMBER,
                    line_start=line_start,
                    line_end=line_end,
                    file_path=path,
                    visibility=Visibility.PUBLIC,
                )
            )

            # DEFINES from enum -> member
            parent_qn = _build_qn(module_qn, namespace_parts, class_stack, enum_name)
            relationships.append(
                ParsedRelationship(
                    from_qualified_name=f"{project_name}:{parent_qn}",
                    rel_type=RelType.DEFINES,
                    to_name=f"{project_name}:{qn}",
                )
            )


def _process_typedef(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> None:
    """Process a type_definition (typedef) node."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return
    name = _get_declarator_name(declarator)
    if not name:
        return

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    docstring = _extract_doxygen_comment(node, source)
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.TYPE_DEF,
            kind=TypeDefKind.TYPE_ALIAS,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            visibility=current_visibility,
        )
    )

    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )


def _macro_invocation_name(node: Node, declarator: Node, class_stack: list[str]) -> str | None:
    """Return the macro name if this ``function_definition`` is really a
    function-like macro followed by a block, rather than a function.

    Only a constructor, a destructor or a conversion operator may omit its
    return type, and all three are distinguishable: a destructor declarator is
    a ``destructor_name``, a conversion operator is an ``operator_cast``, and a
    constructor's name is either the enclosing class or is qualified with
    ``Class::``. Anything else with no return type is a macro the preprocessor
    would have removed.
    """
    if node.child_by_field_name("type") is not None:
        return None
    if declarator.type != "function_declarator":
        return None
    inner = declarator.child_by_field_name("declarator")
    if inner is None or inner.type != "identifier":
        return None
    name = node_text(inner)
    if not name or (class_stack and name == class_stack[-1]):
        return None
    return name


def _gtest_case_name(declarator: Node) -> str | None:
    """Name a googletest case ``Suite.Case`` from the macro's arguments.

    The arguments parse as parameter declarations carrying only a type, since
    ``TEST(FormatTest, Escape)`` is indistinguishable from a two-parameter
    prototype. Anything that does not look like two bare identifiers is not a
    shape worth guessing at.

    Checked before the missing-return-type test rather than after it, because a
    stray macro on the preceding line — fmt ends every file with
    ``FMT_END_NAMESPACE`` — is absorbed as the definition's *return type*, and
    the case would otherwise fall through and collide as ``TEST``.
    """
    if declarator.type != "function_declarator":
        return None
    inner = declarator.child_by_field_name("declarator")
    if inner is None or inner.type != "identifier" or node_text(inner) not in _GTEST_CASE_MACROS:
        return None
    params = declarator.child_by_field_name("parameters")
    if params is None:
        return None
    args = [node_text(p).strip() for p in params.named_children]
    expected_args = 2
    if len(args) != expected_args or not all(a.isidentifier() for a in args):
        return None
    return f"{args[0]}.{args[1]}"


# ---------------------------------------------------------------------------
# Overload disambiguation (ADR-0032)
# ---------------------------------------------------------------------------

# Parameter forms inside a `parameter_list`. A bare `...` (C variadic) is an
# anonymous token, so it is matched by literal type rather than listed here.
_PARAM_NODES = frozenset({"parameter_declaration", "optional_parameter_declaration", "variadic_parameter_declaration"})

# Trailing qualifiers on a function_declarator that C++ resolves overloads on.
# `noexcept` is not one of them and is deliberately absent.
_QUALIFIER_NODES = frozenset({"type_qualifier", "ref_qualifier"})


def _function_name_parts(node: Node, declarator: Node, class_stack: list[str]) -> tuple[list[str], str, bool] | None:
    """``(extra scope parts, name, is_gtest_case)`` for a ``function_definition``.

    ``None`` where the definition claims no qualified name at all: a
    function-like macro the preprocessor would have removed, or a declarator
    with no recoverable identifier.  Those two want different handling of the
    body, so ``_process_function`` re-asks which one it was — that path is rare,
    and the alternative is a second return channel nobody else needs.
    """
    gtest_case = _gtest_case_name(declarator)
    if gtest_case is not None:
        return [], gtest_case, True
    if _macro_invocation_name(node, declarator, class_stack) is not None:
        return None
    scope_parts, name = _get_qualified_declarator_name(declarator)
    return None if name is None else (scope_parts, name, False)


def _declared_qn(member: Node, class_stack: list[str], *, module_qn: str, namespace_parts: list[str]) -> str | None:
    """The unsuffixed qualified name *member* would claim as a Callable, if any.

    Mirrors the only two places that emit one: ``_process_function`` for a
    ``function_definition`` — which is also how the grammar shapes ``= delete``,
    so a deleted copy constructor counts — and ``_process_field_declaration``
    for a body-less method declaration.  A prototype at namespace or file scope
    is deliberately absent, because the walker emits no entity for one.
    """
    declarator = member.child_by_field_name("declarator")
    if declarator is None:
        return None
    if member.type == "function_definition":
        parts = _function_name_parts(member, declarator, class_stack)
        if parts is None:
            return None
        scope_parts, name, _ = parts
    elif member.type == "field_declaration":
        if _prototype_declarator(declarator) is None:
            return None
        scope_parts = []
        name = _get_declarator_name(declarator) or ""
        if not name:
            return None
    else:
        return None
    stack = scope_parts if (not class_stack and scope_parts) else class_stack
    return _build_qn(module_qn, namespace_parts, stack, name)


def _scope_members(node: Node, class_stack: list[str]) -> Iterator[tuple[Node, list[str]]]:
    """*node*'s declarations, each paired with the class stack it is named under.

    Flattens exactly what ``_walk_translation_unit`` recurses through without
    changing scope, so an overload set split across an ``#ifdef`` boundary or
    wrapped in ``template <...>`` is still one overload set.  A ``friend``
    function belongs to the enclosing namespace rather than to the class, so it
    is yielded with the class dropped — the same adjustment the walker makes.
    """
    for child in node.children:
        if child.type in _TRANSPARENT_CONTAINERS:
            yield from _scope_members(child, class_stack)
        elif child.type == "friend_declaration":
            yield from _scope_members(child, class_stack[:-1])
        else:
            yield child, class_stack


def _overloaded_qns(
    node: Node,
    *,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
) -> frozenset[str]:
    """Qualified names claimed by two or more Callables declared directly in this scope.

    C++ permits two definitions of one name in one scope, so a repeated name is
    an overload set and each member needs its own uid (ADR-0032).  A name
    declared once keeps its plain uid, which is what bounds the churn to the
    names that were already ambiguous.
    """
    counts: Counter[str] = Counter()
    for member, stack in _scope_members(node, class_stack):
        qn = _declared_qn(member, stack, module_qn=module_qn, namespace_parts=namespace_parts)
        if qn is not None:
            counts[qn] += 1
    return frozenset(qn for qn, n in counts.items() if n > 1)


def _normalize_cpp_type(text: str) -> str:
    """Normalize a C++ type for an overload suffix — dot-free, whitespace-free, unqualified.

    A parameter pack or C variadic is rendered ``[]`` rather than ``...``, for
    the reason ``jvm.py`` renders Java varargs the same way: a dot in a
    qualified_name separates scope segments, so an ellipsis in the suffix would
    manufacture two fake ones.  The collapse is done before qualifier stripping,
    because ``T...`` otherwise looks like a qualified ``T.`` to that regex.
    Anything else that survives with a dot in it — a floating-point default
    template argument is the only real source — loses it for the same reason.
    """
    return normalize_type_text("".join(text.split()).replace("...", "[]")).replace(".", "")


def _declarator_identifier(declarator: Node | None) -> Node | None:
    """The parameter-name identifier inside a declarator, or None when unnamed.

    Only declarator-shaped children are followed, so a function-pointer
    parameter's own parameter list cannot be mistaken for the name.
    """
    if declarator is None:
        return None
    if declarator.type == "identifier":
        return declarator
    inner = declarator.child_by_field_name("declarator")
    if inner is not None:
        return _declarator_identifier(inner)
    for child in declarator.named_children:
        if child.type == "identifier" or child.type.endswith("declarator"):
            found = _declarator_identifier(child)
            if found is not None:
                return found
    return None


def _param_type_text(param: Node) -> str:
    """One parameter's type, with its name and any default value removed.

    C++ overloads on the whole parameter type, so ``const S&`` must not reduce
    to ``S``: neither the ``const`` nor the ``&`` is inside the ``type`` field —
    they are a sibling ``type_qualifier`` and part of the declarator.  Taking the
    node's own bytes and cutting the name out is what keeps them.  The default
    value goes because it is not part of the signature, and an out-of-line
    definition repeats the parameter without it.
    """
    raw = param.text
    if raw is None:
        return ""
    base = param.start_byte
    eq = next((c for c in param.children if c.type == "="), None)
    if eq is not None:
        raw = raw[: eq.start_byte - base]
    name = _declarator_identifier(param.child_by_field_name("declarator"))
    if name is not None and name.end_byte - base <= len(raw):
        raw = raw[: name.start_byte - base] + raw[name.end_byte - base :]
    return _normalize_cpp_type(raw.decode("utf-8", errors="replace"))


def _parameter_list(declarator: Node) -> Node | None:
    """The ``parameter_list`` of a function declarator, unwrapping pointer/reference returns."""
    node = declarator
    while node.type in ("pointer_declarator", "reference_declarator", "operator_cast"):
        inner = node.child_by_field_name("declarator") or next(
            (c for c in node.named_children if c.type.endswith("declarator")), None
        )
        if inner is None:
            return None
        node = inner
    return node.child_by_field_name("parameters")


def _overload_suffix(node: Node, declarator: Node) -> str:
    """Disambiguating qn suffix for an overloaded callable.

    ``<template params>(<param types>)<cv/ref qualifiers>`` — everything C++
    resolves an overload on, and nothing it does not.

    The template parameter list is in because overloads differing only there are
    real and common: fmt's ``test_value`` is two zero-argument templates told
    apart solely by an ``enable_if_t`` in the template header, and two of
    scan.h's eight ``read`` overloads take identical parameters and differ only
    by ``is_signed`` versus ``is_unsigned``.

    The trailing ``const``/``&``/``&&`` are in for the same reason:
    ranges-test.cc declares ``value() &``, ``value() const&``, ``value() &&``
    and ``value() const&&`` on one type.  ``noexcept`` is deliberately out — C++
    does not overload on it, so a header that spells it where the out-of-line
    definition does not would split one function into two nodes.
    """
    template = _template_wrapper(node)
    tp = template.child_by_field_name("parameters") if template is not None else None
    tp_text = _normalize_cpp_type(node_text(tp)) if tp is not None else ""
    params = _parameter_list(declarator)
    if params is None:
        return f"{tp_text}()"
    types: list[str] = []
    for child in params.children:
        if child.type in _PARAM_NODES:
            types.append(_param_type_text(child))
        elif child.type == "...":
            types.append("[]")
    fn = params.parent
    quals = "" if fn is None else "".join(node_text(c) for c in fn.children if c.type in _QUALIFIER_NODES)
    return f"{tp_text}({','.join(types)}){quals}"


def _process_function(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    overloaded: frozenset[str],
) -> None:
    """Process a function_definition node."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return

    tags = _extract_tags(node)
    template_parent = _template_wrapper(node)
    if template_parent is not None:
        tags.append("template")

    parts = _function_name_parts(node, declarator, class_stack)
    if parts is None:
        if _macro_invocation_name(node, declarator, class_stack) is not None:
            # A function-like macro whose expansion we cannot see. The only name
            # available is the macro's own, which every invocation in the file
            # shares — and one graph node claiming to be forty-seven definitions,
            # carrying an arbitrary body and the union of their edges, is worse
            # than forty-seven absences. Emit nothing; the body's calls are still
            # real and attribute to the module (ADR-0031).
            body = node.child_by_field_name("body")
            if body is not None:
                _extract_calls(body, f"{project_name}:{module_qn}", relationships)
        return
    scope_parts, name, is_gtest_case = parts
    if is_gtest_case:
        tags.append("test")

    # An overloaded name cannot identify one definition on its own, so it takes
    # its signature into the uid (ADR-0032). A name declared once does not.
    naming_stack = scope_parts if (not class_stack and scope_parts) else class_stack
    qn_name = name
    if _build_qn(module_qn, namespace_parts, naming_stack, name) in overloaded:
        qn_name = f"{name}{_overload_suffix(node, declarator)}"

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    # Doc comments for templates sit above the template_declaration wrapper
    docstring = _extract_doxygen_comment(template_parent or node, source)
    signature = _extract_function_signature(node, source)

    # Determine if this is a method (inside class body or qualified with class scope)
    is_method = bool(class_stack) or (is_cpp and bool(scope_parts))

    parent_type_name: str | None = None
    if is_method:
        # Method inside class body or qualified (ClassName::method, or
        # Outer::Inner::method for nested-scope definitions)
        actual_class = class_stack[-1] if class_stack else (scope_parts[-1] if scope_parts else "")
        kind = _method_callable_kind(name, actual_class, tags)

        if not class_stack and scope_parts:
            # Out-of-line definition — the class usually lives in another file
            # (header/impl split), so emit its NAME for post-batch resolution
            # (GraphClient.resolve_member_defines).  Fallback parent is this
            # file's Module (namespaces have no nodes).  parent_type_name is
            # the bare innermost class name (last scope part) — the resolver
            # matches on TypeDef name, not a '::'-qualified chain.
            qn = _build_qn(module_qn, namespace_parts, scope_parts, qn_name)
            parent_qn_str = f"{project_name}:{module_qn}"
            parent_type_name = scope_parts[-1]
        else:
            qn = _build_qn(module_qn, namespace_parts, class_stack, qn_name)
            parent_qn_str = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
        visibility = current_visibility
    else:
        kind = CallableKind.FUNCTION
        qn = _build_qn(module_qn, namespace_parts, class_stack, qn_name)
        parent_qn_str = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
        # File-scope static → PRIVATE (internal linkage)
        visibility = Visibility.PRIVATE if "static" in tags and not class_stack else current_visibility

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.CALLABLE,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            signature=signature,
            source=node_text(node),
            visibility=visibility,
            tags=tags,
        )
    )

    # DEFINES relationship
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_qn_str,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
            properties={"parent_type_name": parent_type_name} if parent_type_name else {},
        )
    )

    # Extract CALLS from function body
    body = node.child_by_field_name("body")
    if body is not None:
        _extract_calls(body, f"{project_name}:{qn}", relationships)


def _process_declaration(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    module_qn: str,
    is_cpp: bool,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
) -> bool:
    """Process a declaration node (global variables, forward declarations, etc.).

    Also handles function declarations embedded inside declarations and
    struct/class/enum specifiers within declarations.

    Returns True when the declaration was really an inline type definition and
    its subtree has been walked structurally — the caller must then not extract
    calls from it again.
    """
    # Check if this declaration contains a type specifier with a body (inline struct/enum/union/class def)
    for child in node.children:
        if child.type in _TYPE_DEF_NODES and child.child_by_field_name("body") is not None:
            _process_type_def(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            return True

    # Check if this is a function declaration (has a function_declarator but no body on this node)
    # We only care about actual variable declarations here
    for child in node.children:
        if child.type == "function_declarator":
            # This is a function forward declaration — skip (we only track definitions)
            return False

    # Extract declarator name for variable declarations
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return False

    # Skip function declarations, including pointer/reference-returning prototypes
    # (`int* alloc(int);` wraps the function_declarator in a pointer_declarator)
    if declarator.type == "function_declarator" or _prototype_declarator(declarator) is not None:
        return False

    name = _get_declarator_name(declarator)
    if not name:
        return False

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    docstring = _extract_doxygen_comment(node, source)
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)
    tags = _extract_tags(node)

    # File-scope static → PRIVATE
    visibility = Visibility.PRIVATE if "static" in tags and not class_stack else current_visibility

    entities.append(
        ParsedEntity(
            name=name,
            qualified_name=f"{project_name}:{qn}",
            label=NodeLabel.VALUE,
            kind=ValueKind.VARIABLE,
            line_start=line_start,
            line_end=line_end,
            file_path=path,
            docstring=docstring,
            source=node_text(node),
            visibility=visibility,
            tags=tags,
        )
    )

    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )
    return False


def _process_field_declaration(
    node: Node,
    *,
    path: str,
    source: bytes,
    project_name: str,
    is_cpp: bool,
    module_qn: str,
    namespace_parts: list[str],
    class_stack: list[str],
    current_visibility: str,
    entities: list[ParsedEntity],
    relationships: list[ParsedRelationship],
    overloaded: frozenset[str],
) -> bool:
    """Process a field_declaration inside a struct/class body.

    Returns True when the field_declaration was really a nested type definition
    and its subtree has been walked structurally — the caller must then not
    extract calls from it again.
    """
    # A type nested inside a class body arrives wrapped in a field_declaration
    # rather than the `declaration` that wraps one at file scope, so it needs
    # the same unwrapping — otherwise the nested type and all its methods
    # vanish.
    for child in node.children:
        if child.type in _TYPE_DEF_NODES and child.child_by_field_name("body") is not None:
            _process_type_def(
                child,
                path=path,
                source=source,
                project_name=project_name,
                module_qn=module_qn,
                is_cpp=is_cpp,
                namespace_parts=namespace_parts,
                class_stack=class_stack,
                current_visibility=current_visibility,
                entities=entities,
                relationships=relationships,
            )
            return True

    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return False
    name = _get_declarator_name(declarator)
    if not name:
        return False

    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    qn = _build_qn(module_qn, namespace_parts, class_stack, name)

    if _prototype_declarator(declarator) is not None:
        # Method declaration without a body (`void draw() const;`) — a Callable,
        # not a field.  Function-pointer members stay on the field path.
        if qn in overloaded:
            qn = _build_qn(module_qn, namespace_parts, class_stack, f"{name}{_overload_suffix(node, declarator)}")
        tags = _extract_tags(node)
        kind = _method_callable_kind(name, class_stack[-1] if class_stack else "", tags)
        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=f"{project_name}:{qn}",
                label=NodeLabel.CALLABLE,
                kind=kind,
                line_start=line_start,
                line_end=line_end,
                file_path=path,
                docstring=_extract_doxygen_comment(node, source),
                signature=_extract_function_signature(node, source),
                visibility=current_visibility,
                tags=tags,
            )
        )
    else:
        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=f"{project_name}:{qn}",
                label=NodeLabel.VALUE,
                kind=ValueKind.FIELD,
                line_start=line_start,
                line_end=line_end,
                file_path=path,
                visibility=current_visibility,
            )
        )

    parent_full_qn = _parent_qn(project_name, module_qn, namespace_parts, class_stack)
    relationships.append(
        ParsedRelationship(
            from_qualified_name=parent_full_qn,
            rel_type=RelType.DEFINES,
            to_name=f"{project_name}:{qn}",
        )
    )
    return False


def _call_relationship(func: Node | None, from_qn: str) -> ParsedRelationship | None:
    """Build the CALLS relationship for a call's ``function`` expression.

    Returns None when the callee has no name that could ever resolve —
    ``fp[i]()``, ``(*fp)()``, a chained ``f()()`` — or when it is a C++ named
    cast, which the grammar shapes exactly like a call but which is a keyword,
    not a callable.
    """
    if func is None:
        return None

    # `(T::min)()` — parenthesised to dodge a min/max macro. Common enough in
    # Windows-targeting C++ to be worth unwrapping.
    while func.type == "parenthesized_expression":
        inner = func.named_children[0] if func.named_children else None
        if inner is None:
            return None
        func = inner

    if func.type == "template_function":
        # `max_value<T>()` — the callee is the template's name; the angle
        # brackets are type arguments, not part of it.
        name_node = func.child_by_field_name("name")
        if name_node is None or node_text(name_node) in _NAMED_CASTS:
            return None
        func = name_node

    if func.type in ("identifier", "qualified_identifier"):
        # qualified: ns::func() keeps the full path, which is what resolution matches on
        call_name = node_text(func)
        return (
            ParsedRelationship(from_qualified_name=from_qn, rel_type=RelType.CALLS, to_name=call_name)
            if call_name
            else None
        )

    if func.type == "field_expression":
        # obj.method() — extract method name
        field = func.child_by_field_name("field")
        call_name = node_text(field) if field is not None else None
        return (
            ParsedRelationship(
                from_qualified_name=from_qn,
                rel_type=RelType.CALLS,
                to_name=call_name,
                properties=call_receiver_props(func.child_by_field_name("argument")),
            )
            if call_name
            else None
        )

    return None


def _extract_calls(
    node: Node,
    from_qn: str,
    relationships: list[ParsedRelationship],
) -> None:
    """Recursively extract every call under *node*, attributed to *from_qn*.

    Descends into everything, including lambda bodies, local classes, and the
    nested ``function_definition`` nodes tree-sitter produces where an
    unexpandable macro defeated it.  A call is always attributed to the nearest
    enclosing named scope (ADR-0031), and for anything reachable from here that
    scope is *from_qn* — legal C++ cannot nest a function definition inside a
    function body, so a nested one is either a GNU extension or parse debris,
    and in both cases the calls belong to whoever wrote the enclosing code.

    Callers must not also walk this subtree structurally; ``_walk_translation_unit``
    guarantees that by never entering a function body itself.
    """
    for child in node.children:
        if child.type == "call_expression":
            rel = _call_relationship(child.child_by_field_name("function"), from_qn)
            if rel is not None:
                relationships.append(rel)
        _extract_calls(child, from_qn, relationships)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------


def _resolve_header_dialect(source: bytes) -> str:
    """Route a ``.h`` file to the C++ grammar only if it contains C++.

    Registered on the C config because C owns ``.h`` by default; anything the
    sniff cannot call is left exactly where it was.
    """
    return "cpp" if (_CPP_AVAILABLE and sniff_header_is_cpp(source)) else "c"


if _C_AVAILABLE:
    register_language(
        LanguageConfig(
            name="c",
            extensions=_C_EXTENSIONS,
            language=_C_LANGUAGE,
            query=_C_QUERY,
            parse_func=_parse_cpp,
            ambiguous_extensions=frozenset({".h"}),
            resolve_dialect=_resolve_header_dialect,
        )
    )

if _CPP_AVAILABLE:
    register_language(
        LanguageConfig(
            name="cpp",
            extensions=_CPP_EXTENSIONS,
            language=_CPP_LANGUAGE,
            query=_CPP_QUERY,
            parse_func=_parse_cpp,
        )
    )
