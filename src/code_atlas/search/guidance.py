"""Subagent guidance tools for Code Atlas.

Provides Cypher validation, search strategy advice, and usage guides
for AI coding agents consuming the MCP interface.  All functions are
deterministic — no LLM calls, no external API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

from code_atlas.schema import NodeLabel, RelType
from code_atlas.search.engine import analyze_query

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient

# ---------------------------------------------------------------------------
# Schema-derived constants
# ---------------------------------------------------------------------------

_LABEL_NAMES: frozenset[str] = frozenset(lbl.value for lbl in NodeLabel)
_REL_NAMES: frozenset[str] = frozenset(r.value for r in RelType)

# ---------------------------------------------------------------------------
# Relationship summary (one-line per RelType)
# ---------------------------------------------------------------------------

_RELATIONSHIP_SUMMARY: dict[str, str] = {
    RelType.CONTAINS: "Parent -> child structural nesting (Project->Package->Module)",
    RelType.DEFINES: "Type/module -> method/field/value it defines",
    RelType.INHERITS: "Subclass -> base class",
    RelType.IMPLEMENTS: "Class -> interface/protocol it implements",
    RelType.CALLS: "Caller -> callee function/method invocation",
    RelType.IMPORTS: "Module/entity -> imported dependency",
    RelType.USES_TYPE: "Entity -> type it references in annotations/signatures",
    RelType.OVERRIDES: "Method -> parent method it overrides",
    RelType.DEPENDS_ON: "Project -> project dependency (monorepo)",
    RelType.DOCUMENTS: "Doc section -> code entity it documents",
    RelType.MOTIVATED_BY: "Code entity -> ADR/doc that motivated it",
    RelType.SIMILAR_TO: "Entity -> semantically similar entity (embedding cosine)",
    RelType.HANDLES_ROUTE: "Callable -> HTTP route it handles (pattern-detected)",
    RelType.HANDLES_EVENT: "Callable -> event it handles (pattern-detected)",
    RelType.REGISTERED_BY: "Entity -> factory/registry that registers it (pattern-detected)",
    RelType.INJECTED_INTO: "Dependency -> class it is injected into (pattern-detected)",
    RelType.TESTS: "Test function -> entity it tests (pattern-detected)",
    RelType.HANDLES_COMMAND: "Callable -> CLI/bot command it handles (pattern-detected)",
}

# ---------------------------------------------------------------------------
# Cypher examples (cookbook for agents)
# ---------------------------------------------------------------------------

CYPHER_EXAMPLES: list[dict[str, str]] = [
    {
        "description": "Find all methods defined by a class",
        "query": "MATCH (c:TypeDef {name: $name})-[:DEFINES]->(m:Callable) RETURN m.name, m.signature, m.kind",
    },
    {
        "description": "Find callers of a function",
        "query": (
            "MATCH (caller)-[:CALLS]->(f:Callable {name: $name}) "
            "RETURN caller.name, caller.qualified_name, caller.file_path"
        ),
    },
    {
        "description": "Class inheritance chain (ancestors)",
        "query": "MATCH (c:TypeDef {name: $name})-[:INHERITS*]->(base:TypeDef) RETURN base.name, base.qualified_name",
    },
    {
        "description": "Module imports (internal and external)",
        "query": (
            "MATCH (m:Module {name: $name})-[:IMPORTS]->(dep) "
            "RETURN dep.name, dep.qualified_name, labels(dep)[0] AS type"
        ),
    },
    {
        "description": "Tests covering an entity",
        "query": "MATCH (t)-[:TESTS]->(target {name: $name}) RETURN t.name, t.qualified_name, t.file_path",
    },
    {
        "description": "All entities in a file",
        "query": (
            "MATCH (n) WHERE n.file_path = $path "
            "RETURN n.name, n.qualified_name, labels(n)[0] AS label, n.kind ORDER BY n.line_start"
        ),
    },
]

# ---------------------------------------------------------------------------
# Usage guide content
# ---------------------------------------------------------------------------

_USAGE_GUIDE: dict[str, str] = {
    "": (
        "# Code Atlas Quick Start\n"
        "\n"
        "**Find by name:** `get_node(name)` -> `get_context(uid)` to expand.\n"
        "**Search by meaning:** `hybrid_search(query)` — fuses graph, BM25, and vector.\n"
        "**Custom traversal:** `schema_info` for schema + examples -> "
        "`validate_cypher(query)` to check -> `cypher_query(query)` to run.\n"
        "**Explore project:** `index_status` for entity counts, `list_projects` for dependencies.\n"
        "\n"
        "hybrid_search is the default — it auto-adjusts weights based on query shape. "
        "Use get_node when you know an exact entity name. "
        "Use cypher_query for structural traversals (callers, inheritance, imports). "
        "Always call validate_cypher before cypher_query to catch errors early."
    ),
    "searching": (
        "# Search Strategy\n"
        "\n"
        "**hybrid_search** (default): Fuses graph name-matching, BM25 keyword, and vector "
        "semantic search via RRF. Auto-boosts channels by query shape.\n"
        "**text_search**: BM25 only. Use for exact keyword/phrase matching. "
        "Supports field-specific queries (name:Foo, docstring:auth).\n"
        "**vector_search**: Semantic only. Use when looking for conceptually similar code "
        "without knowing exact names.\n"
        "\n"
        "Scope filtering: pass `scope='project-name'` to restrict to one project. "
        "Label filtering: pass `label='Callable'` to text/vector search.\n"
        "Weight overrides: `weights='{\"vector\": 2.0}'` in hybrid_search.\n"
        "Exclude toggles: hybrid_search excludes tests/stubs/generated by default. "
        "Set `exclude_tests=false` to include test entities (needed to find tests for a function)."
    ),
    "cypher": (
        "# Cypher Query Guide\n"
        "\n"
        "**UID format:** `{project_name}:{qualified_name}` — qualified_name mirrors the "
        "source tree path (e.g., `myproj:src.mypackage.module.MyClass`). "
        "Use get_node to discover actual UIDs before writing Cypher with uid filters.\n"
        "**Labels:** Project, Package, Module, TypeDef, Callable, Value, DocFile, DocSection, "
        "ExternalPackage, ExternalSymbol. Common mistakes: Function→Callable, Class→TypeDef.\n"
        "**Relationships:** Always directed. Check schema_info for full list.\n"
        "**Common patterns:**\n"
        "- `MATCH (n:Callable {name: $name})` — find by name\n"
        "- `MATCH (a)-[:CALLS]->(b)` — call graph traversal\n"
        "- `MATCH path = (a)-[:INHERITS*]->(b)` — variable-length for hierarchies\n"
        "\n"
        "LIMIT is auto-appended (default 20, max 100). Write ops are rejected. "
        "Use $params for values — don't interpolate strings into Cypher."
    ),
    "navigation": (
        "# Navigation Guide\n"
        "\n"
        "**get_node** cascade: exact uid -> exact name -> suffix (.Name) -> prefix (Name.) "
        "-> contains. First match wins. Pass `label` to filter by type.\n"
        "**get_context** expands a uid into: parent, siblings, callers, callees, docs. "
        "Toggle sections with `include_hierarchy`, `include_calls`, `include_docs`. "
        "`call_depth` (1-3) controls CALLS traversal hops.\n"
        "\n"
        "Workflow: get_node to find -> pick uid from results -> get_context to explore."
    ),
    "patterns": (
        "# Pattern-Detected Relationships\n"
        "\n"
        "Code Atlas detects cross-cutting patterns and creates typed edges:\n"
        "- **TESTS**: test function -> entity under test (pytest naming conventions)\n"
        "- **HANDLES_ROUTE**: handler -> HTTP route (Flask/FastAPI/Django decorators)\n"
        "- **HANDLES_EVENT**: handler -> event type (event-driven frameworks)\n"
        "- **HANDLES_COMMAND**: handler -> CLI/bot command\n"
        "- **INJECTED_INTO**: dependency -> class it's injected into (DI frameworks)\n"
        "- **REGISTERED_BY**: entity -> factory/registry that registers it\n"
        "\n"
        "Query these like any relationship: "
        "`MATCH (t)-[:TESTS]->(target {name: 'MyClass'}) RETURN t`"
    ),
}

# ---------------------------------------------------------------------------
# Validation types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error" | "warning" | "info"
    message: str


# ---------------------------------------------------------------------------
# Static Cypher validation
# ---------------------------------------------------------------------------

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|DELETE|SET|MERGE|REMOVE|DROP|DETACH)\b",
    re.IGNORECASE,
)

_RETURN_RE = re.compile(r"\bRETURN\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

# Common label mistakes → correct label (supplements fuzzy matching)
_LABEL_ALIASES: dict[str, str] = {
    "Function": "Callable",
    "Method": "Callable",
    "Class": "TypeDef",
    "Interface": "TypeDef",
    "Enum": "TypeDef",
    "Constant": "Value",
    "Variable": "Value",
    "File": "Module",
    "Doc": "DocSection",
    "Dependency": "ExternalPackage",
}

# Regex to extract labels from :Label patterns in Cypher (exclude [:REL] inside brackets)
_LABEL_REF_RE = re.compile(r"(?<!\[):([A-Z][A-Za-z_]+)")

# Regex to extract relationship types from [:REL_TYPE] patterns
_REL_REF_RE = re.compile(r"\[:([A-Z_]+(?:\*[0-9.]*)?)\]")


def _extract_rel_type(match_str: str) -> str:
    """Strip variable-length suffixes like *1..3 from relationship type."""
    return match_str.split("*", maxsplit=1)[0]


def _check_balanced_brackets(query: str) -> list[ValidationIssue]:
    """Check that parentheses, brackets, and braces are balanced (ignoring strings)."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    issues: list[ValidationIssue] = []
    stack: list[str] = []
    in_string = False
    string_char = ""
    for char in query:
        if in_string:
            if char == string_char:
                in_string = False
            continue
        if char in ("'", '"'):
            in_string = True
            string_char = char
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack[-1] != char:
                issues.append(ValidationIssue("error", f"Unbalanced '{char}' in query."))
                return issues
            stack.pop()
    if stack:
        expected = stack[-1]
        issues.append(ValidationIssue("error", f"Unbalanced brackets — expected closing '{expected}'."))
    return issues


def validate_cypher_static(query: str) -> list[ValidationIssue]:
    """Validate a Cypher query statically without database access.

    Checks:
    - Write keyword detection
    - Balanced parentheses, brackets, braces
    - RETURN clause present
    - Labels vs NodeLabel enum (with close-match suggestions)
    - Relationship types vs RelType enum
    - LIMIT clause present
    """
    issues: list[ValidationIssue] = []

    if _WRITE_KEYWORDS.search(query):
        issues.append(
            ValidationIssue(
                "error",
                "Query contains write operations (CREATE/DELETE/SET/MERGE/REMOVE/DROP/DETACH)"
                " which are rejected by the MCP server.",
            )
        )

    issues.extend(_check_balanced_brackets(query))

    if not _RETURN_RE.search(query):
        issues.append(ValidationIssue("warning", "Query has no RETURN clause — results may be empty."))

    for match in _LABEL_REF_RE.finditer(query):
        label = match.group(1)
        if label not in _LABEL_NAMES:
            # Check explicit aliases first, then fuzzy match
            alias = _LABEL_ALIASES.get(label)
            if alias:
                suggestion = f" Did you mean '{alias}'?"
            else:
                close = get_close_matches(label, list(_LABEL_NAMES), n=1, cutoff=0.6)
                suggestion = f" Did you mean '{close[0]}'?" if close else ""
            issues.append(ValidationIssue("error", f"Unknown label '{label}'.{suggestion}"))

    for match in _REL_REF_RE.finditer(query):
        rel_raw = _extract_rel_type(match.group(1))
        if rel_raw and rel_raw not in _REL_NAMES:
            close = get_close_matches(rel_raw, list(_REL_NAMES), n=1, cutoff=0.6)
            suggestion = f" Did you mean '{close[0]}'?" if close else ""
            issues.append(ValidationIssue("error", f"Unknown relationship type '{rel_raw}'.{suggestion}"))

    if not _LIMIT_RE.search(query):
        issues.append(ValidationIssue("info", "No LIMIT clause — the server will auto-append LIMIT 20."))

    return issues


# ---------------------------------------------------------------------------
# EXPLAIN-based validation (requires DB)
# ---------------------------------------------------------------------------


async def validate_cypher_explain(graph: GraphClient, query: str) -> ValidationIssue | None:
    """Run EXPLAIN on a query to catch syntax errors the static check misses.

    Returns a ValidationIssue on failure, or None if the query is valid.
    Gracefully returns None if the DB is unreachable.
    """
    try:
        await graph.execute(f"EXPLAIN {query}")
    except Exception as exc:
        msg = str(exc)
        # Filter out connection errors — those aren't query problems
        if "connection" in msg.lower() or "refused" in msg.lower():
            return None
        return ValidationIssue("error", f"EXPLAIN failed: {msg}")
    return None


# ---------------------------------------------------------------------------
# Search strategy planner
# ---------------------------------------------------------------------------

# Patterns for identifying structural / relationship questions
_STRUCTURAL_PATTERNS = re.compile(
    r"\b(call(?:s|ed|er|ing)|inherit|base|parent|child|import|depend|override|implement|test(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)

_DOC_PATTERNS = re.compile(
    r"\b(doc(?:s|umentation|string)?|readme|adr|explained|description)\b",
    re.IGNORECASE,
)


def plan_strategy(question: str) -> dict[str, Any]:
    """Analyze a question and recommend which search tool to use.

    Returns a dict with recommended_tool, params, explanation, and alternatives.
    """
    stripped = question.strip()
    weights = analyze_query(stripped)

    # Check if it's a structural/relationship question
    if _STRUCTURAL_PATTERNS.search(stripped):
        # Identify the relevant relationship type
        rel_hint = ""
        lower = stripped.lower()
        if "call" in lower:
            rel_hint = "CALLS"
        elif "inherit" in lower or "base" in lower or "parent" in lower:
            rel_hint = "INHERITS"
        elif "import" in lower:
            rel_hint = "IMPORTS"
        elif "depend" in lower:
            rel_hint = "DEPENDS_ON"
        elif "override" in lower:
            rel_hint = "OVERRIDES"
        elif "implement" in lower:
            rel_hint = "IMPLEMENTS"
        elif "test" in lower:
            rel_hint = "TESTS"

        explanation = "Structural/relationship question — Cypher traversal is most precise."
        if rel_hint:
            explanation += f" Relevant relationship: {rel_hint}."

        return {
            "recommended_tool": "cypher_query",
            "params": {},
            "explanation": explanation,
            "alternatives": [
                "hybrid_search — if you don't know exact names yet",
                "get_context — if you have a uid and want callers/callees",
            ],
        }

    # Check for documentation questions
    if _DOC_PATTERNS.search(stripped):
        return {
            "recommended_tool": "hybrid_search",
            "params": {
                "search_types": "bm25,vector",
                "weights": '{"bm25": 1.5, "vector": 2.0}',
            },
            "explanation": "Documentation question — vector + BM25 search across doc entities.",
            "alternatives": [
                "text_search with label='DocSection' — for exact keyword matching in docs",
            ],
        }

    # Identifier-like queries → get_node
    is_identifier = weights.get("graph", 0) >= 2.0

    if is_identifier:
        return {
            "recommended_tool": "get_node",
            "params": {},
            "explanation": "Identifier-like query — direct name lookup is fastest.",
            "alternatives": [
                "hybrid_search — if get_node returns too many or zero results",
            ],
        }

    # Fall through to hybrid_search
    weight_json = "{" + ", ".join(f'"{k}": {v}' for k, v in sorted(weights.items())) + "}"
    return {
        "recommended_tool": "hybrid_search",
        "params": {
            "weights": weight_json,
        },
        "explanation": "Natural language query — hybrid search with auto-adjusted weights.",
        "alternatives": [
            "get_node — if you know an exact entity name",
            "cypher_query — for structural traversals (callers, inheritance)",
        ],
    }


# ---------------------------------------------------------------------------
# Usage guide accessor
# ---------------------------------------------------------------------------


def get_guide(topic: str = "") -> dict[str, Any]:
    """Return a usage guide for the given topic.

    Valid topics: '', 'searching', 'cypher', 'navigation', 'patterns'.
    Returns ``{"topic": ..., "guide": ..., "available_topics": [...]}``
    """
    key = topic.strip().lower()
    available = sorted(t for t in _USAGE_GUIDE if t)

    if key not in _USAGE_GUIDE:
        return {
            "topic": key,
            "guide": f"Unknown topic '{key}'. Available topics: {', '.join(available)}",
            "available_topics": available,
        }

    return {
        "topic": key or "quickstart",
        "guide": _USAGE_GUIDE[key],
        "available_topics": available,
    }


# ---------------------------------------------------------------------------
# Import-time validation — ensure schema enums are fully covered
# ---------------------------------------------------------------------------


def _validate_subagent_completeness() -> None:
    """Verify all RelType values have entries in _RELATIONSHIP_SUMMARY."""
    all_rels = {r.value for r in RelType}
    summary_rels = set(_RELATIONSHIP_SUMMARY.keys())
    missing = all_rels - summary_rels
    if missing:
        raise RuntimeError(f"RelType values missing from _RELATIONSHIP_SUMMARY: {missing}")
    extra = summary_rels - all_rels
    if extra:
        raise RuntimeError(f"Extra keys in _RELATIONSHIP_SUMMARY not in RelType: {extra}")


_validate_subagent_completeness()
