# Repository Guidelines for Code Atlas

These practices improve both human readability and how Code Atlas indexes, searches, and serves your codebase to AI
agents.

### 1. Write Meaningful First-Line Doc Comments

The first line of every public doc comment (Python docstrings, JSDoc, Javadoc, `///`, etc.) should be a concise summary
of what the entity does. Code Atlas embeds doc comments for vector search — a clear first line dramatically improves
semantic recall and is often all an agent sees (truncated to 200 chars in results).

```python
# Do
def parse_file(path: Path) -> ParsedFile | None:
    """Parse a single source file into an AST and extract entities."""

# Don't
def parse_file(path: Path) -> ParsedFile | None:
    """This function takes a path and does parsing."""
```

### 2. Use Type Annotations on Signatures

Annotate function parameters and return types. The parser extracts the full signature including annotations — typed
signatures create `USES_TYPE` edges in the graph and produce more distinctive embeddings.

```python
# Do
async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:

# Don't
async def execute(self, query, params=None):
```

### 3. Use Named Imports

Import specific symbols rather than using wildcards or re-exporting everything. Each named import creates an `IMPORTS`
edge — wildcard or barrel imports produce a single vague edge that hides actual dependencies from graph analysis.

```python
# Do
from code_atlas.schema import NodeLabel, RelType, SCHEMA_VERSION

# Don't
from code_atlas.schema import *
```

### 4. One Concept per Module

Each module should have a single responsibility, grouped into packages. Focused modules produce clean graph
neighborhoods and efficient delta detection — a change in one function doesn't re-evaluate a 2000-line god module.

```
# Do
src/auth/
    tokens.py       # JWT creation and validation
    middleware.py    # Authentication middleware
    permissions.py   # Permission checking

# Don't
src/auth.py  # 2000 lines covering everything
```

### 5. Follow Naming Conventions Consistently

Use descriptive, consistent names following language conventions. Graph search uses exact/suffix/contains matching and
BM25 tokenizes names — `calculate_user_score` is found by "user score" but `calc_us` is not.

```python
# Do
class EventBus:
    async def publish(self): ...
    async def subscribe(self): ...

# Don't
class EB:
    async def pub(self): ...
```

### 6. Use Conventional Patterns for Frameworks

Use standard decorators and naming for routes, CLI commands, tests, and event handlers. Code Atlas has pluggable pattern
detectors that recognize `@app.route()`, `@app.command()`, `test_*` prefixes, etc. — detected patterns create
`HANDLES_ROUTE`, `HANDLES_COMMAND`, and `TESTS` edges.

```python
# Do — detectable
@router.get("/api/users")
async def list_users(): ...

def test_parse_file(): ...

# Don't — invisible to detectors
for name, func in commands.items():
    app.register(name, func)
```

### 7. Keep Inheritance Hierarchies Explicit

Use explicit class inheritance rather than dynamic mixins or metaclass magic. `class Foo(Bar)` creates an `INHERITS`
edge that powers hierarchy traversal and Mermaid diagrams — dynamic inheritance via `type()` or conditional bases is
invisible to AST parsing.

```python
# Do
class ASTConsumer(TierConsumer): ...

# Don't
Consumer = make_consumer(TierConsumer, features=["ast"])
```

### 8. Use `.atlasignore` to Exclude Noise

Exclude generated code, vendored deps, and build artifacts from indexing. These files pollute the graph with meaningless
entities, inflate search results, and waste parsing time.

```gitignore
# .atlasignore (same syntax as .gitignore)
*_pb2.py
*.generated.py
vendor/
dist/
build/
```

Code Atlas already excludes `.git/`, `__pycache__/`, `.venv/`, `node_modules/` by default.

### 9. Write File-Level Doc Comments

Every file/module should have a doc comment at the top explaining its purpose. Module nodes are embedded for vector
search — without a doc comment the embedding is just the qualified name, which has limited semantic value.

```python
"""Async Memgraph client for Code Atlas.

Wraps the neo4j Bolt driver with schema management, delta-aware upserts,
and multi-channel search (graph, vector, BM25).
"""
```

### 10. Prefer Small, Focused Functions

Keep functions focused on one task; extract helpers for complex logic. Each function is a separate node with its own
content hash — small functions mean granular delta detection and individually searchable entities.

### 11. Group Tests Near What They Test

Mirror source structure in your test directory and name test functions to match their targets. The test detector creates
`TESTS` edges by matching test naming conventions to targets — e.g. `test_graph_search_returns_results` links to
`graph_search`.

```
src/code_atlas/graph/client.py  →  tests/test_graph.py
src/code_atlas/search/engine.py →  tests/test_search.py
```

## Summary

| Guideline               | Coding Benefit            | Code Atlas Benefit                                 |
| ----------------------- | ------------------------- | -------------------------------------------------- |
| First-line doc comments | Readable API docs         | Better vector search, richer MCP responses         |
| Type annotations        | Catches bugs, IDE support | `USES_TYPE` edges, better signatures in search     |
| Named imports           | Clear dependencies        | `IMPORTS` edges for graph analysis                 |
| Focused modules         | Maintainability           | Clean graph neighborhoods, efficient delta         |
| Consistent naming       | Readability               | Better graph/BM25 search matches                   |
| Conventional patterns   | Framework interop         | Pattern detection (routes, commands, tests)        |
| Explicit inheritance    | Clear design              | `INHERITS` edges for hierarchy analysis            |
| `.atlasignore`          | Clean repo                | Less noise in graph and search                     |
| File-level doc comments | Onboarding                | Module-level vector search, agent context          |
| Small functions         | Testability               | Granular delta detection, individual searchability |
| Test naming             | Test discovery            | `TESTS` relationship edges                         |
