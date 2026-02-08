# ADR-0007: Qualified Name Resolution Strategy

## Status

Accepted

## Date

2026-02-08

## Context

Qualified name resolution is the hardest recurring problem in code graph systems. It affects search, type inference,
cross-file calls, and NL-to-Cypher translation. Prior art shows this consistently:

- **code-graph-rag #278**: NL-to-Cypher generated `{name: 'VatManager'}` but data was stored as `qualified_name`
- **code-graph-rag #275**: Java CamelCase conflicts across packages
- **codegraph-rust**: Cross-file reference resolution was the primary source of incorrect edges

Code Atlas needs a naming convention that works across languages, supports multiple matching modes for AI agents, and
disambiguates results when multiple entities share the same short name.

## Decision

### Naming Scheme: Hybrid with Multiple Properties (Approach C)

Each graph node stores three name-related properties, each serving a distinct role:

| Property         | Purpose                        | Example                              |
| ---------------- | ------------------------------ | ------------------------------------ |
| `uid`            | Global unique identifier       | `myproject:auth.service.UserService` |
| `name`           | Short unqualified name         | `UserService`                        |
| `qualified_name` | Dotted path from project root  | `auth.service.UserService`           |
| `file_path`      | Filesystem path to source file | `auth/service.py`                    |

**`uid` format**: `{project_name}:{qualified_name}`

This provides project isolation (multi-project graphs) via the `uid` prefix, while `qualified_name` alone is sufficient
for within-project lookups.

### Language-Specific Qualified Name Construction

#### Python (implemented)

```
qualified_name = dotted_file_path + "." + nesting
```

Rules:

- File path segments become dotted components: `auth/service.py` → `auth.service`
- `__init__.py` collapses to the package name: `auth/__init__.py` → `auth`
- Class/function nesting is appended: `auth.service.UserService.validate`
- The `name` property is always the leaf: `validate`

#### Java (future)

```
qualified_name = package_declaration + "." + nesting
```

Rules:

- Use the `package` declaration from source, not file path (Java packages are authoritative)
- `com.company.auth.UserService.validate`
- Inner classes: `UserService.Builder`

#### TypeScript / JavaScript (future)

```
qualified_name = dotted_file_path + "." + nesting
```

Rules:

- File path with `/` → `.` conversion: `src/auth/service.ts` → `src.auth.service`
- Default exports use file name as entity name
- Named exports use their declared name

#### Go (future)

```
qualified_name = package_name + "." + symbol
```

Rules:

- Go has flat packages — no nesting beyond package
- `http.ListenAndServe`, `auth.ValidateToken`
- Receiver methods: `auth.UserService.Validate`

### Matching Cascade

The `get_node` tool uses a 5-stage cascade that short-circuits on the first stage with results:

| Stage | Mode       | Cypher Pattern                                             | Use Case                                          |
| ----- | ---------- | ---------------------------------------------------------- | ------------------------------------------------- |
| 1     | Exact uid  | `n.uid = $name`                                            | Precise lookup from prior result                  |
| 2     | Exact name | `n.name = $name`                                           | Simple entity lookup                              |
| 3     | Suffix     | `n.qualified_name ENDS WITH '.' + $name`                   | Agent shorthand (`UserService.validate`)          |
| 4     | Prefix     | `n.qualified_name STARTS WITH $name + '.'`                 | "All members of..." (`auth.service.UserService.`) |
| 5     | Contains   | `n.qualified_name CONTAINS $name OR n.name CONTAINS $name` | Fuzzy exploration                                 |

**Rationale for ordering**: Exact matches are fastest (indexed property lookup) and most precise. Suffix matches are the
most common agent pattern ("find UserService" when qn is `auth.service.UserService`). Prefix finds children of a scope.
Contains is the fallback for broad exploration.

### Disambiguation Ranking

When a cascade stage returns multiple results, they are ranked by relevance:

1. **Source over test**: Entities whose `file_path` does NOT contain "test" rank higher. Users typically want production
   code, not test doubles.
2. **Visibility**: Public > protected > internal > private. Users typically want public API surfaces.
3. **Shorter qualified name**: More canonical entities rank higher (e.g., `UserService` over
   `test_user.MockUserService`).

This is implemented as a stable sort via `_rank_results()`, a pure function applied after each cascade stage.

### Index Strategy

All three name properties are indexed for fast lookup:

| Property         | Index Type             | Purpose                                      |
| ---------------- | ---------------------- | -------------------------------------------- |
| `uid`            | Unique constraint      | Primary key, O(1) exact lookup               |
| `name`           | Label + property index | Fast exact name match (Stage 2)              |
| `qualified_name` | Label + property index | Suffix/prefix/contains matching (Stages 3-5) |
| `file_path`      | Label + property index | File-based queries                           |

These are defined in `schema.py` and applied via `ensure_schema()`.

## Consequences

### Positive

- **Agent-friendly**: The cascade means agents can use short names and still find entities, without needing to know the
  full qualified name upfront
- **Unambiguous**: `uid` guarantees uniqueness across projects; `qualified_name` is unique within a project
- **Extensible**: New languages only need to define their `qualified_name` construction rule — the matching cascade and
  ranking are language-agnostic
- **No re-index needed**: The property set and index strategy are already implemented; this ADR documents and ratifies
  the existing design

### Negative

- **String matching performance**: `CONTAINS` and `ENDS WITH` on `qualified_name` require full index scans in Memgraph.
  Acceptable for graphs under ~100K nodes; may need optimization for larger codebases
- **Language-specific construction**: Each new language requires implementing its own `qualified_name` builder in the
  parser

### Risks

- **Name collisions in CONTAINS mode**: Broad queries like `CONTAINS "Service"` may return too many results. Mitigated
  by the cascade (CONTAINS is last resort) and the limit parameter
- **Test file heuristic**: Ranking by "test" in file path is a simple heuristic that could misrank legitimate code in
  paths like `testing_utils/`. Acceptable for v1

## Alternatives Considered

### Alternative 1: Dotted Path from Project Root Only

```
qualified_name = project.src.auth.service.UserService.validate
```

- Simple, unambiguous, mirrors file structure exactly
- Rejected because: excessively long paths, file structure doesn't always match logical structure, `src/` prefixes add
  noise

### Alternative 2: Logical Package Path

```
qualified_name = auth.UserService.validate
```

- Matches developer mental model, shorter names
- Rejected because: requires language-specific package detection heuristics, Python's package model (based on file path)
  doesn't always align with logical grouping, harder to implement consistently across languages

### Alternative 3: Single `name` Property with Full Qualified Name

- Store only `qualified_name`, derive `name` at query time
- Rejected because: forces every query to parse the qualified name, no fast exact-name lookup, poor agent UX

## References

- [ADR-0001: Memgraph as Database](./0001-memgraph-as-database.md) — index and constraint support
- [Schema implementation](../../src/code_atlas/schema.py) — property indices
- [Parser implementation](../../src/code_atlas/parser.py) — `_module_qualified_name()` builds Python qualified names
- [MCP Server](../../src/code_atlas/mcp_server.py) — `get_node` cascade and `_rank_results()`
