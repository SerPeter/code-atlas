# ADR-0015: Embedded Backend Option (SQLite Graph + Queue)

## Status

Accepted

## Date

2026-07-18

## Context

Code Atlas hard-requires two external services: Memgraph (graph + vector + BM25 search) and Valkey (event bus). This is
a real adoption barrier for users who want to try or run the tool without standing up Docker infrastructure first. The
idea was scoped down from the open P2 item "Embedded/serverless backend option" in
`.tasks/research/2026-07-17_competitor_consolidated-insights.md`.

The explicit ask: replace **both** Memgraph and the Valkey queue with in-process equivalents, config-selected as a
fallback rather than a wholesale replacement — accepting reduced functionality and slower queries in exchange for zero
external dependencies. The graph backend must cache to disk (not run purely in-memory), so a restart doesn't force a
full reindex.

Five parallel research agents were run against the actual codebase and the current library ecosystem before committing
to an approach:

1. **Kùzu is disqualified.** Archived by its maintainers October 2025 — no active development, no named successor. Its
   PyPI wheels stop at `cp313` on Windows/macOS: **no Python 3.14 Windows wheel exists**, which blocks
   `pip install kuzu` on this exact target (code-atlas requires Python 3.14+, dev box is Windows). It also requires
   upfront `CREATE NODE TABLE`/`CREATE REL TABLE` schema DDL, unlike Memgraph's schema-optional model — it was never
   going to be a drop-in Cypher swap; it needs its own translation/bootstrap layer regardless of the wheel problem.

2. **DuckDB + DuckPGQ was investigated as an explicit follow-up and also rejected**, for three independent reasons:
   - DuckDB is an OLAP/columnar engine. Code-atlas's write path is high-frequency, small, per-entity `MERGE` upserts
     driven by file-change events (`ASTConsumer.process_batch` → `GraphClient.upsert_file_entities`) — an OLTP row-store
     access pattern DuckDB isn't built for, unlike SQLite.
   - DuckDB's `vss` (vector) extension is explicitly documented by DuckDB itself as **not recommended for production**,
     due to WAL-recovery gaps that can cause data loss/corruption on an unexpected shutdown — directly undermining the
     "safe disk cache across restarts" goal motivating this decision.
   - `DuckPGQ` (the SQL/PGQ / Cypher-like MATCH-clause extension) is a **community extension pinned to an older DuckDB
     core release** (v1.4.4; not available on the current 1.5.x line) — an extra version-coupling risk on top of
     everything else. It's also SQL/PGQ, not Cypher, so — like Kùzu — it still needs a translation layer, not a
     pass-through.

3. **SQLite is the sound choice for both roles.** `sqlite3` is stdlib (zero new runtime dependency for the core engine).
   `sqlite-vec` (PyPI `sqlite-vec`, MIT/Apache-2.0, actively maintained) ships a **universal `py3-none-*` wheel**
   (Windows included) — because it's a loadable SQLite extension, not a compiled CPython module, it has no
   per-Python-version ABI coupling, sidestepping exactly the problem that disqualified Kùzu. SQLite's built-in FTS5
   extension provides a native `bm25()` ranking function, already compiled into Python's stdlib `sqlite3` module on all
   common distributions. Row-store transactional semantics match the write pattern already in use.

4. `GraphClient` (`src/code_atlas/graph/client.py`, ~3000 lines) turned out to be **mostly portable**: of its ~50 public
   methods, ~46 are plain openCypher with no Memgraph-specific syntax and port to direct SQL mechanically. Only 6 are
   Memgraph/MAGE-specific (`text_search`, `vector_search`, `get_vector_index_info`, `get_text_index_info`,
   `rebuild_vector_indices`, `_create_vector_indices`), plus one external call site
   (`server/analysis.py:_analyze_communities`) that calls MAGE's `leiden_community_detection.get()` directly — no Leiden
   equivalent exists in either Kùzu's or DuckDB's ecosystem either, confirming this is a capability gap for any embedded
   engine, not specific to SQLite.

5. `atlas mcp` has **zero readiness enforcement today** — every MCP tool is servable before any indexing has happened,
   confirmed via `app_lifespan` launching the daemon's catch-up index as an unawaited background task. This gap exists
   independent of which graph backend is in use, but adding a slower, cold-cache-by-default embedded backend makes it
   more likely to bite in practice, so it's addressed alongside this work rather than left as a separate follow-up.

## Decision

Add a config-selected, fully in-process fallback: **SQLite for the graph store (+ `sqlite-vec` for vector search, FTS5
for BM25) and SQLite for the event queue**, selected via `[backend]` settings (`graph`/`queue`: `"memgraph"`/`"valkey"`
| `"sqlite"` | `"auto"`), reusing code-atlas's existing Pydantic settings loader rather than building a second
configuration mechanism. `"auto"` probes the network backend at startup and falls back to SQLite if unreachable; an
explicit `"memgraph"`/`"valkey"` selection fails loudly if unreachable rather than silently falling back. The SQLite
graph store persists to a configurable on-disk directory (default `.atlas/`) so a process restart reuses the existing
cache instead of forcing a full reindex.

Kùzu and DuckDB+DuckPGQ were both evaluated and rejected for the concrete reasons in the Context section above (no
Python 3.14 Windows wheel and an unmaintained project for Kùzu; OLAP-mismatched write path, a vector extension DuckDB
itself warns against for production, and a version-pinned community extension for DuckDB+DuckPGQ).

Community detection (Leiden, via MAGE) is explicitly **out of scope** for the embedded backend. `find_communities`/
`analyze_repo(analysis="communities")` return a clear "unsupported without Memgraph+MAGE" error on the SQLite backend
rather than a silent empty result or an attempted query that fails ungracefully.

Reduced functionality and slower queries on the embedded backend are accepted by design — this is a fallback for
zero-external-dependency operation, not a parity replacement for the Memgraph+Valkey path, which remains the default and
the fully-supported configuration.

## Consequences

### Positive

- Code Atlas can run fully in-process with zero external services (no Docker, no Memgraph, no Valkey) for users who want
  to try it or run it in constrained environments.
- The graph cache survives process restarts, avoiding a full reindex — the core requirement motivating this work.
- No new required runtime dependency for the core engine (`sqlite3` is stdlib); `sqlite-vec` avoids the exact
  ABI-coupling problem that disqualified Kùzu.
- `GraphClient`'s ~46 portable methods and centralized schema bootstrap (`_apply_full_schema`, driven by `schema.py`'s
  DDL generators) give the SQLite backend a mechanical, checklist-driven porting path rather than a from-scratch
  reimplementation.

### Negative

- Two backend implementations to maintain in parallel going forward; a change to graph semantics now needs to be
  considered against both Memgraph/Cypher and SQLite/SQL code paths.
- Community detection, and potentially other MAGE-only capabilities, are permanently unavailable on the embedded
  backend, not just deferred — any future analysis that leans on a MAGE procedure inherits the same gap.
- SQLite-backed operation is expected to be slower than Memgraph for graph traversals and large-batch writes; this is
  accepted, not a bug to chase down.
- `block_ms`-style blocking reads on the SQLite queue are emulated via short polling (SQLite has no server-side blocking
  read), a minor latency/CPU tradeoff versus Valkey's native blocking `XREAD`.

### Risks

- CALLS/import/type-ref resolution parity is best-effort on the embedded backend; if the pure-Python matching logic
  proves too invasive to fully extract from Memgraph-flavored `MERGE` writes, the SQLite backend may ship a simpler
  first-pass resolver. This is accepted per the original reduced-functionality scoping, but could surprise a user who
  expects identical resolution behavior across backends.
- If a meaningful share of users end up preferring the embedded backend as their primary mode rather than a fallback,
  the "reduced functionality is fine, it's just a fallback" framing this decision rests on would need revisiting.

## Alternatives Considered

### Alternative 1: Kùzu as the embedded graph backend

- Originally proposed before this research pass, on the assumption that Kùzu's native Cypher support would make it close
  to a drop-in replacement for Memgraph.
- Rejected: archived by its maintainers as of October 2025 with no active development or named successor; no PyPI wheel
  for Python 3.14 on Windows blocks installation outright on this project's exact target; and its upfront-schema DDL
  model means it was never a true drop-in regardless of the wheel problem.

### Alternative 2: DuckDB + DuckPGQ as the embedded graph backend

- Investigated as an explicit follow-up once Kùzu was rejected, on the strength of DuckDB's broader ecosystem and
  DuckPGQ's Cypher-like `MATCH` clause support.
- Rejected: DuckDB's OLAP/columnar design doesn't match code-atlas's high-frequency small-row OLTP upsert write pattern;
  DuckDB's own documentation advises against its `vss` vector extension in production due to WAL-recovery data-loss
  risk; and DuckPGQ is a community extension pinned to an older DuckDB core release, adding version-coupling risk on top
  of still needing a SQL/PGQ translation layer rather than a pass-through.

### Alternative 3: No embedded backend — keep Memgraph/Valkey as hard requirements

- Considered doing nothing, on the grounds that Memgraph+Valkey via `docker compose up -d` is already a single command.
- Rejected: the ask was specifically to lower the barrier to zero external services, and a hard infrastructure
  requirement remains a real adoption barrier independent of how simple the `docker compose` command is — some
  environments can't run Docker at all.

## References

- `.tasks/research/2026-07-17_competitor_consolidated-insights.md` — the P2 "Embedded/serverless backend option" item
  this work scopes down from.
- [ADR-0001: Use Memgraph as the Graph Database](./0001-memgraph-as-database.md) — the default backend this option falls
  back from; still the fully-supported configuration.
- [ADR-0013: MCP Tool Taxonomy](./0013-mcp-tool-taxonomy.md) —
  `find_communities`/`analyze_repo(analysis="communities")`, the tools that gain the "unsupported on this backend"
  guard.
- `src/code_atlas/graph/client.py`, `src/code_atlas/schema.py`, `src/code_atlas/events.py` — the modules the embedded
  backend must match the method surface and schema of.
