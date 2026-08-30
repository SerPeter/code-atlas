# ADR-0020: Env-Var and Referenced-File Nodes, and When a Node Is Global

## Status

Accepted — extends the referenced-only node pattern established by `ExternalPackage`/`ExternalSymbol`

## Date

2026-07-31

## Context

The graph modelled what code _contains_ but not the runtime surface it _reaches for_: the environment variables it reads
and the data files it opens. Both are real coupling — "what breaks if `DATABASE_URL` changes" and "who reads this
fixture" are ordinary questions the graph could not answer.

There was already a precedent for nodes that exist only because something referenced them:
`ExternalPackage`/`ExternalSymbol`, keyed `{project}:ext/{name}`, MERGEd during post-batch import resolution and never
produced by a parser. Env vars and referenced files share exactly that lifecycle.

The open question was **scope**: should such a node be shared across projects or project-scoped?

## Decision

### The rule

**Globalize only identifiers that are globally unique by nature and carry no per-project attributes.** If a node needs a
property whose value differs per project, it cannot be shared — the projects will fight over that property, last writer
wins.

Applying it gives a deliberate asymmetry between the two new kinds:

| kind           | scope              | uid                    | why                                                                                                             |
| -------------- | ------------------ | ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `EnvVar`       | **global**         | `env/{NAME}`           | `DATABASE_URL` means the same thing everywhere; no version, no path, no per-project attribute                   |
| `ResourceFile` | **project-scoped** | `{project}:res/{path}` | a path is only meaningful relative to a project root — `data/fixtures.json` in two repos is two different files |

The global node makes "every callsite of `DATABASE_URL` across every repo" a single node rather than a name-join. The
same rule kept `ExternalPackage` project-scoped, because it carried a per-project `version`. That reason is gone as of
schema v18 (ATL-146): the version moved onto a `Project -[DEPENDS_ON]-> ExternalPackage` edge, which is per-project even
when the node is not. Globalizing the node is now unblocked — see ATL-088 — but not done here.

### A sentinel, because null is impossible

`project_name` carries an existence constraint on every entity label, so a genuinely global node cannot omit it.
`GLOBAL_PROJECT = "_global"` is that sentinel. Verified live on `memgraph-mage:3.7.2`: an entity node without
`project_name` is **rejected at COMMIT, not at statement execution** — the offending write appears to succeed and the
transaction dies afterwards.

### Garbage collection by reference counting

A reference-counted node should disappear when nothing points at it. That is correct here for a non-obvious reason:
`_recreate_batch_relationships` already deletes every outgoing relationship from a file's entities before recreating
them on reparse, so when the last `os.getenv("X")` leaves the source, the last incoming edge leaves with it.
`gc_orphaned_reference_nodes` then sweeps nodes with zero incoming edges.

`_REFERENCE_COUNTED_LABELS` is deliberately narrower than `_EXTERNAL_LABELS`: `ExternalPackage` receives a structural
`CONTAINS` edge from its package and, since v18, a structural incoming `DEPENDS_ON` from its project, so its
incoming-edge count is not a reference count. **Invariant, stated in the code:** never give a reference-counted label a
structural incoming edge — it would make every node permanently referenced and silently disable the sweep.

### Names only, never values

`os.getenv("API_KEY", "sk-live-…")` puts a real secret in the default argument. An indexed entity is an **embedded**
one, so a captured default would leave the machine for the embedding API. Capture is an explicit allowlist at the write
path, with tests asserting the secret appears in no node property, no edge property, and no SQLite FTS document. A
referenced sensitive file gets a path-only node and is never opened.

Both labels are **text-searchable** (an agent asking "where is `DATABASE_URL` read?" needs the keyword hit) and neither
is **embeddable** — by the names-only invariant there is nothing to embed but a bare identifier.

`SCHEMA_VERSION` goes to **7**.

## Consequences

### Positive

- Runtime configuration coupling is queryable, and cross-project for env vars specifically.
- The GC keeps the graph honest without a bespoke lifecycle: reference counting falls out of the existing
  relationship-recreation behaviour.
- Joining `_EXTERNAL_LABELS` buys uid uniqueness, the existence constraints and the property/composite indices
  automatically.

### Negative

- Another schema bump, another forced reindex, one release after v6.
- Extraction is **Python only** this pass. Other languages produce no env-var or file references yet, so absence of a
  node does not mean absence of a reference.
- In a monorepo, two sub-projects referencing one genuinely shared file still produce two `ResourceFile` nodes, because
  the path is resolved per project root.

### Risks

- The sentinel is a reserved value. Anything filtering by project must treat `_global` as a member of every project, and
  anything refusing to touch non-test data must allowlist it — the integration wipe guard needed exactly that, or every
  test run would abort with a message that reads like a production incident.
- File-reference extraction is deliberately conservative (plain string literals only, no f-strings, no concatenation). A
  false reference mints a node for a path that does not exist, which is worse than a missing one.
- Reference-counted GC is only as correct as the relationship-recreation invariant it rests on. If a future change makes
  relationship deletion partial or lazy, nodes will be swept while still referenced.

## Alternatives Considered

### Make env vars project-scoped like everything else

Rejected on the user's argument: the whole value of an env-var node is that every callsite converges on it, including
across project boundaries. Project-scoping turns "who reads `DATABASE_URL`" into a name-join over N nodes, which is what
the graph exists to avoid.

### A truly project-less node (no `project_name`)

Not possible — the existence constraint rejects it at commit. Measured, not assumed.

### Lump referenced files in with env vars as global

Rejected. A path has no meaning outside its project root, so a global `res/data/fixtures.json` would conflate unrelated
files across repos. The two kinds look similar and are not.

### Reference-count `ExternalPackage` too

Rejected: it has a structural `CONTAINS` edge and, since v18, a structural incoming `DEPENDS_ON` carrying the manifest
version, so zero-incoming-edges is not a reference count for it. Sweeping it would delete live nodes.

## References

- `src/code_atlas/schema.py` — `ENV_VAR`, `RESOURCE_FILE`, `GLOBAL_PROJECT`, `_REFERENCE_COUNTED_LABELS`
- `src/code_atlas/graph/client.py` — `resolve_config_refs`, `gc_orphaned_reference_nodes`,
  `_migrate_v7_clear_freshness_markers`
- [ADR-0018: Non-code file parsing](./0018-non-code-file-parsing.md) — the secret deny-list this reinforces
- ATL-146 — moved `version` onto `Project -[DEPENDS_ON]-> ExternalPackage` (schema v18)
- ATL-088 (backlog) — globalizing `ExternalPackage`; was blocked on ATL-146, no longer is
