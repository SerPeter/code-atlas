---
title: "ADR-0053: Declaring a backend is selecting it"
tags: [adr, settings, backends, breaking]
kind: decision
---

# ADR-0053: Declaring a backend is selecting it

## Status

Accepted (2026-09-11) — amends [ADR-0015](./0015-embedded-backend-option.md), replacing its
`"memgraph" | "sqlite" | "auto"` selector with the presence of a configuration section.

## Context

ADR-0015 gave each axis a string selector — `backend.graph = "memgraph" | "sqlite" | "auto"`, defaulting to `"auto"` —
while the connection settings lived in separate top-level `[memgraph]` and `[redis]` sections. Configuring a backend and
choosing one were two independent statements, and nothing checked that they agreed.

They did not agree, on the machine this project is developed on. `atlas.toml` carried a fully specified `[memgraph]` —
host, port, and deliberately raised `write_timeout_s` / `query_timeout_s` with comments explaining the monorepo
contention they were tuned for — and **no `[backend]` section at all**. So both axes sat at `"auto"`, and every one of
those carefully chosen values was parsed and then not used.

`"auto"` probes, and on failure logs one `WARNING` and substitutes an **empty** embedded graph. The next index rebuilds
the entire graph into `.atlas/` and, because the graph is the embedding dedup layer
([ADR-0036](./0036-the-graph-is-the-embedding-dedup-layer.md)), re-buys every vector from the provider. Found on disk:
214 MB of `graph.sqlite3` written while Memgraph was running and reachable. The trigger only has to be momentary, and
this project has a known Memgraph GC segfault under full-index churn.

The failure is not the fallback. It is that a one-line warning preceded an expensive, silent divergence, and that the
configuration which should have prevented it was being read all along.

## Decision

**A backend is selected by being configured. There is no separate selector.**

```toml
[backend.graph.memgraph]   # declaring it selects it
host = "localhost"
port = 7687

[backend.queue.valkey]
host = "localhost"
```

| state on an axis | meaning                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| no section       | `auto` — probe the network backend, fall back to embedded SQLite. **The only path that may fall back** |
| one section      | that backend; unreachable is an error, and nothing is rebuilt into `.atlas/`                           |
| two sections     | a configuration error, not a precedence rule                                                           |

ADR-0015's zero-config promise is untouched: someone with no Docker and no `[backend]` section still gets a working
embedded index. What changed is that the fallback is now reserved for the case where nothing was asked for, rather than
being the default answer to a question the user had already answered.

Two backends on one axis is rejected **per axis**, so Memgraph with a SQLite queue remains legal — that is a real
configuration, not a mistake.

### `false` un-declares

`atlas.local.toml` merges **per key** over `atlas.toml`. So a machine that wants the embedded backend cannot simply
declare `[backend.graph.sqlite]`: the committed `[backend.graph.memgraph]` is still there, both are set, and that is the
two-backends error. Declaring-is-selecting would otherwise make a committed choice impossible to override locally, which
is a regression against the layering ADR-0015 relies on.

TOML has no null, so `false` is how a machine opts out without editing a shared file:

```toml
# atlas.local.toml
[backend.graph]
memgraph = false
sqlite = {}
```

`true` is rejected rather than read as "with defaults" — it would be a second way to spell a declaration, and this ADR
exists to remove the second way.

### What deliberately did not move

`settings.memgraph` and `settings.redis` remain as read accessors, returning the configured section or defaults. All
eleven readers in `src/` are unchanged. `"auto"` still has to probe an address for a backend it may not end up using, so
"no section" cannot mean "no settings" — only the file shape and the selection moved.

`sqlite_data_dir` stays on `[backend]` rather than moving under each `[backend.*.sqlite]`: the graph, the queue and the
rate limiter all write into one directory, and a per-axis path would let them drift apart for no reason anybody wants.

## Consequences

- **Breaking for every existing config.** `StrictSection` would reject a top-level `[memgraph]` anyway, which is the
  right failure — loud, never silent. But "extra inputs are not permitted" names the key and not the rewrite, so a
  `mode="before"` validator intercepts and prints the section back in its new shape, including the environment-variable
  spellings (`ATLAS_MEMGRAPH__HOST` → `ATLAS_BACKEND__GRAPH__MEMGRAPH__HOST`). A user with one config per project in a
  monorepo should not have to derive the mapping six times.
- **The strictness guard had to learn to recurse.** `TestEverySectionRejectsUnknownKeys` discovers sections by walking
  `AtlasSettings.model_fields`. Nesting `MemgraphSettings` would have silently dropped it from that guard — it would no
  longer be checked for rejecting typos — and the "at least 15 sections" assertion would have gone quietly to 14. The
  walk now recurses and unwraps `X | None` unions.
- A config that declares nothing behaves exactly as before. This is only a behaviour change for a config that declares
  something, which is the population that was being ignored.
- The embedded backend remains a fallback rather than a peer, so ADR-0015's revisit condition ("if a meaningful share of
  users prefer the embedded backend as their primary mode") still stands and is now easier to observe: preferring it
  means declaring it.
