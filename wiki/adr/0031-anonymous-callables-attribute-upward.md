# ADR-0031: Anonymous Callables Get No Entity and Attribute Their Calls Upward

## Status

Accepted. Establishes the rule every language walker follows; ATL-096 applies it across the eight languages that did
not.

## Context

A walker meets four kinds of function form:

1. **Named definitions** — `def`, `function foo`, `fn`, `method_declaration`. These are entities.
2. **Name-bound anonymous forms** — `const foo = () => {}`, an object-literal method, `$x = function () {}`. Anonymous
   in the grammar, but the codebase refers to them by the binding.
3. **Truly anonymous forms** — a callback argument, a Ruby block, a Go `func_literal`, a C++ lambda.
4. **Signatures with no body** — TypeScript ambient declarations, trait method declarations.

Measured over eight real repositories (2026-08-05), forms 2 and 3 dominate real code and were the largest source of
missing graph edges — but almost none of the loss was the missing _entity_. It was the calls inside the body:

| language   | call nodes inside a function that produced no entity |
| ---------- | ---------------------------------------------------: |
| TypeScript |                                                83.5% |
| Ruby       |                                                77.3% |
| C#         |                                                42.2% |
| C++        |                                                33.0% |
| PHP        |                                                26.5% |
| Rust       |                                                17.3% |
| Go         |                                                10.0% |
| Java       |                                                 7.9% |

Every walker except Python's stops recursing at an anonymous form —
`# Recurse but don't descend into nested function literals` appears near-verbatim in `rust.py`, `go.py`, `cpp.py`,
`ruby.py` and `php.py`. The calls in those bodies are not attributed to something wrong; they are attributed to nobody,
and vanish.

Two other options were considered for the anonymous forms:

- **Synthesize a positional name** (`parent.<lambda@L42>`). Rejected: a uid must survive re-indexing, and a line number
  does not. Every edit above the lambda churns the graph.
- **Emit an entity for every anonymous form.** Rejected: it inflates the node count with callbacks nobody will ever look
  up by name, and `find_dead_code` reports each one as dead — the exact failure the Python nested-function work had to
  add an exclusion for (ATL-096, commit 89a91e1).

## Decision

**A callable becomes an entity if and only if it has a name that a developer could use to refer to it.** Categories 1
and 2 are entities. Category 3 is not. Category 4 is not, and is excluded from coverage measurement entirely.

**A call is always attributed to the nearest enclosing named scope.** Walking into an anonymous body is mandatory, not
optional. Where there is no enclosing named callable, the call belongs to the module — the same rule that made
import-time Python calls visible.

The `from_qn` of a call inside a callback is therefore the named function that lexically contains the callback, or the
module. This is not an approximation standing in for something better: a callback's calls genuinely are made by whoever
defines it, because that is the code that decided to pass it.

## Consequences

- Calls inside callbacks stop vanishing. On the corpus this is the single largest source of missing edges in six of
  eight languages.
- Node counts stay stable. No new uid scheme, no churn on re-index, no new `find_dead_code` false positives — the change
  is edges-only for category 3.
- A name-bound arrow or object-literal method **does** become an entity, so TypeScript's existing behaviour for
  top-level `const foo = () => {}` is preserved rather than reverted. The distinction between categories 2 and 3 is
  whether a binding name is recoverable, not whether the grammar node is called "anonymous".
- Two callbacks passed to the same function from the same caller produce two edges to the same target, which
  `_combine_call_edge_facts` merges with a summed `site_count` (ADR-0028). Attribution upward therefore makes
  `site_count` meaningful for callback-heavy code rather than double-counting it.
- **A deeply nested callback loses its intermediate structure.** A call five callbacks deep attributes to the same named
  function as one at the top of the body. That is a real loss of precision and it is accepted: the alternative is
  category-3 entities, whose cost is above.
- Coverage is measured with two separate ratios so this decision cannot hide a walker bug. `named_funcs` covers
  categories 1 and 2 only, and should approach 1.0; `calls` covers every call node regardless of what encloses it, and
  is what catches a skipped callback body. See `tests/fixtures/langcov/README.md`.

## References

- ATL-096
- [ADR-0028](./0028-every-resolved-edge-states-its-evidence.md) — `site_count` merging
- `tests/support/langcov.py` — the measurement; `named` vs `anon` per language encodes this decision
- Python is the reference implementation: 1.000 named-function capture, 0.966 call coverage, 0 calls lost inside an
  uncaptured function.
