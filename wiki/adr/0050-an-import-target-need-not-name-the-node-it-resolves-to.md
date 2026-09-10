---
title: "ADR-0050: An import target need not name the node it resolves to"
tags: [adr, graph, resolution, salesforce]
kind: decision
---

# ADR-0050: An import target need not name the node it resolves to

## Status

Accepted (2026-09-07)

## Context

`resolve_imports` matched an import target against a node's `qualified_name` exactly, and minted an `ext/` stub on a
miss. That is correct for every language whose import names are the names of the things they import. Salesforce is not
such a language, in two distinct ways, and the epic that indexed it (ATL-144) hit both.

**A `<c:foo>` reference cannot say which kind of component it names.** Salesforce shares one `c` namespace between Aura
and LWC and forbids the two from holding the same name — "A custom Lightning web component and a custom Aura component
in the same namespace can't have the same name" (LWC Dev Guide, Component Namespaces). So the reference is unambiguous
_in an org_ and completely ambiguous _in a file_, and a parser sees one file. The same is true of a FlexiPage's
`componentName`. The two kinds are minted under different namespaces (`aura.foo`, `lwc.foo`), so nothing the parser can
write names the node.

ATL-182 shipped this by emitting **both** targets. That resolved the true edge and left an `ext/` stub asserting a
component of the other kind that was never referenced — 38 of 132 Aura `<c:X>` references in a measured corpus point at
LWC bundles, so neither side is rare enough to simply pick.

**Salesforce API names are case-insensitive.** `FROM ACCOUNT` in Apex and `Account.object-meta.xml` are the same object;
measured, 19 of 121 resolvable SOQL `FROM` names in one corpus differ from their object file only by case, across 76
sites, including in Salesforce's own apex-recipes.

## Decision

### 1. A target may name a _class_ of nodes, and the resolver widens it

`cmp.<Name>` is minted by nothing. It is a parser saying "a custom component named X, whose kind this file cannot
determine", and `resolve_imports` tries `aura.<Name>` then `lwc.<Name>`. First match wins: a repo holding both cannot
deploy.

This is the general form worth naming — **the parser writes what the file says, and the resolver knows what the platform
permits.** Widening at the resolver keeps the ambiguity in the one place that can see the whole graph, instead of
forcing every reference site to guess.

A `cmp.` target matching neither still becomes a stub, because a managed-package or platform component genuinely is
external. Only the false twin was the problem.

### 2. Case folding is gated on the node, by namespace

A second index, keyed on the lowercased `qualified_name`, consulted only when the exact match misses. A node enters it
only if its first dotted segment is one of the 23 the Salesforce parsers coin.

Gating on the **node** side rather than the target side is the stronger form at the same cost: in a repo with no
Salesforce metadata the map is empty and the feature costs one `partition()` per node. Six of the namespaces (`app`,
`page`, `component`, `layout`, `tab`, `profile`) are also plausible directory names, but a path-derived qualified name
is rooted at a top-level directory (`src.`, `force-app.`), so a bare one never appears first — measured across 77,540
entities in three non-Salesforce repos, zero entered the map.

**Only Salesforce works this way.** Python, TypeScript, Go and Rust imports are case-sensitive, and folding those would
mint edges the language itself rejects.

### 3. Normalising belongs in the resolver, never in a parser

`salesforce.py`'s docstring said fixing case-insensitivity "has to be fixed in all three at once" (`salesforce.py`,
`apex.py`, `typescript.py`). That is right about _normalising_ and wrong about where the fix goes.
`uid = f"{project_name}:{qualified_name}"`, so normalising in a parser would rewrite every SObject, Apex class and
custom label uid — minting new nodes, orphaning the old ones, and changing what every `get_context` uid in an agent's
scrollback points at. A resolver fold changes no stored string.

### 4. An ambiguous fold refuses; it never picks

Two objects differing only by case cannot both exist in one Salesforce org, but two _nodes_ can — a monorepo, a managed
package, a parse artefact. A key that two nodes fold to maps to `None` and the target falls through to a stub.

This is ADR-0032's "a uid must identify exactly one definition" applied to what an edge _points at_. It is not
hypothetical: six such collisions inside `apex.` were measured in one production org, including `apex.Class` against
`apex.class`.

### 5. Both helpers are shared, against this repo's usual preference for mirrors

`build_case_folded_map` and `resolve_component_alias` live in `graph/client.py` and are imported by
`backends/sqlite_graph.py`. The two `resolve_imports` implementations are otherwise hand-written mirrors, and the house
rule is that a small duplication beats an abstraction.

The rule is suspended here because the conformance ledger classifies resolution passes as **not output-compared**, so a
divergence between the two would resolve an edge on one backend and stub it on the other, with nothing failing. ATL-171
reached main through exactly that gap.

## Consequences

- The matching ladder is now: exact → `cmp.` alias (which itself falls back to the fold) → case fold → the Python
  dotted-prefix walk. Exact always wins.
- **The two mechanisms compose, and had to be made to.** `cmp.` can never be a fold key, because nothing mints a `cmp.`
  qualified*name — so `<c:MyPanel>` against a bundle folder named `myPanel` routed \_around* the fold that should have
  caught it. Two correct fixes composed into a gap, closed by passing the fold map into the alias resolver.
- A folded match still enters `inexact`, so it stays replayable: it is a guess a later batch could improve on, and the
  docstring's promise that every rel whose exact name matched nothing comes back stays literally true.
- `EXTRACTION_EPOCH` 9 → 10, because the emitted targets changed.
- Not done: `apex.` is in the fold set, but no measurement yet shows how many Apex-side references it recovers. The
  19-of-121 figure is SOQL `FROM` names only.
