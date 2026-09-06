---
title: "ADR-0048: An extension can be claimed by the format inside it"
tags: [adr, parsing, extensibility]
kind: decision
---

# ADR-0048: An extension can be claimed by the format inside it

## Status

Accepted (2026-09-06)

## Context

`.json` routes to one handler, which parses any JSON structurally into `config_file` / `config_section` /
`config_setting`. That is the right floor and the wrong ceiling. A Power BI PBIR `visual.json` announces what it is in
its own first bytes —

```json
{ "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.11.0" }
```

— and so do a great many application formats. Parsed structurally, a report visual becomes a tree of key/value leaves
addressed by GUID path; parsed by something that knows the format, it becomes a named visual with edges to the columns
and measures it displays.

The mechanism to fix this already existed and had exactly one user. `LanguageConfig.ambiguous_extensions` plus
`resolve_dialect` route a suffix by content, and `.h` (C versus C++) was the only caller. But it hard-codes a single
sniff into the C registration, so a second dialect means editing the router.

XML answers the same question a **second** way: `config.py`'s `_parse_xml` offers every document to
`salesforce.parse_salesforce_metadata` first and parses structurally only when that declines. Two mechanisms for one
question is one too many, and neither generalises — one hard-codes a sniff, the other hard-codes a callee.

## Decision

**1. `register_dialect(extension, language_name, sniff)`.**

A registry in `parsing/ast.py`, keyed by extension. The extension's owner opts in by listing it in
`ambiguous_extensions` and taking its `resolve_dialect` from `dialect_resolver(extension, default)`; `config.py` does
that for `.json`. The resolver reads the registry at call time, so a dialect registered by a module imported later than
the extension's owner still wins.

Keyed by extension rather than written for JSON because the same question is already being asked of Salesforce `.xml`
(ATL-144). When that surface is revisited, the hand-off inside `_parse_xml` moves here and there is one mechanism again.

**2. The generic handler is the floor, and it is always reachable.**

Nothing matched → the registered default. An unknown dialect name → the registered default
(`_LANGUAGES.get(name, config)`). A sniff that raises → logged once, treated as no-match, and the next sniff still runs.
A file no dialect claims produces byte-for-byte what it produced before the route existed, and that is a test, not a
promise.

The failure being designed against is not loud. A route that swallowed unmatched files would quietly replace every
config entity in the graph, and nothing would error.

**3. First registered match wins, and the order is stated.**

Registration order is `_BUILTIN_LANGUAGE_MODULES` in `languages/__init__.py`. "First match wins" is only a rule if the
order is predictable; two overlapping sniffs is the normal case, not the pathological one.

**4. A sniff sees 4 KiB and never parses.**

It runs before any grammar does, on every file with that suffix in the repo. A format that cannot identify itself in its
first 4 KiB is not identifying itself; it is being parsed.

**5. `get_language_for_file(..., resolve_content=False)`, and `FileScope.scan` uses it.**

The lookup reads the file from disk when the caller passes no source, because the answer must not depend on who is
asking. Exactly one caller on the hot path passed none: the scan gate, asking whether a file is indexable **at all** — a
question every dialect of a suffix answers identically.

This was already costing a full read of every `.h` in the repo, invisibly, because falling back to the default was also
the correct answer. Adding `.json` to the ambiguous set would have made it a read of every JSON file too. So the
resolution-free path is part of this change rather than a follow-up: without it the mechanism makes scanning slower for
every user, whether or not they own a single file any dialect claims.

## Consequences

- The registry ships with **no built-in dialects**. `.json` behaves exactly as it did; the first consumer is the PBIR
  parser (ATL-170), and ATL-144 is the second.
- One more way to be wrong: a dialect that claims a file and then declines it gets an _empty_ `ParsedFile`, not a
  fallback to the generic handler. Claiming is a commitment — sniff conservatively.
- `resolve_dialect` remains the lower-level hook. A language with one fixed rule (`.h`) has no reason to move to the
  registry, and `cpp.py` has not.
- The XML/Salesforce hand-off still exists. It is not migrated here, deliberately — ATL-144 owns that surface — but no
  _new_ handler may copy it.
