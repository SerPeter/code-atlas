# Extraction-coverage corpus

Trimmed excerpts of real open-source projects, used by
`tests/unit/parsing/test_extraction_coverage.py` to hold each language walker to a recorded floor.

## Why real code

These files exist because synthetic snippets lied. Every language passed its hand-written unit tests while
TypeScript was dropping nine calls in ten on a real codebase (ATL-096). The shapes that break a walker —
`#ifdef`-guarded definitions, namespace-nested partial classes, DSL blocks, callback pyramids — are the ones
nobody writes when authoring a test fixture.

## Layout

One directory per language, discovered automatically. Adding a language means adding a directory; no shared
table to edit, so language work can proceed on separate branches without colliding.

```
<language>/
  floor.json     the recorded floor and where the code came from
  *.<ext>        vendored sources, byte-identical to upstream
```

`floor.json` requires `lang`, `named_funcs`, `calls`, `source_repo`, `source_commit`, `license`, and
`rationale`. The rationale is not decoration — a floor below 1.0 is a claim that the remainder is not worth
extracting, and that claim needs to be readable by whoever next sees the number and assumes it is a bug.

## The two ratios

| ratio         | measures                                                      |
| ------------- | ------------------------------------------------------------- |
| `named_funcs` | function forms that carry a name and became Callable entities |
| `calls`       | call nodes in the AST that became CALLS relationships         |

Anonymous forms — arrows, lambdas, closures, blocks, func literals — are excluded from `named_funcs` on
purpose: they get no entity, and their calls attribute to the nearest named enclosing scope (ADR-0031). They
are still visible in `calls`, which is what catches a walker that skips a callback body.

## Vendored code is not ours

Each directory's sources keep their upstream license, recorded in `floor.json` and reproduced in
`LICENSE-<name>` where the upstream project ships a license file. They are excluded from ruff, ruff-format,
ty, prettier, the whitespace hooks, and pytest collection — reformatting them would silently change what the
walkers are being measured against.

## Running it by hand

The floors are a ratchet, not a diagnostic. When one trips, the harness that produced it says where:

```bash
uv run --no-sync python -m tests.support.langcov tests/fixtures/langcov/<language> <language>
uv run --no-sync python -m tests.support.langcov /path/to/full/checkout <language>
uv run --no-sync python -m tests.support.langcov /path/to/checkout <language> --census
```

`--census` dumps a raw node-type histogram, which is how the node types in `tests/support/langcov.py` were
established — they were verified against real code rather than read out of grammar documentation.
