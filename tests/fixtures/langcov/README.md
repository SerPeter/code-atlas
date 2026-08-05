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

### Do not compare these numbers across languages

`calls` measures **walker reach, not edge usefulness**, and the languages are not on the same scale:

- Go spells type conversions and builtins as `call_expression`, so `len`, `append` and `make` count as
  covered and emit edges that resolve to nothing. Its 0.99 is easier than it looks.
- Rust counts `macro_invocation`, 9.6% of its denominator, of which only ~30% name a macro declared in the
  repo. Excluding macros its number would read 0.903 rather than 0.999.
- Ruby's denominator excludes bare-identifier implicit calls that the walker nonetheless emits, so its ratio
  can legitimately exceed 1.0.
- C++ has a genuine ceiling below the others because tree-sitter has no preprocessor.

A floor is a per-language ratchet against that language's own past. It is not a score, and a lower number is
not a worse walker.

### `named_funcs` rewards emitting entities, including wrong ones

It counts nodes captured over nodes present. It cannot see whether the entity produced was _correct_, and
three languages hit that from three directions during ATL-096:

- **C#** — 50 of 62 `local_function_statement` nodes in Newtonsoft.Json are tree-sitter recovering
  `else if (x) { }` at the head of a `#if` branch as a local function named `if`. Capturing them scores
  0.993; rejecting them scores 0.988. **The higher number was bought with Callables named `if`.**
- **C++** — suppressing 40 macro invocations that no metric could know the preprocessor deletes
  (`FMT_CATCH(...) { }`) moved `named_funcs` **down**, 0.653 → 0.641. Removing a confidently-wrong node
  reads as a coverage regression.
- **TypeScript** — declining an entity for an unbound object-literal method took it from 1.000 to 0.254,
  while `calls` did not move by a thousandth. The decline was correct.

So do not read `named_funcs` alone, and do not maximise it. The three numbers are a set:

| number           | pressure                                            |
| ---------------- | --------------------------------------------------- |
| `named_funcs`    | rewards capturing more                              |
| `duplicate_uids` | punishes capturing things that cannot be told apart |
| `calls`          | form-agnostic, and unmoved by any naming decision   |

`calls` is the honest one when a change is about naming rather than reach: if it holds steady while entity
counts drop, entities were declined, not lost. That is the argument TypeScript and PHP both used, and it is
checkable rather than assertable.

### Known limit: name-bound anonymous forms are not asserted

ADR-0031 category 2 — `var handler = func() {...}`, `$loader = static function () {}` — is an entity, but the
harness places the whole grammar form in `anon`, so `named_funcs` never asserts its capture. Go shows 10 of
207 `func_literal` nodes captured: correct behaviour, invisible to the floor. Detecting bindability needs
per-language logic in the measurement, which would make the harness a second parser to keep in sync. `calls`
still catches the part that matters, since an unwalked body loses its calls either way.

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
