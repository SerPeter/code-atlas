# ADR-0032: A uid Must Identify Exactly One Definition

## Status

Accepted. Extends [ADR-0031](./0031-anonymous-callables-attribute-upward.md), which decided _whether_ a callable becomes
an entity. This decides what happens when two callables that both qualify would claim the same uid. ATL-107.

## Context

ATL-096 added a `duplicate_uids` ceiling to the extraction-coverage floors, because both coverage ratios can improve
while the graph gets worse. A uid is the graph's identity: two definitions emitting the same one upsert into a single
node carrying an arbitrary winner's source and the union of both edge sets. That is a confident wrong answer, and worse
than the silence of a missing entity.

Measured across the vendored corpus once all eight languages landed, the collisions are **three unrelated problems
wearing one symptom**:

| language                | count | cause                                                    | category |
| ----------------------- | ----: | -------------------------------------------------------- | -------- |
| C++                     |    49 | genuine overloads — 9 `basic_scan_arg` ctors, 8 `read`   | A        |
| Ruby                    |     5 | `def self.settings` beside `def settings`                | B        |
| Ruby                    |     3 | `def call` inside `superclass.class_eval do ... end`     | C        |
| TypeScript              |    10 | `const customFetch` inside sibling `test(...)` callbacks | C        |
| Python                  |    11 | `@t.overload` declarations — **must keep merging**       | —        |
| Go, Rust, PHP, Java, C# |     0 |                                                          |          |

Java and C# are already at zero because ATL-096 gave them an overload suffix locally, computed over a `#if`-flattened
member view. That mechanism is sound and this ADR generalises it rather than replacing it.

**Python is the constraint, not an exception.** Its eleven are four `@t.overload` declarations ahead of each
implementation of `command` and `group`, plus a platform-conditional `_get_argv_encoding`. Those really are one
function, and one node is the right answer. Any rule shaped like "a repeated name gets a discriminator" breaks the
reference floor every other language is measured against.

What separates them is not the repetition. It is that **Python has no function overloading at all** — a second `def foo`
in one scope replaces the first, so merging is always correct. C++ permits two definitions of one name in one scope;
Python does not.

## Decision

**A callable becomes an entity only if its qualified name identifies exactly one definition.** Where that fails, one of
two things applies, chosen by _why_ it failed.

### Category A/B — the language permits two definitions of one name in one scope: discriminate

Applied **only in languages that actually permit it**, which is what preserves Python's merging without a special case
for `@overload`:

- **Overload sets** (C++, Java, C#) take a suffix of normalised parameter types, `(<types>)`, and only when the name is
  declared two or more times in that scope. A name declared once keeps its plain uid, so the change churns ambiguous
  names and nothing else. This is `jvm.py`'s existing `_overload_suffix`, generalised to C++ and to namespace and
  translation-unit scope, not just class bodies.
- **Ruby singleton methods** take a `self` segment: `Sinatra::Base.settings` stays the instance method and the class
  method becomes `Sinatra.Base.self.settings`. `self` is a Ruby keyword, so no module or class can ever be named it and
  the segment cannot collide with a real scope. It mirrors the source (`def self.x`), keeps uids dot-separated and
  alphanumeric, and changes the rarer of the two.

### Category C — an enclosing scope is anonymous: emit no entity

A definition inside an anonymous callback has no qualifiable path. `const customFetch` inside
`test('...', async t => {...})` cannot be referred to from outside, and eight sibling callbacks each declaring one
produce eight claims on `http-error.customFetch`. ADR-0031's own test already rejects it: a name a developer could use
to refer to it means referable from outside.

So an entity requires **every enclosing scope to be named**. A nested function inside a _named_ function is still an
entity — Python's `module.outer.inner` is unaffected — because that path is qualifiable. Only an anonymous link in the
chain disqualifies.

This is the rule PHP already applies to a closure bound to a local, arrived at from the other direction: a local binding
dies with the call, and rebinding proves the name does not identify a function.

Category C emits no entity and is therefore **not a uid change**: the body is still walked and its calls still attribute
to the nearest named enclosing scope.

## Consequences

- **Churn is bounded to what was ambiguous.** Only overloaded names and Ruby singleton methods get a new uid. Every
  unambiguous callable in every language keeps the uid it has.
- **Schema v12 with a hash-clear migration, not a purge.** `_classify_file` already deletes uids present in the graph
  but absent from a re-parse, so the stale node disappears on its own once the file is re-read. Clearing file and git
  hashes forces that re-read. Verified by reading the classification path rather than assumed — the first draft of this
  ADR called for an explicit delete, which would have been wrong.
- **`find_dead_code` and `blast_radius` output moves**, because one node becomes several and each carries only its own
  edges. That is the point, and it needs the same re-baseline ATL-096 took.
- **Python, Go, Rust and PHP are untouched.** They have no overloading and no singleton/instance split, so the rule has
  nothing to apply.
- **Category C reduces entity counts.** TypeScript loses the callback-local consts; Ruby loses the `class_eval do`
  methods. Both were already unreachable by name, and `calls` must not move — if it does, something was walked away
  rather than declined, which is the check to run.
- **A signature suffix is not a mangled name.** Parameter types are normalised by stripping whitespace and namespace
  qualifiers, so `file::dup2(int,error_code&)` reads as source rather than as a linker symbol. It is stable across edits
  above the definition, which a positional ordinal would not be — ADR-0031 bans those for the same reason.
- **Two same-named overloads in different files still merge** when their module qn coincides — a C++ member declared in
  `foo.h` and defined out-of-line in `foo.cc` beside it. That is arguably correct, since they are one function, but the
  surviving node may carry the declaration's empty body. Left as-is and recorded.

## References

- ATL-107, and ATL-096 which surfaced it
- [ADR-0031](./0031-anonymous-callables-attribute-upward.md) — whether a callable becomes an entity; bans positional
  names in uids
- `jvm.py` `_overload_suffix`, `_overloaded_callable_names` — the mechanism being generalised
- `tests/fixtures/langcov/*/floor.json` — `max_duplicate_uids`, which should reach 0 everywhere except Python's 11
