---
title: "ADR-0043: Benchmarks count work, and account for waiting separately"
tags: [adr, performance, testing, observability]
kind: decision
---

# ADR-0043: Benchmarks count work, and account for waiting separately

## Status

Accepted (2026-09-03)

## Context

`tests/bench/` existed and did not catch four regressions found in production on 2026-08-30/31. That is worse than
having no suite, because it implies coverage that is not there. Two structural causes sat behind all four:

- **Synthetic corpora.** The generator emitted Python only, in one shape, so no `DocSection` ever existed and the
  doc-link path never ran; no entity approached a model's input cap, so chunking was unreachable; `EmbedClient` was an
  `AsyncMock` returning one vector for any batch, so batching, the rate limiter and retry were all bypassed.
- **Almost nothing asserted.** One benchmark had an assertion, with a 900s budget that documents itself as a complexity
  tripwire rather than a regression detector. The rest printed JSON and returned.

Building the replacement surfaced a third cause, which is the one this ADR is mostly about: **a wall-clock number blends
things that mean different things when they move.** On the very first real run of the new harness, a 10-file corpus took
14.0s of which 4.1s was accounted work and **9.9s was deliberate waiting** — the drain settle floor, batch windows and
poll intervals. A suite comparing that 14.0s between runs is comparing mostly pacing.

## Decision

### 1. Prefer counters to clocks, in a fixed order

| Tier | Metric                                                                                | Reproducible?           | Use                                                            |
| ---- | ------------------------------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------- |
| 1    | Work counters — round-trips by `op`, rows, statements, events, bytes, tokenizer calls | Exactly, on any machine | The comparison                                                 |
| 2    | Instruction counts (`pytest-codspeed`)                                                | Yes                     | Pure-CPU stages only                                           |
| 3    | Wall clock                                                                            | No                      | Read by a human; compared only against a same-machine baseline |

Tier 1 is what a comparison asserts on. This is why the suite can claim "every work counter identical" across two
independent runs and mean it.

### 2. Work, pacing and contention are three numbers

- **Work** — parsing, writing, resolving. The number to compare.
- **Contention** — a lock or semaphore wait. A rise is real but says the pacing constants are now wrong, not that the
  code got slower.
- **Pacing** — drain settle, batch windows, poll intervals, rate limiting. Deliberate and correct; it moves because
  someone changed a constant.

**Production pacing constants stay at production values and are reported, never lowered.** A benchmark that speeds up
its own drain floor measures a system nobody runs.

The residual (`total − accounted`) is shown rather than discarded, so an *un*instrumented stage has somewhere visible to
appear.

### 3. Harvest existing telemetry; never monkeypatch to measure

`timed_phase` already brackets ten pipeline steps and carries each one's work-unit count as a span attribute;
`atlas_graph_query_seconds{op, kind}` already attributes every round-trip to the calling method, and its **sample count
per `op` is the round-trip count**. The harness installs in-memory exporters and reads what comes back. A monkeypatch
would reimplement `_timed_query` worse and rot at the next refactor.

The one thing a benchmark may substitute rather than observe is an **external dependency**.

### 4. Graph size is swept independently of batch size

Several stages cost in proportion to the whole project rather than the batch. A benchmark varying only file count moves
both variables together and can never separate "we wrote more" from "the graph got bigger". Measured: a fixed 5-file
batch costs 623 round-trips into a 50-entity graph and 2,028 into a 146-entity one.

### 5. Every measured stage carries a non-vacuity guard

Each stage has a degenerate path that produces a stable, meaningless number. Filters that reject nothing, fusion over
disjoint channels, a corpus that parsed nothing, a traversal over a graph with no edges, a query matching no token — all
of these are fast, quiet and wrong. So the corpus assertions run **before** the timings, and a control arm accompanies
every regression arm.

### 6. The suite measures; it does not fix

Where it surfaces a cost — SQLite's partial-index scans, Memgraph's per-label multipliers — the number is the
deliverable. Whether to change the design is a separate decision, taken against the number.

### 7. Baselines: counters compare exactly, timings never compare

The suite is not run by CI. It is invoked deliberately after significant changes, so the stored baseline _is_ the signal
— there is no continuous series to spot a trend in.

- Counters compare **exactly**. A percentage band on a deterministic number only hides small real regressions.
- Timings are stored for a human to read and **never** drive a verdict. A run that is slower with identical counters is
  the machine, not the code.
- A **cross-machine difference produces no comparison at all**, plus a report naming the fields that differ. Not a
  lenient comparison. A threshold that fires because somebody changed laptops teaches people to ignore it, which costs
  more than the regression it was meant to catch.
- A counter that **shrank** is reported and never failed. Work removed is what an optimisation is for; a suite that
  fails on improvement gets its thresholds raised until it means nothing.
- Recording is only ever explicit. A suite that rewrites its own baseline on a passing run cannot detect slow drift —
  every run compares against the previous one and every step looks like no change.

## Consequences

The suite found things on its first runs, which is the point: 71 tokenizer calls for 36 texts (the splitter re-encodes
walking the border ladder, and nothing measured it); 12 statements that plan as a full scan of `nodes` including the
file-hash gate itself, at 1,280 → 3,680 row visits across a size sweep; and `atlas bench` making **real embedding
calls** on its first end-to-end run, fixed by stubbing at `litellm.aembedding`, the single line on that path that leaves
the machine.

It also produced three wrong readings before producing right ones, and the pattern is worth recording because it is the
failure mode this ADR's guards exist to prevent. Each time, a metric that _correlated_ with the thing being measured was
used instead of one that _measures_ it: a call count read as a vector count; `new_vectors` used to detect rewrites,
which it structurally cannot see; a token cap set below a character truncation so both arms of a regression test behaved
identically. In every case a control arm or a non-vacuity guard caught it, and in no case would a one-sided assertion
have.

Six of twenty-two registered languages — apex, hcl, shell, sql, tsx, xml — have no real source anywhere in this
repository. Their grammars are installed and their handlers run on every index, so a regression in any of them is
currently invisible. Recorded in an `UNCOVERED` ledger with a reason per language rather than silently skipped; closing
it means vendoring third-party sources the way `tests/fixtures/langcov` already does.

The cost is that the suite is slower to write than one that prints JSON, because every number needs a guard proving it
is not measuring an empty case. That is the trade this ADR takes deliberately: the previous suite was cheap to write and
did not work.

## References

- ATL-154 and children (158, 159, 160, 161, 162, 163, 164, 155)
- ADR-0036 — the graph is the dedup layer; the provenance split this reports
- ADR-0040 — oversized nodes; the chunking path a truncation made unreachable
- ADR-0042 — reindex scope and destruction; `--full` is what a benchmark drives
