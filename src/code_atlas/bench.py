"""Per-stage accounting for an indexing run: work, contention, and what was waiting.

Nothing here instruments anything. The pipeline already emits what a benchmark needs —
`timed_phase(stage, phase, **attrs)` brackets ten steps and carries each one's work-unit
count as a span attribute, and `atlas_graph_query_seconds{op, kind}` attributes every
database round-trip to the method that issued it. This module installs in-memory
exporters, reads what comes back, and arranges it so a person can see where the time
went.

**The distinction this exists for.** A single wall-clock number blends three things that
mean different things when they move:

- *work* — parsing, writing, resolving. The number to compare between runs.
- *contention* — a lock or semaphore wait. A rise here is real, but it says the pacing
  constants are now wrong, not that the code got slower.
- *pacing* — the drain settle floor, batch windows, poll intervals. Deliberate, correct,
  and not a regression when it moves; it moves because someone changed a constant.

They are reported apart, and the pacing constants stay at their production values. A
benchmark that lowers them to make its numbers look better is measuring a system nobody
runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# Phases whose time is work: the pipeline doing the thing it exists to do.
_WORK: frozenset[tuple[str, str]] = frozenset(
    {
        ("ast", "parse"),
        ("ast", "upsert"),
        ("ast", "detectors"),
        ("ast", "enrich"),
        ("ast", "resolve"),
        ("embed", "read_entities"),
        ("embed", "dedup"),
        ("embed", "write"),
    }
)

# Waiting on something another coroutine holds. Reported apart because the fix is a
# constant, not a query.
_CONTENTION: frozenset[tuple[str, str]] = frozenset({("embed", "write_lock_wait")})

# Someone else's latency. Reported, never gated -- it is not ours to optimise, and
# under the default stub it is not even real.
#
# CAVEAT, which the report states out loud rather than implying otherwise: the
# `embed.provider` span (consumers.py:2217) brackets `embed_batch`, which also holds
# the concurrency gate and the rate limiter. So this line is provider time PLUS
# whatever those waited for. Measured on this harness's first real run: 4.08s of a
# 14s indexing run, with the provider stubbed to return instantly and Valkey down --
# `EmbedClient` is built with `settings.redis` unconditionally (orchestrator.py:2198),
# so on the embedded backend the limiter still dials Valkey and pays a connection
# timeout per batch before degrading. Separating the two needs a `timed_phase` inside
# `embed_batch`, which belongs to the embed story rather than to the harness.
_EXTERNAL: frozenset[tuple[str, str]] = frozenset({("embed", "provider")})

# The work-unit attribute each phase carries, so a duration is never printed without the
# count it applies to. Taken from the `timed_phase` call sites; a phase missing here
# reports its time with no unit rather than guessing one.
_UNITS: dict[tuple[str, str], str] = {
    ("ast", "parse"): "files",
    ("ast", "upsert"): "files",
    ("ast", "detectors"): "detectors",
    ("ast", "enrich"): "count",
    ("ast", "resolve"): "rels",
    ("embed", "read_entities"): "entities",
    ("embed", "dedup"): "candidates",
    ("embed", "provider"): "entities",
    ("embed", "write"): "vectors",
}


def classify(stage: str, phase: str) -> str:
    """Which of the three lines a phase belongs on."""
    key = (stage, phase)
    if key in _CONTENTION:
        return "contention"
    if key in _EXTERNAL:
        return "external"
    if key in _WORK:
        return "work"
    return "work"


@dataclass(frozen=True)
class PhaseTiming:
    """One `timed_phase` step, summed over every batch that ran it."""

    stage: str
    phase: str
    seconds: float
    calls: int
    units: int
    unit_name: str

    @property
    def kind(self) -> str:
        return classify(self.stage, self.phase)

    @property
    def rate(self) -> float | None:
        """Units per second, or None when the phase carries no unit."""
        if not self.unit_name or self.seconds <= 0:
            return None
        return self.units / self.seconds


@dataclass(frozen=True)
class ConsumerBreakdown:
    """One consumer's phases. Summing *within* a consumer is valid; across is not.

    The AST and embed consumers run as separate asyncio tasks against the same graph, so
    their phases overlap in wall time. Within one consumer the `timed_phase` blocks are
    strictly sequential inside a single `process_batch`, which is what makes these sums
    meaningful at all.
    """

    stage: str
    phases: tuple[PhaseTiming, ...]

    def _sum(self, kind: str) -> float:
        return sum(p.seconds for p in self.phases if p.kind == kind)

    @property
    def work_s(self) -> float:
        return self._sum("work")

    @property
    def contention_s(self) -> float:
        return self._sum("contention")

    @property
    def external_s(self) -> float:
        return self._sum("external")

    @property
    def accounted_s(self) -> float:
        return self.work_s + self.contention_s + self.external_s


@dataclass(frozen=True)
class BenchReport:
    """What one indexing run cost, arranged so the three lines stay apart."""

    total_s: float
    consumers: tuple[ConsumerBreakdown, ...]
    round_trips: dict[tuple[str, str], int] = field(default_factory=dict)
    corpus: str = ""
    backend: str = ""
    notes: tuple[str, ...] = ()

    def consumer(self, stage: str) -> ConsumerBreakdown | None:
        return next((c for c in self.consumers if c.stage == stage), None)

    @property
    def accounted_s(self) -> float:
        """The longest single consumer, not the sum of both.

        Adding two overlapping wall-clock spans produces a number larger than the run
        itself, which is how a report starts claiming negative idle time. The floor a
        concurrent pipeline can be held to is its slowest participant.
        """
        return max((c.accounted_s for c in self.consumers), default=0.0)

    @property
    def unaccounted_s(self) -> float:
        """Wall clock no phase claimed: pacing, plus any stage nobody instrumented.

        Dominated by deliberate pacing on a healthy run — `_wait_for_drain` alone holds
        `lag == 0` for `_DRAIN_SETTLE_S` behind a 1.5x poll backoff and first passes at
        roughly 3.6s. It is shown rather than discarded precisely so an *un*instrumented
        stage has somewhere visible to appear: if this grows and no phase did, something
        is running that nobody is measuring.
        """
        return max(0.0, self.total_s - self.accounted_s)

    def round_trips_by_op(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for (op, _kind), count in self.round_trips.items():
            totals[op] = totals.get(op, 0) + count
        return totals

    @property
    def total_round_trips(self) -> int:
        return sum(self.round_trips.values())

    def work_counters(self) -> dict[str, int]:
        """Every deterministic count in the run, for comparing two runs exactly.

        These are what reproduce across machines. Times do not, which is why the
        equality check a benchmark makes is against this and never against a duration.
        """
        counters = {
            f"{p.stage}.{p.phase}.{p.unit_name or 'calls'}": p.units or p.calls
            for c in self.consumers
            for p in c.phases
        }
        counters.update({f"rt.{op}": n for op, n in self.round_trips_by_op().items()})
        return counters


def _fmt(seconds: float) -> str:
    return f"{seconds:7.2f}s"


def render(report: BenchReport) -> str:
    """A table for a person to read after a significant change.

    This suite is invoked deliberately, not by CI, so the output *is* the deliverable —
    nothing downstream parses it. Legibility is the requirement.
    """
    lines: list[str] = []
    head = f"bench: {report.corpus or 'unnamed corpus'}"
    if report.backend:
        head += f"  [backend: {report.backend}]"
    lines.append(head)
    lines.append("=" * max(len(head), 72))

    for consumer in report.consumers:
        if not consumer.phases:
            continue
        lines.append("")
        lines.append(f"{consumer.stage} consumer")
        lines.append(f"  {'phase':<16}{'time':>9}  {'calls':>6}  {'units':>10}  {'rate':>12}   class")
        for phase in sorted(consumer.phases, key=lambda p: -p.seconds):
            rate = phase.rate
            rate_s = f"{rate:,.0f}/s" if rate is not None else "-"
            unit_s = f"{phase.units:,} {phase.unit_name}" if phase.unit_name else "-"
            lines.append(
                f"  {phase.phase:<16}{_fmt(phase.seconds):>9}  {phase.calls:>6}  "
                f"{unit_s:>10}  {rate_s:>12}   {phase.kind}"
            )
        lines.append(
            f"  {'-> work':<16}{_fmt(consumer.work_s):>9}"
            f"   contention {_fmt(consumer.contention_s)}   external {_fmt(consumer.external_s)}"
        )

    lines.append("")
    lines.append("run")
    lines.append(f"  {'wall clock':<16}{_fmt(report.total_s):>9}")
    lines.append(
        f"  {'accounted':<16}{_fmt(report.accounted_s):>9}   (slowest consumer; consumers overlap, so not a sum)"
    )
    lines.append(
        f"  {'pacing + gaps':<16}{_fmt(report.unaccounted_s):>9}   (drain settle, batch windows, poll intervals)"
    )

    if report.round_trips:
        lines.append("")
        lines.append(f"database round-trips ({report.total_round_trips:,} total)")
        for op, count in sorted(report.round_trips_by_op().items(), key=lambda kv: -kv[1])[:15]:
            lines.append(f"  {op:<44}{count:>8,}")

    lines.extend(f"\nnote: {note}" for note in report.notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telemetry capture
# ---------------------------------------------------------------------------


class TelemetryCapture:
    """In-memory span and metric capture for one process.

    Five things silently produce zero data if got wrong, so they are done here once
    rather than at each call site:

    1. `set_tracer_provider` is once per process — a second call is ignored with a log
       line, so the provider is installed once and isolation comes from clearing.
    2. `telemetry._initialized` must be set, or a later `init_telemetry` rebinds
       `telemetry._metrics` to its own meter and detaches this reader. Spans keep
       working, so the failure reads as "metrics stopped, traces fine".
    3. `SimpleSpanProcessor`, not `BatchSpanProcessor` — spans land on end with no flush
       to race.
    4. `shutdown_telemetry()` must never be called mid-run; it resets `_enabled`.
    5. The lazy tracer caches once resolved while enabled, so the provider must be in
       place before the first measured span.
    """

    def __init__(self) -> None:
        from opentelemetry import trace as otel_trace  # noqa: PLC0415
        from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: PLC0415

        import code_atlas.telemetry as tel  # noqa: PLC0415

        self._tel = tel
        self.spans = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.spans))
        otel_trace.set_tracer_provider(provider)

        self.reader = InMemoryMetricReader()
        meter = MeterProvider(metric_readers=[self.reader]).get_meter("code_atlas")
        # Only the instruments this module reads. The rest stay no-ops, which is correct:
        # an instrument nobody harvests costs nothing and hides nothing.
        # Reaching into module state on purpose: these globals are the switch every
        # lazy tracer and `_timed_query` guard reads, and there is no public setter.
        tel._metrics = tel._Metrics(  # noqa: SLF001
            stage_seconds=meter.create_histogram("atlas_stage_seconds", unit="s"),
            graph_query_seconds=meter.create_histogram("atlas_graph_query_seconds", unit="s"),
            parse_seconds=meter.create_histogram("atlas_parse_seconds", unit="s"),
            embeddings_total=meter.create_counter("atlas_embeddings_total"),
            parse_bytes=meter.create_histogram("atlas_parse_bytes", unit="By"),
        )
        tel._enabled = True  # noqa: SLF001
        tel._initialized = True  # noqa: SLF001

    def clear(self) -> None:
        self.spans.clear()

    def phases(self) -> list[PhaseTiming]:
        """Aggregate `timed_phase` spans into one row per (stage, phase).

        Read from spans rather than from `atlas_stage_seconds`, because only the span
        carries the work-unit attribute. The histogram agrees on the duration; it just
        cannot say how many files that duration covered.
        """
        acc: dict[tuple[str, str], dict[str, Any]] = {}
        for span in self.spans.get_finished_spans():
            name = span.name
            if "." not in name:
                continue
            stage, _, phase = name.partition(".")
            if (stage, phase) not in _UNITS and (stage, phase) not in _CONTENTION:
                continue
            seconds = (span.end_time - span.start_time) / 1e9 if span.end_time and span.start_time else 0.0
            unit_name = _UNITS.get((stage, phase), "")
            attrs = span.attributes or {}
            entry = acc.setdefault((stage, phase), {"seconds": 0.0, "calls": 0, "units": 0, "unit_name": unit_name})
            entry["seconds"] += seconds
            entry["calls"] += 1
            if unit_name:
                # A span attribute has no useful static type. Every `timed_phase` call
                # site passes an int; anything else counts as zero rather than raising,
                # because a benchmark that dies on a stray attribute is worse than one
                # that reports a missing unit.
                raw = attrs.get(unit_name, 0)
                entry["units"] += int(raw) if isinstance(raw, (int, float)) else 0
        return [
            PhaseTiming(
                stage=stage,
                phase=phase,
                seconds=v["seconds"],
                calls=v["calls"],
                units=v["units"],
                unit_name=v["unit_name"],
            )
            for (stage, phase), v in acc.items()
        ]

    def round_trips(self) -> dict[tuple[str, str], int]:
        """(op, kind) -> round-trip count.

        The histogram's *sample count* is the number of round-trips: one sample is
        recorded per statement. This is the hardware-independent half of the report and
        the half a comparison should lean on.
        """
        counts: dict[tuple[str, str], int] = {}
        data = self.reader.get_metrics_data()
        if data is None:  # no collection has happened yet
            return counts
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name != "atlas_graph_query_seconds":
                        continue
                    for point in metric.data.data_points:
                        attrs = point.attributes or {}
                        key = (str(attrs.get("op")), str(attrs.get("kind")))
                        # Only histogram points carry a sample count; the union this
                        # iterates includes number points, which do not.
                        counts[key] = counts.get(key, 0) + int(getattr(point, "count", 0))
        return counts

    def embedding_provenance(self) -> dict[str, int]:
        """Where each vector came from: unchanged / dedup / api.

        The split ADR-0036 turns on. "2,346 API calls, 0 cache hits" is the measurement
        that justified making the graph the dedup layer, and nothing was reporting it at
        the time -- it had to be reconstructed from logs afterwards.
        """
        out: dict[str, int] = {}
        data = self.reader.get_metrics_data()
        if data is None:
            return out
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name != "atlas_embeddings_total":
                        continue
                    for point in metric.data.data_points:
                        source = str((point.attributes or {}).get("source"))
                        out[source] = out.get(source, 0) + int(getattr(point, "value", 0))
        return out

    def report(
        self, *, total_s: float, corpus: str = "", backend: str = "", notes: tuple[str, ...] = ()
    ) -> BenchReport:
        by_stage: dict[str, list[PhaseTiming]] = {}
        for phase in self.phases():
            by_stage.setdefault(phase.stage, []).append(phase)
        return BenchReport(
            total_s=total_s,
            consumers=tuple(
                ConsumerBreakdown(stage=stage, phases=tuple(phases)) for stage, phases in sorted(by_stage.items())
            ),
            round_trips=self.round_trips(),
            corpus=corpus,
            backend=backend,
            notes=notes,
        )


_capture: TelemetryCapture | None = None


def capture() -> TelemetryCapture:
    """The process's capture, installed on first use.

    A module-level singleton because the tracer provider is a process-level resource:
    a second `TracerProvider` would be refused and its exporter would stay empty, which
    looks like "the pipeline emitted nothing" rather than like a mistake.
    """
    global _capture  # noqa: PLW0603
    if _capture is None:
        _capture = TelemetryCapture()
    return _capture


def pin_determinism(consumer: Any) -> Iterator[None] | None:
    """Make an AST consumer's flush cadence reproducible.

    The resolution flush is partly wall-clock driven: `_resolve_time_interval_s` is 30s
    and the adaptive gap is computed from the *previous* flush's measured duration. Left
    alone, two runs over identical bytes issue different numbers of round-trips, and the
    run-to-run equality this suite depends on is not available at all.

    Pins the cadence to batch count only. Returns nothing; mutates in place, because the
    consumer is constructed inside `_run_pipeline` and handed out already running.
    """
    consumer._resolve_adaptive = False  # noqa: SLF001
    consumer._resolve_time_interval_s = float("inf")  # noqa: SLF001
    return None


# ---------------------------------------------------------------------------
# Provider seam
# ---------------------------------------------------------------------------


@dataclass
class StubStats:
    """What the run asked of the embedding provider, had one been there."""

    calls: int = 0
    texts: int = 0
    tokenizer_calls: int = 0
    """`litellm.encode` invocations.

    Counted because it is the embed stage's largest piece of pure CPU and nothing
    measures it: `_truncate_texts` encodes every text, and `split_text` encodes the same
    text repeatedly as the splitter walks down the border ladder looking for a cut. On a
    corpus with oversized entities this is not a rounding error, and it is entirely our
    own code -- exactly the kind of cost this suite exists to make visible.
    """

    def summary(self) -> dict[str, int]:
        return {
            "provider_calls": self.calls,
            "texts_embedded": self.texts,
            "tokenizer_calls": self.tokenizer_calls,
        }


def _deterministic_vector(text: str, dimension: int) -> list[float]:
    """A stable unit-ish vector derived from the text.

    Deterministic on purpose: the same text must yield the same vector, or the graph
    dedup layer (ADR-0036) and the freshness check behave differently under the stub
    than they do in production, and the stage stops measuring what it claims to.
    """
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * ((dimension // len(digest)) + 1))[:dimension]
    return [(b - 127.5) / 127.5 for b in raw]


@contextmanager
def stub_provider(dimension: int) -> Iterator[StubStats]:
    """Replace the one line that reaches the network, and nothing above it.

    `litellm.aembedding` at `search/embeddings.py:289` is the only call in the
    repository that leaves the machine on this path. Stubbing *there* rather than
    replacing `EmbedClient` keeps everything above it real and measured: text building,
    hashing, tokenizing, chunking, truncation, batch packing, the rate limiter, the
    concurrency gate, the graph dedup lookup and the vector writes.

    This is not the monitoring-by-monkeypatch this module otherwise refuses. Measurement
    is harvested from telemetry; this is substituting an external dependency, which is
    the one thing a benchmark must do rather than observe. Cost is the smaller reason —
    the provider's latency is not ours to optimise and it swamps the numbers that are.

    Arity is the property worth guarding: exactly one vector per input text. The
    `AsyncMock` this replaces returned a single vector for any batch size, which would
    hide a fan-out bug rather than expose one.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    import litellm  # noqa: PLC0415

    from code_atlas.search.embeddings import EmbedClient  # noqa: PLC0415

    stats = StubStats()
    real = EmbedClient._embed_call  # noqa: SLF001 - substituting the network boundary is the point
    real_encode = litellm.encode

    def _counting_encode(*args: Any, **kwargs: Any) -> Any:
        stats.tokenizer_calls += 1
        return real_encode(*args, **kwargs)

    async def _stub(_self: Any, kwargs: dict[str, Any]) -> Any:
        texts = list(kwargs.get("input") or [])
        stats.calls += 1
        stats.texts += len(texts)
        return SimpleNamespace(data=[{"embedding": _deterministic_vector(t, dimension)} for t in texts])

    EmbedClient._embed_call = _stub  # ty: ignore[invalid-assignment]  # noqa: SLF001 - substituting the network boundary is the point
    litellm.encode = _counting_encode
    try:
        yield stats
    finally:
        EmbedClient._embed_call = real  # noqa: SLF001 - substituting the network boundary is the point
        litellm.encode = real_encode


# ---------------------------------------------------------------------------
# Corpus profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusProfile:
    """What a corpus actually contained, read back from the graph after indexing.

    Asserted rather than assumed, because a corpus that quietly stopped producing a
    shape is indistinguishable from a benchmark that got faster. Three of the four
    regressions this suite exists to catch lived in a shape the corpus did not have:
    no `DocSection` meant the doc-link path never ran at all, and no entity over the
    embedding cap meant chunking was unreachable.
    """

    labels: dict[str, int]
    files: int
    entities: int
    files_by_language: dict[str, int] = field(default_factory=dict)
    """Files per registered language.

    Derived from `get_language_for_file`, the same lookup indexing uses, rather than from
    bare extensions: `.h` routes by content and `Dockerfile` has no extension at all, so
    an extension histogram would disagree with what was actually parsed.
    """

    @property
    def doc_sections(self) -> int:
        return self.labels.get("DocSection", 0)

    @property
    def notes(self) -> int:
        return self.labels.get("Note", 0)

    @property
    def embed_chunks(self) -> int:
        """Overflow vectors for entities too large for one provider call (ADR-0040).

        Zero on a small corpus, and that is correct — so it is not required by default.
        A corpus meant to exercise chunking asks for it explicitly via `missing`.
        """
        return self.labels.get("EmbedChunk", 0)

    def missing(self, *, require: tuple[str, ...] = ("DocSection",)) -> tuple[str, ...]:
        """Required labels the corpus did not produce."""
        return tuple(label for label in require if self.labels.get(label, 0) == 0)

    def summary(self) -> dict[str, int]:
        return {
            "files": self.files,
            "entities": self.entities,
            **{k: v for k, v in sorted(self.labels.items()) if v},
            **{f"lang:{k}": v for k, v in sorted(self.files_by_language.items()) if v},
        }


async def profile_corpus(graph: Any, project_name: str) -> CorpusProfile:
    """Read back what the indexed corpus contained.

    Uses only `GraphBackend` methods, so it reports identically on either backend --
    a profile that worked on one and silently returned zeros on the other would be
    worse than none, since zero reads as "the corpus lacks this shape".
    """
    from code_atlas.parsing.ast import get_language_for_file  # noqa: PLC0415

    labels = await graph.get_label_counts()
    paths = await graph.get_project_file_paths(project_name)
    entities = await graph.count_entities(project_name)

    by_language: dict[str, int] = {}
    for path in paths:
        # Source is deliberately not passed. This counts which language *claims* the
        # file; resolving the dialect of a shared suffix would need the bytes, which is
        # not worth a read per file for a histogram.
        config = get_language_for_file(path)
        name = config.name if config is not None else "(none)"
        by_language[name] = by_language.get(name, 0) + 1

    return CorpusProfile(labels=dict(labels), files=len(paths), entities=entities, files_by_language=by_language)


# ---------------------------------------------------------------------------
# Corpus resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """A corpus to measure, pinned to a commit so results stay comparable."""

    path: Any
    name: str
    commit: str
    source: str
    reused_cache: bool = False

    def label(self) -> str:
        short = self.commit[:12] if self.commit else "no-commit"
        return f"{self.name}@{short}"

    def comparable_with(self, other_commit: str) -> bool:
        """Two results may only be compared when they measured the same bytes.

        An unpinned corpus moving under a benchmark is the quiet way a suite starts
        reporting improvements nobody made.
        """
        return bool(self.commit) and self.commit == other_commit


def cache_root() -> Any:
    """Where cloned corpora live.

    Outside the project tree deliberately: a clone under `.atlas/` would be swept by
    the very indexing the benchmark is measuring, and would make the corpus part of the
    repository it is meant to be independent of.

    The project has no cache convention to follow -- everything else it writes lands in
    `.atlas/` relative to the project root -- so this introduces one, honouring
    `ATLAS_BENCH_CACHE` first so a machine with a small home volume can move it.
    """
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    override = os.environ.get("ATLAS_BENCH_CACHE")
    if override:
        return Path(override)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "code-atlas" / "bench-corpora"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "code-atlas" / "bench-corpora"


def _git(args: list[str], cwd: Any, timeout: int = 300) -> tuple[int, str, str]:
    """Run git the way the rest of this codebase does: list args, never a shell."""
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _head_commit(path: Any) -> str:
    code, out, _ = _git(["rev-parse", "HEAD"], cwd=path, timeout=10)
    return out if code == 0 else ""


def _slug(url: str) -> str:
    """A filesystem-safe directory name for a clone, stable across runs."""
    import hashlib  # noqa: PLC0415
    import re  # noqa: PLC0415

    tail = re.sub(r"[^A-Za-z0-9._-]+", "-", url.rstrip("/").removesuffix(".git").split("/")[-1] or "repo")
    # The hash disambiguates two forks that share a repository name; the readable half
    # is kept so the cache directory can be understood without consulting anything.
    return f"{tail}-{hashlib.sha256(url.encode()).hexdigest()[:8]}"


def resolve_corpus(spec: str, *, ref: str = "", cache: Any = None) -> Corpus:
    """A local path or a git URL, resolved to a pinned directory on disk.

    A local path is used where it is and never fetched -- a benchmark must not mutate
    the working tree someone is sitting in.

    A URL is cloned shallow into the cache. A second call for the same (url, ref) with
    the clone already present **does not fetch**: the commit already on disk is what was
    measured last time, and re-fetching would silently change the corpus under a
    comparison.
    """
    from pathlib import Path  # noqa: PLC0415

    if "://" not in spec and not spec.startswith("git@"):
        path = Path(spec).resolve()
        return Corpus(path=path, name=path.name, commit=_head_commit(path), source="local")

    root = Path(cache) if cache is not None else cache_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / _slug(spec) / (ref or "HEAD")

    if (target / ".git").exists():
        commit = _head_commit(target)
        if commit:
            return Corpus(path=target, name=target.parent.name, commit=commit, source=spec, reused_cache=True)

    target.mkdir(parents=True, exist_ok=True)
    code, _, err = _git(["init", "-q"], cwd=target, timeout=30)
    if code != 0:
        raise RuntimeError(f"git init failed for {target}: {err}")
    _git(["remote", "add", "origin", spec], cwd=target, timeout=30)
    # Fetch the single commit rather than a branch: `--branch` cannot take a raw sha,
    # and a benchmark corpus is pinned to a sha far more often than to a branch tip.
    code, _, err = _git(["fetch", "--depth", "1", "origin", ref or "HEAD"], cwd=target)
    if code != 0:
        raise RuntimeError(f"git fetch failed for {spec} at {ref or 'HEAD'}: {err}")
    code, _, err = _git(["checkout", "-q", "FETCH_HEAD"], cwd=target, timeout=120)
    if code != 0:
        raise RuntimeError(f"git checkout failed for {spec}: {err}")
    return Corpus(path=target, name=target.parent.name, commit=_head_commit(target), source=spec)
