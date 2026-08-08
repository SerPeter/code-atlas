# ADR-0033: The Graph Renderer Must Be Permissively Licensed

## Status

Accepted. Governs every vendored asset the web UI ships (ATL-109, ATL-115).

## Context

`code-atlas-mcp` is distributed on PyPI under **Apache-2.0**. The web UI vendors a JavaScript graph-rendering library
and serves it from `atlas ui`, so that library is redistributed inside an Apache-2.0 package and its license becomes
ours to honour.

Cosmograph was the obvious candidate — GPU force simulation, purpose-built for graphs at this scale. Reading the actual
shipped license files rather than the registry metadata turned up a trap that a reasonable person would walk straight
into:

| package                  |   version | license          |
| ------------------------ | --------: | ---------------- |
| `@cosmograph/cosmos`     |     3.4.1 | **CC-BY-NC-4.0** |
| `@cosmograph/cosmograph` |     2.4.1 | **CC-BY-NC-4.0** |
| `cosmograph` (PyPI)      |     0.5.3 | **GPL-3.0**      |
| **`@cosmos.gl/graph`**   | **3.4.0** | **MIT** ✅       |

**The last two rows are the same engine.** `@cosmograph/cosmos` was MIT for seven versions (`2.0.0-beta.20` through
`2.0.0-beta.26`, Dec 2024 – May 2025); that lineage was donated to the **OpenJS Foundation** and continues as
`@cosmos.gl/graph`. The original scope then **reverted to CC-BY-NC-4.0** at 3.4.0. So today there are two packages with
near-identical names, the same version number, released three days apart, differing by a license that decides whether
this project can be distributed at all.

The CC grant is explicitly **non-sublicensable** and limited to "NonCommercial purposes". Shipping it would purport to
grant downstream users rights we do not hold. Creative Commons themselves
[recommend against using CC licenses for software](https://creativecommons.org/faq/).

Separately, `@cosmograph/cosmograph` ships `licensing-manager.js` and `telemetry-manager.js` that POST
`navigator.userAgent`, hostname and graph size to a hardcoded Supabase endpoint. Telemetry is **on unless a license key
is present**, and an unlicensed instance gets a clickable watermark injected into the DOM. The skip list covers
`localhost`/`127.0.0.1`/`::1` — but a `file://` static export (ATL-120) has an **empty** hostname, which is not in that
list. Our offline export would have phoned home.

## Decision

**Every vendored front-end asset must be MIT, BSD, Apache-2.0, or ISC.** No CC-licensed, GPL/LGPL, source-available, or
"free for non-commercial use" dependency, whether vendored, CDN-linked, or pulled at build time.

Each vendored asset records, beside the file, its **name, version, license, and upstream source commit** — the same
provenance discipline `tests/fixtures/langcov/` already uses for the vendored corpus, and for the same reason:
attribution that lives somewhere else stops travelling with the thing it describes.

**The renderer is [sigma.js](https://www.sigmajs.org/) v3 + graphology (MIT).** Not cosmos.gl, despite cosmos.gl being
the stronger engine, and the reason is labels: sigma renders them with collision avoidance built in, cosmos.gl renders
points and links only. Three of the four planned views are module dependency graphs, blast-radius exploration and entity
detail, where **the module names _are_ the content** — an unlabelled map of them is a decorative hairball. Layout is
precomputed server-side from Memgraph, which neutralises cosmos.gl's main advantage.

**Revisit if** the opt-in "show everything" view (30k+ nodes) becomes a headline feature. At that density labels stop
mattering and GPU simulation starts to; `@cosmos.gl/graph` is MIT and is then the right answer. It is also the cleaner
single-file export: one 177 KB gzipped UMD bundle exposing a global, with zero external imports.

## Consequences

- **`@cosmograph/*` is permanently out of bounds**, and the near-name collision is the hazard to guard. A future reader
  searching npm for "cosmograph" finds the CC-BY-NC package first. Pin and review the exact scope: `@cosmos.gl/graph`,
  never `@cosmograph/cosmos`. Never `pip install cosmograph` — that one is GPL-3.0.
- **No runtime network access.** Assets are served from the package, not a CDN, so `atlas ui` works offline and the
  static export cannot leak. This also means bundle size is a packaging cost we accept deliberately.
- **Labels are a first-class requirement**, not a nice-to-have. Any future renderer swap has to keep them.
- **License checking belongs in CI eventually.** Right now the guard is this ADR plus review. That is thin for something
  with a wrong-answer cost this high, but a license scanner over a vendored `.js` file is not a solved problem, and the
  vendored set is currently one library.
- This constrains only **front-end** assets. Python dependencies are governed by the existing pinning discipline in
  `pyproject.toml`.

## References

- ATL-115, ATL-109
- [cosmos.gl at the OpenJS Foundation](https://openjsf.org/blog/introducing-cosmos-gl) — the MIT lineage
- [Cosmograph licensing](https://cosmograph.app/licensing/) — the CC-BY-NC terms
- `tests/fixtures/langcov/README.md` — the provenance pattern this reuses
