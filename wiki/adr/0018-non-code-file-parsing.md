# ADR-0018: Parsing Non-Code Files — Dispatch, Content-Aware Config, and Pre-Parse Safety

## Status

Accepted

## Date

2026-07-30

## Context

Indexing covered nine programming languages plus Markdown. Real repositories are not only code: the infrastructure that
deploys them (Terraform, Dockerfiles, Kubernetes manifests, Ansible), the scripts that build them, the SQL that shapes
their data, and the config that wires them together all carry structure and cross-references worth having in the graph.

Extending to those formats surfaced three problems the existing framework could not express:

1. **Not every format is identified by extension.** `PurePosixPath("Dockerfile").suffix` is `""`, and dispatch was
   suffix-only, so the canonical Dockerfile was unreachable. Registering `""` as an extension would hijack every
   extensionless file in the repo (`LICENSE`, `Makefile`, `.gitignore`).
2. **One extension can be many formats.** A `.yml` file may be a Kubernetes manifest, a docker-compose file, an Ansible
   playbook, a GitHub Actions workflow, or arbitrary application config. The registry maps one extension to one
   language, so this cannot be a registry concern.
3. **Some grammars die natively on pathological input.** Discovered by measurement, not by reasoning.

A fourth problem was latent and became urgent: registering a language did **not** cause its files to be indexed.
`_DEFAULT_INCLUDE` (the scan allowlist) and `_EXTENSION_MAP` (the parser registry) were separately hand-maintained, and
the guard test compared `_DEFAULT_INCLUDE` against a _third_ hardcoded list, so drift passed CI green and the failure
was silent.

## Decision

### Filename dispatch alongside extensions

`LanguageConfig` gains `filenames: frozenset[str]`, matched on the lowercased whole basename **before** the suffix map.
Whole-basename, so `dockerfile.txt` does not route to the container language.

### Content-aware dialects live inside `parse_func`, not the registry

One registration per file format, whose `parse_func` branches internally on the bytes it already receives. Dialects are
detected **structurally** — `apiVersion`+`kind` for Kubernetes, `services:` for compose, `hosts:`/`tasks:` for Ansible
playbooks — with path convention only ever a secondary signal. This needs zero registry change and keeps "one extension,
one language" intact.

### Unrecognised config falls back; data files are rejected

An unfingerprinted JSON/YAML/TOML file still yields a generic key tree rather than being dropped, because searchable
structure has value even when nothing can interpret the schema. Data files are excluded ahead of that fallback by a
cheapest-first ladder: `.jsonl`/`.ndjson` by extension, a size ceiling, unparseability, top-level-array shape, and — the
strongest signal — a key-repetition ratio, which catches nested record dumps like `{"rows": [...]}` that a
top-level-array check misses entirely.

### A pre-parse safety guard on raw bytes

Some grammars die _natively_ inside `Parser.parse()` — a scanner-buffer overflow that kills the process (Windows
`0xC0000005`, POSIX SIGSEGV) with no Python exception. The kill happens before `parse_func` is called, so the only place
it can be stopped is on the raw bytes before tree-sitter sees them. Input over a measured block-nesting depth or size
ceiling is declined via the existing "unsupported" path that the AST consumer already logs and skips.

### Deny-list for secret-bearing files

Widening the allowlist to `*.yaml`/`*.json`/`*.toml`/`*.tfvars` made `secrets.yaml`, `gcp-key.json`,
`service-account.json`, `local.settings.json`, `terraform.tfvars` and `credentials.toml` reachable by the scanner for
the first time. An indexed entity is an **embedded** one, so their contents would leave the machine for the embedding
API. Denied by name in `_DEFAULT_EXCLUDE`.

### The drift guard derives from the live registry

The scan-allowlist guard test now calls `discover_plugins()` and asserts `_DEFAULT_INCLUDE` is a superset of every
registered extension and filename, naming the missing entries on failure.

## Consequences

### Positive

- Infrastructure and config are first-class graph content, with statically-resolvable cross-references.
- Registry/allowlist drift is now a loud CI failure instead of silent non-indexing.
- The pre-parse guard closes a **pre-existing** process-kill vector: Markdown block constructs crash at the same depth,
  so the knowledge vault was already exposed before any of this work.

### Negative

- Config-node volume rises sharply — this repo went from 19 to 173 entities across 17 config files (~9x), each carrying
  an embedding. Bounded per file, unbounded in file count.
- `*.tfvars` is denied wholesale. Terraform convention puts secrets there, but teams that keep non-secret tfvars lose
  that structure, and `scope.extend_include` cannot re-include it (exclude is evaluated first).
- `_DEFAULT_INCLUDE` stays hand-maintained. Deriving it from the registry was rejected deliberately: it would make
  indexing scope depend on which optional grammar wheels are installed, so a grammar that fails to load would _silently
  shrink_ scope — trading a loud failure for exactly the silent one this ADR exists to kill.

### Risks

- Dialect detection is heuristic. A file that looks structurally like a Kubernetes manifest but isn't will be modelled
  as one.
- The datafile ladder's thresholds are calibrated, not proven. Validated against realistic input (a 150-dep
  `package.json` and a 120-key `appsettings.json` both survive; 500 uniform records and `.jsonl` are rejected), but a
  genuinely large hand-written config could still be misread as data.
- The guard's depth limit (64) sits well below the lowest measured crash (250), but the limit is a heuristic over
  indentation and markers, not a real parse — a construct that nests without a leading marker run would not be counted.

## Alternatives Considered

### Register each YAML dialect as its own language

Rejected: dispatch resolves one extension to exactly one `LanguageConfig`, so `.yml` cannot map to five languages. It
would require the registry to run content detection, pushing format-specific logic into the framework.

### Return `None` for unrecognised config

The original decision, reversed on user direction. Dropping unfingerprinted config discards searchable structure; the
datafile triage is the narrower instrument that removes the actual cost driver.

### Bound bracket nesting as well as block nesting

Rejected on measurement. Tree-sitter's core parser is iterative and bracket nesting never crashed at any depth up to
25600 — it only ever surfaces as a `RecursionError` from a handler's own walk, which is caught normally. A byte
heuristic on brackets would reject legitimate files: minified JS in `site-packages` measures bracket depth 270.

### Rely on `.gitignore` to keep secrets out

Rejected. `FileScope` never consults git — it parses `.gitignore` as a pattern file — and it never reads
`.git/info/exclude` or the global `core.excludesFile`, drops the repo-root `.gitignore` entirely when a monorepo
sub-project is rooted in a subdirectory, and matches case-sensitively where git on Windows does not. Any one of those
re-exposes a file the user believes is hidden.

## References

- `src/code_atlas/parsing/ast.py` — `LanguageConfig.filenames`, `_FILENAME_MAP`, `_block_depth`, `_parse_hazard`
- `src/code_atlas/parsing/languages/{hcl,shell,containerfile,sql,config}.py`
- `src/code_atlas/indexing/orchestrator.py` — `_DEFAULT_INCLUDE`, `_DEFAULT_EXCLUDE`
- [ADR-0006](./0006-pure-python-tree-sitter.md) — the in-process py-tree-sitter decision this builds on
- [ADR-0011](./0011-note-vault-schema.md) — Markdown/Note precedent for a non-code format
