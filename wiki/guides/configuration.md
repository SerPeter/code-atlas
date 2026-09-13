# Configuration

## atlas.toml

Create an `atlas.toml` in your project root to customize Code Atlas behavior:

```toml
[scope]
include_paths = ["services/auth", "services/billing", "libs/shared"]
exclude_patterns = ["*.generated.ts", "testdata/"]

[libraries]                          # see "Indexing external libraries" below
stubs = true                         # read installed packages' public entrypoints
full_index = ["my_company_shared_lib"]

[monorepo]
auto_detect = true
always_include = ["libs/shared"]

[embeddings]
model = "nomic-ai/CodeRankEmbed"     # see "Choosing an Embedding Model" below
base_url = "http://localhost:8080"   # self-hosted TEI; omit for cloud providers

[search]
default_token_budget = 8000
test_filter = true  # exclude test files from results by default

[detectors]
enabled = ["decorator_routing", "event_handlers", "test_mapping", "class_overrides", "di_injection", "cli_commands"]
```

## Indexing external libraries

An `ExternalPackage` is otherwise a bare name: `from pathlib import Path` puts a node called `Path` in the graph that
says nothing about what `Path` is. Stub indexing reads each installed package's **public entrypoints** — the names its
top-level module exports, with signatures and docstrings where they can be read — so an agent can see what a library
offers without the graph absorbing the library itself.

```toml
[libraries]
stubs = true          # default. Read public entrypoints of installed external packages
stub_index = []       # default: every package that resolves. Non-empty restricts to those names
full_index = []       # import names whose *whole module tree* is read, not just the entrypoint
embed_stubs = true    # default. Give stub symbols vectors
introspect = false    # default. Import each package and read signatures with inspect
```

### What it costs

The entrypoint of a package is around 1 KB; its whole tree can be 25 MB. Measured on this repo's own environment,
reading entrypoints for 96 packages is ~180 files and about 6 seconds, and it produces roughly 3,300 symbols. Reading
every tree instead would be 3,287 files and 37 MB, two-thirds of it `litellm` alone. That difference is why `full_index`
is an explicit list rather than a default.

The cost is paid once per dependency version. Each package records the version its stub was read from, and an index
skips one whose installed version has not moved — the check happens _before_ anything is read. Standard library modules
use the interpreter version, which is exactly what changes when they do.

### What resolves, and what does not

Only packages installed in **atlas's own environment** can be stubbed. Indexing somebody else's repo, most imports
resolve to nothing; those keep the provenance weight they already have (see
[ADR-0054](../adr/0054-an-external-name-is-weighted-by-how-deliberately-it-was-chosen.md)) and gain no signatures. Names
from other ecosystems — Ruby `require` paths, GitHub Actions, container images — never resolve at all.

Resolution prefers a hand-written type stub over source: a bundled `.pyi`, then a `-stubs` distribution, then the
package's own `__init__`. A `py.typed` marker is _not_ a separate case; it says the source is annotated, which is a
statement about type checkers rather than a different file to read.

### `introspect`, and why it is off

A static read finds every public entrypoint **name**, but only some of their signatures. The rest are unreachable by
construction:

| pattern                      | example                | static                                                 |
| ---------------------------- | ---------------------- | ------------------------------------------------------ |
| `from .x import *`           | `asyncio`, `sqlite3`   | reached — the star targets are scanned                 |
| re-export via `__all__`      | `pydantic`, `mcp`      | reached — one hop to the defining module               |
| lazy `__getattr__` (PEP 562) | `litestar`             | **no** — the name-to-module map is computed at runtime |
| compiled extension           | `orjson`, numpy ufuncs | **no** — there is no Python source                     |
| runtime-generated class      | metaclass output       | **no** — it does not exist until import runs           |

`introspect = true` imports each package and reads signatures off the live objects, which reaches all three. It also
runs that package's import-time code inside the indexer, and in the wild that means network calls, CUDA initialisation
and thread spawning. Turn it on only for an environment whose dependencies you trust. A package that raises on import is
skipped and keeps its static read.

## .atlasignore

File exclusions use `.atlasignore` (same syntax as `.gitignore`):

```
# Generated code
*_pb2.py
*_pb2_grpc.py
# Vendored deps
vendor/
# Migration history
migrations/
```

## Environment Variables

All settings can be overridden with environment variables using the `ATLAS_` prefix and double-underscore nesting:

```bash
ATLAS_EMBEDDINGS__MODEL=nomic-ai/CodeRankEmbed
ATLAS_EMBEDDINGS__BASE_URL=http://localhost:8080
ATLAS_SEARCH__DEFAULT_TOKEN_BUDGET=8000
```

## Choosing an Embedding Model

Code Atlas generates embeddings locally via [TEI](https://github.com/huggingface/text-embeddings-inference) (Text
Embeddings Inference). Any TEI-compatible model works — swap it by changing `[embeddings] model` in `atlas.toml` and
restarting the TEI container.

The right model depends on your hardware and priorities. All recommendations below are open-source, Apache 2.0 or MIT
licensed, and natively supported by TEI.

### Small — fast, runs on CPU

**[CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)** (Nomic AI) — 137M params, 768-dim, MIT

The default recommendation. Punches far above its weight: its CodeSearchNet MRR of 77.9 beats models 10x its size,
including CodeSage-Large at 1.3B parameters. Trained on 21M code examples from the CoRNStack dataset. Runs comfortably
on CPU or any GPU.

```toml
[embeddings]
model = "nomic-ai/CodeRankEmbed"   # 137M, 768-dim
```

Runner-up: [jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) (137M, 768-dim,
Apache 2.0) — covers 30 programming languages, slightly lower code retrieval scores.

### Medium — balanced quality and speed

**[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** (Alibaba) — 509M params, up to 1024-dim,
Apache 2.0

A decoder-based model with 32K context length and Matryoshka dimension support (configurable 32–1024). Trained on 100+
natural and programming languages. Strong on mixed workloads where you search both code and documentation. Requires a
modest GPU (4–6 GB VRAM).

```toml
[embeddings]
model = "Qwen/Qwen3-Embedding-0.6B"   # 509M, 1024-dim
```

Runner-up: [Snowflake Arctic Embed M v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) (305M,
768-dim, Apache 2.0) — general-purpose retrieval model with native TEI support; good if you want to stay at 768
dimensions.

### Large — best quality, dedicated GPU

**[nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code)** (Nomic AI) — 7B params, 4096-dim, Apache 2.0

Current open-source state of the art for code retrieval. Outperforms proprietary models like Voyage Code 3 and OpenAI
Embed 3 Large on CodeSearchNet benchmarks. Based on Qwen2 architecture with native TEI support. Requires a GPU with 16+
GB VRAM (or quantized via [GGUF](https://huggingface.co/nomic-ai/nomic-embed-code-GGUF)).

```toml
[embeddings]
model = "nomic-ai/nomic-embed-code"   # 7B, 4096-dim
```

Runner-up: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (7.6B, up to 4096-dim, Apache 2.0) —
ranked #2 on the overall MTEB leaderboard; best choice when you need a single model for mixed code, documentation, and
natural-language workloads.

**Shared inference for teams.** In an enterprise setting, a single machine with a capable GPU (e.g., NVIDIA A100 or
H100) can run TEI as a shared service for the entire team. Point every developer's `base_url` at the central server
instead of running a local container — one powerful machine replaces dozens of individual GPU setups, and embedding
results are cached by content hash so repeated queries across team members are essentially free.

```toml
[embeddings]
model = "nomic-ai/nomic-embed-code"
base_url = "https://tei.internal.company.com"   # shared on-premise TEI instance
```

### Quick Comparison

|                   | Small         | Medium               | Large            |
| ----------------- | ------------- | -------------------- | ---------------- |
| **Model**         | CodeRankEmbed | Qwen3-Embedding-0.6B | nomic-embed-code |
| **Parameters**    | 137M          | 509M                 | 7B               |
| **Dimensions**    | 768           | 1024                 | 4096             |
| **Context**       | 8K tokens     | 32K tokens           | 32K tokens       |
| **Hardware**      | CPU / any GPU | GPU (4–6 GB)         | GPU (16+ GB)     |
| **Code-specific** | Yes           | Mixed (code + NL)    | Yes              |
| **License**       | MIT           | Apache 2.0           | Apache 2.0       |
