# Configuration

## atlas.toml

Create an `atlas.toml` in your project root to customize Code Atlas behavior:

```toml
[scope]
include_paths = ["services/auth", "services/billing", "libs/shared"]
exclude_patterns = ["*.generated.ts", "testdata/"]

[libraries]
full_index = ["my_company_shared_lib"]
stub_index = ["fastapi", "sqlalchemy"]

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
