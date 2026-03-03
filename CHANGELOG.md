# CHANGELOG

<!-- version list -->

## v0.2.0-dev.2 (2026-02-23)

### Bug Fixes

- **indexing**: Label-constrained queries, PEL reclaim, and drain progress
  ([`9ec9150`](https://github.com/SerPeter/code-atlas/commit/9ec915088058b05396a506ca23bfd0a0492d67f6))


## v0.2.0-dev.1 (2026-02-23)

### Bug Fixes

- Address PR #2 review — schema logging, concurrency tests, duration
  ([#2](https://github.com/SerPeter/code-atlas/pull/2),
  [`5107b24`](https://github.com/SerPeter/code-atlas/commit/5107b24a7dfbcb44cadc7917f632ae6a9743c057))

- **ci**: Resolve ty check failures with --all-extras in CI
  ([`3f74816`](https://github.com/SerPeter/code-atlas/commit/3f7481635091d2d676aed75c3fbcaa5db4332242))

- **consumers**: Group batches by project in AST/Embed consumers
  ([#2](https://github.com/SerPeter/code-atlas/pull/2),
  [`5107b24`](https://github.com/SerPeter/code-atlas/commit/5107b24a7dfbcb44cadc7917f632ae6a9743c057))

### Build System

- **release**: Add python-semantic-release v10 for automated releases
  ([#5](https://github.com/SerPeter/code-atlas/pull/5),
  [`de9c06d`](https://github.com/SerPeter/code-atlas/commit/de9c06d8c1e1744a92ff7ef160e06f859b5aa768))

### Features

- **analysis**: Add quality sub-analysis with health score
  ([#4](https://github.com/SerPeter/code-atlas/pull/4),
  [`37f2113`](https://github.com/SerPeter/code-atlas/commit/37f2113efa2ed776dec97a685b9e199014fb7f4a))

- **analysis**: Quality sub-analysis with health score
  ([#4](https://github.com/SerPeter/code-atlas/pull/4),
  [`37f2113`](https://github.com/SerPeter/code-atlas/commit/37f2113efa2ed776dec97a685b9e199014fb7f4a))

- **docker**: Add McpSettings, multi-stage Dockerfile, and .dockerignore
  ([`88ac0ba`](https://github.com/SerPeter/code-atlas/commit/88ac0ba93d2fc4d56fb362805c257c77e978351e))

- **parser**: Type-only import distinction + USES_TYPE edges
  ([#4](https://github.com/SerPeter/code-atlas/pull/4),
  [`37f2113`](https://github.com/SerPeter/code-atlas/commit/37f2113efa2ed776dec97a685b9e199014fb7f4a))

- **parser**: Type-only import distinction + USES_TYPE edges
  ([#3](https://github.com/SerPeter/code-atlas/pull/3),
  [`23ba25c`](https://github.com/SerPeter/code-atlas/commit/23ba25cbe0417b7d4816387744e5c51f4331a618))

- **scope**: Ruff-style include/exclude pattern system
  ([#2](https://github.com/SerPeter/code-atlas/pull/2),
  [`5107b24`](https://github.com/SerPeter/code-atlas/commit/5107b24a7dfbcb44cadc7917f632ae6a9743c057))

- **search**: Add label boost and code_only filter to hybrid search
  ([#2](https://github.com/SerPeter/code-atlas/pull/2),
  [`5107b24`](https://github.com/SerPeter/code-atlas/commit/5107b24a7dfbcb44cadc7917f632ae6a9743c057))

### Performance Improvements

- **indexing**: Concurrent embeddings, shared monorepo pipeline, clean logs
  ([#2](https://github.com/SerPeter/code-atlas/pull/2),
  [`5107b24`](https://github.com/SerPeter/code-atlas/commit/5107b24a7dfbcb44cadc7917f632ae6a9743c057))

### Refactoring

- **cli**: Shift verbosity levels — silent default, logs behind -v
  ([#2](https://github.com/SerPeter/code-atlas/pull/2),
  [`5107b24`](https://github.com/SerPeter/code-atlas/commit/5107b24a7dfbcb44cadc7917f632ae6a9743c057))


## v0.1.2 (2026-02-17)

### Bug Fixes

- Bump version to 0.1.2 in pyproject.toml and update uv.lock
  ([`f0baf60`](https://github.com/SerPeter/code-atlas/commit/f0baf6030a522b869e5b4cfcc066b8321714ee8a))

- Prevent Memgraph OOM crashes with Docker memory limit
  ([`e4dc6a9`](https://github.com/SerPeter/code-atlas/commit/e4dc6a93b3784a420cdd496e9c9b0d08ae2505ff))

- **cli**: Clean up atlas index output with progress bar and quieter logs
  ([`a0104cf`](https://github.com/SerPeter/code-atlas/commit/a0104cf978f469cf7702fad7f67ab82eed773c92))

- **doctor**: Show loaded config file paths in diagnostics
  ([`21f7bf2`](https://github.com/SerPeter/code-atlas/commit/21f7bf26ea36837a9a42b51b76ef2dfb488bb0e7))

- **graph**: Replace OR scans with uid-only lookups in embedding queries
  ([`71e5cef`](https://github.com/SerPeter/code-atlas/commit/71e5cefbb0ddc027392d41422b5c326c952ebeb5))

### Chores

- Update cryptography package version to 46.0.5 in uv.lock
  ([`aea84db`](https://github.com/SerPeter/code-atlas/commit/aea84dbcd73914b8423b5f3b4f3a215ff01455a0))


## v0.1.1 (2026-02-16)

### Bug Fixes

- Bump version to 0.1.1 in pyproject.toml
  ([`9889860`](https://github.com/SerPeter/code-atlas/commit/988986038b9faefe994f8cf21f99d8a531cc9c9b))

- Load .env from cwd so uvx installs pick up API keys
  ([`68e1f2c`](https://github.com/SerPeter/code-atlas/commit/68e1f2c234ab07fcc19f9973bc7760b670c21cc5))

### Documentation

- Update Quick Start for PyPI install via uvx
  ([`5ce16cb`](https://github.com/SerPeter/code-atlas/commit/5ce16cbda7fab88184205c7f65f12fbbf22c15f5))


## v0.1.0 (2026-02-15)

### Bug Fixes

- Bump version to 0.1.0 in pyproject.toml
  ([`222542a`](https://github.com/SerPeter/code-atlas/commit/222542ab5a698c8c32bbe4f69473567114fce712))

- Replace placeholder username with SerPeter and rename PyPI package to code-atlas-mcp
  ([`08a438f`](https://github.com/SerPeter/code-atlas/commit/08a438fa62a2a77b729a3787367484a0f8b69898))

- Replace placeholder username with SerPeter and rename PyPI package to code-atlas-mcp
  ([`0a83fd4`](https://github.com/SerPeter/code-atlas/commit/0a83fd4b1fb4f4d424d3170e27b2815ee072ea5f))

- Resolve end-to-end indexing and search pipeline issues
  ([`04bce9a`](https://github.com/SerPeter/code-atlas/commit/04bce9a239162923987882e8a32ba42f7ca9de90))

- Resolve live MCP tool testing issues
  ([`0a4a9ec`](https://github.com/SerPeter/code-atlas/commit/0a4a9ecda02bc07e01d5c8f78ec69c163ed5556a))

- **graph**: Tighten CALLS strategy 4 to unique-name-only matching
  ([`e3d8238`](https://github.com/SerPeter/code-atlas/commit/e3d82385f9255742073f7ee1cd621cbc05022a69))

- **mcp**: Resolve CALLS edges, ranking noise, staleness, and visibility boost
  ([`c7e788e`](https://github.com/SerPeter/code-atlas/commit/c7e788ef4a107fdaec44368c86083e85eeea9201))

- **search**: Finalize Epic 2 — migrate to Memgraph 3.7 DDL and fix integration bugs
  ([`bceff4c`](https://github.com/SerPeter/code-atlas/commit/bceff4c34a92066dcffcfa5b11ab17cb0716e2e1))

### Chores

- Update dependencies, pre-commit hooks, and Docker images to latest
  ([`80f87eb`](https://github.com/SerPeter/code-atlas/commit/80f87ebacb679159e5241ff9d0ab08aa1a2d0085))

- Update setup-uv and CodeQL action versions in CI/CD workflows
  ([`25572cf`](https://github.com/SerPeter/code-atlas/commit/25572cf10cd5a4d7ac60286d2109525cf7661332))

### Documentation

- Add landscape, configuration, usage guides and enrich MCP tool schemas
  ([`951ddc8`](https://github.com/SerPeter/code-atlas/commit/951ddc8095ce4a0b01b7798d779ae7cbf182ce9f))

- Add task tracking section to CLAUDE.md and project documentation
  ([`8ddda04`](https://github.com/SerPeter/code-atlas/commit/8ddda044432322e6f1795ed03e40059c9337d113))

- Clean up references and move local planning to CLAUDE.local.md
  ([`b6c8788`](https://github.com/SerPeter/code-atlas/commit/b6c8788240474b09dd2b0c57ace14632afdf7b25))

- **adr**: Finalize ADR-0004 and ADR-0005, update architecture
  ([`0e8ad11`](https://github.com/SerPeter/code-atlas/commit/0e8ad11febe7ac6c43bff313456d138af235c3b9))

- **adr**: Refine ADR-0005 with deployment model research findings
  ([`8453162`](https://github.com/SerPeter/code-atlas/commit/8453162e67a62abb47c307e1eedb5bc61421e626))

- **guidelines**: Add repository guidelines for Code Atlas indexing
  ([`3ec3412`](https://github.com/SerPeter/code-atlas/commit/3ec34129a2eeb15576b551053e143b8a5e7be1c2))

### Features

- Scaffold project structure
  ([`1d9c2bb`](https://github.com/SerPeter/code-atlas/commit/1d9c2bbfc267c750ba22e280d5db731ee424388a))

- **cli**: Add global --quiet, --json, --verbose, --no-color output modes
  ([`098cfeb`](https://github.com/SerPeter/code-atlas/commit/098cfebf86f73fdd909cd7325e2ffc6f4eacf80b))

- **detectors**: Add pluggable pattern detector framework
  ([`fa5f39d`](https://github.com/SerPeter/code-atlas/commit/fa5f39df9304b44c866c1e6fcf7dac0d7aca9d90))

- **detectors**: Implement 6 core pattern detectors
  ([`eae3340`](https://github.com/SerPeter/code-atlas/commit/eae3340a3bc2b57e98b14bcf958a65956d806570))

- **docs**: Add heuristic doc-code linking via DOCUMENTS edges
  ([`f8672bd`](https://github.com/SerPeter/code-atlas/commit/f8672bd0487d7063b7fdbb81a05d84bf8a5efd76))

- **docs**: Add markdown parser with tree-sitter-markdown
  ([`e8d372c`](https://github.com/SerPeter/code-atlas/commit/e8d372c162652d6d73d1f66da5e14a61fcb2136a))

- **embeddings**: Add EmbedClient with litellm routing and embed pipeline
  ([`ad7c972`](https://github.com/SerPeter/code-atlas/commit/ad7c9726f2e48fdb8746b50547089c5c483bcb75))

- **embeddings**: Add three-tier embedding cache with Valkey backend
  ([`9bc9b4c`](https://github.com/SerPeter/code-atlas/commit/9bc9b4ce6997e1a47fa70b96d1a632cff84de152))

- **embeddings**: Handle embedding model changes robustly
  ([`c03990b`](https://github.com/SerPeter/code-atlas/commit/c03990bea435fb3ade935ef91a364aa0c9fa40d0))

- **graph**: Add library stub resolution for external dependencies
  ([`699fc1b`](https://github.com/SerPeter/code-atlas/commit/699fc1bc2163053a3d2e78ffe111992ce6ecd113))

- **graph**: Add query timeout protection for read queries
  ([`4c15338`](https://github.com/SerPeter/code-atlas/commit/4c153386fe40ce86015d92513ae4e48d8686b604))

- **health**: Add health/doctor CLI commands and MCP health_check tool
  ([`3f0a761`](https://github.com/SerPeter/code-atlas/commit/3f0a76189106997161cf24ff887577b9a4622b6c))

- **indexer**: Add FileScope class with nested .gitignore support
  ([`becbe91`](https://github.com/SerPeter/code-atlas/commit/becbe91bea050a2d353d6e543a83998200c855c6))

- **indexer**: Add git-based staleness detection for query results
  ([`7ddd19f`](https://github.com/SerPeter/code-atlas/commit/7ddd19fdf0d337c29f8bd4d61ea95aae218405fc))

- **indexer**: Implement Python indexer with atlas index and status commands
  ([`8a8c3dd`](https://github.com/SerPeter/code-atlas/commit/8a8c3dd8caacfa9fc61c08c44a429900a11112f6))

- **infra**: Add Valkey to Docker stack and Redis settings
  ([`a566e01`](https://github.com/SerPeter/code-atlas/commit/a566e01cffcf3cb9abfce8151eccd8a1653ea03c))

- **mcp**: Add 5-stage matching cascade and disambiguation ranking to get_node
  ([`5a0043f`](https://github.com/SerPeter/code-atlas/commit/5a0043ffd4a34322f1e129fb0cb7a11fff43c655))

- **mcp**: Add detail parameter to search tools for compact/full output modes
  ([`e6756c1`](https://github.com/SerPeter/code-atlas/commit/e6756c10122865c1bb27abfb5dd538c241fc5973))

- **mcp**: Add MCP Roots support with git root fallback
  ([`546b243`](https://github.com/SerPeter/code-atlas/commit/546b243dec3fa9f1bd9c48c535ee8fbd004c823d))

- **mcp**: Add subagent guidance tools for AI coding agents
  ([`3bd3ab2`](https://github.com/SerPeter/code-atlas/commit/3bd3ab26d057c2de2fcabd5dbec77ec6e404c02d))

- **mcp**: Implement MCP server with 7 tools for AI agent access
  ([`ea19cf8`](https://github.com/SerPeter/code-atlas/commit/ea19cf81c29ea08d964c41612f26c79b9bcea700))

- **monorepo**: Add monorepo support with sub-project detection and cross-project resolution
  ([`5d8c029`](https://github.com/SerPeter/code-atlas/commit/5d8c029d849f46d929be57f7a7614dcf57f07a68))

- **naming**: Worktree-aware naming and monorepo sub-project prefixing
  ([`2acdfb3`](https://github.com/SerPeter/code-atlas/commit/2acdfb33ba4b486f966272a01cf8a37f670661f6))

- **parser**: Add py-tree-sitter parser, implement AST pipeline, drop Rust
  ([`d56e7d2`](https://github.com/SerPeter/code-atlas/commit/d56e7d2a686ec279a52d85bbc4903f4d85f51a4e))

- **parsing**: Add multi-language support (10 languages, 7 modules)
  ([`2c53ec8`](https://github.com/SerPeter/code-atlas/commit/2c53ec8b5c406fc21e1484aadfba723e2966c664))

- **parsing**: Add Python meta-programming quick wins
  ([`7e57c1d`](https://github.com/SerPeter/code-atlas/commit/7e57c1da0b347f906b1d54b995a70af6ca973869))

- **pipeline**: Add AST diffing with content-hash delta upsert
  ([`c265cd1`](https://github.com/SerPeter/code-atlas/commit/c265cd1dd68df376726d987841176df8d08f8b2d))

- **pipeline**: Add event bus and tiered consumer prototype
  ([`42d5983`](https://github.com/SerPeter/code-atlas/commit/42d5983a4f28bbc8a21593801c496b7e7c48bc48))

- **pipeline**: Add git-based delta indexing with file-level change detection
  ([`368973c`](https://github.com/SerPeter/code-atlas/commit/368973ca46baaa99a2d40ddc339edb6bf55b29be))

- **pipeline**: Compute actual significance level for delta gating
  ([`62b5076`](https://github.com/SerPeter/code-atlas/commit/62b507618e8f83e07d3635b952f115e10cdf6a4d))

- **schema**: Implement Memgraph graph schema with language-agnostic model
  ([`ec310b2`](https://github.com/SerPeter/code-atlas/commit/ec310b268a156061e55a9a1b335ec28ff85b0c56))

- **search**: Add configurable post-fusion result filtering
  ([`39f9ac3`](https://github.com/SerPeter/code-atlas/commit/39f9ac36ce1a1498319f68cda9180ade0fdd25f8))

- **search**: Add context expander with parallel Cypher queries
  ([`6edb1f2`](https://github.com/SerPeter/code-atlas/commit/6edb1f29c26aa6f89b8358d57fc3daa8ccd44611))

- **search**: Add full-body source text to entity indexing
  ([`3fbe54b`](https://github.com/SerPeter/code-atlas/commit/3fbe54bf30b65d57e6a2d7ae731583b45af061f3))

- **search**: Add hybrid search with RRF fusion across graph, vector, and BM25 channels
  ([`11977f8`](https://github.com/SerPeter/code-atlas/commit/11977f8387fb0054d70ad9d3f28da7f56171cde3))

- **search**: Add token-budget context assembly with priority ordering
  ([`e4b0bd2`](https://github.com/SerPeter/code-atlas/commit/e4b0bd216a48f3738a2fff6707fa28d29f137958))

- **search**: Add vector index DDL, scope/threshold filtering, and query cache
  ([`8f7e053`](https://github.com/SerPeter/code-atlas/commit/8f7e05305c2b4f692c0a59cc94f70aeeb248e14b))

- **server**: Add analyze_repo and generate_diagram MCP tools
  ([`ff875c5`](https://github.com/SerPeter/code-atlas/commit/ff875c5f6e91a5579e6c2b021d1fd0b9667a161a))

- **settings**: Add lightweight mode (no embeddings)
  ([`57d290c`](https://github.com/SerPeter/code-atlas/commit/57d290c12b5fa2ff8111d1e2ded647dc47562cb8))

- **telemetry**: Add OpenTelemetry tracing and metrics integration
  ([`e1544ce`](https://github.com/SerPeter/code-atlas/commit/e1544ce490532a8a32fbdc8d23b2313c9f6a7193))

- **watcher**: Add file watcher with hybrid debounce for real-time indexing
  ([`95b4757`](https://github.com/SerPeter/code-atlas/commit/95b4757593c4e81556b0ed43e38e20a5145d77d0))

### Performance Improvements

- Fix event loop blocking + add benchmark suite
  ([`0774451`](https://github.com/SerPeter/code-atlas/commit/0774451ef1ec2145bce3aa8b5332fd536e2a903f))

- **graph**: Optimize graph_search with UNION ALL query
  ([`b90cd19`](https://github.com/SerPeter/code-atlas/commit/b90cd191a12b425694c90a807866a0a941ae819e))

### Refactoring

- Reorganize flat layout into feature-based package structure
  ([`e8c8366`](https://github.com/SerPeter/code-atlas/commit/e8c83664d42288de0439ecaca0d26b7dfc8cffde))

- **parsing**: Extract language-specific code into plugin system
  ([`be44dc7`](https://github.com/SerPeter/code-atlas/commit/be44dc7f716d598dafbf41e100a455e8a1d39925))

- **schema**: Remove unused ADR node label
  ([`2b54b86`](https://github.com/SerPeter/code-atlas/commit/2b54b86247ce562416d7881ee1a860364425967d))

- **tests**: Reorganize test directory to mirror source layout
  ([`fb63c57`](https://github.com/SerPeter/code-atlas/commit/fb63c577e8d075e4c2dd1986d81b3ece110585a8))

### Testing

- Prune 27 trivial unit tests that violate high-gear principle
  ([`a18af95`](https://github.com/SerPeter/code-atlas/commit/a18af950a419215e16891a5c0066a10bc8023d1e))

- **infra**: Add testcontainers for auto-managed integration test infrastructure
  ([`a6958d2`](https://github.com/SerPeter/code-atlas/commit/a6958d2bbbaccb825bf09d11f9fec3b0d5e95d35))

- **mcp**: Add lock and ignore staleness mode tests
  ([`69646d8`](https://github.com/SerPeter/code-atlas/commit/69646d84c46756e160d3c31e9286599544ac2ff1))

- **watcher**: Add end-to-end file change integration test
  ([`6c594df`](https://github.com/SerPeter/code-atlas/commit/6c594df1a6328de189f643c548457479927e9ae9))
