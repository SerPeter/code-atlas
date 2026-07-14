# CHANGELOG

<!-- version list -->

## v0.4.0 (2026-07-14)

### Bug Fixes

- **analysis**: Derive cross-package coupling from real package names
  ([`655de39`](https://github.com/SerPeter/code-atlas/commit/655de398f32d30a7f198d3cc6827fd537792fc10))

- **graph**: Detect body-only edits and resolve cross-file references
  ([`3e417af`](https://github.com/SerPeter/code-atlas/commit/3e417afa92cd6f8a4a4c5176906dc82f1ef4cfcd))

- **graph**: Preserve cross-file edges on delete and stop BM25 crashes
  ([`6f30442`](https://github.com/SerPeter/code-atlas/commit/6f30442463377a06d4058118b57dce3cc1ad9c45))

- **indexing**: Load nested gitignores for the watcher, honor monorepo scope
  ([`ab3128b`](https://github.com/SerPeter/code-atlas/commit/ab3128ba06d78f291f947963edcd7b72e95ac830))

- **indexing**: Make the event pipeline durable and path-consistent
  ([`a8b0246`](https://github.com/SerPeter/code-atlas/commit/a8b02460043e3aaeab38513bfeb1bd47128f1334))

- **indexing**: Scope monorepo indexing and gate empty-scan deletion
  ([`ba1405d`](https://github.com/SerPeter/code-atlas/commit/ba1405d5d7de281341522cc056c7f96e9bddd7a8))

- **indexing**: Withhold file hashes until deferred edges resolve
  ([`dcdeecc`](https://github.com/SerPeter/code-atlas/commit/dcdeecc64137589e13765e6c7965cecb273b8f41))

- **infra**: Enable AOF persistence for the production Valkey event bus
  ([`3751c20`](https://github.com/SerPeter/code-atlas/commit/3751c20b81ce9ef24d64d044a03c0587ab445337))

- **parser**: Attach cross-file Go methods via receiver type name
  ([`b24b0d3`](https://github.com/SerPeter/code-atlas/commit/b24b0d353cb63de930468af5fd6a415b515808f3))

- **parser**: Capture multi-name Go var/const specs and exclude generics from USES_TYPE
  ([`81ca9d2`](https://github.com/SerPeter/code-atlas/commit/81ca9d2cf38ea947709e9b84276be1dc3bb1c9d4))

- **parser**: Disambiguate duplicate Markdown headings and fix line_end
  ([`a5c89c4`](https://github.com/SerPeter/code-atlas/commit/a5c89c4526f6373fb73b70712201b0a985f5fffb))

- **parser**: Disambiguate JVM overload uids with parameter signatures
  ([`e0fe188`](https://github.com/SerPeter/code-atlas/commit/e0fe188875dd492f7b49b1ee81a05a8b364cdf7c))

- **parser**: Extract C++ operator overloads and out-of-line nested definitions
  ([`5acea2a`](https://github.com/SerPeter/code-atlas/commit/5acea2a38745535a407558233b8ef01fdd96df8b))

- **parser**: Extract C++ templates, prototypes, and out-of-line methods
  ([`805ca54`](https://github.com/SerPeter/code-atlas/commit/805ca5447aed09e659630fed941a46e6354e1ef5))

- **parser**: Extract Ruby inline-visibility methods
  ([`b76f0bd`](https://github.com/SerPeter/code-atlas/commit/b76f0bdd87eb40ce8235c5c2f8b4a865f1696f58))

- **parser**: Extract TS interface heritage, re-exports, and decorators
  ([`1278201`](https://github.com/SerPeter/code-atlas/commit/1278201731ea058fb5f4a5f5f06e691b1c4a0eb9))

- **parser**: Fold JVM namespaces into qualified names, resolve Java imports
  ([`6199504`](https://github.com/SerPeter/code-atlas/commit/619950430c739301517154a69f2882cb539a587b))

- **parser**: Parse .tsx files with the TSX grammar
  ([`82a3db3`](https://github.com/SerPeter/code-atlas/commit/82a3db3acc9ddf458af4afbd3a1e080f988c158b))

- **parser**: Preserve compact Ruby class-path names for INHERITS matching
  ([`6449af5`](https://github.com/SerPeter/code-atlas/commit/6449af5a97ce6ab089f4b1cba688bb301b96ecd3))

- **parser**: Resolve nested-class names and stop plugin-load lockout
  ([`136e78c`](https://github.com/SerPeter/code-atlas/commit/136e78cd31e47d7c3c1be40b6df0292b1e1e7872))

- **parser**: Scope Rust associated types to their impl block
  ([`7cdc801`](https://github.com/SerPeter/code-atlas/commit/7cdc8011edae9b06c8f91fd4c17aaaacf5359011))

- **parser**: Walk braced PHP namespaces and namespace-qualify entity names
  ([`16efc39`](https://github.com/SerPeter/code-atlas/commit/16efc39b44bed00376d2a37c7a0112d7a3ca1cff))

- **parser**: Walk inline Rust modules and cross-file impl parents
  ([`a4c1d5e`](https://github.com/SerPeter/code-atlas/commit/a4c1d5eefcf4f4bfce63f0f350aec09d396b274e))

- **search**: Stop suppressing vector search and silently emptying scope filters
  ([`d78f86d`](https://github.com/SerPeter/code-atlas/commit/d78f86d136fd3f2122991d224b23f32448bd0dc0))

- **server**: Correct diagram scoping, cycle detection, and node-cap edge handling
  ([`d919f1e`](https://github.com/SerPeter/code-atlas/commit/d919f1e958f8be02220f0fd97c10b6eff6626b8d))

- **server**: Fix cypher_query serialization and scope defaults for monorepos
  ([`e5a8c7f`](https://github.com/SerPeter/code-atlas/commit/e5a8c7fc5019545c3c63bb7955dfdc998850d215))

- **server**: Validate label params and surface pipeline health honestly
  ([`5c35cbe`](https://github.com/SerPeter/code-atlas/commit/5c35cbea4f54a87d425030b371e3272bc8222601))

- **settings**: Resolve atlas.toml against project_root, not cwd
  ([`3c3b21c`](https://github.com/SerPeter/code-atlas/commit/3c3b21cdbf3f1d6e284465e5d157b2f532a34a30))

- **settings**: Scope nested config sections to prefixed env vars
  ([`f1aa4d9`](https://github.com/SerPeter/code-atlas/commit/f1aa4d9d0f809cc5913969c501d3572edd2bfda1))

### Chores

- Drop unused type-ignore directives
  ([`0f7b85a`](https://github.com/SerPeter/code-atlas/commit/0f7b85ab3cb8a037093f1eae3c1ab89f92000319))

- Sync uv.lock to the 0.3.1 version bump
  ([`5b5189b`](https://github.com/SerPeter/code-atlas/commit/5b5189bd4af3055fff224bf616a2d29e9a7e4034))

### Documentation

- Add atlas dream to the CLI commands list
  ([`b52d921`](https://github.com/SerPeter/code-atlas/commit/b52d921b107a886790fdffea00ec6e80d79c1342))

- **adr**: Record cross-file resolution, pipeline durability, and test isolation decisions
  ([`ada1ce1`](https://github.com/SerPeter/code-atlas/commit/ada1ce113111ef781b4cf7cbebea64de1f6b053f))

- **adr**: Record the Note vault schema decision
  ([`d93215c`](https://github.com/SerPeter/code-atlas/commit/d93215cc383e39732648cd4a966978be7095085b))

### Features

- **knowledge**: Add anchors + staleness resolution (Phase 3)
  ([`c22ebd6`](https://github.com/SerPeter/code-atlas/commit/c22ebd61677f098b5a3c727d9fb0640f98ade272))

- **knowledge**: Add dream-mode deterministic report (Phase 4)
  ([`47c4a6d`](https://github.com/SerPeter/code-atlas/commit/47c4a6d86851f73c9abcd4535f8e42186cd546ff))

- **knowledge**: Add Note vault foundations to the code graph (Phase 1)
  ([`1706a0a`](https://github.com/SerPeter/code-atlas/commit/1706a0acfeae3958b82f3f004390a0331f0dea4b))

- **knowledge**: Live global vault + polish (Phase 5)
  ([`9b40136`](https://github.com/SerPeter/code-atlas/commit/9b40136091ef147deac443c55b06e2e478facac4))

- **knowledge**: Poll and index extra vaults from the daemon (Phase 2)
  ([`ef750bb`](https://github.com/SerPeter/code-atlas/commit/ef750bbb46d658c0232b4a8f32734169d915969e))

### Testing

- **integration**: Isolate tests from production databases
  ([`a1bdecd`](https://github.com/SerPeter/code-atlas/commit/a1bdecd4bdfa8225e894442bd62c0facc79e6654))


## v0.3.1 (2026-03-07)

### Bug Fixes

- **ci**: Use GitHub App token to push to protected main branch
  ([`13b46b9`](https://github.com/SerPeter/code-atlas/commit/13b46b972f900e4d1170b9c42af0c40d9c0cf35c))


## v0.3.0 (2026-03-04)

### Features

- **indexing**: Add file hash gate to skip unchanged files
  ([#9](https://github.com/SerPeter/code-atlas/pull/9))
- **indexing**: Add per-file cooldown for daemon mode
  ([#9](https://github.com/SerPeter/code-atlas/pull/9))

### Performance Improvements

- Reduce RTTs across indexing and query pipelines
  ([`cf3a519`](https://github.com/SerPeter/code-atlas/commit/cf3a519))

### Refactoring

- **indexing**: Eliminate Tier 1 consumer, simplify to two-tier pipeline
  ([#9](https://github.com/SerPeter/code-atlas/pull/9))
- **indexing**: Rename Tier 2/3 to AST/Embed stage across code and docs
  ([#9](https://github.com/SerPeter/code-atlas/pull/9))

### Bug Fixes

- **ci**: Remove detached HEAD checkout in release workflow

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
