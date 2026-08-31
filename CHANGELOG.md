# CHANGELOG

<!-- version list -->

## v0.11.0 (2026-08-31)

### Documentation

- **adr**: Accept ADR-0042 and document the flag split
  ([`e80fdf5`](https://github.com/SerPeter/code-atlas/commit/e80fdf5a6220a115b06840b916a0969aaa80f925))

### Features

- **cli**: Separate reindex scope from destruction
  ([`ffdb7a8`](https://github.com/SerPeter/code-atlas/commit/ffdb7a8665e14306394614f60177324649dea15e))

- **graph**: Count what a destructive run would remove, before it starts
  ([`4b7f9f8`](https://github.com/SerPeter/code-atlas/commit/4b7f9f850434b3559bd7ed2c1d051f4a74aeb733))


## v0.10.4 (2026-08-31)

### Documentation

- **adr**: Separate reindex scope from destruction
  ([`3ede0dc`](https://github.com/SerPeter/code-atlas/commit/3ede0dcad4d27d3777776245c3007b9d8d6263bd))

### Performance Improvements

- **graph**: Resolve import-scope references from the module, not the name
  ([`6cbb4c0`](https://github.com/SerPeter/code-atlas/commit/6cbb4c0bb3f1974c1c4f7933c3458e4fe2536ea2))


## v0.10.3 (2026-08-30)

### Bug Fixes

- **indexing**: Raise max_source_chars so oversized code can actually chunk
  ([`73f8b3e`](https://github.com/SerPeter/code-atlas/commit/73f8b3eefa9306448868db8ccf2ca184fca8d02b))


## v0.10.2 (2026-08-30)

### Bug Fixes

- **indexing**: Extend the file-hash gate to documents
  ([`31e7e3b`](https://github.com/SerPeter/code-atlas/commit/31e7e3bdd5cd38df9c605a20248f4f7fb4fc4476))

- **watcher**: Stop a directory's modified event re-publishing its whole subtree
  ([`2ce20f3`](https://github.com/SerPeter/code-atlas/commit/2ce20f322f59842c9cfd246bf525cccebed10ec8))


## v0.10.1 (2026-08-30)

### Performance Improvements

- **indexing**: Resolve DOCUMENTS in the flush, not per file
  ([`5ab225e`](https://github.com/SerPeter/code-atlas/commit/5ab225e418ceea6c5a1d3758284e73aed6b3de80))


## v0.10.0 (2026-08-30)

### Bug Fixes

- **chunking**: Stop losing the tail past max_chunks, and keep fences balanced
  ([`1c4651c`](https://github.com/SerPeter/code-atlas/commit/1c4651c77e954263586db0a3f31cd6360f649f46))

- **cli**: Pair telemetry shutdown with init on every exit path
  ([`21af285`](https://github.com/SerPeter/code-atlas/commit/21af285a2f1458037bda7852a3795328a1c745d2))

- **embeddings**: Give EmbedClient a lifecycle and every site an owner
  ([`152c13a`](https://github.com/SerPeter/code-atlas/commit/152c13addd1afe917589adf27c83131035372a93))

- **graph**: Keep embed chunks out of the queries that ask about code
  ([`c584a7f`](https://github.com/SerPeter/code-atlas/commit/c584a7f9560ed2f54f554aaa4b9001d772fde273))

- **graph**: Recheck pooled Bolt connections before reuse
  ([`02a859b`](https://github.com/SerPeter/code-atlas/commit/02a859bf68d1834d62e5abf2db64e713483cb754))

- **graph**: Refuse to drop vector indices of an embedded graph
  ([`deab48c`](https://github.com/SerPeter/code-atlas/commit/deab48c1c2f11c1fcd7992d15a5bde38f9f89c35))

- **indexing**: Bound the daemon's catch-up lease wait, not the CLI's
  ([`e5c9df3`](https://github.com/SerPeter/code-atlas/commit/e5c9df37a93d108453e60b9b9e488dc766332e17))

- **mcp**: Let --index/--no-index override the configured value in both directions
  ([`8fc9825`](https://github.com/SerPeter/code-atlas/commit/8fc98258a4cb979c08b125b3ab4b958cb10c3878))

- **parsing**: A constant's docstring lands on its Value node
  ([`29cfe85`](https://github.com/SerPeter/code-atlas/commit/29cfe859daf45956fe7bddd5dba972acbf3d4844))

- **parsing**: A grandchild span was eating its parent's own code
  ([`f45366b`](https://github.com/SerPeter/code-atlas/commit/f45366bda58342819b47320c8fa8840cbcd9dc4f))

- **parsing**: Dbt is Jinja-templated SQL, so run the whole SQL path on it
  ([`3c32f6d`](https://github.com/SerPeter/code-atlas/commit/3c32f6dc85f753de6668fd1fa2470b8f4133109f))

- **parsing**: Index every string a Salesforce component declares, not one
  ([`2066693`](https://github.com/SerPeter/code-atlas/commit/2066693146f7e65b1d813af73727f00a6e2860bc))

- **parsing**: Module docstrings land on the module node
  ([`8ab463e`](https://github.com/SerPeter/code-atlas/commit/8ab463eaf44afc72833d0fec24fd210c598e7669))

- **parsing**: Recognise .each suites, and stop suites duplicating their cases
  ([`1fd906e`](https://github.com/SerPeter/code-atlas/commit/1fd906ebe7e9fb693dba5fa3fad195a93139d988))

- **parsing**: Store markdown frontmatter as one queryable map
  ([`ed64600`](https://github.com/SerPeter/code-atlas/commit/ed6460036bfc1c148a6bdbfef56d2d749a057b42))

- **telemetry**: Resolve tracers lazily so spans survive import-before-init
  ([`5ec1abd`](https://github.com/SerPeter/code-atlas/commit/5ec1abdd0e85154dd0b18065bd3083ede56c9480))

- **test**: Assert the port contract, not two consecutive numbers
  ([`30309bc`](https://github.com/SerPeter/code-atlas/commit/30309bc65a45939d94edeb9fae88daad0fd5b90a))

- **test**: Close infra clients before skipping, and stop erroring on ResourceWarning
  ([`0b4a81c`](https://github.com/SerPeter/code-atlas/commit/0b4a81cbc94627e12eebfa4c90099b47355adf6c))

- **test**: Scope the ResourceWarning exemption the port tests need
  ([`1ba3b34`](https://github.com/SerPeter/code-atlas/commit/1ba3b34c9d734e5119898a4565c5ec0cc1ccb151))

- **web**: Drop SO_REUSEADDR, which let two UIs claim one port on Linux
  ([`86b7ea1`](https://github.com/SerPeter/code-atlas/commit/86b7ea1a6abc12e2770d9612369c4fccc9cd29db))

### Build System

- **compose**: Add optional victoria telemetry profile
  ([`c4a95e6`](https://github.com/SerPeter/code-atlas/commit/c4a95e6135d783e093358124483781e1c95da63e))

- **deps**: Upgrade the toolchain and adopt ty's own suppression prefix
  ([`4160860`](https://github.com/SerPeter/code-atlas/commit/41608605b761b62f95a40d16169ab48cfb019ea7))

### Chores

- Shorten the ty pin comment
  ([`19eb4cf`](https://github.com/SerPeter/code-atlas/commit/19eb4cf7cc8fd1ae61306a762946895394eae5ea))

### Continuous Integration

- Assert the lockfile instead of silently rewriting it
  ([`ca6f9e6`](https://github.com/SerPeter/code-atlas/commit/ca6f9e6b3b0db526666d926416b47cb8311d2afb))

### Documentation

- Document `atlas index --watch` as the way to keep indexing out of sessions
  ([`fdfae81`](https://github.com/SerPeter/code-atlas/commit/fdfae81542f6a9ef04dba74c7ddf1ccbdb89bedc))

- Extend ADR-0037 with the multi-process story and the web UI
  ([`c32a598`](https://github.com/SerPeter/code-atlas/commit/c32a598b94040e5d90db7ad614243c2cd7f9d180))

- Record ADR-0037 and document the new telemetry and indexing options
  ([`3676388`](https://github.com/SerPeter/code-atlas/commit/3676388d6a21b5ef0fbe6b5fd21b2bc0a8a0ae67))

- Record the new test guardrails and the one still missing
  ([`752ce4c`](https://github.com/SerPeter/code-atlas/commit/752ce4c2a37dd8bdcf74ee62fe32637617396f7d))

- **adr**: Amend ADR-0040 — chunks are dedup sources after all
  ([`704fe0f`](https://github.com/SerPeter/code-atlas/commit/704fe0ffe51520bf22875ab9c57817fbe146b0a2))

- **adr**: Record ADR-0039 on frontmatter storage and importance
  ([`7b32b64`](https://github.com/SerPeter/code-atlas/commit/7b32b64bf24f3d42df6388cdc99eebffa662e882))

- **adr**: Record ADR-0040 on splitting oversized nodes
  ([`1db261e`](https://github.com/SerPeter/code-atlas/commit/1db261e563ce24720699599537ad65dcdbe1da17))

- **adr**: Record backend ownership and why ResourceWarning is fatal
  ([`b61c772`](https://github.com/SerPeter/code-atlas/commit/b61c7727a6d49134421aedbbe9aed06dc631c912))

- **wiki**: Close out the driver leaks, record the EmbedClient one
  ([`a06f426`](https://github.com/SerPeter/code-atlas/commit/a06f426c37fc46756a77673ee2ff4d346d9cab7b))

- **wiki**: Close out the leak note -- all of it was one class of bug
  ([`b86d1db`](https://github.com/SerPeter/code-atlas/commit/b86d1dbbea648d2e0883484b70ad77c381385875))

- **wiki**: Correct the driver claim, record the EmbedClient result
  ([`3ff4932`](https://github.com/SerPeter/code-atlas/commit/3ff49320601a1cafbdca610d905cbe006a8efa96))

- **wiki**: Record the aiosqlite thread-warning flake and its root cause
  ([`a6ad6c8`](https://github.com/SerPeter/code-atlas/commit/a6ad6c81886fd259f1705bd7e9dcd1b51fc6e7b8))

- **wiki**: Record what the ResourceWarning hunt actually found
  ([`9352041`](https://github.com/SerPeter/code-atlas/commit/935204183f016abe70b6599a0effc60b20593db1))

### Features

- **backends**: Let every client be used as an async context manager
  ([`cd5b8e9`](https://github.com/SerPeter/code-atlas/commit/cd5b8e95155cb71d619ce4ae916d9b0f09164232))

- **backends**: One owner for connections, and health-check the idle ones
  ([`80d57b2`](https://github.com/SerPeter/code-atlas/commit/80d57b222f9b80524069465dafe01276c8b2cb00))

- **cli**: Add `atlas index --watch` so a checkout can have a persistent indexer
  ([`302fa8f`](https://github.com/SerPeter/code-atlas/commit/302fa8f9fa7fc88f9b9f57d884d7157947ecb600))

- **embeddings**: Border-aware chunker and an explicit input-token cap
  ([`279a6b6`](https://github.com/SerPeter/code-atlas/commit/279a6b690b150e671849ad99c2752a18c3e2c519))

- **embeddings**: Several vectors per node, scored at its best chunk
  ([`89dd4f4`](https://github.com/SerPeter/code-atlas/commit/89dd4f49a54317b1103623b8e1fead44d38d2615))

- **events**: Wait for the indexer lease with jittered polling
  ([`3d69b4d`](https://github.com/SerPeter/code-atlas/commit/3d69b4d37459f39ee76d198d657236078833013f))

- **graph**: A version belongs to the dependency, not the package
  ([`a94fdca`](https://github.com/SerPeter/code-atlas/commit/a94fdca826ed6deb5a4c43924dc0eb78c24205e7))

- **mcp**: Add --no-index for extra sessions sharing a worktree
  ([`cbe42a7`](https://github.com/SerPeter/code-atlas/commit/cbe42a7b1ff124afd150b1db93aa697d9014c703))

- **parsing**: A test's title is the name that makes it findable
  ([`e6a4743`](https://github.com/SerPeter/code-atlas/commit/e6a4743f8028f55371690d64e11681365b063a23))

- **parsing**: Long Python string literals become nodes of their own
  ([`cec6adb`](https://github.com/SerPeter/code-atlas/commit/cec6adb64a57e8ade181bb1857983058593ef7c6))

- **parsing**: Recover macro-hidden C/C++ scopes by keeping the better parse
  ([`98839f7`](https://github.com/SerPeter/code-atlas/commit/98839f7f7a3e89e7495dbbe37cbdd79179c3caff))

- **parsing**: Split oversized doc sections into consecutive nodes
  ([`6fa2488`](https://github.com/SerPeter/code-atlas/commit/6fa248859b0ef50f1e8121c5c02adb3f3f903853))

- **parsing**: SQL CTEs become nodes of their own
  ([`1eb856c`](https://github.com/SerPeter/code-atlas/commit/1eb856c330590eb39cad37ee4d4d956389f7baab))

- **ratelimit**: Give RateLimiter the context-manager protocol
  ([`8b3f0de`](https://github.com/SerPeter/code-atlas/commit/8b3f0ded9e946d9c0ed8adde6ff2beaff74b3d33))

- **search**: A chunk hit says which part matched, and where
  ([`db881c8`](https://github.com/SerPeter/code-atlas/commit/db881c8e2b3fd39d8275e3ca387e2db65fa6998c))

- **search**: Path and frontmatter importance multipliers
  ([`e8eb231`](https://github.com/SerPeter/code-atlas/commit/e8eb23106b2e3ccf9ac938139b7f07294e27681f))

- **telemetry**: Instrument MCP tools, log export, and the indexing pipeline
  ([`651f4b2`](https://github.com/SerPeter/code-atlas/commit/651f4b2d002a92ce7cbcbb6e5de231a25398b4b8))

- **telemetry**: Tell overlapping atlas processes apart, and instrument the web UI
  ([`5b1a257`](https://github.com/SerPeter/code-atlas/commit/5b1a25720b54a96d32b61d09c31f302cb80e84da))

- **telemetry**: Time every stage, every parse and every graph round-trip
  ([`aee3515`](https://github.com/SerPeter/code-atlas/commit/aee351521a886f4034f12bc996c1887725753085))

- **web**: Stop concurrent atlas ui invocations competing for one port
  ([`27eee1f`](https://github.com/SerPeter/code-atlas/commit/27eee1fee731f09b585287cc654e498b01a7ebaa))

### Performance Improvements

- **embeddings**: Let overflow chunks be dedup sources
  ([`3d5c0fd`](https://github.com/SerPeter/code-atlas/commit/3d5c0fdfee4ab62c4f55f14828af71a12f085514))

- **parsing**: Look up a claimed text-block name by index
  ([`51b1574`](https://github.com/SerPeter/code-atlas/commit/51b1574ef225f0e7b20762a9ae992f40a5ed1ca5))

- **parsing**: Stop indexing the same Python bytes under two entities
  ([`b4b638d`](https://github.com/SerPeter/code-atlas/commit/b4b638d1bf962375802bc3b699c7e963f86d511a))

### Refactoring

- **chunking**: Move the splitter where parsing can reach it
  ([`3336770`](https://github.com/SerPeter/code-atlas/commit/33367708c8e42b38b3644017335bdd6b5726e2cc))

- **cli**: Give index an explicit AsyncExitStack, finishing the conversion
  ([`0ee7a57`](https://github.com/SerPeter/code-atlas/commit/0ee7a57fed8685876165338e37aba4bf03fd8098))

- **cli**: Move search onto connected()
  ([`c502f03`](https://github.com/SerPeter/code-atlas/commit/c502f0381af6f67aa3e6f423f391370146847fa3))

- **cli**: Move status, project rm, mine-git-history and dream onto connected()
  ([`90d1f55`](https://github.com/SerPeter/code-atlas/commit/90d1f557b9d03fbb3429f0d26cd813fe116f7d3b))

- **cli**: Open backends through a scope instead of per-command plumbing
  ([`4233b21`](https://github.com/SerPeter/code-atlas/commit/4233b21ab6ccc03f6e0f8032bcec3ece1634506a))

- **health**: Require the connections instead of creating them
  ([`4cfacde`](https://github.com/SerPeter/code-atlas/commit/4cfacde95af8d0680392c81229f49b38d7bd374a))

- **indexing**: Inject the bus into DaemonManager instead of building one
  ([`93b9c7f`](https://github.com/SerPeter/code-atlas/commit/93b9c7fecd3855bacf29e379b200c9fe906779b6))

### Testing

- Add pytest-timeout and pytest-testmon
  ([`9a964fc`](https://github.com/SerPeter/code-atlas/commit/9a964fc48f6e9ca97b8acfa6afcc3756b352a10a))

- Adopt the pytest baseline, and fix the three deprecations it caught
  ([`d611212`](https://github.com/SerPeter/code-atlas/commit/d61121295836408679efe5ee385ac02f0dfb215a))

- Block off-box network access in unit tests
  ([`b1aa283`](https://github.com/SerPeter/code-atlas/commit/b1aa283ca5fbb507db4f155c1ed1e676b504ea4c))

- Let fixtures and blocks own the clients, not trailing close() lines
  ([`af218f7`](https://github.com/SerPeter/code-atlas/commit/af218f715be1786f3d1f891505e4a9c3e7ff9347))

- Put hypothesis, time-machine and codspeed to work
  ([`3ff1b5e`](https://github.com/SerPeter/code-atlas/commit/3ff1b5ef8dcdb975a166da9c99d9a75e31c07dd6))

- Stop ignoring ResourceWarning, now that nothing leaks
  ([`fdbc22a`](https://github.com/SerPeter/code-atlas/commit/fdbc22a9d31fab02ebeeb525ad0407c884ce6be6))

- Stop the suite waiting out production timeouts
  ([`f9cfd53`](https://github.com/SerPeter/code-atlas/commit/f9cfd53fb3a0debbe11caeaa3ef09c92805c7533))

- **cli**: Teach the graph doubles the new ensure_schema kwarg
  ([`625b865`](https://github.com/SerPeter/code-atlas/commit/625b8651561dcf5921f65dcb3cd43bbb553ae4d8))

- **daemon**: Stop the CLI wiring test reaching the production graph
  ([`944762f`](https://github.com/SerPeter/code-atlas/commit/944762ffcf35e7f4eae49bdfbabdbcfaf4a41475))

- **git-signals**: Let the CLI own its client instead of a patch that missed
  ([`8753e43`](https://github.com/SerPeter/code-atlas/commit/8753e43f9826563ea92cd50f7db785de729cb65a))

- **integration**: Scope the three clients whose setup ran outside the finally
  ([`17349c9`](https://github.com/SerPeter/code-atlas/commit/17349c96b98e38187dbe47280f880685a483b401))

- **langcov**: A third floor, for whether a file is findable at all
  ([`66e0f7e`](https://github.com/SerPeter/code-atlas/commit/66e0f7ede04016370561329c53607c4b6f021f42))

- **mcp**: Close the client the stubbed backend scope hands over
  ([`d592eaf`](https://github.com/SerPeter/code-atlas/commit/d592eafa8a9cb3a57cf63f58ce23b289fe824530))


## v0.9.0 (2026-08-28)

### Bug Fixes

- **embeddings**: Scope the model lock per project, keep dimension global
  ([`47b4263`](https://github.com/SerPeter/code-atlas/commit/47b4263d6dd85e7d1cb66b9ffd0497a6a505f58f))

- **graph**: Heal the :Entity marker instead of assuming it, on every ensure_schema
  ([`04b12cf`](https://github.com/SerPeter/code-atlas/commit/04b12cf340bed05a1723d9ab549a817a97b096c6))

- **graph**: Index uid-only lookups instead of scanning the whole graph
  ([`466deb8`](https://github.com/SerPeter/code-atlas/commit/466deb8e50b791a80f059e94dfdc2724f9f5461c))

- **graph**: Make the :Entity marker safe to read and cheap to write
  ([`06760f4`](https://github.com/SerPeter/code-atlas/commit/06760f41bd974195348a531d6eb6c04f4857bdbe))

- **graph**: Make the patient write path actually patient
  ([`7906c17`](https://github.com/SerPeter/code-atlas/commit/7906c17dfd3a495246ffe32187ab86cf541bf9d5))

- **graph**: Source typedef_names from every TypeDef, not the DEFINES join
  ([`d3b4237`](https://github.com/SerPeter/code-atlas/commit/d3b42377c67e60f5baaedf7d96a288148c8d48a0))

- **infra**: Give vector-index DDL a storage-access budget it can actually meet
  ([`e11672e`](https://github.com/SerPeter/code-atlas/commit/e11672e4746597992adf4dc48c72a915ff8d124d))

### Continuous Integration

- **release**: Regenerate uv.lock as part of the release commit
  ([`deae16e`](https://github.com/SerPeter/code-atlas/commit/deae16e00c5145e73029aa4bd39fc313dc2665ae))

### Features

- **dream**: Split dedup candidates into two bands and headline fragmentation
  ([`3a1eac9`](https://github.com/SerPeter/code-atlas/commit/3a1eac93cb11d3954355da60d26641f8bf4e7e6f))

- **embeddings**: Make the graph the dedup layer and delete the Valkey cache
  ([`3ff4a13`](https://github.com/SerPeter/code-atlas/commit/3ff4a135c9faeb3e036f653154e4f47a4c506e8a))

- **embeddings**: Pace provider calls with a shared budget and adaptive backoff
  ([`b633e2f`](https://github.com/SerPeter/code-atlas/commit/b633e2fba29d4b9444cfd288db65d994d2233922))

- **knowledge**: Demote superseded notes and flag unresolved contradictions
  ([`4e95c11`](https://github.com/SerPeter/code-atlas/commit/4e95c11ecc8770b46650e96479ff3ed62398bbfd))

- **mcp**: Every tool result names the backend that produced it
  ([`3ba83b4`](https://github.com/SerPeter/code-atlas/commit/3ba83b45963f5212000e6d45d205852e9cf8fea5))

- **parsing**: Index dbt models, sources and macros as a real DAG
  ([`a6da0aa`](https://github.com/SerPeter/code-atlas/commit/a6da0aa0adc2ca585f66859c36162001e5c50e2d))

### Performance Improvements

- **graph**: Index eleven more uid lookups that were scanning the whole graph
  ([`acbad29`](https://github.com/SerPeter/code-atlas/commit/acbad29e0143c286dbc462ed8c1616a3b35320e6))

- **graph**: Stop scanning the whole graph on the indexing hot path
  ([`b8b83f9`](https://github.com/SerPeter/code-atlas/commit/b8b83f9f4b6afddff9d4e2d09b6a0399e067755d))

- **indexing**: Bound embed writes instead of serialising them
  ([`96af241`](https://github.com/SerPeter/code-atlas/commit/96af2418a3deca54089221e342a51191bdeab6e9))

- **indexing**: Defer backlog replay to the final flush during a reindex
  ([`876be0d`](https://github.com/SerPeter/code-atlas/commit/876be0ddb64d91443dbc22d972e14ca9573be9d7))

- **indexing**: Pace resolution flushes by what the last one cost
  ([`24fb341`](https://github.com/SerPeter/code-atlas/commit/24fb341bbda2e89d6c4da3cc8c497bb402954505))

- **indexing**: Skip the no-op entity transaction on the detector pass
  ([`e34b2f9`](https://github.com/SerPeter/code-atlas/commit/e34b2f9b3bdec727aa86c820dbcb9d0e25800ef8))

### Testing

- **backends**: Conformance suite for the shared GraphBackend surface
  ([`4de6ba3`](https://github.com/SerPeter/code-atlas/commit/4de6ba36408598bce3fbc4e7b87512f9dfb39e9a))

- **bench**: Make the vector-search tripwire measure complexity, not co-tenancy
  ([`ef237c5`](https://github.com/SerPeter/code-atlas/commit/ef237c58cefe8728550ebf18b847a25a918aec46))

- **bench**: Size the vector-search budget for the load it actually runs under
  ([`a4d186b`](https://github.com/SerPeter/code-atlas/commit/a4d186b7690899edba4f58175808d601fe88ef1d))

- **integration**: Stamp :Entity on fixture nodes built by raw Cypher
  ([`530283e`](https://github.com/SerPeter/code-atlas/commit/530283e7d67af29902984a77baebfff599b47bcf))

- **integration**: Stamp :Entity on the remaining raw-Cypher fixtures
  ([`4ef9c03`](https://github.com/SerPeter/code-atlas/commit/4ef9c03550046c3c7a6c48b8e90481180fed6364))

- **parsing**: Pin the Jinja shim's offset invariant, and tell agents about dbt
  ([`d6f6878`](https://github.com/SerPeter/code-atlas/commit/d6f687878e72afe7cebbc99800b4ca6ae9d0bfe7))


## v0.8.4 (2026-08-27)

### Bug Fixes

- **indexing**: Make the drain timeout configurable, was hardcoded at 600s
  ([`29d0340`](https://github.com/SerPeter/code-atlas/commit/29d03403d413b28d4818d1d49af89fff5543c8f4))


## v0.8.3 (2026-08-27)

### Bug Fixes

- **indexing**: Translate scope.paths into each monorepo sub-project's own root
  ([`d6e46e4`](https://github.com/SerPeter/code-atlas/commit/d6e46e40179b11691b3988c0405dd945d7a90f8a))


## v0.8.2 (2026-08-27)

### Bug Fixes

- **indexing**: Gate daemon startup catch-up behind the indexer lease
  ([`29618f8`](https://github.com/SerPeter/code-atlas/commit/29618f841023a1b9ba9068b1f9fcd80a13d1afd7))


## v0.8.1 (2026-08-26)

### Bug Fixes

- **events**: Size stream_maxlen as a memory budget, not just a backlog guard
  ([`3054404`](https://github.com/SerPeter/code-atlas/commit/3054404e0bda8432b1ff3b6ba819a77476ba528b))

- **infra**: Raise Valkey ceiling to 2g/1536mb for the bus it actually carries
  ([`db6b351`](https://github.com/SerPeter/code-atlas/commit/db6b351d5a39eb13dd14ae3267ef8b264a32bbea))

- **search**: Forward configured dimension to litellm embedding calls
  ([`eb855f3`](https://github.com/SerPeter/code-atlas/commit/eb855f352a5ae6534802a278336491ac2c5a5533))

### Continuous Integration

- **release**: Pin semantic-release action to v10.6.1, not floating v10
  ([`294d54e`](https://github.com/SerPeter/code-atlas/commit/294d54e412b284673d02f14d9d3749b4e44c172e))

- **release**: Run semantic-release via uvx with a GitPython pin
  ([`ee53a6e`](https://github.com/SerPeter/code-atlas/commit/ee53a6efdbc38c0e8739a878366d2e59e329f1df))


## v0.8.0 (2026-08-09)

### Bug Fixes

- **backends**: Close three SQLite defects that produce wrong answers (ATL-112)
  ([`3dd231b`](https://github.com/SerPeter/code-atlas/commit/3dd231bb29ca294dc98772c9fd1b094ecdcaf232))

- **cli**: Make a partial index say so instead of reporting success (ATL-110)
  ([`f01fa61`](https://github.com/SerPeter/code-atlas/commit/f01fa6179085dedb0398930b101932471c171a57))

- **graph**: Resolve calls only within a call-namespace group (ATL-113)
  ([`abaa4f9`](https://github.com/SerPeter/code-atlas/commit/abaa4f984f34245c35eb7a70db345a4625b6a308))

- **index**: Verify the embedding dimension and fail closed on unknown staleness (ATL-111)
  ([`06aa2bb`](https://github.com/SerPeter/code-atlas/commit/06aa2bbc2bb3ffb5a7f532f45d2e1504310f7e70))

- **indexing**: Anchor staleness survives full re-indexes
  ([`eecf7f6`](https://github.com/SerPeter/code-atlas/commit/eecf7f67acdceece00d76f23ff597eb0adbe2aa7))

- **infra**: Size Valkey for the embedding cache it actually holds
  ([`9f9c1cc`](https://github.com/SerPeter/code-atlas/commit/9f9c1ccf65dd7479094d1f68f12c8b1e91bb470c))

- **layout**: Scale the simulation constants with node count
  ([`bfe43f3`](https://github.com/SerPeter/code-atlas/commit/bfe43f394d8c8bfe28e9d67f794e584dbfc4cb7d))

- **map**: Directory scopes get the whole graph, filters reach every view
  ([`d011d90`](https://github.com/SerPeter/code-atlas/commit/d011d90155f0de71457b9d69ea755dd3752f1505))

- **map**: Methods on by default, stranded externals out, honest names
  ([`95c6ef5`](https://github.com/SerPeter/code-atlas/commit/95c6ef57028f04447ff9685899995d3919ec6cdf))

- **map**: One node cap for every entity view
  ([`f97b3fa`](https://github.com/SerPeter/code-atlas/commit/f97b3facee446a4044cf4dbecd9ea085c063bb36))

- **parsing**: Attribute Go closure calls to the enclosing named scope (ATL-096)
  ([`43419d1`](https://github.com/SerPeter/code-atlas/commit/43419d175a64b9291939abf7c07c3debbeb6de6b))

- **parsing**: Decline an entity for a PHP closure bound to a local (ATL-096)
  ([`50c60c4`](https://github.com/SerPeter/code-atlas/commit/50c60c4bfdb0b7fbf6b804f78b10835755d8e02b))

- **parsing**: Decline TypeScript bindings local to an anonymous callback (ATL-107)
  ([`95b0d05`](https://github.com/SerPeter/code-atlas/commit/95b0d05ead5d0af35e738515e854199c69f2f5ff))

- **parsing**: Give C++ overloads distinct uids (ATL-107)
  ([`e6736a7`](https://github.com/SerPeter/code-atlas/commit/e6736a747405fe97491bf553e1c24760ac6c9c54))

- **parsing**: Name gtest cases from their macro arguments (ATL-096)
  ([`771ffa3`](https://github.com/SerPeter/code-atlas/commit/771ffa3cd9056c921ac63a47ed7436ca0ea689e2))

- **parsing**: Reach the Java and C# callables the walkers never visited (ATL-096)
  ([`74708e5`](https://github.com/SerPeter/code-atlas/commit/74708e58415faf06ca0ef100a9039b9e7de9b819))

- **parsing**: Route .h to C or C++ by sniffing its content (ATL-096)
  ([`e17d91e`](https://github.com/SerPeter/code-atlas/commit/e17d91e60e9820b54bdf2462e5772cef462f65d2))

- **parsing**: Separate Ruby singleton methods and decline block-nested defs (ATL-107)
  ([`020e34d`](https://github.com/SerPeter/code-atlas/commit/020e34d460c8375f4d69740480c6c6cafa5b7df3))

- **parsing**: Walk C++ preprocessor branches, class bodies and linkage blocks (ATL-096)
  ([`02d9e6b`](https://github.com/SerPeter/code-atlas/commit/02d9e6b15fef07c12db3d3c2c006172cdfc3b1fa))

- **parsing**: Walk PHP blocks and emit an edge for `new` (ATL-096)
  ([`c7c9714`](https://github.com/SerPeter/code-atlas/commit/c7c97149f5cda49bcb5b80693452184f7cb40221))

- **parsing**: Walk Ruby blocks and class<<self instead of stepping over them (ATL-096)
  ([`001965e`](https://github.com/SerPeter/code-atlas/commit/001965ecf9a650e9d9cc60a6c73abd11d3da6454))

- **parsing**: Walk Rust closures, nested fns and module scope (ATL-096)
  ([`adddc33`](https://github.com/SerPeter/code-atlas/commit/adddc33682c454634c6c6c9c9c09ce3e7de2cecb))

- **parsing**: Walk TypeScript scopes so every call reaches a named owner (ATL-096)
  ([`6f53a88`](https://github.com/SerPeter/code-atlas/commit/6f53a88440defb8facddde576538d1552afa35b1))

- **search**: Report the score that actually determined the order
  ([`ea1672d`](https://github.com/SerPeter/code-atlas/commit/ea1672da8ec407782b4115c8f4f7a85ac01911f6))

- **server**: Make the SQLite fallback announce itself (ATL-112)
  ([`751ff20`](https://github.com/SerPeter/code-atlas/commit/751ff20cd429aa85a9d3e5e1768f4b370cca7a18))

- **server**: Stop reporting a truncation count the search never computed (ATL-111)
  ([`7fdc537`](https://github.com/SerPeter/code-atlas/commit/7fdc537e8c19e0279bab794922090b4d265d6368))

- **settings**: Reject unknown keys inside a config section (ATL-111)
  ([`aba1b00`](https://github.com/SerPeter/code-atlas/commit/aba1b00f4993ea45ebc4e331d2e46baad190837f))

- **ui**: Port the design's own markup instead of rebuilding it from memory
  ([`20f7aec`](https://github.com/SerPeter/code-atlas/commit/20f7aecf41a6901a539adbfac12887e71b808725))

- **ui**: Read the real get_project_status shape, not the one the fake invented
  ([`28a1246`](https://github.com/SerPeter/code-atlas/commit/28a1246c9e480e5f69b66d86733bcfbf832d27fa))

- **ui**: The web search page showed the pre-boost score too
  ([`6e7ad82`](https://github.com/SerPeter/code-atlas/commit/6e7ad826e070a0f2c943b2c45c41f3ea6bf4c969))

### Build System

- Keep prettier off the Jinja templates and vendored bundles
  ([`6bbbd7e`](https://github.com/SerPeter/code-atlas/commit/6bbbd7ec6a1f0f8d93e0a44dab290e6e8464d942))

### Chores

- Catch uv.lock up to the released version
  ([`73dd4a6`](https://github.com/SerPeter/code-atlas/commit/73dd4a62c2b570bbe900d15116f549c8808e1ae7))

### Continuous Integration

- Gate releases on a CI run that actually tests the product (ATL-114)
  ([`8e4cd98`](https://github.com/SerPeter/code-atlas/commit/8e4cd986d78d39cd26cd1a71e273b8a8e4775d49))

- Isolate integration services from production ports, unshallow history
  ([`2da949c`](https://github.com/SerPeter/code-atlas/commit/2da949c12d1118ea5337f83961a882970fb3a0c6))

### Documentation

- **adr**: A uid must identify exactly one definition (ADR-0032)
  ([`b73d5bc`](https://github.com/SerPeter/code-atlas/commit/b73d5bc1e4944e2e9e20bea0eb4ff008a5e1edaf))

- **adr**: Correct what the ADR-0031 loss table actually measures
  ([`9282116`](https://github.com/SerPeter/code-atlas/commit/928211640ef905b5f208dd29fc963203ae38fcbf))

- **adr**: Record why architecture snapshots live on the Project node
  ([`02cb0cf`](https://github.com/SerPeter/code-atlas/commit/02cb0cf0860efcacd36e5e1fa99167b0637b4244))

- **adr**: Record why candidate hygiene is asymmetric (ADR-0030)
  ([`011b818`](https://github.com/SerPeter/code-atlas/commit/011b818eff023ac3aa38024099ed9056805e478c))

- **readme**: Document all 23 MCP tools and make the token figures reproducible
  ([`452c6ea`](https://github.com/SerPeter/code-atlas/commit/452c6eadad18f09821e9937d567c6a9fa710173e))

- **test**: Record what named_funcs cannot see
  ([`c2b7ce4`](https://github.com/SerPeter/code-atlas/commit/c2b7ce430b51ec687bf9fed9057948e28fc60372))

### Features

- **analysis**: Carry per-pair evidence up to the module graph
  ([`986f5d8`](https://github.com/SerPeter/code-atlas/commit/986f5d85a5405a194cf4d6088bccf1ce8ddd3824))

- **graph**: Schema v12 — re-index for uid discriminators (ATL-107)
  ([`7fe929b`](https://github.com/SerPeter/code-atlas/commit/7fe929b00d1375872387595d825ab2b0467ad2c9))

- **layout**: Community blobs, per-edge affinity, and measured spacing
  ([`dc2c4fa`](https://github.com/SerPeter/code-atlas/commit/dc2c4fabfff570d1af4b3eadca80ca393adc9305))

- **map**: Documentation files join the module level
  ([`e5e6182`](https://github.com/SerPeter/code-atlas/commit/e5e61828a52271d6238075a590d769b2e3479d16))

- **map**: The third-party boundary, entity communities, and a scope tree
  ([`c015b99`](https://github.com/SerPeter/code-atlas/commit/c015b9923ac52adb632e5e9ecd0589ed9af8084e))

- **parsing**: Constants get the REFERENCES edges they always deserved
  ([`79bf8f1`](https://github.com/SerPeter/code-atlas/commit/79bf8f1c7ad719914f2f7f03f9da1167b1232fce))

- **ui**: Add architecture-health view with DSM and propagation cost
  ([`c9f7542`](https://github.com/SerPeter/code-atlas/commit/c9f7542cab7a813bf33b57279f8b0d5132ac484d))

- **ui**: Add atlas ui --export for a self-contained HTML snapshot
  ([`66d0e3d`](https://github.com/SerPeter/code-atlas/commit/66d0e3d76fd5c5cec8add7772de3ab31e88cbcd2))

- **ui**: Add the blast radius and trace path explorer
  ([`31e9b54`](https://github.com/SerPeter/code-atlas/commit/31e9b54d3ffd0e9c37b78c28f171dbb0f065cff8))

- **ui**: Add the community map with a vendored sigma.js renderer
  ([`28c6796`](https://github.com/SerPeter/code-atlas/commit/28c679689ae657e87acfb27d0e44ae996d206537))

- **ui**: Adopt Claude Design v1.1 — two map levels and the real rail
  ([`4c6e2d1`](https://github.com/SerPeter/code-atlas/commit/4c6e2d1469f1426ccb0d3c970bfb069a4247f650))

- **ui**: Adopt the Claude Design visual system and app shell
  ([`60212d5`](https://github.com/SerPeter/code-atlas/commit/60212d581abf9d47d3d78b3a357ad654dfce4310))

- **ui**: Make the map show real structure — direction, position, names, scope
  ([`18f1491`](https://github.com/SerPeter/code-atlas/commit/18f14911ab3333c661bc31c09765f8065df26ece))

- **ui**: Map interaction polish and dense-scope readability
  ([`d2b84c4`](https://github.com/SerPeter/code-atlas/commit/d2b84c4521923d0f0fd69abd97d21ca64ba77d08))

- **ui**: Name the edges that close each dependency cycle
  ([`7aa00bd`](https://github.com/SerPeter/code-atlas/commit/7aa00bd8ce872348fd7b9b02a0b6646423e1f4e2))

- **ui**: Rebuild the web UI as a verbatim port of the v1.1 design
  ([`e783ca3`](https://github.com/SerPeter/code-atlas/commit/e783ca366419a4d6e9be90486d1b55a64a342b68))

- **ui**: Record architecture snapshots per index run and show the trend
  ([`c282892`](https://github.com/SerPeter/code-atlas/commit/c282892d8820496babcf81b6fa84846fec4e1e30))

- **ui**: Search entry and entity detail with edge evidence (ATL-116)
  ([`e948b17`](https://github.com/SerPeter/code-atlas/commit/e948b1749a113738aa3e07ac2fe84dd16f3dce06))

- **ui**: The entity level draws the whole project
  ([`6df3e80`](https://github.com/SerPeter/code-atlas/commit/6df3e8010404fa4822fff606d1d988539c01931e))

- **ui**: The map is the homepage, and projects are picked from a real page
  ([`a258ce7`](https://github.com/SerPeter/code-atlas/commit/a258ce73e95d8be13c0120ead866a86a47ed803e))

- **ui**: Three-layer web skeleton for atlas ui (ATL-115)
  ([`2bd07dd`](https://github.com/SerPeter/code-atlas/commit/2bd07dd5a1a4369be541438589b8e5e37acb91db))

- **ui**: Wire the renderer, theme axes, and map chrome to v1.1
  ([`ba33e32`](https://github.com/SerPeter/code-atlas/commit/ba33e32b8872ac207af67bf068ca5128d9d01264))

- **web**: Tests and guessed calls become stated, togglable populations
  ([`6b475f0`](https://github.com/SerPeter/code-atlas/commit/6b475f073b5d96db50880805e1513437aac645ea))

### Performance Improvements

- **graph**: Match CALLS sources on their own label index
  ([`0dd3f6c`](https://github.com/SerPeter/code-atlas/commit/0dd3f6c71e34d5fdd12f1c24fb77abd37afe4c4b))

- **graph**: Match citation DOCUMENTS endpoints on their own label index
  ([`b473727`](https://github.com/SerPeter/code-atlas/commit/b4737278b0b01c3f060ca7cdaf8df9bace82bff5))

- **graph**: Match IMPORTS endpoints on their own label index
  ([`eff5948`](https://github.com/SerPeter/code-atlas/commit/eff59483a214f7906dfa48cba39ae719635550fd))

### Refactoring

- **ui**: Extract the view sections into partials two renderers can share
  ([`8e60e47`](https://github.com/SerPeter/code-atlas/commit/8e60e474b1e14cfc4281915f147d502aef5a6da5))

### Testing

- **bench**: Give the synthetic corpus calls that actually resolve
  ([`6af9ff8`](https://github.com/SerPeter/code-atlas/commit/6af9ff8435848f177e42f4a6bfd264b00f3c19d3))

- **build**: Verify the distributions by building and reading them (ATL-114)
  ([`09db0f4`](https://github.com/SerPeter/code-atlas/commit/09db0f4066b97f757835314d9a1827c6020582a2))

- **graph**: Pin the call-candidate narrowing rules (ATL-113)
  ([`9b6bfef`](https://github.com/SerPeter/code-atlas/commit/9b6bfef0c9723e6135ae75b7ce27f2cd61dba817))

- **parsing**: Assert a per-language ceiling on colliding uids (ATL-096)
  ([`6d90358`](https://github.com/SerPeter/code-atlas/commit/6d9035800cfa1158a0f6ea3f025d3a6f970fb0b3))

- **parsing**: Count only forms that must be entities, and calls in statement position
  ([`dddd1a6`](https://github.com/SerPeter/code-atlas/commit/dddd1a69c315daea811358d9939ce250428ef83d))

- **parsing**: Measure extraction coverage against real code (ATL-096)
  ([`a896c14`](https://github.com/SerPeter/code-atlas/commit/a896c14a3fdeec1aa9911421c73c2c28a5c6ab1c))

- **parsing**: Stop asserting capture of definitions ADR-0032 declines
  ([`6acd308`](https://github.com/SerPeter/code-atlas/commit/6acd308b71e32c0959ba1cb3515b97eb7b889498))

- **settings**: Prove every section rejects unknown keys (ATL-111)
  ([`2674625`](https://github.com/SerPeter/code-atlas/commit/2674625b8a2c1deceae74824989594550b1965fc))


## v0.7.0 (2026-08-05)

### Bug Fixes

- **graph**: Stop test fixtures from diluting real call edges (ATL-103)
  ([`5d38d61`](https://github.com/SerPeter/code-atlas/commit/5d38d61c5622dd5fb933b443bd3b2a1101a904b5))

### Features

- **graph**: Point CALLS edges at the call site, not the def (ATL-105)
  ([`4ec2a1c`](https://github.com/SerPeter/code-atlas/commit/4ec2a1c6ecd65e364e0e3b2b8fd35c6841018288))

- **server**: Say what a truncated result withheld (ATL-104)
  ([`e7ed769`](https://github.com/SerPeter/code-atlas/commit/e7ed769e6fd64dd9e76370f20bf25f6046911429))


## v0.6.0 (2026-08-04)

### Bug Fixes

- **graph**: Stop vector search under-returning against a polluted index
  ([`87898f8`](https://github.com/SerPeter/code-atlas/commit/87898f838bb6a926a88c20d2cfe979f3c2fa7464))

- **server**: Make blast_radius traverse dependency, not execution (ADR-0029)
  ([`a5c387d`](https://github.com/SerPeter/code-atlas/commit/a5c387d1379581b072f321e35464448fb8e9a3bc))

### Features

- **graph**: Make every resolved edge state its evidence (ADR-0028)
  ([`26e5d16`](https://github.com/SerPeter/code-atlas/commit/26e5d16974eeab7a2c9bc41a01b9d879bc95dd4f))


## v0.5.0 (2026-08-04)

### Bug Fixes

- **graph**: A method that implements a Protocol is not dead code
  ([`f581e0c`](https://github.com/SerPeter/code-atlas/commit/f581e0c1d4f620df503fd050795f6417ab641a75))

- **graph**: Ask "is this a stub" of the method, not of its class
  ([`8e7d7e0`](https://github.com/SerPeter/code-atlas/commit/8e7d7e092b4705ecb5056cb19f1472b9799a58ac))

- **graph**: Damp the weight of an unverified-receiver call edge
  ([`0c8c9b2`](https://github.com/SerPeter/code-atlas/commit/0c8c9b21c1d501f852d3c4f6d17203e4d97ee944))

- **graph**: Five more dead-code exclusions from the triage fanout
  ([`ce29b60`](https://github.com/SerPeter/code-atlas/commit/ce29b60e5c83b610f143698017a8053880ee046a))

- **graph**: Give ResourceFile nodes their path and stop capturing directories
  ([`c856083`](https://github.com/SerPeter/code-atlas/commit/c8560835c4a3b2c3a382edc48607d5d906b333c1))

- **graph**: Recreate search indices that vanish at the current schema version
  ([`c17993d`](https://github.com/SerPeter/code-atlas/commit/c17993d4506e4a7d6619660788c65a13abe316d6))

- **graph**: Replay resolution that ran against a partial graph (ADR-0026)
  ([`d86f137`](https://github.com/SerPeter/code-atlas/commit/d86f1371e0ec9639f91abb6d01cdaf6075be9d2a))

- **graph**: Require a grounded receiver for the lexical strategies (ADR-0027)
  ([`c8edd3b`](https://github.com/SerPeter/code-atlas/commit/c8edd3b65a310b500f29cbc095e4ad42e746bd64))

- **graph**: Resolve a class's base even when the base is imported
  ([`fac5d1f`](https://github.com/SerPeter/code-atlas/commit/fac5d1f6235cec1158d5a11bec8b3c4f22575258))

- **graph**: Stop calling a nested function dead because nothing calls it by name
  ([`89a91e1`](https://github.com/SerPeter/code-atlas/commit/89a91e126afa4d4d11897bdf55298759440ae54f))

- **graph**: Stop resolving a call whose receiver was never identified
  ([`404ebf8`](https://github.com/SerPeter/code-atlas/commit/404ebf8f99b5f1844374c5ba1f5a280fde324175))

- **graph**: Write and read every file hash in a batch, not just the first
  ([`50a1bee`](https://github.com/SerPeter/code-atlas/commit/50a1bee077d6469505a5bbf085b9690bf5cb6e5f))

- **indexing**: Deregister consumers instead of leaking one per index run
  ([`3e333c4`](https://github.com/SerPeter/code-atlas/commit/3e333c4b3ab7d84a5ed71a72d732b3d6884f260d))

- **indexing**: Keep sweeping for abandoned work, and stop guessing past a known receiver type
  ([`70d060a`](https://github.com/SerPeter/code-atlas/commit/70d060a4a3556ecb625ae97c95023e633729d284))

- **indexing**: Port git-signals write path to GraphBackend
  ([`8fc1f18`](https://github.com/SerPeter/code-atlas/commit/8fc1f18147fb6df5998461f4af5598c42e9f0b6d))

- **indexing**: Reconcile lost embeddings, and fail the exit code when the index is incomplete
  ([`2164370`](https://github.com/SerPeter/code-atlas/commit/2164370cc752869afe4b805b4269ea016513e77e))

- **indexing**: Run protocol conformance once, after every batch is written
  ([`d012670`](https://github.com/SerPeter/code-atlas/commit/d0126703d887a5b73bf4c3332f60046b67e8f1c9))

- **indexing**: Stop a crash from stranding a stale citation edge
  ([`b67b19c`](https://github.com/SerPeter/code-atlas/commit/b67b19c330fb144719ae0114ed6972587f229f07))

- **indexing**: Stop the teardown timeout truncating the final flush
  ([`68ecddd`](https://github.com/SerPeter/code-atlas/commit/68ecddd26d473a8ac81f8fe94d8b3ef9a8d6a916))

- **indexing**: Stop two processes sharing one consumer identity
  ([`52430a0`](https://github.com/SerPeter/code-atlas/commit/52430a047cb162bf30774d496a39125b019e2e75))

- **indexing**: Wait on consumer progress instead of a teardown deadline
  ([`6b30b76`](https://github.com/SerPeter/code-atlas/commit/6b30b7699d8a94349a11e46b6f3efe1a66c57ed8))

- **indexing**: Withhold every file hash until the deferred flush
  ([`2671499`](https://github.com/SerPeter/code-atlas/commit/26714992ae3b29f3dd48feab0d5d751928a5d542))

- **parsing**: Resolve a re-exported name, not just a locally defined one
  ([`dc46ab4`](https://github.com/SerPeter/code-atlas/commit/dc46ab415738cba5cfcba374fe83cada3f5f3876))

- **parsing**: Stop hash from meaning three things in one outline line
  ([`fc1d69f`](https://github.com/SerPeter/code-atlas/commit/fc1d69fdacfb9b98485c95d19a5c878184dde80b))

- **parsing**: Walk the whole body for nested defs, and stop reading Any as a class
  ([`b45750b`](https://github.com/SerPeter/code-atlas/commit/b45750b927830f0e562ba7e98782d9458ba736e8))

- **search**: Apply test/stub/generated filters to text_search and vector_search
  ([`a62b760`](https://github.com/SerPeter/code-atlas/commit/a62b760c2a2e485b37cd43055780812a8a32c3ad))

- **search**: Retry embedding API calls on transient provider errors
  ([`d2a7cc1`](https://github.com/SerPeter/code-atlas/commit/d2a7cc1b5afd4ca4527778af02efa30dd3f4fe0c))

- **server**: Apply test_filter consistently across all analyze_repo sub-analyses
  ([`a1b1d5c`](https://github.com/SerPeter/code-atlas/commit/a1b1d5c4a5904759560e49214a7e484607edf181))

- **server**: Correct outline defects found by blind-reading the format
  ([`ed5587a`](https://github.com/SerPeter/code-atlas/commit/ed5587a2c59a85b15f160c6790f4e19611cf1e55))

- **server**: Distinguish an unpartitionable subgraph from a missing MAGE procedure
  ([`a700b38`](https://github.com/SerPeter/code-atlas/commit/a700b38a366abea7beed6da9a5b22e4c0f1aa6ee))

- **server**: Exclude ExternalSymbol/ExternalPackage nodes from community detection
  ([`5d2c20d`](https://github.com/SerPeter/code-atlas/commit/5d2c20dd117d7f8516e884aac5117bad5892eada))

- **server**: Exclude test entities from Leiden's input graph, not just its output
  ([`384f94d`](https://github.com/SerPeter/code-atlas/commit/384f94dcd9beba7da5351db9f2d3860c8bebaf1a))

- **server**: Exclude test modules from analyze_repo quality scoring
  ([`bc39f16`](https://github.com/SerPeter/code-atlas/commit/bc39f1690d7c109a91669b6acd28923dc06b026e))

- **server**: Get_node no longer silently hides fuzzy-match siblings
  ([`6dd60f2`](https://github.com/SerPeter/code-atlas/commit/6dd60f245a7614b7ad5873d8d3f56c73af79a25d))

- **server**: Make module_summary's header and legend match what it rendered
  ([`755dbc5`](https://github.com/SerPeter/code-atlas/commit/755dbc50e66336836ab2681c2a14180466bca0a2))

- **server**: Report the real .env path from the MCP health check
  ([`36111ce`](https://github.com/SerPeter/code-atlas/commit/36111ce78e589e6a855fbb12e88e13f53e5dff66))

- **server**: Run MCP daemon startup catchup in the background
  ([`3e7aa42`](https://github.com/SerPeter/code-atlas/commit/3e7aa42d0698cc3fc17b01896de29616c8cc3702))

- **server**: Stop config-derived entities swamping dead-code analysis
  ([`dfbd90e`](https://github.com/SerPeter/code-atlas/commit/dfbd90ebdc3b73850d7db8c74d4b59337915427a))

- **server**: Stop find_dead_code calling live classes dead
  ([`0b73107`](https://github.com/SerPeter/code-atlas/commit/0b7310704addc2f82aa72b5d1de463db244ddb7c))

- **tests**: Make infra fixtures reachable from the bench tier
  ([`84f9c71`](https://github.com/SerPeter/code-atlas/commit/84f9c713cf93999fd7cab12b94dc601a7068abed))

### Build System

- **ci**: Stop the ty pre-commit hook from re-syncing the venv
  ([`a87c79a`](https://github.com/SerPeter/code-atlas/commit/a87c79a129731a870dbc25b28825ba1c1900acc8))

- **deps**: Lock the new tree-sitter grammar extras
  ([`67fd7cc`](https://github.com/SerPeter/code-atlas/commit/67fd7cce59a21961b5ff2c89ecc9fcbd0e6003fb))

- **infra**: Upgrade Memgraph 3.7.2 -> 3.12.0 to stop the vector-index GC segfault
  ([`2d59778`](https://github.com/SerPeter/code-atlas/commit/2d5977887de69f9f9f57719fe41a13f0aef7d783))

### Chores

- Point local planning at .specs/ instead of .tasks/
  ([`0ba1737`](https://github.com/SerPeter/code-atlas/commit/0ba17372106f760586635026830d56ccb9b3afd3))

- **deps**: Upgrade dependencies, cap litellm and ty at known-good versions
  ([`af3e47b`](https://github.com/SerPeter/code-atlas/commit/af3e47b8557f79efbfbf20d41c9f335b628a2b77))

### Code Style

- **docs**: Apply prettier formatting to ADR-0016
  ([`b064f86`](https://github.com/SerPeter/code-atlas/commit/b064f8672eda956652d9fd50d26d74ca03813221))

### Documentation

- Fix docs/->wiki path references dropped from the rename commit
  ([`1e585de`](https://github.com/SerPeter/code-atlas/commit/1e585de68c5254caab95030d578be1675d1d38ff))

- Record how to run the test suite without burning hours
  ([`36c814c`](https://github.com/SerPeter/code-atlas/commit/36c814c688df7a20a4ca984e848770c5f1d5ce38))

- Rename docs/ vault to wiki/, add ADR-0012
  ([`2a3ed86`](https://github.com/SerPeter/code-atlas/commit/2a3ed86625a7fdecdebec25491a85685f3e39132))

- **adr**: Add ADR-0015 for the embedded SQLite backend option
  ([`8beeb7a`](https://github.com/SerPeter/code-atlas/commit/8beeb7ac01e19b218dae30e14cc802a4cda4d65b))

- **adr**: Add ADR-0018 for non-code file parsing
  ([`ff0d3c8`](https://github.com/SerPeter/code-atlas/commit/ff0d3c8840e5bfbfa1f23a444bb8ddd2f043d7f5))

- **adr**: Record the outline changes the blind-read evaluation produced
  ([`dd160da`](https://github.com/SerPeter/code-atlas/commit/dd160dafbc646d2da8017f8a3f49a15f011425b6))

- **adr**: Record type-directed call resolution
  ([`dbee463`](https://github.com/SerPeter/code-atlas/commit/dbee46379af8be1e1c0299dbbc2972b5412938dc))

- **server**: Record the dead-code shape that is permanently unresolvable
  ([`87278f8`](https://github.com/SerPeter/code-atlas/commit/87278f851a3e0b25d82f00dda928112193afd099))

### Features

- **backends**: Add SQLite-backed embedded graph and queue as a Memgraph/Valkey fallback
  ([`ecbe1f7`](https://github.com/SerPeter/code-atlas/commit/ecbe1f7f5ec15a22617eaaff4f720f6124ca0e67))

- **cli**: Add --with-git-signals flag to atlas index
  ([`aaa4bc4`](https://github.com/SerPeter/code-atlas/commit/aaa4bc4e1fbee62dece3f70f2f0a431fe7d6b8e5))

- **graph**: Env-var and referenced-file nodes, citation edges, manifest parity
  ([`2d09467`](https://github.com/SerPeter/code-atlas/commit/2d094672e957ba0066154beb5464bf545e37669d))

- **graph**: Follow a dispatch table to its handlers, without fanning out
  ([`0178c08`](https://github.com/SerPeter/code-atlas/commit/0178c084387486edbf7fd42787331c302e6594e5))

- **graph**: Give a builtin base class a node so exception hierarchies are visible
  ([`28ac31b`](https://github.com/SerPeter/code-atlas/commit/28ac31b24d1abed22c0746e438c231296a03682f))

- **graph**: Recognise structural Protocol conformance (ADR-0025)
  ([`198f03c`](https://github.com/SerPeter/code-atlas/commit/198f03c41884b2e9ae5905730968c910b6924383))

- **graph**: Record a callable named as a value, distinct from calling it
  ([`a38d204`](https://github.com/SerPeter/code-atlas/commit/a38d204e2dc038d7d8f545e4edbc571cafd98b32))

- **graph**: Resolve a call through its receiver's declared type
  ([`e91ccf9`](https://github.com/SerPeter/code-atlas/commit/e91ccf9a1281ef2b3771976c2dd2fd0a639ce374))

- **graph**: Resolve calls past Protocol stubs, and damp every unverified receiver
  ([`3ab3a68`](https://github.com/SerPeter/code-atlas/commit/3ab3a6800c591aba7c43110b2383868aa9002829))

- **graph**: Resolve constructor calls (ClassName(...)) to __init__
  ([`244d4b9`](https://github.com/SerPeter/code-atlas/commit/244d4b9ea1fcebbc50732ec0c900893fb41215fc))

- **graph**: Surface CALLS edge resolution confidence instead of discarding ambiguous matches
  ([`3015c5a`](https://github.com/SerPeter/code-atlas/commit/3015c5ac13519a301b5f95ba517b49acd2c6b2d9))

- **graph**: Weighted CALLS edges, module summaries, and rationale extraction
  ([`b12b31d`](https://github.com/SerPeter/code-atlas/commit/b12b31dfd1e24db7153a4c7734e4cd641ce408cf))

- **indexing**: Mine git-derived signals (hotspots, bus factor, co-change) via GitPython
  ([`1a2bd1a`](https://github.com/SerPeter/code-atlas/commit/1a2bd1a8bcfe802698565b0333f6e41fa6136ed4))

- **infra**: Swap Memgraph image to memgraph-mage for community-detection support
  ([`6568f63`](https://github.com/SerPeter/code-atlas/commit/6568f639f97754fb8607ec723791f3b196072f31))

- **parsing**: Add HCL, shell, Dockerfile, SQL and context-aware config parsers
  ([`d922fb0`](https://github.com/SerPeter/code-atlas/commit/d922fb0ef4f753494717c0e53782e8964da58310))

- **parsing**: Add Salesforce support via a Java-grammar shim
  ([`d26ecd7`](https://github.com/SerPeter/code-atlas/commit/d26ecd71f4418a2bae260f069e053da502244c74))

- **parsing**: Enable the module_exports detector so __all__ becomes an edge
  ([`aa9e98c`](https://github.com/SerPeter/code-atlas/commit/aa9e98cd8cb7e26b1be973850a303eb8acd064c4))

- **parsing**: Give a class its fields, and give a field its type
  ([`26add0d`](https://github.com/SerPeter/code-atlas/commit/26add0da252b19e3ef8f23b576be942949d65158))

- **parsing**: Index functions defined inside other functions
  ([`a23d528`](https://github.com/SerPeter/code-atlas/commit/a23d5283d11b7242aa7435a1522ac5874981e319))

- **parsing**: Link a decorated definition to the decorator that registers it
  ([`5fab287`](https://github.com/SerPeter/code-atlas/commit/5fab287f950dbdcccc57533c476e7f3cf01a8bb3))

- **parsing**: Make a constructor-injected collaborator a typed field
  ([`cbd4743`](https://github.com/SerPeter/code-atlas/commit/cbd474327abb73337885ed9d49d3bd627fbc2b06))

- **parsing**: Record the call receiver in seven more languages
  ([`1335006`](https://github.com/SerPeter/code-atlas/commit/133500623653b8d0b50c76e388f12c23566d24c8))

- **parsing**: See the code that runs at import time
  ([`04c2ce1`](https://github.com/SerPeter/code-atlas/commit/04c2ce1705cf2d158a02193770f885baa40df100))

- **server**: Add exclude_tests to blast_radius and get_node
  ([`4afc9cc`](https://github.com/SerPeter/code-atlas/commit/4afc9cca43d5b9880acff8f821fa0e2fb5ec7b0c))

- **server**: Add offset pagination to hybrid_search, text_search, vector_search, get_node
  ([`5fc50eb`](https://github.com/SerPeter/code-atlas/commit/5fc50ebbfc430e6e470822b4d0284dffef69e875))

- **server**: Add trace_path/blast_radius and dead-code/complexity/community analyses
  ([`819356c`](https://github.com/SerPeter/code-atlas/commit/819356c4e150d249b6d143172236b73da167a338))

- **server**: Cluster communities at module granularity
  ([`a00410e`](https://github.com/SerPeter/code-atlas/commit/a00410e92306a05586f9e8bc0d86e41529580443))

- **server**: Drop Mermaid for grouped adjacency on large import graphs
  ([`02beb10`](https://github.com/SerPeter/code-atlas/commit/02beb10785a09e8de3c73f5320b8b741389af1d0))

- **server**: Hide find_communities from tools/list on the sqlite backend
  ([`704a7a0`](https://github.com/SerPeter/code-atlas/commit/704a7a06ae9004b2e8fc1ad8683a6675cbdc0e38))

- **server**: Make the outline boundary sections answer the question asked of them
  ([`9755101`](https://github.com/SerPeter/code-atlas/commit/9755101c29e5e38149982e3ebafd3441c904029a))

- **server**: Pick module_summary detail by rendered size, not by scope shape
  ([`bf0d5fe`](https://github.com/SerPeter/code-atlas/commit/bf0d5fe976e6432029fd62388ff2f011c8b1bb68))

- **server**: Surface cycles, file locations and scope in the import diagram
  ([`68c0e7f`](https://github.com/SerPeter/code-atlas/commit/68c0e7f0c54a5bbaa41a71e52adb6bb6011d2b57))

- **server**: Surface live indexing backlog in staleness reporting
  ([`7d5927c`](https://github.com/SerPeter/code-atlas/commit/7d5927ce83492fe613fa6b1d59217ec9a6e53bb9))

- **settings**: Add config-driven backend selection
  ([`449be65`](https://github.com/SerPeter/code-atlas/commit/449be651956a5ee75d15d4be3c3d6d0a9b3d6c42))

- **settings**: Default the knowledge vault to wiki/ instead of docs/
  ([`07a64b3`](https://github.com/SerPeter/code-atlas/commit/07a64b397403cbb14bff2fbbca460e0e354d30f5))

### Refactoring

- **parsing**: Record a decorator's registration surface generically, delete three framework
  detectors
  ([`739f140`](https://github.com/SerPeter/code-atlas/commit/739f140c783d3b06db19b6d9ab6dac1aac0d9c27))

### Testing

- **integration**: Measure which Python design patterns the graph can actually answer
  ([`74c3b1e`](https://github.com/SerPeter/code-atlas/commit/74c3b1edbf9ecea289f45714ef5bef583712e710))

- **parsing**: Assemble the PEM marker so detect-private-key stays useful
  ([`8ecf469`](https://github.com/SerPeter/code-atlas/commit/8ecf46931345045666231690a719246792795f33))


## v0.4.1 (2026-07-15)

### Bug Fixes

- Resolve 14 findings from the knowledge-convergence review
  ([`e500c27`](https://github.com/SerPeter/code-atlas/commit/e500c27a800e5f6eab89859d9d8657750db2a5a1))


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
