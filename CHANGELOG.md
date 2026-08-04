# CHANGELOG

<!-- version list -->

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
