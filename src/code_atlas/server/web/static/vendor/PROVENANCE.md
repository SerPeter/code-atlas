# Vendored browser assets

Everything in this directory is committed, not fetched. `atlas ui` must work with no network beyond the local Memgraph,
and the static export (ATL-120) must not reach the network at all — see
[ADR-0033](../../../../../../wiki/adr/0033-graph-renderer-must-be-permissively-licensed.md).

**Both are MIT.** That is a hard requirement, not a preference. The obvious candidate for this view was Cosmograph, and
the trap there is that `@cosmograph/*` on npm is **CC-BY-NC** — a non-commercial licence that would make this project
undistributable. Its sibling `@cosmos.gl/graph` is MIT. Check the licence of the exact package name before adding
anything here.

| File                           | Package    | Version | Licence | Source                                   |
| ------------------------------ | ---------- | ------- | ------- | ---------------------------------------- |
| `sigma-3.0.3.min.js`           | sigma      | 3.0.3   | MIT     | https://github.com/jacomyal/sigma.js     |
| `graphology-0.26.0.umd.min.js` | graphology | 0.26.0  | MIT     | https://github.com/graphology/graphology |

Each ships with its upstream `LICENSE.txt` alongside it, unmodified.

## Verifying these files

The bytes here are the untouched `dist/` builds from the published npm tarballs. To re-derive them:

```bash
npm install --prefix /tmp/vendorjs sigma@3.0.3 graphology@0.26.0
cp /tmp/vendorjs/node_modules/sigma/dist/sigma.min.js            sigma-3.0.3.min.js
cp /tmp/vendorjs/node_modules/graphology/dist/graphology.umd.min.js  graphology-0.26.0.umd.min.js
```

SHA-512 of the vendored files, in Subresource Integrity form:

```
sigma-3.0.3.min.js            sha512-Pl31Mn3QNmO8wHTaB9fdt/0YGpGtHHqcvT8SuwSK4/E1VLgkiPoluigM5BxlC3irGAzfLjSv9/L1m9dyVEl4Bw==
graphology-0.26.0.umd.min.js  sha512-Hqa5FKQ53pYDWaRnytoNvRT3JXRac7dcH+kB3RUCX69CGNrnz5LE76Mp0z186qDv0LBWrwx5QipEoenZB5CE4w==
```

These hash the individual files. npm's own `dist.integrity` covers the whole package tarball and is a different value —
do not expect the two to match:

```
sigma@3.0.3       sha512-5H0zFlx6/NTQpqBg4Rm569ZOpnBOXMaS25UQThIWMU3XyzI5AhmorK/gnl87BvJBLhQd0tW4C0LIp3enWzMoNw==
graphology@0.26.0 sha512-8SSImzgUUYC89Z042s+0r/vMibY7GX/Emz4LDO5e7jYXhuoWfHISPFJYjpRLUSJGq6UQ6xlenvX1p/hJdfXuXg==
```

## Why sigma

Module names are the content of this view, and sigma has collision-aware label rendering built in — labels appear and
disappear with zoom so the map stays readable at every scale. A renderer that draws beautiful nodes but cannot label 400
of them legibly would not answer the question the map exists for.

`sigma.min.js` is a UMD bundle exposing `Sigma`; graphology is a **peer** dependency, not bundled, so
`graphology.umd.min.js` must load first and expose `graphology`.

## Archivo (typeface)

Added 2026-08-09 with the Claude Design output for the v1 interface. **SIL Open Font License 1.1** --
permissive, and the licence text ships beside the files as `archivo.LICENSE.txt`, which OFL requires.

| File | Subset | Source |
| --- | --- | --- |
| `archivo-latin.woff2` | latin | https://github.com/Omnibus-Type/Archivo |
| `archivo-latin-ext.woff2` | latin-ext | (via `@fontsource/archivo@5.3.0`, licence `OFL-1.1`) |
| `archivo-vietnamese.woff2` | vietnamese | |

80,776 bytes across the three subsets. They are the woff2 builds embedded in the design bundle, extracted
verbatim -- each begins with the `wOF2` magic number and is byte-identical to what the bundle carried.

Kept as three subsets rather than merged: each `@font-face` declares its own `unicode-range`, so a browser
downloads only the ranges a page actually uses. Latin-only pages fetch 34KB, not 80KB.

```
archivo-latin-ext.woff2        sha512-J32KRMxeSVnEjxg+dMPwNYaKqKmPFmbwKbrHesGWWBvQKfu2PgHC1GMC32DUfrfKVqj31Najr3kwSeBads3x8Q==
archivo-latin.woff2            sha512-1A47GJcwSdmzi16cgyIRY+EDxXzJk5LXYpopNJvfpDii6Rl3qR2C8aS9GXEKNjcOGBD+mPqzPP1ScWLjQftT4g==
archivo-vietnamese.woff2       sha512-CMh56tILNZFI24qfi4Ulxw72wL1gfL9qv+pS/xTI2W7QKXx8tGfK62I/qST2JLFDmfB/zqDlGLQzZSSoB6eNXw==
```
