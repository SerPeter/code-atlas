# Vendored assets

Everything the UI loads is in this directory — no CDN, no network at runtime (ADR-0033).

| File | What | Source | Licence |
| --- | --- | --- | --- |
| `archivo-latin.woff2` | Archivo variable font, latin subset | Extracted from the Claude Design v1.1 bundle (originally Google Fonts) | SIL OFL 1.1 (`archivo.LICENSE.txt`) |
| `archivo-latin-ext.woff2` | latin-ext subset | same | same |
| `archivo-vietnamese.woff2` | vietnamese subset | same | same |

The graph canvas is DOM + SVG, exactly as the reference design draws it, so there is no
renderer library to vendor: nodes are positioned `<span>`s, edges are SVG `<line>`s, and
every colour resolves through CSS custom properties (which is what lets `oklch()` and
`color-mix()` work without a JS colour parser).
