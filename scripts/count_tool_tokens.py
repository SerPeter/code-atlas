"""Count the context-window cost of every MCP tool definition.

The README quotes a per-tool token cost, and until this script existed those numbers
could not be reproduced — which made them a claim rather than a measurement. Run it
after adding or re-describing a tool and paste the output into the README table.

No graph, no network: FastMCP builds tool schemas from the decorators at import time, so
the numbers depend only on the source.

    uv run --no-sync python scripts/count_tool_tokens.py
    uv run --no-sync python scripts/count_tool_tokens.py --markdown
"""

from __future__ import annotations

import argparse
import asyncio
import json

import tiktoken

# cl100k_base rather than a model-specific encoding: it is the tokenizer the original
# figures used, so recounts stay comparable to the ones already published.
ENCODING = "cl100k_base"


async def collect() -> list[tuple[str, int, int]]:
    """``(name, search_tokens, full_tokens)`` for every registered tool.

    *search* is what an agent pays to know the tool exists (name + description);
    *full* adds the parameter schema it needs to actually call it.
    """
    from code_atlas.server.mcp import create_mcp_server
    from code_atlas.settings import AtlasSettings

    # catchup=False: the schemas come from the decorators, so building the server must
    # not touch Memgraph or kick off an index pass.
    mcp = create_mcp_server(AtlasSettings(), catchup=False)

    encoder = tiktoken.get_encoding(ENCODING)
    rows: list[tuple[str, int, int]] = []
    for tool in sorted(await mcp.list_tools(), key=lambda t: t.name):
        description = tool.description or ""
        search = len(encoder.encode(tool.name)) + len(encoder.encode(description))
        schema = json.dumps(tool.inputSchema, separators=(",", ":"), sort_keys=True)
        rows.append((tool.name, search, search + len(encoder.encode(schema))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="emit README table cells")
    args = parser.parse_args()

    rows = asyncio.run(collect())
    for name, search, full in rows:
        if args.markdown:
            print(f"| `{name}` | | ~{search} | ~{full} | |")
        else:
            print(f"{name:<26} search={search:>6}  full={full:>6}")

    print(f"\n{len(rows)} tools — search total {sum(r[1] for r in rows)}, full total {sum(r[2] for r in rows)}")


if __name__ == "__main__":
    main()
