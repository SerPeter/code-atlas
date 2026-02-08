# Code Atlas Agent

## MCP Configuration

```json
{
  "mcpServers": {
    "code-atlas": {
      "command": "atlas",
      "args": ["mcp"]
    }
  }
}
```

## Getting Started

1. Call `schema_info` to see available node types, relationships, and properties.
2. Call `index_status` to see indexed projects and entity counts.
3. Use `get_node` to find entities by name, then `get_context` to expand.
4. Use `cypher_query` for custom graph traversals.
5. Use `text_search` for keyword matching, `vector_search` for semantic similarity.
