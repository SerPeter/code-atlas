# Code Atlas Architecture

This document describes the architecture of Code Atlas, a code intelligence graph system that indexes codebases and exposes them via MCP tools for AI coding agents.

## System Overview

Code Atlas combines three search paradigms in a unified system:

- **Graph traversal** — follow relationships (calls, inheritance, imports)
- **Semantic search** — find code by meaning via embeddings
- **BM25 keyword search** — exact matches for identifiers and strings

All powered by Memgraph as a single backend, exposed through an MCP server.

```mermaid
graph TB
    subgraph Clients
        CC[Claude Code]
        CU[Cursor]
        WS[Windsurf]
        API[API Clients]
    end

    subgraph "Code Atlas"
        MCP[MCP Server]
        QR[Query Router]

        subgraph Search
            GS[Graph Search]
            VS[Vector Search]
            BS[BM25 Search]
        end

        RRF[RRF Fusion]
        CE[Context Expander]

        subgraph Indexing
            FS[File Scanner]
            RP[Rust Parser]
            PD[Pattern Detectors]
            EM[Embedder]
            GW[Graph Writer]
        end
    end

    subgraph Infrastructure
        MG[(Memgraph)]
        TEI[TEI Embeddings]
    end

    CC --> MCP
    CU --> MCP
    WS --> MCP
    API --> MCP

    MCP --> QR
    QR --> GS
    QR --> VS
    QR --> BS

    GS --> RRF
    VS --> RRF
    BS --> RRF

    RRF --> CE
    CE --> MCP

    GS --> MG
    VS --> MG
    BS --> MG

    FS --> RP
    RP --> PD
    PD --> EM
    EM --> GW
    GW --> MG
    EM --> TEI
```

## Component Architecture

### MCP Server

The MCP server is the primary interface for AI agents. It exposes tools for querying the code graph and managing the index.

```mermaid
graph LR
    subgraph "MCP Server"
        direction TB
        QT[Query Tools]
        IT[Index Tools]
        AT[Admin Tools]
    end

    subgraph "Query Tools"
        CS[cypher_query]
        TS[text_search]
        VS[vector_search]
        HS[hybrid_search]
        GN[get_node]
        GC[get_context]
    end

    subgraph "Index Tools"
        IX[index]
        ST[status]
    end

    subgraph "Admin Tools"
        HE[health]
        SC[schema_info]
    end
```

### Indexing Pipeline

The indexing pipeline transforms source code into a searchable graph.

```mermaid
flowchart LR
    subgraph Input
        SRC[Source Files]
        DOC[Documentation]
        CFG[Config Files]
    end

    subgraph "File Scanner"
        GI[.gitignore]
        AI[.atlasignore]
        AT[atlas.toml]
    end

    subgraph "Rust Parser"
        TS[Tree-sitter]
        AST[AST Extraction]
        ENT[Entity Detection]
    end

    subgraph "Pattern Detectors"
        DR[Decorator Routing]
        EH[Event Handlers]
        TM[Test Mapping]
        CO[Class Overrides]
    end

    subgraph "Embedder"
        BC[Batch Collector]
        TE[TEI Client]
        EC[Embedding Cache]
    end

    subgraph "Graph Writer"
        NW[Node Writer]
        EW[Edge Writer]
        IU[Index Updater]
    end

    SRC --> GI
    DOC --> GI
    GI --> AI
    AI --> AT
    AT --> TS

    TS --> AST
    AST --> ENT
    ENT --> DR
    DR --> EH
    EH --> TM
    TM --> CO

    CO --> BC
    BC --> TE
    TE --> EC
    EC --> NW
    NW --> EW
    EW --> IU
    IU --> MG[(Memgraph)]
```

### Query Pipeline

The query pipeline handles search requests and assembles context.

```mermaid
flowchart TB
    Q[Query] --> QA[Query Analyzer]
    QA --> |structural| GS[Graph Search]
    QA --> |semantic| VS[Vector Search]
    QA --> |keyword| BS[BM25 Search]
    QA --> |hybrid| ALL[All Three]

    GS --> RRF[RRF Fusion]
    VS --> RRF
    BS --> RRF
    ALL --> RRF

    RRF --> CE[Context Expander]

    CE --> |hierarchy| HW[Walk UP]
    CE --> |calls| CW[Walk CALLS]
    CE --> |docs| DW[Walk DOCUMENTS]

    HW --> TA[Token Assembler]
    CW --> TA
    DW --> TA

    TA --> |budget| TB[Token Budget]
    TB --> R[Response]
```

## Graph Schema

### Node Types

```mermaid
graph TB
    subgraph "Code Nodes"
        P[Project]
        PK[Package]
        M[Module]
        C[Class]
        F[Function]
        MT[Method]
    end

    subgraph "Documentation Nodes"
        DF[DocFile]
        DS[DocSection]
        ADR[ADR]
    end

    subgraph "Dependency Nodes"
        EP[ExternalPackage]
        ES[ExternalSymbol]
    end

    P --> PK
    PK --> M
    M --> C
    M --> F
    C --> MT
```

### Relationships

```mermaid
graph LR
    subgraph Structural
        CONTAINS[CONTAINS]
        DEFINES[DEFINES]
        INHERITS[INHERITS]
    end

    subgraph "Call/Data"
        CALLS[CALLS]
        IMPORTS[IMPORTS]
        USES_TYPE[USES_TYPE]
    end

    subgraph Documentation
        DOCUMENTS[DOCUMENTS]
        MOTIVATED_BY[MOTIVATED_BY]
    end

    subgraph Patterns
        HANDLES_ROUTE[HANDLES_ROUTE]
        HANDLES_EVENT[HANDLES_EVENT]
        TESTS[TESTS]
        OVERRIDES[OVERRIDES]
    end
```

### Full Schema Diagram

```mermaid
erDiagram
    Project ||--o{ Package : CONTAINS
    Package ||--o{ Module : CONTAINS
    Module ||--o{ Class : DEFINES
    Module ||--o{ Function : DEFINES
    Class ||--o{ Method : DEFINES_METHOD
    Class ||--o{ Class : INHERITS

    Function ||--o{ Function : CALLS
    Method ||--o{ Function : CALLS
    Method ||--o{ Method : CALLS

    Module ||--o{ ExternalPackage : IMPORTS
    Function ||--o{ ExternalSymbol : USES_TYPE

    DocSection ||--o{ Function : DOCUMENTS
    DocSection ||--o{ Class : DOCUMENTS
    ADR ||--o{ Module : AFFECTS

    Function ||--o{ Route : HANDLES_ROUTE
    Function ||--o{ Event : HANDLES_EVENT
    Function ||--o{ Class : TESTS
    Method ||--o{ Method : OVERRIDES
```

## Data Flow

### Indexing Flow

```mermaid
sequenceDiagram
    participant CLI
    participant Scanner
    participant Parser
    participant Detector
    participant Embedder
    participant Writer
    participant Memgraph

    CLI->>Scanner: index /path/to/project
    Scanner->>Scanner: Apply ignore rules
    Scanner->>Parser: Batch of files

    loop For each file
        Parser->>Parser: Parse AST
        Parser->>Detector: Extracted entities
        Detector->>Detector: Detect patterns
        Detector->>Embedder: Entities + patterns
    end

    Embedder->>Embedder: Batch embeddings
    Embedder->>Writer: Entities + embeddings

    Writer->>Memgraph: Batch write nodes
    Writer->>Memgraph: Batch write edges
    Writer->>Memgraph: Update indices
```

### Query Flow

```mermaid
sequenceDiagram
    participant Agent
    participant MCP
    participant Router
    participant Graph
    participant Vector
    participant BM25
    participant Fusion
    participant Expander

    Agent->>MCP: hybrid_search("auth middleware")

    MCP->>Router: Analyze query

    par Parallel Search
        Router->>Graph: Cypher query
        Router->>Vector: Embed + search
        Router->>BM25: Text search
    end

    Graph-->>Fusion: Graph results
    Vector-->>Fusion: Vector results
    BM25-->>Fusion: BM25 results

    Fusion->>Fusion: RRF merge
    Fusion->>Expander: Top results

    Expander->>Expander: Walk hierarchy
    Expander->>Expander: Walk calls
    Expander->>Expander: Assemble context

    Expander-->>MCP: Expanded results
    MCP-->>Agent: Response with context
```

## Deployment Architecture

### Local Development

```mermaid
graph TB
    subgraph "Developer Machine"
        CLI[atlas CLI]
        MCP[MCP Server]

        subgraph Docker
            MG[(Memgraph)]
            TEI[TEI]
        end
    end

    subgraph "AI Client"
        CC[Claude Code]
    end

    CLI --> MG
    MCP --> MG
    MCP --> TEI
    CC --> MCP
```

### Docker Compose Setup

```mermaid
graph TB
    subgraph "docker-compose"
        MG[memgraph:7687]
        TEI[tei:8080]
    end

    subgraph "Host"
        ATLAS[atlas CLI/MCP]
        VOL[./data volume]
    end

    ATLAS --> MG
    ATLAS --> TEI
    MG --> VOL
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| CLI | Typer | Command-line interface |
| MCP | mcp-python | Model Context Protocol server |
| Config | Pydantic | Configuration management |
| Parsing | Tree-sitter (Rust) | Fast AST parsing |
| Graph DB | Memgraph | Graph storage + vector + BM25 |
| Embeddings | TEI / LiteLLM | Code embeddings |
| HTTP | httpx | Async HTTP client |
| Tokens | tiktoken | Token counting |

## Security Considerations

- **Local-first**: All data stays on the developer's machine
- **No external calls by default**: TEI runs locally in Docker
- **Optional cloud embeddings**: LiteLLM fallback requires explicit config
- **No telemetry**: No usage data sent anywhere
- **Git-aware**: Respects .gitignore, never indexes secrets

## Performance Characteristics

| Operation | Target | Notes |
|-----------|--------|-------|
| Full index (10K files) | < 60s | Parallelized parsing |
| Delta index (10% change) | < 10s | Entity-level diffing |
| Simple query (p95) | < 100ms | Single search type |
| Hybrid query (p95) | < 300ms | Three search types + fusion |
| Memory (100K nodes) | < 2GB | Memgraph in-memory |

## Future Considerations

- **Language expansion**: Additional tree-sitter grammars
- **Distributed indexing**: For very large monorepos
- **Remote Memgraph**: Team-shared graph instance
- **Custom detectors**: User-defined pattern plugins
- **IDE integration**: Real-time indexing via file watchers
