# =============================================================================
# Stage 1: Build Rust parser
# =============================================================================
FROM rust:1.84-slim AS rust-builder

WORKDIR /build

# Copy only Rust sources
COPY crates/ crates/

# Build release binary
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cd crates && \
    cargo build --release && \
    cp target/release/atlas-parser /build/atlas-parser || true

# =============================================================================
# Stage 2: Python application
# =============================================================================
FROM python:3.14-slim AS runtime

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy Rust binary from builder (if built)
COPY --from=rust-builder /build/atlas-parser /usr/local/bin/atlas-parser

# Install Python dependencies (cached layer)
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source
COPY src/ src/
COPY README.md ./

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Default: run the MCP server
ENTRYPOINT ["uv", "run", "atlas"]
CMD ["mcp"]
