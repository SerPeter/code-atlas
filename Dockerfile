# =============================================================================
# Code Atlas — CLI + MCP server
#
# Use cases:
#   docker run -v /project:/data/project atlas index /data/project
#   docker run -p 8000:8000 atlas mcp
#   docker run atlas search "query"
#
# Note: stdio MCP transport requires a local process, not a container.
# For AI agent sessions, prefer running `atlas mcp` directly on the host.
# =============================================================================

FROM python:3.14-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies first (cached layer — changes less than source)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source + metadata, then install the project itself
COPY src/ src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =============================================================================
FROM python:3.14-slim AS runtime

WORKDIR /app

# System deps: git for delta indexing / staleness checking
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 atlas && \
    useradd --uid 1000 --gid atlas --create-home atlas

# Copy uv binary and the fully-built virtualenv from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder --chown=atlas:atlas /app /app

# Container-sensible defaults: HTTP transport, bind all interfaces.
# Override at runtime with -e or docker-compose environment.
ENV ATLAS_MCP__TRANSPORT=streamable-http \
    ATLAS_MCP__HOST=0.0.0.0

EXPOSE 8000

USER atlas

LABEL org.opencontainers.image.title="code-atlas" \
      org.opencontainers.image.description="Code intelligence graph — index, search, and serve codebases via MCP" \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["uv", "run", "atlas"]
CMD ["mcp"]
