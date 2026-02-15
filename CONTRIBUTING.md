# Contributing to Code Atlas

Thank you for your interest in contributing to Code Atlas! This guide will help you get started.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Prerequisites

- [Python 3.14+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Node.js](https://nodejs.org/) (for prettier formatting)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/SerPeter/code-atlas.git
cd code-atlas

# Start infrastructure (Memgraph + Valkey)
docker compose up -d

# Install dependencies (including dev tools)
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

## Running Tests

```bash
# All tests (requires running infrastructure)
uv run pytest

# Skip infrastructure-dependent tests
uv run pytest -m "not integration and not bench and not tei"

# Single test
uv run pytest tests/test_foo.py::test_bar
```

## Code Style

Code Atlas uses automated tooling to enforce consistent style:

- **[Ruff](https://docs.astral.sh/ruff/)** for linting and formatting (line length 120)
- **[ty](https://docs.astral.sh/ty/)** for type checking
- **[Prettier](https://prettier.io/)** for markdown formatting

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

Pre-commit hooks run these checks automatically on each commit.

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

Examples:

```
feat(parser): add Go language support
fix(search): handle empty query strings
docs: update installation instructions
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, ensuring tests pass and linting is clean.
3. Write clear commit messages following the conventions above.
4. Open a pull request against `main` and fill out the PR template.
5. Wait for CI checks to pass and a maintainer review.

## Reporting Issues

Use [GitHub Issues](https://github.com/SerPeter/code-atlas/issues) with the provided templates:

- **Bug Report** — for bugs and unexpected behavior
- **Feature Request** — for new features and improvements

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
