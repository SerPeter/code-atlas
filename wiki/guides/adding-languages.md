# Adding Language Support to Code Atlas

This guide covers how to add new programming language support to Code Atlas, either as a built-in language or an
external plugin.

## Overview

Code Atlas uses [tree-sitter](https://tree-sitter.github.io/) grammars for AST parsing. Each language registers a
`LanguageConfig` with:

- A **name** (e.g., `"python"`, `"rust"`)
- File **extensions** it handles
- A tree-sitter **Language** object (from a grammar package)
- A tree-sitter **Query** (can be minimal)
- A **parse function** that extracts entities and relationships from the AST

## Built-in Language Checklist

To add a new language to `src/code_atlas/parsing/languages/`:

### 1. Add the grammar dependency

Add the tree-sitter grammar package to `pyproject.toml` as an optional dependency:

```toml
[project.optional-dependencies]
mylang = ["tree-sitter-mylang~=0.23"]
all-languages = ["code-atlas[typescript,go,rust,java,csharp,cpp,ruby,php,mylang]"]
```

### 2. Create the language module

Create `src/code_atlas/parsing/languages/mylang.py`:

```python
"""MyLang language support — tree-sitter parser."""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

if TYPE_CHECKING:
    from tree_sitter import Node

try:
    import tree_sitter_mylang as ts_mylang
except ImportError:
    pass  # Grammar not installed — language not available
else:
    from tree_sitter import Language, Query

    _LANG = Language(ts_mylang.language())
    _QUERY = Query(_LANG, "(source_file) @root")

    def _module_qualified_name(file_path: str) -> str:
        """Convert file path to a dotted module name."""
        from pathlib import PurePosixPath

        p = PurePosixPath(file_path.replace("\\", "/"))
        parts = list(p.parts)
        if parts:
            parts[-1] = parts[-1].rsplit(".", 1)[0]  # Strip extension
        return ".".join(parts)

    def _parse_mylang(
        path: str,
        source: bytes,
        root: Node,
        project_name: str,
    ) -> ParsedFile:
        """Extract entities and relationships from a MyLang parse tree."""
        module_qn = _module_qualified_name(path)
        entities: list[ParsedEntity] = []
        relationships: list[ParsedRelationship] = []

        # Module entity (always emitted)
        entities.append(
            ParsedEntity(
                name=module_qn.rsplit(".", 1)[-1] if "." in module_qn else module_qn,
                qualified_name=f"{project_name}:{module_qn}",
                label=NodeLabel.MODULE,
                kind="module",
                line_start=1,
                line_end=root.end_point[0] + 1,
                file_path=path,
            )
        )

        # Walk the AST and extract entities/relationships...
        _walk(root, path, source, project_name, module_qn, entities, relationships)

        return ParsedFile(
            file_path=path,
            language="mylang",
            entities=entities,
            relationships=relationships,
        )

    def _walk(root, path, source, project_name, module_qn, entities, relationships):
        """Walk the AST recursively to extract entities."""
        # Implement language-specific extraction here
        pass

    register_language(
        LanguageConfig(
            name="mylang",
            extensions=frozenset({".ml"}),
            language=_LANG,
            query=_QUERY,
            parse_func=_parse_mylang,
        )
    )
```

### 3. Register in plugin discovery

Add the import to `src/code_atlas/parsing/languages/__init__.py`:

```python
import code_atlas.parsing.languages.mylang  # noqa: PLC0415, F401
```

### 4. Write tests

Create `tests/test_parser_mylang.py`:

```python
"""Tests for MyLang parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_mylang", reason="tree-sitter-mylang not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind, ValueKind, Visibility

PROJECT = "test_project"


def _parse(source: str, path: str = "src/example.ml") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity_by_name(parsed: ParsedFile, name: str):
    matches = [e for e in parsed.entities if e.name == name]
    assert len(matches) == 1, f"Expected 1 entity named {name!r}, got {len(matches)}"
    return matches[0]


# ... test cases follow
```

### 5. Verify

```bash
uv run pytest tests/test_parser_mylang.py -v    # Parser tests pass
uv run ruff check src/code_atlas/parsing/languages/mylang.py  # No lint errors
uv run pytest -m "not slow"                      # Existing tests still pass
```

## Parse Function Contract

The parse function must have this signature:

```python
def _parse_func(
    path: str,        # File path (forward slashes, relative to project root)
    source: bytes,    # Raw file content
    root: Node,       # tree-sitter root node (already parsed)
    project_name: str # Project name for qualified name prefixes
) -> ParsedFile:
```

### Entity Requirements

Every parsed file **must** emit at least one `Module` entity representing the file itself:

```python
ParsedEntity(
    name="filename",                           # Short name
    qualified_name=f"{project_name}:{module_qn}",  # Unique identifier
    label=NodeLabel.MODULE,
    kind="module",
    line_start=1,
    line_end=total_lines,
    file_path=path,
)
```

Entity fields:

| Field            | Required | Description                                                             |
| ---------------- | -------- | ----------------------------------------------------------------------- |
| `name`           | Yes      | Short identifier name                                                   |
| `qualified_name` | Yes      | Globally unique: `{project_name}:{dotted.path.Name}`                    |
| `label`          | Yes      | `NodeLabel` enum (MODULE, TYPE_DEF, CALLABLE, VALUE)                    |
| `kind`           | Yes      | Discriminator string (e.g., `TypeDefKind.CLASS`, `CallableKind.METHOD`) |
| `line_start`     | Yes      | 1-indexed start line                                                    |
| `line_end`       | Yes      | 1-indexed end line                                                      |
| `file_path`      | Yes      | Same as the `path` argument                                             |
| `docstring`      | No       | Extracted documentation text                                            |
| `signature`      | No       | Function/method signature (without body)                                |
| `visibility`     | No       | `Visibility` enum (default: PUBLIC)                                     |
| `tags`           | No       | List of tag strings (decorators, attributes, etc.)                      |
| `source`         | No       | Source code text (truncated by `parse_file()`)                          |

### Relationship Conventions

| Relationship | From          | To                   | When                          |
| ------------ | ------------- | -------------------- | ----------------------------- |
| `DEFINES`    | Parent entity | Child entity         | Class→method, module→function |
| `IMPORTS`    | Module        | Import target name   | Import statements             |
| `INHERITS`   | Subtype       | Supertype name       | Class inheritance             |
| `IMPLEMENTS` | Type          | Interface/trait name | Interface implementation      |
| `CALLS`      | Callable      | Called function name | Call expressions              |
| `EXPORTS`    | Module        | Exported entity      | Export statements             |

The `to_name` field can be either a full qualified name or a short name — the graph client resolves it during upsert.

### Qualified Name Format

```
{project_name}:{dotted.module.path.ClassName.method_name}
```

Examples:

- Module: `myproject:src.utils.helpers`
- Class: `myproject:src.utils.helpers.MyClass`
- Method: `myproject:src.utils.helpers.MyClass.my_method`
- Function: `myproject:src.utils.helpers.top_level_func`

## Grouping Similar Languages

Languages with similar syntax can share a module with shared helper functions. Examples:

- **TypeScript + JavaScript** (`typescript.py`): Single module, two `register_language()` calls
- **Java + C#** (`jvm.py`): Shared helpers for visibility, annotations, inheritance extraction
- **C + C++** (`cpp.py`): Single parse function with `is_cpp` flag

This is composition via shared helpers, not inheritance.

## External Plugin Guide

External packages can register languages via entry points without modifying Code Atlas source.

### 1. Create your package

```
code-atlas-kotlin/
├── pyproject.toml
└── src/
    └── code_atlas_kotlin/
        └── __init__.py
```

### 2. Define the entry point

In your `pyproject.toml`:

```toml
[project.entry-points."code_atlas.languages"]
kotlin = "code_atlas_kotlin:register"
```

### 3. Implement the register function

```python
# src/code_atlas_kotlin/__init__.py

from tree_sitter import Language, Query
import tree_sitter_kotlin as ts_kotlin

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedFile,
    register_language,
)

_LANG = Language(ts_kotlin.language())
_QUERY = Query(_LANG, "(source_file) @root")


def _parse_kotlin(path, source, root, project_name):
    # ... implementation
    pass


def register():
    """Entry point called by Code Atlas plugin discovery."""
    register_language(
        LanguageConfig(
            name="kotlin",
            extensions=frozenset({".kt", ".kts"}),
            language=_LANG,
            query=_QUERY,
            parse_func=_parse_kotlin,
        )
    )
```

### 4. Install and use

```bash
pip install code-atlas-kotlin
atlas index /path/to/kotlin/project
```

Code Atlas discovers plugins via `importlib.metadata.entry_points(group="code_atlas.languages")` at startup.

## Languages Without Tree-Sitter Grammars

If no PyPI package exists for a language's tree-sitter grammar:

1. **Check `tree-sitter-languages`** — a package that bundles many grammars as pre-built wheels
2. **Build the grammar manually** from its GitHub repo:
   ```bash
   git clone https://github.com/tree-sitter/tree-sitter-swift
   cd tree-sitter-swift
   tree-sitter generate
   # Compile and package as a Python wheel
   ```
3. **Contribute the grammar to PyPI** — most tree-sitter grammars have Python bindings that just need packaging
4. **Regex-based lightweight parser** (last resort) — extract only module-level entities (functions, classes) using
   regex patterns. Very limited but provides basic BM25 indexing coverage.

## Testing Checklist

Each language parser should have tests covering:

1. **Language detection** — `get_language_for_file()` returns config for all registered extensions
2. **Module entity** — every file produces a Module entity with correct qualified name
3. **Type definitions** — classes, structs, interfaces, enums, traits with correct `kind`
4. **Callables** — functions, methods, constructors with correct `kind` distinction
5. **Visibility** — language-specific visibility rules are correctly applied
6. **Imports** — import statements produce IMPORTS relationships
7. **Inheritance** — extends/implements produce INHERITS/IMPLEMENTS relationships
8. **Docstrings** — language-specific doc comment formats are extracted
9. **Signatures** — function/method signatures captured without body
10. **Values** — constants, variables, fields, enum members with correct `kind`
11. **Tags** — decorators, annotations, attributes extracted as tags
12. **DEFINES** — parent→child relationships emitted correctly
13. **CALLS** — call expressions in function bodies produce CALLS relationships
14. **Content hash** — deterministic (same input → same hash)
15. **Edge cases** — empty files, syntax errors don't crash the parser

## Currently Supported Languages

| Language   | Module          | Extensions                                   | Grammar Package          |
| ---------- | --------------- | -------------------------------------------- | ------------------------ |
| Python     | `python.py`     | `.py`, `.pyi`                                | `tree-sitter-python`     |
| Markdown   | `markdown.py`   | `.md`                                        | `tree-sitter-markdown`   |
| TypeScript | `typescript.py` | `.ts`, `.tsx`                                | `tree-sitter-typescript` |
| JavaScript | `typescript.py` | `.js`, `.jsx`, `.mjs`, `.cjs`                | `tree-sitter-javascript` |
| Go         | `go.py`         | `.go`                                        | `tree-sitter-go`         |
| Rust       | `rust.py`       | `.rs`                                        | `tree-sitter-rust`       |
| Java       | `jvm.py`        | `.java`                                      | `tree-sitter-java`       |
| C#         | `jvm.py`        | `.cs`                                        | `tree-sitter-c-sharp`    |
| C          | `cpp.py`        | `.c`, `.h`                                   | `tree-sitter-c`          |
| C++        | `cpp.py`        | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx`, `.hh` | `tree-sitter-cpp`        |
| Ruby       | `ruby.py`       | `.rb`, `.rake`, `.gemspec`                   | `tree-sitter-ruby`       |
| PHP        | `php.py`        | `.php`                                       | `tree-sitter-php`        |
