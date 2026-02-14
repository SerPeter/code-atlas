"""Parsing package — tree-sitter AST extraction and pattern detectors."""

from __future__ import annotations

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    get_language_for_file,
    parse_file,
    register_language,
)
from code_atlas.parsing.detectors import (
    Detector,
    DetectorResult,
    PropertyEnrichment,
    get_enabled_detectors,
    register_detector,
    run_detectors,
)

__all__ = [
    "Detector",
    "DetectorResult",
    "LanguageConfig",
    "ParsedEntity",
    "ParsedFile",
    "ParsedRelationship",
    "PropertyEnrichment",
    "get_enabled_detectors",
    "get_language_for_file",
    "parse_file",
    "register_detector",
    "register_language",
    "run_detectors",
]
