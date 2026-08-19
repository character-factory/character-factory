"""The character format: model, validation, canonical form, JSON Schema.

This package is the reference implementation of SPEC.md and is deliberately
dependency-free (standard library only): validating, editing, and hashing
character files works on any machine that can run Python.
"""

from __future__ import annotations

import json
from importlib import resources

from character_factory.schema import vocab
from character_factory.schema.canonical import canonical_form, content_id, float32_value
from character_factory.schema.model import Character, CharacterError
from character_factory.schema.validation import (
    ValidationIssue,
    ValidationReport,
    validate_document,
)

__all__ = [
    "Character",
    "CharacterError",
    "ValidationIssue",
    "ValidationReport",
    "canonical_form",
    "character_json_schema",
    "content_id",
    "float32_value",
    "validate_document",
    "vocab",
]


def character_json_schema() -> dict:
    """The published JSON Schema for character format v0.1 (strict flavor).

    Third parties can validate against this directly; it mirrors the closed
    vocabularies in :mod:`character_factory.schema.vocab`.
    """
    data = resources.files("character_factory.schema").joinpath(
        "data/character-0.1.schema.json"
    )
    return json.loads(data.read_text(encoding="utf-8"))
