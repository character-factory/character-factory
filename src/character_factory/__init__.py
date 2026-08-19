"""Character Factory: text description in, rigged 3D human out.

The stable import surface is intentionally small. Today it exposes the
character format (see SPEC.md); the generation and assembly entry points
(`create`, `bake`, `assemble`, `make`) arrive with their modules.
"""

from character_factory.schema import (
    Character,
    ValidationIssue,
    ValidationReport,
    canonical_form,
    character_json_schema,
    content_id,
    validate_document,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Character",
    "ValidationIssue",
    "ValidationReport",
    "canonical_form",
    "character_json_schema",
    "content_id",
    "validate_document",
    "__version__",
]
