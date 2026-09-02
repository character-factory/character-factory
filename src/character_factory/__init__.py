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

__version__ = "0.1.0"

__all__ = [
    "Character",
    "ValidationIssue",
    "ValidationReport",
    "assemble",
    "create",
    "make",
    "canonical_form",
    "character_json_schema",
    "content_id",
    "validate_document",
    "__version__",
]


def __getattr__(name: str):
    # Heavier entry points import lazily so `import character_factory` stays
    # instant and torch-free until something actually needs them.
    if name in ("assemble", "create", "make"):
        from character_factory import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
