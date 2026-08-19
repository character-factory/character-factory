"""The `Character` object: a validated character document with identity.

Design note: `Character` wraps the JSON document rather than exploding it
into a parallel dataclass tree. This is deliberate — the format promises
that documents from a newer schema minor version keep their unrecognized
optional fields through a read/write round-trip (SPEC.md §10), which a fixed
field list would silently drop. Typed access is provided for the fields
code actually touches.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from character_factory.schema.canonical import canonical_form, content_id, float32_value
from character_factory.schema.validation import ValidationReport, validate_document

__all__ = ["Character", "CharacterError"]


class CharacterError(ValueError):
    """Raised when a document fails validation. Carries the full report."""

    def __init__(self, report: ValidationReport):
        self.report = report
        lines = [str(issue) for issue in report.errors]
        super().__init__("invalid character document:\n  " + "\n  ".join(lines))


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid in a character document")


def _normalize_float32(document: dict) -> None:
    """Snap parameter arrays to exact float32 values (SPEC.md §2), in place.

    A no-op for documents written by a conforming writer; it makes the
    content ID well-defined even when a hand-edited file carries excess
    precision.
    """
    body = document.get("body", {})
    for key in ("identity", "resting_expression"):
        values = body.get(key)
        if isinstance(values, list):
            body[key] = [float32_value(v) for v in values]
    hair = document.get("hair")
    if isinstance(hair, dict):
        rgb = hair.get("color", {}).get("rgb") if isinstance(hair.get("color"), dict) else None
        if isinstance(rgb, list):
            hair["color"]["rgb"] = [float32_value(v) for v in rgb]


class Character:
    """An immutable-by-convention, validated character document.

    Construct via :meth:`load` / :meth:`loads` / :meth:`from_document`;
    construction validates (default mode) and raises :class:`CharacterError`
    on any error. Warnings are kept on :attr:`load_report`.
    """

    def __init__(self, document: dict, *, strict: bool = False):
        document = copy.deepcopy(document)
        report = validate_document(document, strict=strict)
        if not report.ok:
            raise CharacterError(report)
        _normalize_float32(document)
        self._document = document
        self.load_report = report

    # -- construction --------------------------------------------------------

    @classmethod
    def from_document(cls, document: dict, *, strict: bool = False) -> "Character":
        return cls(document, strict=strict)

    @classmethod
    def loads(cls, text: str, *, strict: bool = False) -> "Character":
        document = json.loads(text, parse_constant=_reject_constant)
        return cls(document, strict=strict)

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = False) -> "Character":
        return cls.loads(Path(path).read_text(encoding="utf-8"), strict=strict)

    # -- serialization ---------------------------------------------------------

    def to_document(self) -> dict:
        return copy.deepcopy(self._document)

    def dumps(self) -> str:
        return json.dumps(
            self._document, indent=2, ensure_ascii=False, allow_nan=False
        ) + "\n"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.dumps(), encoding="utf-8")
        return path

    # -- identity ---------------------------------------------------------------

    def canonical(self) -> bytes:
        """RFC 8785 canonical form of the document (SPEC.md §2.1)."""
        return canonical_form(self._document)

    @property
    def content_id(self) -> str:
        """SHA-256 of the canonical form: the character's identity."""
        return content_id(self._document)

    def validate(self, *, strict: bool = False) -> ValidationReport:
        return validate_document(self._document, strict=strict)

    # -- typed access -------------------------------------------------------------

    @property
    def schema_version(self) -> str:
        return self._document["schema_version"]

    @property
    def name(self) -> str | None:
        return self._document.get("name")

    @property
    def rig(self) -> str:
        return self._document["body"]["rig"]

    @property
    def topology(self) -> str:
        return self._document["body"]["topology"]

    @property
    def identity(self) -> list[float]:
        return list(self._document["body"]["identity"])

    @property
    def resting_expression(self) -> list[float]:
        return list(self._document["body"]["resting_expression"])

    @property
    def textures(self) -> dict[str, dict]:
        """Texture recipes by slot name, present slots only."""
        return copy.deepcopy(self._document["textures"])

    @property
    def hair(self) -> dict | None:
        return copy.deepcopy(self._document["hair"])

    @property
    def provenance(self) -> dict:
        return copy.deepcopy(self._document["provenance"])

    @property
    def assets(self) -> dict[str, dict]:
        return copy.deepcopy(self._document.get("assets", {}))

    @property
    def prompt(self) -> str | None:
        return self._document["provenance"].get("prompt")

    def __repr__(self) -> str:
        label = self.name or self.content_id[:12]
        slots = ", ".join(sorted(self._document["textures"]))
        return f"Character({label!r}, rig={self.rig!r}, slots=[{slots}])"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Character) and other.canonical() == self.canonical()

    def __hash__(self) -> int:
        return hash(self.content_id)
