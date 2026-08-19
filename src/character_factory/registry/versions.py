"""Semantic-version handling for component versions and schema ranges.

Component versions are plain `major.minor.patch`. Schema compatibility ranges
use the two comparator forms the registry actually emits (`>=X.Y` and
`<X.Y`, space-separated), kept deliberately smaller than full semver range
syntax so third parties can reimplement it in a few lines.
"""

from __future__ import annotations

import re

__all__ = ["Version", "schema_range_allows"]

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_COMPARATOR_RE = re.compile(r"^(>=|<=|>|<|==)(\d+)\.(\d+)$")


class Version(tuple):
    """An orderable `major.minor.patch` version."""

    def __new__(cls, text: str) -> "Version":
        match = _VERSION_RE.match(text)
        if not match:
            raise ValueError(f"not a major.minor.patch version: {text!r}")
        return super().__new__(cls, (int(match.group(1)), int(match.group(2)), int(match.group(3))))

    def __str__(self) -> str:
        return ".".join(str(part) for part in self)


def _schema_tuple(text: str) -> tuple[int, int]:
    major, _, minor = text.partition(".")
    return int(major), int(minor)


def schema_range_allows(range_text: str, schema_version: str) -> bool:
    """Whether a `requires.schema` range (e.g. ``">=0.1 <1.0"``) admits a
    character schema version (e.g. ``"0.1"``)."""
    current = _schema_tuple(schema_version)
    for comparator in range_text.split():
        match = _COMPARATOR_RE.match(comparator)
        if not match:
            raise ValueError(f"unsupported schema range comparator: {comparator!r}")
        op, major, minor = match.group(1), int(match.group(2)), int(match.group(3))
        bound = (major, minor)
        ok = {
            ">=": current >= bound,
            "<=": current <= bound,
            ">": current > bound,
            "<": current < bound,
            "==": current == bound,
        }[op]
        if not ok:
            return False
    return True
