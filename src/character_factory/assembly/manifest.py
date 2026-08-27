"""Versioned machine-readable export-manifest contract."""

from __future__ import annotations

import json
from importlib import resources

MANIFEST_SCHEMA_VERSION = "0.6"
MANIFEST_SCHEMA_PATH = "/v0/schemas/export-manifest-0.6.json"


class ManifestVersionError(ValueError):
    """The document is not the export-manifest contract this reader supports."""


def export_manifest_schema() -> dict:
    path = resources.files("character_factory.assembly").joinpath(
        "data/export-manifest-0.6.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def require_supported_manifest(
    manifest: dict, *, supported_version: str = MANIFEST_SCHEMA_VERSION
) -> None:
    """Fail loudly when a consumer has not adopted a manifest version.

    Downstream importers call this before reading fields they depend on.
    There is one pre-release contract, so support is exact rather than an
    implicit legacy-guessing table.
    """
    if manifest.get("format") != "character-factory/export-manifest":
        raise ManifestVersionError("not a Character Factory export manifest")
    actual = manifest.get("schema_version")
    if actual != supported_version:
        raise ManifestVersionError(
            f"unsupported export manifest {actual!r}; reader supports "
            f"{supported_version!r}"
        )


__all__ = [
    "MANIFEST_SCHEMA_PATH", "MANIFEST_SCHEMA_VERSION", "ManifestVersionError",
    "export_manifest_schema", "require_supported_manifest"
]
