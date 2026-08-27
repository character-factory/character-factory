"""The one current export-manifest schema and downstream fixture."""

import json
from pathlib import Path

import pytest
from jsonschema import validate

from character_factory.assembly import (
    ManifestVersionError,
    export_manifest_schema,
    require_supported_manifest,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "export-manifest-0.6.json"


def test_versioned_downstream_fixture_matches_the_published_schema():
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    require_supported_manifest(manifest)
    validate(manifest, export_manifest_schema())
    assert manifest["topology"] == "mouth-interior"
    assert manifest["humanoid_map"]["map"]
    assert manifest["expression_morphs"]["count"] == 72
    assert manifest["expression_morphs"]["names"] == [
        f"facs_{index:02d}" for index in range(72)
    ]


def test_future_or_missing_version_is_never_guessed():
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manifest["schema_version"] = "0.7"
    with pytest.raises(ManifestVersionError, match="unsupported"):
        require_supported_manifest(manifest)
    del manifest["schema_version"]
    with pytest.raises(ManifestVersionError, match="unsupported"):
        require_supported_manifest(manifest)
