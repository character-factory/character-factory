"""Registry index model and validation.

The registry is a JSON index of components: versioned, hash-pinned artifacts
(model weights, the body rig, static assets) fetched on first use. See
ARCHITECTURE.md §4. Like the character schema, this module is stdlib-only and
document-backed: unknown optional fields are preserved, unknown *values* that
change behavior are errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from character_factory.registry.versions import Version, schema_range_allows
from character_factory.schema import vocab

__all__ = ["ComponentEntry", "RegistryError", "RegistryIndex"]

REGISTRY_FORMAT = "character-factory/registry"
REGISTRY_VERSION = "0.1"

KINDS = frozenset(
    {
        "identity", "interpreter", "texture-adapter", "hair-provider",
        "base-model", "body-rig", "assets",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class RegistryError(ValueError):
    """A malformed registry index or an unsatisfiable request against it."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


@dataclass(frozen=True)
class ComponentEntry:
    """One (name, version) row of the registry, validated."""

    document: dict

    def __post_init__(self) -> None:
        doc = self.document
        _require(isinstance(doc, dict), "component entry must be an object")
        _require(
            isinstance(doc.get("name"), str) and _NAME_RE.match(doc["name"]),
            f"bad component name: {doc.get('name')!r}",
        )
        try:
            Version(doc.get("version", ""))
        except ValueError as error:
            raise RegistryError(f"{self.ref}: {error}") from error
        _require(doc.get("kind") in KINDS, f"{self.ref}: unknown kind {doc.get('kind')!r}")
        slot = doc.get("slot")
        if slot is not None:
            _require(slot in vocab.ALL_SLOTS, f"{self.ref}: unknown slot {slot!r}")
        map_name = doc.get("map")
        if map_name is not None:
            _require(
                isinstance(map_name, str) and map_name,
                f"{self.ref}: map must be a non-empty string",
            )
        requires = doc.get("requires", {})
        _require(isinstance(requires, dict), f"{self.ref}: requires must be an object")
        if "schema" in requires:
            try:
                schema_range_allows(requires["schema"], vocab.SCHEMA_VERSION)
            except ValueError as error:
                raise RegistryError(f"{self.ref}: {error}") from error
        artifacts = doc.get("artifacts", [])
        _require(isinstance(artifacts, list), f"{self.ref}: artifacts must be an array")
        for artifact in artifacts:
            _require(
                isinstance(artifact, dict)
                and isinstance(artifact.get("path"), str)
                and ".." not in artifact["path"]
                and not artifact["path"].startswith("/")
                and isinstance(artifact.get("sha256"), str)
                and _SHA256_RE.match(artifact["sha256"])
                and isinstance(artifact.get("bytes"), int)
                and artifact["bytes"] > 0,
                f"{self.ref}: malformed artifact {artifact!r}",
            )
        source = doc.get("source")
        if source is not None:
            _require(
                isinstance(source, dict)
                and isinstance(source.get("hf_repo"), str)
                and isinstance(source.get("revision"), str),
                f"{self.ref}: source needs hf_repo and revision",
            )
        constraints = doc.get("constraints")
        if constraints is not None:
            _require(isinstance(constraints, dict), f"{self.ref}: constraints must be an object")
            vocabulary = constraints.get("vocabulary")
            if vocabulary is not None:
                _require(
                    isinstance(vocabulary, dict)
                    and all(
                        isinstance(values, list) and values
                        and all(isinstance(v, str) for v in values)
                        for values in vocabulary.values()
                    ),
                    f"{self.ref}: constraints.vocabulary must map names to "
                    f"non-empty string arrays",
                )

    # -- accessors -----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.document["name"]

    @property
    def version(self) -> Version:
        return Version(self.document["version"])

    @property
    def ref(self) -> str:
        return f"{self.document.get('name')}@{self.document.get('version')}"

    @property
    def kind(self) -> str:
        return self.document["kind"]

    @property
    def slot(self) -> str | None:
        return self.document.get("slot")

    @property
    def map(self) -> str:
        """Which named map of its slot a texture component produces.
        Defaults to `albedo`; a future normal-map entry is pure data."""
        return self.document.get("map", "albedo")

    @property
    def artifacts(self) -> list[dict]:
        return list(self.document.get("artifacts", []))

    @property
    def source(self) -> dict | None:
        return self.document.get("source")

    @property
    def inference(self) -> dict:
        return dict(self.document.get("inference", {}))

    @property
    def vocabulary(self) -> dict[str, list[str]]:
        """Declared supported-vocabulary constraints (empty = unconstrained)."""
        constraints = self.document.get("constraints") or {}
        return {k: list(v) for k, v in (constraints.get("vocabulary") or {}).items()}

    def compatible_with_schema(self, schema_version: str) -> bool:
        requires = self.document.get("requires", {})
        if "schema" not in requires:
            return True
        return schema_range_allows(requires["schema"], schema_version)


class RegistryIndex:
    """A validated registry document: every component entry, all versions."""

    def __init__(self, document: dict):
        _require(isinstance(document, dict), "registry index must be an object")
        _require(
            document.get("format") == REGISTRY_FORMAT,
            f'registry format must be "{REGISTRY_FORMAT}"',
        )
        _require(
            document.get("registry_version") == REGISTRY_VERSION,
            f"unsupported registry_version {document.get('registry_version')!r} "
            f"(this implementation supports {REGISTRY_VERSION})",
        )
        components = document.get("components")
        _require(isinstance(components, list), "components must be an array")
        self.document = document
        self.entries: list[ComponentEntry] = [ComponentEntry(c) for c in components]
        seen: set[str] = set()
        for entry in self.entries:
            _require(entry.ref not in seen, f"duplicate component entry {entry.ref}")
            seen.add(entry.ref)

    def versions_of(self, name: str) -> list[ComponentEntry]:
        return sorted(
            (e for e in self.entries if e.name == name), key=lambda e: e.version
        )

    def get(self, name: str, version: str | None = None,
            schema_version: str = vocab.SCHEMA_VERSION) -> ComponentEntry:
        """Resolve a component: exact version if given, else the newest one
        compatible with the running character schema version."""
        candidates = self.versions_of(name)
        _require(bool(candidates), f"unknown component {name!r}")
        if version is not None:
            wanted = Version(version)
            for entry in candidates:
                if entry.version == wanted:
                    return entry
            raise RegistryError(f"component {name}@{version} is not in the registry")
        compatible = [e for e in candidates if e.compatible_with_schema(schema_version)]
        _require(
            bool(compatible),
            f"no version of {name!r} is compatible with schema {schema_version}",
        )
        return compatible[-1]


def parse_ref(ref: str) -> tuple[str, str | None]:
    """Split ``"name@1.2.0"`` / ``"name"`` into (name, version-or-None)."""
    name, _, version = ref.partition("@")
    return name, (version or None)
