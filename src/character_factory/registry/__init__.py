"""The component registry: versioned, hash-pinned models and assets.

Weights never live in this repository or the Python package. Every model and
static asset is a **component**: named, semver-versioned, SHA-256 pinned,
fetched on first use into a local cache, and described by a JSON index. New
capability arrives as registry data, not code (ARCHITECTURE.md §4).

This package is stdlib-only so that the assembly-only install (macOS story,
ARCHITECTURE.md §6.2) can fetch and verify components without the generation
stack.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from character_factory.registry.model import (
    ComponentEntry,
    RegistryError,
    RegistryIndex,
    parse_ref,
)
from character_factory.registry.store import (
    ComponentNotPublished,
    IntegrityError,
    cache_dir,
    ensure_component,
)
from character_factory.schema import vocab

__all__ = [
    "ComponentEntry",
    "ComponentNotPublished",
    "IntegrityError",
    "Registry",
    "RegistryError",
    "RegistryIndex",
    "cache_dir",
    "parse_ref",
]


def _snapshot_document() -> dict:
    data = resources.files("character_factory.registry").joinpath(
        "data/registry-snapshot.json"
    )
    return json.loads(data.read_text(encoding="utf-8"))


class Registry:
    """Component lookup, resolution, and retrieval against one index.

    The index comes from, in order of preference: an explicit path, a
    previously refreshed copy in the local cache, or the snapshot vendored
    into the package (the offline fallback that ships with every release).
    """

    def __init__(self, index: RegistryIndex):
        self.index = index

    @classmethod
    def default(cls) -> "Registry":
        cached = cache_dir() / "registry.json"
        if cached.is_file():
            return cls(RegistryIndex(json.loads(cached.read_text(encoding="utf-8"))))
        return cls(RegistryIndex(_snapshot_document()))

    @classmethod
    def from_path(cls, path: str | Path) -> "Registry":
        return cls(RegistryIndex(json.loads(Path(path).read_text(encoding="utf-8"))))

    @classmethod
    def refresh(cls) -> "Registry":
        """Fetch the configured alternate registry index, validate it, cache
        it, and return a Registry over it. Requires a configured
        ``registry_url`` (environment or config file — see
        :mod:`character_factory.registry.config`)."""
        from character_factory.registry.config import load_config
        from character_factory.registry.store import fetch_json

        config = load_config()
        if not config.registry_url:
            raise RegistryError(
                "no registry_url configured: set CHARACTER_FACTORY_REGISTRY_URL "
                "or add registry_url to the config file"
            )
        index = RegistryIndex(fetch_json(config.registry_url, config.headers()))
        target = cache_dir() / "registry.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(index.document, indent=2) + "\n", encoding="utf-8"
        )
        return cls(index)

    # -- resolution -----------------------------------------------------------

    def get(self, ref_or_name: str, version: str | None = None) -> ComponentEntry:
        """Resolve ``"name"``, ``"name@1.2.0"``, or (name, version)."""
        name, ref_version = parse_ref(ref_or_name)
        return self.index.get(name, version or ref_version, vocab.SCHEMA_VERSION)

    def resolve_slots(
        self, slots: list[str], map_name: str = "albedo"
    ) -> dict[str, ComponentEntry]:
        """The component that serves each requested slot's named map, newest
        compatible version of each — the same resolution `create` pins into
        a character's provenance. Multiple components may register against
        one slot (SPEC.md §5); the (slot, map) pair is what resolves."""
        resolved: dict[str, ComponentEntry] = {}
        for slot in slots:
            candidates = [
                entry
                for entry in self.index.entries
                if entry.kind == "texture-adapter" and entry.slot == slot
                and entry.map == map_name
                and entry.compatible_with_schema(vocab.SCHEMA_VERSION)
            ]
            if not candidates:
                raise RegistryError(
                    f"no component serves texture slot {slot!r} (map {map_name!r})"
                )
            resolved[slot] = max(candidates, key=lambda e: e.version)
        return resolved

    # -- constraints (the interpreter's view) ------------------------------------

    def vocabulary_for(self, ref_or_name: str, version: str | None = None
                       ) -> dict[str, list[str]]:
        """One component's declared supported vocabularies (empty dict =
        unconstrained)."""
        return self.get(ref_or_name, version).vocabulary

    def vocabulary_by_slot(self, slots: list[str]) -> dict[str, dict[str, list[str]]]:
        """What the interpreter queries before writing slot prompts: for each
        requested slot, the resolved component's declared vocabularies. Slots
        whose component declares nothing map to an empty dict (unconstrained).
        The resolution used here is the same one pinned into provenance, so
        the interpreter clamps against exactly the components that will run.
        """
        return {
            slot: entry.vocabulary
            for slot, entry in self.resolve_slots(slots).items()
        }

    # -- retrieval ---------------------------------------------------------------

    def ensure(self, ref_or_name: str, version: str | None = None, **kwargs) -> Path:
        """Local directory of the component's verified artifacts, fetching
        anything missing. Raises ComponentNotPublished for entries whose
        weights have not been published yet. Configured auth headers apply
        to every download unless a custom `fetch` is injected."""
        if "fetch" not in kwargs and "headers" not in kwargs:
            from character_factory.registry.config import load_config

            kwargs["headers"] = load_config().headers()
        return ensure_component(self.get(ref_or_name, version), **kwargs)
