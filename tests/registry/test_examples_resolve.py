"""Every committed example must build against the vendored snapshot.

The example characters are the first documents a new user touches, and
their recipe pins are real pins: a version that leaves the registry (say,
in a renumbering) turns an example into a guaranteed first-run failure.
This suite makes that class of drift a CI failure instead — every
committed example must validate strictly and every version it pins must
resolve in the registry snapshot packaged with this very tree.
"""

import pathlib

import pytest

from character_factory.registry import Registry, RegistryIndex, _snapshot_document
from character_factory.schema import Character

EXAMPLES = sorted(
    (pathlib.Path(__file__).parents[2] / "examples/characters").glob("*.char.json")
)


@pytest.fixture(scope="module")
def snapshot():
    # The packaged snapshot specifically — never the local cache index,
    # which may carry staged declarations this tree does not ship.
    return Registry(RegistryIndex(_snapshot_document()))


def test_examples_exist():
    assert EXAMPLES, "examples/characters holds no example documents"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_validates_strictly(path):
    Character.load(path, strict=True)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_recipes_resolve_in_snapshot(path, snapshot):
    character = Character.load(path)
    for slot, maps in character.texture_maps().items():
        recipe = maps["albedo"]
        entry = snapshot.get(recipe["component"], recipe["component_version"])
        assert entry.kind == "texture-adapter"
        assert entry.slot == slot
        # The whole dependency chain must exist: the adapter's declared
        # base model has to be a component the snapshot knows.
        base = entry.document.get("requires", {}).get("base_model")
        if base is not None:
            snapshot.get(base)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_provenance_versions_resolve_in_snapshot(path, snapshot):
    # Provenance pins are records, not build inputs — but a committed
    # example should still name only versions that exist. A backend-variant
    # suffix ("0.1.0+…") pins the base version.
    character = Character.load(path)
    components = character.to_document()["provenance"]["components"]
    assert components
    for name, info in components.items():
        snapshot.get(name, info["version"].partition("+")[0])
