"""UV compositing: keying, region masks, and the normative layer order."""

import numpy as np
import pytest

from character_factory.assembly.composite import (
    AtlasDefinition,
)

SIZE = 32


def solid(r, g, b):
    layer = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = r, g, b
    return layer


def atlas(**kwargs):
    return AtlasDefinition(resolution=SIZE, feather_radius=0, **kwargs)


TOP = (slice(0, 8), slice(None))
BOTTOM = (slice(24, 32), slice(None))
MIDDLE = (slice(12, 20), slice(None))


def test_atlas_metadata_round_trip(tmp_path):
    from PIL import Image

    head = np.zeros((SIZE, SIZE), dtype=np.uint8)
    head[TOP] = 255
    Image.fromarray(head, mode="L").save(tmp_path / "head.png")
    (tmp_path / "atlas.json").write_text(
        '{"format": "character-factory/atlas-metadata", "resolution": 32,'
        ' "masks": {"head": "head.png"}, "key": {"cutoff": 24}}'
    )
    definition = AtlasDefinition.load(tmp_path)
    assert definition.resolution == SIZE
    assert definition.key_cutoff == 24
    assert definition.head_mask[TOP].all() and not definition.head_mask[BOTTOM].any()
