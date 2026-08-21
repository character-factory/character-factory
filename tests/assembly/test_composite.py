"""UV compositing: keying, region masks, and the normative layer order."""

import numpy as np
import pytest

from character_factory.assembly.composite import (
    AtlasDefinition,
    composite_albedo,
    coverage_mask,
)

SIZE = 32


def solid(r, g, b):
    layer = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = r, g, b
    return layer


def painted(region, color):
    layer = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    layer[region] = color
    return layer


def overlay(region, color):
    """A shoe overlay: RGBA with authoritative alpha (footwear bake output)."""
    layer = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    layer[region] = (*color, 255)
    return layer


def atlas(**kwargs):
    return AtlasDefinition(resolution=SIZE, feather_radius=0, **kwargs)


TOP = (slice(0, 8), slice(None))
BOTTOM = (slice(24, 32), slice(None))
MIDDLE = (slice(12, 20), slice(None))


def test_key_recovers_painted_region():
    layer = painted(MIDDLE, (200, 40, 40))
    mask = coverage_mask(layer, cutoff=20, feather_radius=0)
    assert mask[MIDDLE].min() == 1.0
    assert mask[TOP].max() == 0.0


def test_feather_softens_inward_never_bleeds_outward():
    layer = painted(MIDDLE, (200, 40, 40))
    mask = coverage_mask(layer, cutoff=20, feather_radius=1.5)
    assert mask[15, 16] == 1.0                 # fully interior stays 1
    assert 0.0 < mask[12, 16] < 1.0            # covered edge is softened
    assert mask[TOP].max() == 0.0              # never paints outside the key
    assert mask[BOTTOM].max() == 0.0


def test_dark_garment_survives_the_key():
    # The generator's value floor keeps real cloth above the cutoff; a very
    # dark garment (max channel 64) must still key as covered.
    layer = painted(MIDDLE, (64, 60, 58))
    mask = coverage_mask(layer, cutoff=20, feather_radius=0)
    assert mask[MIDDLE].min() == 1.0


def test_compositing_order_skin_garments_footwear():
    skin = solid(100, 80, 60)
    garments = painted(MIDDLE, (10, 200, 10))
    garments[BOTTOM] = (200, 10, 10)          # garment also paints the feet…
    footwear = overlay(BOTTOM, (10, 10, 200))  # …but footwear wins there
    result = composite_albedo(skin, garments, footwear, atlas())
    assert tuple(result[16, 16]) == (10, 200, 10)   # garment over skin
    assert tuple(result[28, 16]) == (10, 10, 200)   # footwear over garment
    assert tuple(result[4, 16]) == (100, 80, 60)    # bare skin untouched


def test_head_mask_blocks_garments():
    head = np.zeros((SIZE, SIZE), dtype=bool)
    head[TOP] = True
    garments = painted(TOP, (200, 200, 200))
    result = composite_albedo(solid(90, 90, 90), garments, None, atlas(head_mask=head))
    assert tuple(result[4, 16]) == (90, 90, 90)


def test_feet_mask_confines_footwear():
    feet = np.zeros((SIZE, SIZE), dtype=bool)
    feet[BOTTOM] = True
    footwear = overlay(MIDDLE, (200, 200, 200))   # tries to paint the torso
    footwear[BOTTOM] = (30, 30, 220, 255)
    result = composite_albedo(
        solid(90, 90, 90), np.zeros((SIZE, SIZE, 3), np.uint8), footwear,
        atlas(feet_mask=feet),
    )
    assert tuple(result[16, 16]) == (90, 90, 90)    # torso untouched
    assert tuple(result[28, 16]) == (30, 30, 220)   # feet shod


def test_barefoot_is_no_footwear_layer():
    skin = solid(90, 90, 90)
    result = composite_albedo(skin, np.zeros((SIZE, SIZE, 3), np.uint8), None, atlas())
    assert (result == skin).all()


def test_wrong_resolution_rejected():
    with pytest.raises(ValueError):
        composite_albedo(
            solid(1, 1, 1),
            np.zeros((16, 16, 3), np.uint8),
            None,
            atlas(),
        )


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
