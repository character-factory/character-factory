"""The gallery thumbnail renderer: a software rasterization of the exported
scene. Tested on the synthetic rig's export — the point is that a figure
lands on the canvas, lit and textured, with a transparent background."""

import io

import numpy as np
import pytest

pytest.importorskip("PIL")

from character_factory.assembly import export_character_glb
from character_factory.assembly.thumbnail import render_thumbnail


def test_renders_a_figure_on_transparent_background(rig, tmp_path):
    from PIL import Image

    result = export_character_glb(
        rig, [0.0, 0.0], [0.0, 0.0], tmp_path / "scene.glb",
        generator="character-factory/test", _body_only_test=True,
    )
    png = render_thumbnail(result.glb_path.read_bytes(), width=160, height=200)
    image = np.asarray(Image.open(io.BytesIO(png)))
    assert image.shape == (200, 160, 4)
    coverage = (image[:, :, 3] > 0).mean()
    # Something rendered, but not a full-bleed wash (the synthetic rig is
    # a six-triangle stick figure, so coverage is legitimately small).
    assert 0.002 < coverage < 0.9
    # The corners stay transparent (the background is the tile's own).
    assert image[0, 0, 3] == 0 and image[-1, -1, 3] == 0
    # Lit, shaded pixels: some variation, nothing blown out.
    lit = image[image[:, :, 3] > 0][:, :3]
    assert lit.std() > 1.0


def test_render_is_deterministic(rig, tmp_path):
    result = export_character_glb(
        rig, [0.0, 0.0], [0.0, 0.0], tmp_path / "scene.glb",
        generator="character-factory/test", _body_only_test=True,
    )
    data = result.glb_path.read_bytes()
    assert render_thumbnail(data, width=96, height=128) == \
        render_thumbnail(data, width=96, height=128)
