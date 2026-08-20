"""The HairProvider boundary over the vendored engine."""

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from character_factory.hair import HeadGeometry, WigProvider  # noqa: E402

ASSETS = Path(__file__).parent / "assets"

INTENT = {
    "schema_version": 1, "seed": 11, "family": "crop",
    "part": {"kind": "none", "side": "wearer_left", "position": "moderate",
             "extent": "to_crown", "width": "narrow"},
    "hairline": {"height": "natural", "shape": "rounded",
                 "temple_recession": "natural", "sideburns": "natural",
                 "nape": "natural", "irregularity": "natural"},
    "length": {"overall": "cropped", "cut_line": "soft"},
    "shape": {"volume": "low", "density": "medium", "texture": "straight",
              "wave_size": "medium", "wave_strength": "medium",
              "root_lift": "medium"},
    "drape": {"gravity": "natural", "stiffness": "natural",
              "shoulder_routing": "split", "body_clearance": "natural"},
    "color": {"family": "dark_brown"},
}


@pytest.fixture(scope="module")
def head() -> HeadGeometry:
    mesh = trimesh.load(ASSETS / "mannequin.obj")
    return HeadGeometry(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        eye_level=-1.0,
    )


def test_synthesize_returns_textured_mesh(head):
    result = WigProvider().synthesize(INTENT, head)
    assert len(result.mesh.vertices) > 1000
    assert len(result.mesh.faces) > 1000
    assert result.mesh.visual.material.baseColorTexture is not None
    assert result.manifest["provider"] == "make-wig"
    assert result.manifest["compiler_version"]


def test_synthesize_is_deterministic(head):
    provider = WigProvider()
    first = provider.synthesize(INTENT, head)
    second = provider.synthesize(INTENT, head)
    assert np.array_equal(
        np.asarray(first.mesh.vertices), np.asarray(second.mesh.vertices)
    )


def test_intent_errors_surface(head):
    from character_factory.hair.wig import HairIntentError

    bad = dict(INTENT, family="tonsure")
    with pytest.raises(HairIntentError):
        WigProvider().synthesize(bad, head)
