"""End-to-end assemble: character file + baked assets → validated rigged GLB.
Runs only when the real body-rig component is cached locally."""

from pathlib import Path

import numpy as np
import pytest

from character_factory import Character
from character_factory.assembly.gltf import parse_glb
from character_factory.assembly.validate import validate_glb

from tests.assembly.test_export import _real_rig_dir

pytestmark = pytest.mark.skipif(
    _real_rig_dir() is None,
    reason="body-rig component not present in the local cache",
)

EXAMPLES = Path(__file__).parents[2] / "examples" / "characters"


def solid_png(path: Path, color, size=64):
    from PIL import Image

    Image.new("RGB", (size, size), color).save(path)


def test_assemble_end_to_end(tmp_path):
    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (180, 140, 110))
    solid_png(assets / "garments.png", (0, 0, 0))       # nothing keyed
    solid_png(assets / "footwear.png", (0, 0, 0))
    out = assemble(
        EXAMPLES / "marathon-runner.char.json", assets, tmp_path / "runner.glb"
    )
    data = out.read_bytes()
    report = validate_glb(data, expected_joints=127)
    assert report["reparse_max_error_mm"] < 1e-2

    gltf, binary = parse_glb(data)
    assert gltf["images"][0]["mimeType"] == "image/png"
    material = gltf["materials"][0]["pbrMetallicRoughness"]
    assert material["baseColorTexture"]["index"] == 0


def test_assemble_is_deterministic(tmp_path):
    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "garments.png", (0, 0, 0))
    character = Character.load(EXAMPLES / "freediver.char.json")
    first = assemble(character, assets, tmp_path / "a.glb").read_bytes()
    second = assemble(character, assets, tmp_path / "b.glb").read_bytes()
    assert first == second


def test_barefoot_character_needs_no_footwear_asset(tmp_path):
    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "garments.png", (0, 0, 0))
    out = assemble(
        EXAMPLES / "freediver.char.json", assets, tmp_path / "diver.glb"
    )
    assert np.frombuffer(out.read_bytes()[:4], dtype=np.uint32)[0] == 0x46546C67
