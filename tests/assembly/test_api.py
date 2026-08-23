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
    solid_png(assets / "eye.png", (90, 60, 40))
    solid_png(assets / "garment.png", (0, 0, 0))       # nothing keyed
    solid_png(assets / "shoe.png", (0, 0, 0))
    # Re-pin the example's texture components to what the active registry
    # resolves — the example pins release versions; a staged local index
    # may carry a different version line, and the pipeline (not the pin's
    # absolute value) is what this test exercises.
    import json as jsonlib

    from character_factory.registry import Registry
    from character_factory.schema import Character

    document = jsonlib.loads((EXAMPLES / "marathon-runner.char.json").read_text())
    resolved = Registry.default().resolve_slots(sorted(document["textures"]))
    for slot, recipe in document["textures"].items():
        recipe["component_version"] = str(resolved[slot].version)
    character_path = tmp_path / "runner.char.json"
    Character.from_document(document).save(character_path)
    out = assemble(character_path, assets, tmp_path / "runner.glb")
    data = out.read_bytes()
    report = validate_glb(data, expected_joints=127)
    assert report["reparse_max_error_mm"] < 1e-2

    gltf, binary = parse_glb(data)
    assert gltf["images"][0]["mimeType"] == "image/png"
    # The embedded export manifest makes the file self-describing.
    manifest = gltf["asset"]["extras"]
    assert manifest["format"] == "character-factory/export-manifest"
    assert manifest["joint_count"] == 127
    assert manifest["units"] == "meters"
    material = gltf["materials"][0]["pbrMetallicRoughness"]
    assert material["baseColorTexture"]["index"] == 0

    from character_factory.assembly.gltf import read_accessor as _ra

    # The full assembly: eyes attached to the eye joints, hair to the head.
    mesh_names = {m["name"] for m in gltf["meshes"]}
    assert {"eye_left", "eye_right", "hair",
            "eye_left_backing", "eye_right_backing"} <= mesh_names
    # The hair provider emits inside-out winding; the exporter's decisive
    # winding guard must have flipped it to majority-outward.
    hair = next(m for m in gltf["meshes"] if m["name"] == "hair")
    hair_positions = _ra(gltf, binary, hair["primitives"][0]["attributes"]["POSITION"])
    hair_normals = _ra(gltf, binary, hair["primitives"][0]["attributes"]["NORMAL"])
    outward = hair_positions - hair_positions.mean(axis=0)
    assert ((hair_normals * outward).sum(axis=1) > 0).mean() > 0.75
    names = {node.get("name"): i for i, node in enumerate(gltf["nodes"])}
    for joint, child in (("l_eye", "eye_left"), ("r_eye", "eye_right"),
                         ("c_head", "hair")):
        assert names[child] in gltf["nodes"][names[joint]]["children"]
    # Socket faces were removed from the body: fewer than the full triangle
    # count, and the body still skins exactly (validated above).
    body = next(m for m in gltf["meshes"] if m["name"] == "body")
    indices = _ra(gltf, binary, body["primitives"][0]["indices"])
    assert len(indices) // 3 == 36874 - 64


def test_assemble_is_deterministic(tmp_path):
    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "eye.png", (90, 60, 40))
    solid_png(assets / "garment.png", (0, 0, 0))
    character = Character.load(EXAMPLES / "freediver.char.json")
    first = assemble(character, assets, tmp_path / "a.glb").read_bytes()
    second = assemble(character, assets, tmp_path / "b.glb").read_bytes()
    assert first == second


def test_barefoot_character_needs_no_footwear_asset(tmp_path):
    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "eye.png", (90, 60, 40))
    solid_png(assets / "garment.png", (0, 0, 0))
    out = assemble(
        EXAMPLES / "freediver.char.json", assets, tmp_path / "diver.glb"
    )
    assert np.frombuffer(out.read_bytes()[:4], dtype=np.uint32)[0] == 0x46546C67


def test_mouth_interior_refused_without_rig_mouth_data(tmp_path):
    # SPEC.md §4.2: assembling a mouth-interior document against a body-rig
    # version that declares no mouth data is a defined error — never a
    # silent fall back to the closed surface.
    import json as jsonlib

    from character_factory.api import assemble

    rig_metadata = jsonlib.loads(
        (Path(_real_rig_dir()) / "rig.json").read_text()
    )
    if "mouth" in rig_metadata:
        pytest.skip("resolved body-rig declares mouth data")

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "eye.png", (90, 60, 40))
    solid_png(assets / "garment.png", (0, 0, 0))
    document = jsonlib.loads((EXAMPLES / "freediver.char.json").read_text())
    document["body"]["topology"] = "mouth-interior"
    with pytest.raises(ValueError, match="mouth data"):
        assemble(Character.from_document(document), assets, tmp_path / "m.glb")
