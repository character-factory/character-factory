"""End-to-end assemble: character file + baked assets → validated rigged GLB.
Runs only when the real body-rig component is cached locally."""

from pathlib import Path

import numpy as np
import pytest

from character_factory import Character
from character_factory.assembly.gltf import parse_glb
from character_factory.assembly.validate import validate_glb

from tests.assembly.test_export import _real_rig_dir, source_topology_registry

pytestmark = pytest.mark.skipif(
    _real_rig_dir() is None,
    reason="body-rig component not present in the local cache",
)

EXAMPLES = Path(__file__).parents[2] / "examples" / "characters"


def solid_png(path: Path, color, size=64):
    from PIL import Image

    Image.new("RGB", (size, size), color).save(path)


def band_garment_png(path, resolution=1024, color=(140, 60, 60)):
    """A fitted torso-band garment synthesized from the real rig's own
    geometry, so it keys cleanly and extracts on any identity."""
    from PIL import Image

    from character_factory.assembly import garment_shell as gs
    from character_factory.assembly.rig import load_rig

    rig = load_rig(_real_rig_dir())
    v = rig.evaluate([0.0] * 45, [0.0] * 72).vertices
    band = ((v[rig.faces][:, :, 1] > 95)
            & (v[rig.faces][:, :, 1] < 135)).all(axis=1)
    mask = np.full((resolution, resolution), -np.inf)
    for face in rig.texcoords[rig.texcoord_faces][band] * (resolution - 1):
        gs._fill(mask, face[0], face[1], face[2], 1.0, 1.0, 1.0)
    rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    rgb[mask > -np.inf] = color
    Image.fromarray(rgb).save(path)


def test_assemble_end_to_end(tmp_path):
    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (180, 140, 110))
    solid_png(assets / "eye.png", (90, 60, 40))
    band_garment_png(assets / "garment.png")
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
    out = assemble(character_path, assets, tmp_path / "runner.glb",
                   registry=source_topology_registry())
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
    # Every slot ships as a shell — there is no painted mode. The shoe
    # overlay's alpha is authoritative occupancy, so even a black canvas
    # bakes real coverage: black shoes, two feet, upper + sole islands.
    assert manifest["garments"]["garment"]["render_mode"] == "shell"
    shoe_entry = manifest["garments"]["shoe"]
    assert shoe_entry["render_mode"] == "shell"
    assert shoe_entry["shell"]["hidden_body_faces"] > 0
    material = gltf["materials"][0]["pbrMetallicRoughness"]
    assert material["baseColorTexture"]["index"] == 0

    # Grounding describes what ships: with the barefoot sole deleted
    # under the shoe shell, the declared plane is the minimum over ALL
    # render geometry — the shoe sole — never the truncated body's edge.
    from character_factory.assembly.gltf import read_accessor as _ra_g

    minima = {}
    for mesh in gltf["meshes"]:
        if mesh["name"] in ("body", "garment", "shoe"):
            pos = _ra_g(gltf, binary,
                        mesh["primitives"][0]["attributes"]["POSITION"])
            minima[mesh["name"]] = float(pos[:, 1].min())
    plane = gltf["asset"]["extras"]["grounding"]["plane_height_m"]
    assert abs(plane - min(minima.values())) < 1e-5
    assert minima["shoe"] == min(minima.values())  # the shoe is the floor

    # Delivery compression on the real thing: the structural inventory a
    # consumer relies on survives, and only the images change.
    from character_factory.assembly.compress import compress_glb

    for target, mime in (("web", "image/webp"), ("unity", "image/jpeg")):
        compressed = compress_glb(data, target)
        assert len(compressed) < len(data)
        small, _ = parse_glb(compressed)
        body = [m for m in small["meshes"] if m["name"] == "body"][0]
        assert len(body["primitives"][0]["targets"]) == 72
        assert len(body["extras"]["targetNames"]) == 72
        assert sum("skin" in node for node in small["nodes"]) == 3
        assert small["animations"][0]["name"] == "idle"
        assert "humanoid_map" in small["asset"]["extras"]
        assert {image["mimeType"] for image in small["images"]} == {mime}
        assert small["meshes"] == gltf["meshes"]
        assert small["nodes"] == gltf["nodes"]

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
    # The body primitive removes 64 eye faces, the 288-face mouth portal,
    # and the faces under both shells, then appends the 806-face socket
    # strip. It still skins exactly (validated above).
    body = next(m for m in gltf["meshes"] if m["name"] == "body")
    indices = _ra(gltf, binary, body["primitives"][0]["indices"])
    hidden = (shoe_entry["shell"]["hidden_body_faces"]
              + manifest["garments"]["garment"]["shell"]["hidden_body_faces"])
    assert len(indices) // 3 == 36874 - 64 - 288 + 806 - hidden


def test_assemble_is_deterministic(tmp_path):
    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "eye.png", (90, 60, 40))
    band_garment_png(assets / "garment.png")
    character = Character.load(EXAMPLES / "freediver.char.json")
    first = assemble(character, assets, tmp_path / "a.glb").read_bytes()
    second = assemble(character, assets, tmp_path / "b.glb").read_bytes()
    assert first == second


def test_barefoot_character_needs_no_footwear_asset(tmp_path):
    from character_factory.api import assemble
    from character_factory.assembly.gltf import read_accessor as _ra_g

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "eye.png", (90, 60, 40))
    band_garment_png(assets / "garment.png")
    out = assemble(
        EXAMPLES / "freediver.char.json", assets, tmp_path / "diver.glb"
    )
    assert np.frombuffer(out.read_bytes()[:4], dtype=np.uint32)[0] == 0x46546C67
    # Barefoot grounding semantics are unchanged: the plane is the body's
    # own sole.
    gltf, binary = parse_glb(out.read_bytes())
    body = next(m for m in gltf["meshes"] if m["name"] == "body")
    pos = _ra_g(gltf, binary, body["primitives"][0]["attributes"]["POSITION"])
    plane = gltf["asset"]["extras"]["grounding"]["plane_height_m"]
    assert abs(plane - float(pos[:, 1].min())) < 1e-5


def test_unclothed_character_needs_no_garment_asset(tmp_path):
    """No garment slot, no garment shell: the body renders skin alone."""
    import json as jsonlib

    from character_factory.api import assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (128, 128, 128))
    solid_png(assets / "eye.png", (90, 60, 40))
    document = jsonlib.loads((EXAMPLES / "freediver.char.json").read_text())
    del document["textures"]["garment"]
    out = assemble(Character.from_document(document), assets, tmp_path / "bare.glb")
    gltf, _ = parse_glb(out.read_bytes())
    names = {m["name"] for m in gltf["meshes"]}
    assert "body" in names and "garment" not in names
    assert "garment" not in gltf["asset"]["extras"].get("garments", {})


def test_mouth_interior_refused_without_rig_mouth_data(tmp_path):
    # SPEC.md §4.2: assembling a mouth-interior document against a body-rig
    # version that declares no mouth data is a defined error — never a
    # silent reduced-quality fallback.
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
    band_garment_png(assets / "garment.png")
    document = jsonlib.loads((EXAMPLES / "freediver.char.json").read_text())
    document["body"]["topology"] = "mouth-interior"
    with pytest.raises(ValueError, match="mouth data"):
        assemble(Character.from_document(document), assets, tmp_path / "m.glb")


def _repinned_character(tmp_path, name="marathon-runner"):
    import json as jsonlib

    from character_factory.registry import Registry

    document = jsonlib.loads((EXAMPLES / f"{name}.char.json").read_text())
    resolved = Registry.default().resolve_slots(sorted(document["textures"]))
    for slot, recipe in document["textures"].items():
        recipe["component_version"] = str(resolved[slot].version)
    path = tmp_path / f"{name}.char.json"
    Character.from_document(document).save(path)
    return path


def test_unextractable_garment_is_a_defined_error(tmp_path):
    """A garment texture that cannot key (all black) is a broken bake:
    assembly raises a named AssetError — the body is never painted."""
    from character_factory.api import AssetError, assemble

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (180, 140, 110))
    solid_png(assets / "eye.png", (90, 60, 40))
    solid_png(assets / "garment.png", (0, 0, 0))
    solid_png(assets / "shoe.png", (0, 0, 0))
    character_path = _repinned_character(tmp_path)

    with pytest.raises(AssetError, match="alpha-coverage-small"):
        assemble(character_path, assets, tmp_path / "broken.glb",
                 registry=source_topology_registry())


def test_garment_shell_ships_when_every_gate_passes(tmp_path):
    """The full path on the real rig: a fitted torso-band garment texture
    (synthesized from the body's own geometry, so it keys cleanly)
    extracts and ships as a skinned
    closed solid riding the body's skin, with the covered body faces
    omitted."""
    from PIL import Image

    from character_factory.api import assemble
    from character_factory.assembly import garment_shell as gs
    from character_factory.assembly.rig import load_rig

    rig = load_rig(_real_rig_dir())
    canonical = rig.evaluate([0.0] * 45, [0.0] * 72)
    v = canonical.vertices
    band = ((v[rig.faces][:, :, 1] > 95)
            & (v[rig.faces][:, :, 1] < 135)).all(axis=1)
    resolution = 1024
    mask = np.full((resolution, resolution), -np.inf)
    for face in rig.texcoords[rig.texcoord_faces][band] * (resolution - 1):
        gs._fill(mask, face[0], face[1], face[2], 1.0, 1.0, 1.0)
    garment_rgb = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    garment_rgb[mask > -np.inf] = (140, 60, 60)

    assets = tmp_path / "assets"
    assets.mkdir()
    solid_png(assets / "skin.png", (180, 140, 110), size=resolution)
    solid_png(assets / "eye.png", (90, 60, 40))
    Image.fromarray(garment_rgb).save(assets / "garment.png")
    solid_png(assets / "shoe.png", (0, 0, 0), size=resolution)
    character_path = _repinned_character(tmp_path)

    out = assemble(character_path, assets, tmp_path / "shelled.glb",
                   registry=source_topology_registry())
    data = out.read_bytes()
    report = validate_glb(data, expected_joints=127)
    assert report["reparse_max_error_mm"] < 1e-2

    gltf, binary = parse_glb(data)
    manifest = gltf["asset"]["extras"]
    entry = manifest["garments"]["garment"]
    assert entry["render_mode"] == "shell"
    shell_info = entry["shell"]
    assert shell_info["hidden_body_faces"] > 0
    assert shell_info["solid_vertices"] > 0

    # The shell mesh: skinned (rides skin 0), child of the character node.
    names = {node.get("name"): i for i, node in enumerate(gltf["nodes"])}
    garment_node = gltf["nodes"][names["garment"]]
    assert garment_node["skin"] == 0
    assert names["garment"] in gltf["nodes"][0]["children"]
    garment_mesh = gltf["meshes"][garment_node["mesh"]]
    attributes = garment_mesh["primitives"][0]["attributes"]
    assert {"POSITION", "NORMAL", "TEXCOORD_0",
            "JOINTS_0", "WEIGHTS_0"} <= set(attributes)

    from character_factory.assembly.gltf import read_accessor as _ra

    # UV inheritance holds by construction — asserted anyway: every shell
    # UV lands in the garment's own keyed region or its narrow boundary
    # halo (the feather plus the surface-graph smoothing can move the 0.5
    # crossing a few texels past the hard key), and never in an unrelated
    # atlas island.
    from scipy import ndimage

    uv = _ra(gltf, binary, attributes["TEXCOORD_0"])
    keyed = garment_rgb.max(axis=2) > 0
    texel_distance = ndimage.distance_transform_edt(~keyed)
    at_uv = texel_distance[
        np.clip((uv[:, 1] * (resolution - 1)).astype(int), 0, resolution - 1),
        np.clip((uv[:, 0] * (resolution - 1)).astype(int), 0, resolution - 1)]
    assert at_uv.max() <= 6.0
    # Boundary vertices are heavily UV-split, so they are overrepresented
    # in the vertex population; the body of the shell samples strictly
    # inside the key.
    assert (at_uv == 0).mean() > 0.8

    weights = _ra(gltf, binary, attributes["WEIGHTS_0"])
    assert np.abs(weights.sum(axis=1) - 1.0).max() < 1e-4

    # Covered body faces (garment and shoe shells alike) are omitted
    # alongside the 64 eye faces and the 288-face mouth portal; the
    # 806-face socket strip is then appended.
    hidden = shell_info["hidden_body_faces"]
    shoe_entry = manifest["garments"].get("shoe", {})
    if shoe_entry.get("render_mode") == "shell":
        hidden += shoe_entry["shell"]["hidden_body_faces"]
    body = next(m for m in gltf["meshes"] if m["name"] == "body")
    indices = _ra(gltf, binary, body["primitives"][0]["indices"])
    assert len(indices) // 3 == 36874 - 64 - 288 + 806 - hidden
