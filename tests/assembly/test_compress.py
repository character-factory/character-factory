"""The opt-in delivery compression: textures change, nothing else does."""

import io
import json

import numpy as np
import pytest

from character_factory.assembly import export_character_glb, validate_glb
from character_factory.assembly.compress import (
    TARGETS,
    compress_glb,
    compress_glb_file,
    compressed_path,
)
from character_factory.assembly.export import Attachment
from character_factory.assembly.gltf import GlbWriter, parse_glb, read_accessor
from character_factory.cli import _COMPRESS_TARGETS


def png(color, size=64):
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (size, size), color).save(out, format="PNG")
    return out.getvalue()


def export_with_textures(rig, tmp_path):
    """A body with an albedo plus two rigid accessories: one textured with
    the SAME image bytes as the body (dedup), one carrying a normal map."""
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32) + 100.0
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv = np.zeros((3, 2), dtype=np.float32)
    skin = png((180, 140, 110))
    attachments = [
        Attachment("badge", tri, faces, uv, 0, albedo_png=skin),
        Attachment("visor", tri, faces, uv, 0, albedo_png=png((20, 20, 20)),
                   normal_png=png((128, 128, 255))),
    ]
    return export_character_glb(
        rig, [0.0, 0.0], [0.0, 0.0], tmp_path / "scene.glb",
        generator="character-factory/test", _body_only_test=True,
        albedo_png=skin, attachments=attachments,
    )


def geometry(gltf, binary):
    """Every accessor's data plus the whole document minus images/textures/
    bufferViews/accessors — what compression must carry across unchanged."""
    arrays = [read_accessor(gltf, binary, i).tobytes()
              for i in range(len(gltf["accessors"]))]
    rest = {k: v for k, v in gltf.items()
            if k not in ("images", "textures", "bufferViews", "accessors",
                         "buffers", "extensionsUsed", "extensionsRequired")}
    return arrays, json.dumps(rest, sort_keys=True)


def test_exporter_embeds_identical_textures_once(rig, tmp_path):
    gltf, _ = parse_glb(export_with_textures(rig, tmp_path).glb_path.read_bytes())
    # body albedo == badge albedo → one image, one texture, two materials.
    assert len(gltf["images"]) == 3
    body = gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    badge = [m for m in gltf["materials"] if m["name"] == "badge"][0]
    assert badge["pbrMetallicRoughness"]["baseColorTexture"]["index"] == body


@pytest.mark.parametrize("target", TARGETS)
def test_compression_changes_only_the_images(rig, tmp_path, target):
    original = export_with_textures(rig, tmp_path).glb_path.read_bytes()
    compressed = compress_glb(original, target)

    before, after = parse_glb(original), parse_glb(compressed)
    assert geometry(*before) == geometry(*after)
    assert validate_glb(compressed, expected_joints=7)["reparse_max_error_mm"] < 1e-3

    gltf, binary = after
    mime = {"web": "image/webp", "unity": "image/jpeg"}[target]
    assert [image["mimeType"] for image in gltf["images"]] == [mime] * 3
    magic = {"web": b"RIFF", "unity": b"\xff\xd8\xff"}[target]
    for image in gltf["images"]:
        view = gltf["bufferViews"][image["bufferView"]]
        assert binary[view["byteOffset"]:].startswith(magic)
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0

    if target == "web":
        # WebP is an extension and there is no PNG fallback: required.
        assert gltf["extensionsUsed"] == ["EXT_texture_webp"]
        assert gltf["extensionsRequired"] == ["EXT_texture_webp"]
        for texture in gltf["textures"]:
            assert "source" not in texture
            assert texture["extensions"]["EXT_texture_webp"]["source"] < 3
    else:
        # JPEG is core glTF — a plain file for loaders without WebP.
        assert "extensionsUsed" not in gltf
        assert "extensionsRequired" not in gltf
        assert all("extensions" not in t for t in gltf["textures"])


def test_normal_maps_are_still_identified_after_dedup(rig, tmp_path):
    from PIL import Image

    original = export_with_textures(rig, tmp_path).glb_path.read_bytes()
    gltf, binary = parse_glb(compress_glb(original, "unity"))
    visor = [m for m in gltf["materials"] if m["name"] == "visor"][0]
    source = gltf["textures"][visor["normalTexture"]["index"]]["source"]
    view = gltf["bufferViews"][gltf["images"][source]["bufferView"]]
    picture = Image.open(io.BytesIO(
        binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    ))
    # The flat normal survives the (higher-quality) lossy encode.
    assert np.abs(np.asarray(picture).astype(int) - [128, 128, 255]).max() <= 2


def test_duplicate_images_in_older_files_are_merged():
    # Files exported before the exporter deduplicated carry the eye
    # texture twice; compression merges them and drops the orphan view.
    writer = GlbWriter()
    eye = png((90, 60, 40))
    image_a = writer.add_image(eye)
    positions = writer.add_accessor(
        np.zeros((3, 3), dtype=np.float32), "VEC3", minmax=True
    )
    image_b = writer.add_image(eye)
    gltf = {
        "asset": {"version": "2.0"},
        "images": [image_a, image_b],
        "textures": [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}],
        "samplers": [{}],
        "materials": [
            {"name": "eye_left",
             "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
            {"name": "eye_right",
             "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": positions}}]}],
    }
    data = writer.finish(gltf)
    out, binary = parse_glb(compress_glb(data, "unity"))
    assert len(out["images"]) == 1
    assert [t["source"] for t in out["textures"]] == [0, 0]
    assert len(out["bufferViews"]) == 2
    # The accessor still finds its (renumbered) view.
    assert read_accessor(out, binary, 0).shape == (3, 3)


def test_unknown_target_is_a_defined_error():
    with pytest.raises(ValueError, match="unknown compression target"):
        compress_glb(b"", "draco")


def test_compressed_path_and_file_helper(rig, tmp_path):
    result = export_with_textures(rig, tmp_path)
    assert compressed_path(result.glb_path, "web") == tmp_path / "scene.web.glb"
    out = compress_glb_file(result.glb_path, "web")
    assert out == tmp_path / "scene.web.glb"
    assert parse_glb(out.read_bytes())[0]["images"][0]["mimeType"] == "image/webp"
    # The canonical file is untouched.
    (tmp_path / "again").mkdir()
    assert result.glb_path.read_bytes() == export_with_textures(
        rig, tmp_path / "again").glb_path.read_bytes()


def test_cli_mirrors_the_target_list():
    assert _COMPRESS_TARGETS == TARGETS
