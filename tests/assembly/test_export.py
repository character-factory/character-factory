"""Exporter unit tests on the synthetic rig, plus the real-rig integration
test (skipped unless the body-rig component is present in the local cache)."""

import numpy as np
import pytest

from character_factory.assembly import export_character_glb, validate_glb
from character_factory.assembly.export import SCALE
from character_factory.assembly.gltf import parse_glb, read_accessor
from character_factory.assembly.restpose import (
    KNEE_FLEXION_DEGREES,
    Skeleton,
    bake_knee_flexion,
    quat_from_matrix,
    quat_to_matrix,
    reauthor_orientations,
)

IDENTITY = [0.0, 0.0]
EXPRESSION = [0.0, 0.0]


def export(rig, tmp_path, **kwargs):
    return export_character_glb(
        rig, IDENTITY, EXPRESSION, tmp_path / "out.glb",
        generator="character-factory/test", **kwargs
    )


def test_export_passes_validation(rig, tmp_path):
    result = export(rig, tmp_path)
    report = validate_glb(result.glb_path.read_bytes(), expected_joints=7)
    assert report["reparse_max_error_mm"] < 1e-3
    assert report["mirror_pairs"] == 2  # l_upleg/r_upleg, l_lowleg/r_lowleg
    assert report["idle_channels"] == 21          # T, R, and S per joint
    assert report["idle_clip_rest_error_mm"] < 1e-3
    # The clip is a real (if subtle) motion loop, not a statue hold.
    assert report["idle_clip_peak_deviation_mm"] > 0.05
    assert result.manifest["format"] == "character-factory/export-manifest"
    # The manifest's own shape is versioned like character.json: a shape
    # change to any field bumps this, so consumers detect it instead of
    # silently falling back.
    assert result.manifest["schema_version"] == "0.5"
    assert result.manifest["idle_clip"]["starts_at_rest"] is True


def test_export_is_byte_deterministic(rig, tmp_path):
    first = export(rig, tmp_path).glb_path.read_bytes()
    second = export_character_glb(
        rig, IDENTITY, EXPRESSION, tmp_path / "second.glb",
        generator="character-factory/test",
    ).glb_path.read_bytes()
    assert first == second
    # The embedded manifest is inside those bytes; assert it explicitly —
    # a pure function of rig version + exporter constants, identical on
    # every export, and never separable from the mesh it describes.
    manifest_a = parse_glb(first)[0]["asset"]["extras"]
    manifest_b = parse_glb(second)[0]["asset"]["extras"]
    assert manifest_a == manifest_b
    assert manifest_a["format"] == "character-factory/export-manifest"
    assert manifest_a["joint_count"] == 7


def test_motionless_idle_clip_is_rejected(rig, tmp_path, monkeypatch):
    # A fully-driven statue must fail validation: with every motion
    # amplitude zeroed, the exporter produces exactly the old constant-hold
    # clip, and the validator's variance requirement catches it.
    import character_factory.assembly.export as export_module

    monkeypatch.setattr(export_module, "IDLE_MOTION", {})
    result = export(rig, tmp_path)
    with pytest.raises(AssertionError, match="motionless"):
        validate_glb(result.glb_path.read_bytes(), expected_joints=7)


def test_idle_clip_loops_and_starts_at_rest(rig, tmp_path):
    # Frame 0 == rest pose == last frame, per channel: the clip loops
    # seamlessly and the t=0 substitution check is meaningful.
    data = export(rig, tmp_path).glb_path.read_bytes()
    gltf, binary = parse_glb(data)
    idle = gltf["animations"][0]
    node_locals = {}
    for index, node in enumerate(gltf["nodes"]):
        node_locals[index] = {
            "translation": np.asarray(node.get("translation", [0, 0, 0])),
            "rotation": np.asarray(node.get("rotation", [0, 0, 0, 1])),
            "scale": np.asarray(node.get("scale", [1, 1, 1])),
        }
    saw_motion = False
    for channel in idle["channels"]:
        sampler = idle["samplers"][channel["sampler"]]
        output = read_accessor(gltf, binary, sampler["output"])
        target = channel["target"]
        rest = node_locals[target["node"]][target["path"]]
        assert np.allclose(output[0], rest, atol=1e-6)
        assert np.allclose(output[-1], output[0], atol=1e-6)
        if len(output) > 2 and np.abs(output - output[0]).max() > 1e-4:
            saw_motion = True
    assert saw_motion


def test_uv_seam_unwelds_with_weights(rig, tmp_path):
    data = export(rig, tmp_path).glb_path.read_bytes()
    gltf, binary = parse_glb(data)
    attributes = gltf["meshes"][0]["primitives"][0]["attributes"]
    positions = read_accessor(gltf, binary, attributes["POSITION"])
    joints = read_accessor(gltf, binary, attributes["JOINTS_0"])
    weights = read_accessor(gltf, binary, attributes["WEIGHTS_0"])
    uvs = read_accessor(gltf, binary, attributes["TEXCOORD_0"])

    # Vertex 8 has two texcoords → 11 output vertices from 10 positions.
    assert positions.shape[0] == 11
    # The two copies share position and skin weights but differ in UV.
    chest = positions[np.linalg.norm(positions - positions.mean(0), axis=1) >= 0]
    matches = np.where(
        (np.abs(positions - positions[:, None]).sum(axis=2) < 1e-9)
        & ~np.eye(len(positions), dtype=bool)
    )
    assert len(matches[0]) == 2  # exactly one duplicated pair
    a, b = matches[0][0], matches[1][0]
    assert (joints[a] == joints[b]).all() and (weights[a] == weights[b]).all()
    assert not np.allclose(uvs[a], uvs[b])
    assert chest is not None  # silence unused warning path


def test_no_uv_flip(rig, tmp_path):
    data = export(rig, tmp_path).glb_path.read_bytes()
    gltf, binary = parse_glb(data)
    uvs = read_accessor(
        gltf, binary, gltf["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]
    )
    # Source texcoords pass straight through — no 1-v anywhere.
    assert np.isclose(uvs.min(), 0.1) and np.isclose(uvs.max(), 0.9)


def test_scale_is_exactly_centimeters_to_meters(rig, tmp_path):
    data = export(rig, tmp_path).glb_path.read_bytes()
    gltf, binary = parse_glb(data)
    positions = read_accessor(
        gltf, binary, gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    )
    # Head vertex at 162 cm → 1.62 m; X sign preserved (no axis flip).
    assert np.isclose(positions[:, 1].max(), 1.62, atol=0.01)
    assert positions[:, 0].max() > 0 and positions[:, 0].min() < 0
    assert SCALE == 0.01


def test_world_root_weights_are_rejected(rig, tmp_path):
    bad_joints = rig.vertex_joints.copy()
    bad_weights = rig.vertex_weights.copy()
    bad_joints[9] = [0, 0, 0, 0]   # weight the world root
    bad_weights[9] = [1, 0, 0, 0]
    from tests.assembly.conftest import make_rig

    with pytest.raises(ValueError):
        export(make_rig(bad_joints, bad_weights), tmp_path)


def test_knee_flexion_moves_feet_backward(rig):
    evaluation = rig.evaluate(IDENTITY, EXPRESSION)
    skeleton = Skeleton.from_rig_state(evaluation.skeleton, rig.parents)
    knees = [rig.role_index("left_knee"), rig.role_index("right_knee")]
    flexed = bake_knee_flexion(
        skeleton, evaluation.vertices, rig.vertex_joints, rig.vertex_weights,
        knees, [rig.subtree(k) for k in knees],
    )
    # Foot vertices (fully weighted to the knee subtrees) move backward in Z
    # and stay left/right symmetric; head vertices do not move.
    for foot in (0, 1, 2, 3):
        assert flexed[foot][2] < evaluation.vertices[foot][2]
    assert np.allclose(flexed[0][1:], flexed[2][1:])   # same y/z left vs right
    assert np.allclose(flexed[0][0], -flexed[2][0])    # mirrored x
    for head in (6, 7):
        assert np.allclose(flexed[head], evaluation.vertices[head])
    assert KNEE_FLEXION_DEGREES > 0


def test_reauthored_frames_are_mirror_consistent(rig):
    evaluation = rig.evaluate(IDENTITY, EXPRESSION)
    skeleton = Skeleton.from_rig_state(evaluation.skeleton, rig.parents)
    # Give the source frames a deliberately unmirrored roll: re-authoring
    # must erase it.
    skeleton.rotations[2] = quat_to_matrix(np.array([0.0, 0.3, 0.0, 0.954]))
    reauthor_orientations(skeleton)
    mirror, flip_z = np.diag([-1.0, 1, 1]), np.diag([1.0, 1, -1])
    left, right = skeleton.rotations[2], skeleton.rotations[4]
    assert np.allclose(left, mirror @ right @ flip_z, atol=1e-9)
    for joint in range(7):
        assert np.isclose(np.linalg.det(skeleton.rotations[joint]), 1.0)


def test_quaternion_round_trip():
    rng = np.random.default_rng(7)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        m = quat_to_matrix(q)
        q2 = quat_from_matrix(m)
        assert np.allclose(quat_to_matrix(q2), m, atol=1e-9)
        assert q2[3] >= 0  # canonical hemisphere, for byte determinism


def test_embedded_albedo_texture(rig, tmp_path):
    # A 1×1 PNG (smallest valid) — enough to prove the embed path.
    import struct
    import zlib

    def chunk(kind, payload):
        data = kind + payload
        return struct.pack(">I", len(payload)) + data + struct.pack(
            ">I", zlib.crc32(data)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )
    result = export(rig, tmp_path, albedo_png=png)
    gltf, binary = parse_glb(result.glb_path.read_bytes())
    assert gltf["images"][0]["mimeType"] == "image/png"
    material = gltf["materials"][0]["pbrMetallicRoughness"]
    assert material["baseColorTexture"]["index"] == 0
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    start = view["byteOffset"]
    assert binary[start : start + 8] == b"\x89PNG\r\n\x1a\n"


def test_buffer_views_are_aligned(rig, tmp_path):
    gltf, _ = parse_glb(export(rig, tmp_path).glb_path.read_bytes())
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
    position = gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    assert "min" in gltf["accessors"][position] and "max" in gltf["accessors"][position]


# --- integration against the real rig component -------------------------------


def source_topology_registry():
    """A registry pinned to the tier these batteries assert against.

    The live index serves the newest compatible component, which is the
    right behaviour for the product and the wrong behaviour for a test
    that hard-codes a surface's triangle counts or vertex indices. This
    resolves the newest body-rig whose render topology IS its source
    topology, and the assembly-assets that pairs with it.
    """
    import json

    from character_factory.registry import Registry, RegistryIndex
    from character_factory.registry.store import component_dir

    index = Registry.default().index
    keep = []
    for entry in index.entries:
        if entry.name == "body-rig":
            directory = component_dir(entry)
            if not (directory / "rig.json").is_file():
                continue
            if json.loads((directory / "rig.json").read_text()).get("render"):
                continue          # a declared render LOD is a different tier
        keep.append(entry.document)
    document = dict(index.document, components=keep)
    return Registry(RegistryIndex(document))


def _real_rig_dir(render_lod=None):
    """A cached body-rig component directory.

    `render_lod=None` (the default) asks for one whose render topology is
    its source topology — the surface these tests were written against.
    Face-index data (mouth portals, eye apertures) belongs to whichever
    surface a component renders, so a battery has to say which it means
    rather than taking whatever resolution happens to serve today.
    """
    import json

    from character_factory.registry import Registry
    from character_factory.registry.store import component_dir

    for entry in reversed(Registry.default().index.versions_of("body-rig")):
        directory = component_dir(entry)
        if not (directory / "mhr_model.pt").is_file() \
                or not (directory / "rig.json").is_file():
            continue
        declared = json.loads((directory / "rig.json").read_text()).get("render")
        if (declared or {}).get("lod") == render_lod:
            return directory
    return None


real_rig = pytest.mark.skipif(
    _real_rig_dir() is None,
    reason="body-rig component not present in the local cache",
)


@real_rig
def test_real_rig_export_passes_acceptance(tmp_path):
    from character_factory import Character
    from character_factory.assembly import load_rig

    rig = load_rig(_real_rig_dir())
    examples = __import__("pathlib").Path(__file__).parents[2] / "examples/characters"
    character = Character.load(examples / "marathon-runner.char.json")
    result = export_character_glb(
        rig, character.identity, character.resting_expression,
        tmp_path / "runner.glb", name="marathon-runner",
        generator="character-factory/test",
    )
    report = validate_glb(result.glb_path.read_bytes(), expected_joints=127)
    assert report["reparse_max_error_mm"] < 1e-2
    assert report["mirror_worst_deviation_degrees"] < 0.5
    assert result.vertex_count == 19455  # the documented unweld count

    # The embedded humanoid map: 54 engine roles mapped, the jaw carried as
    # a structured flag (mappable, default-unmapped), and every joint
    # accounted for exactly once across map + jaw + leave-unmapped groups.
    humanoid = result.manifest["humanoid_map"]
    assert humanoid["map"]["Hips"] == "root"
    assert humanoid["map"]["LeftHand"] == "l_wrist"
    assert len(humanoid["map"]) == 54
    # The convention label is made true: role keys are HumanBodyBones enum
    # member names, and the spaced HumanTrait normalization is documented
    # in the map itself.
    assert humanoid["convention"] == "unity-humanoid"
    assert "Left Thumb Proximal" in humanoid["naming"]
    assert humanoid["jaw"]["mappable"] and not humanoid["jaw"]["default_mapped"]
    assert humanoid["fingers"]["mapped"] and humanoid["fingers"]["verify_in_engine"]
    joint_names = set(rig.joint_names)
    mapped = set(humanoid["map"].values())
    unmapped = {j for group in humanoid["unmapped"].values() for j in group}
    covered = mapped | {humanoid["jaw"]["joint"]} | unmapped
    assert covered == joint_names
    assert len(mapped) + 1 + len(unmapped) == len(joint_names)


def test_proportion_pose_resolves_named_channels(rig):
    # Inert plumbing for the proportions schema event: named parameters
    # resolve through the rig metadata table into pose channels;
    # articulation channels stay zero; unknown names are hard errors (a
    # proportion ignored is a different skeleton).
    pose = rig.proportion_pose({"leg_length": 0.9, "spine_length": -0.4})
    assert pose[1] == 0.9 and pose[2] == -0.4
    assert (pose[[0]] == 0).all() and pose.shape == (3,)
    with pytest.raises(ValueError, match="unknown proportion"):
        rig.proportion_pose({"leg_lenght": 1.0})
    # evaluate() without proportions is the template path, unchanged.
    a = rig.evaluate([0.0, 0.0], [0.0, 0.0])
    b = rig.evaluate([0.0, 0.0], [0.0, 0.0], proportions=None)
    assert (a.vertices == b.vertices).all()


@real_rig
def test_real_rig_proportioned_export_passes_acceptance(tmp_path):
    # The full varied-skeleton matrix: tall, short-broad, long-armed —
    # deliberately at the ±0.40 format bound. Every export must pass the
    # complete validator (re-parse, mirror, upright, idle), and stature
    # must actually move.
    from character_factory import Character
    from character_factory.assembly import load_rig

    rig = load_rig(_real_rig_dir())
    examples = __import__("pathlib").Path(__file__).parents[2] / "examples/characters"
    character = Character.load(examples / "marathon-runner.char.json")

    statures = {}
    for label, proportions in (
        ("tall", {"leg_length": 0.4, "spine_length": 0.4}),
        ("short_broad", {"leg_length": -0.4, "shoulder_width": 0.4,
                         "hip_width": 0.4}),
        ("long_arms", {"arm_length": 0.4}),
        ("template", None),
    ):
        result = export_character_glb(
            rig, character.identity, character.resting_expression,
            tmp_path / f"{label}.glb", generator="character-factory/test",
            evaluation=rig.evaluate(
                character.identity, character.resting_expression,
                proportions=proportions,
            ),
        )
        report = validate_glb(result.glb_path.read_bytes(), expected_joints=127)
        assert report["mirror_worst_deviation_degrees"] < 0.1  # the bone floor
        assert report["idle_clip_rest_error_mm"] < 1e-3
        statures[label] = result.manifest["stature_m"]

    # Proportions moved the skeleton: legs+spine at +0.4 ≈ +8 cm; legs at
    # -0.4 ≈ -4 cm; arm length leaves stature alone.
    assert statures["tall"] > statures["template"] + 0.06
    assert statures["short_broad"] < statures["template"] - 0.03
    assert abs(statures["long_arms"] - statures["template"]) < 0.01
