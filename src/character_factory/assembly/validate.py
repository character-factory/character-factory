"""Exported-artifact validation (ARCHITECTURE.md §7).

Every check here re-parses the .glb from bytes and works only with what a
consumer's engine would see — never the exporter's in-memory state. The
central check is the re-parse acceptance test: walking the node hierarchy,
applying the inverse bind matrices, and skinning the rest mesh must
reproduce the exported vertex positions essentially exactly.
"""

from __future__ import annotations

import numpy as np

from character_factory.assembly.gltf import parse_glb, read_accessor
from character_factory.assembly.restpose import quat_to_matrix

__all__ = ["validate_glb"]

_MIRROR = np.diag([-1.0, 1.0, 1.0])
_FLIP_Z = np.diag([1.0, 1.0, -1.0])


def _node_matrix(node: dict) -> np.ndarray:
    m = np.eye(4)
    rotation = quat_to_matrix(np.asarray(node.get("rotation", [0, 0, 0, 1]), float))
    scale = np.asarray(node.get("scale", [1, 1, 1]), float)
    m[:3, :3] = rotation * scale
    m[:3, 3] = np.asarray(node.get("translation", [0, 0, 0]), float)
    return m


def _global_matrices(gltf: dict) -> list[np.ndarray]:
    nodes = gltf["nodes"]
    parents = [-1] * len(nodes)
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = index
    order = sorted(range(len(nodes)), key=lambda i: 0 if parents[i] < 0 else 1)
    matrices: list[np.ndarray | None] = [None] * len(nodes)

    def resolve(index: int) -> np.ndarray:
        if matrices[index] is None:
            local = _node_matrix(nodes[index])
            parent = parents[index]
            matrices[index] = local if parent < 0 else resolve(parent) @ local
        return matrices[index]

    for index in order:
        resolve(index)
    return matrices  # type: ignore[return-value]


def validate_glb(data: bytes, *, expected_joints: int | None = None) -> dict:
    """Validate one exported character .glb. Returns a report of measured
    values; raises AssertionError on any contract violation."""
    gltf, binary = parse_glb(data)
    report: dict = {}

    skin = gltf["skins"][0]
    joints = skin["joints"]
    if expected_joints is not None:
        assert len(joints) == expected_joints, (
            f"expected {expected_joints} joints, found {len(joints)}"
        )
    report["joint_count"] = len(joints)

    globals_ = _global_matrices(gltf)
    ibms = read_accessor(gltf, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4)
    ibms = ibms.transpose(0, 2, 1).astype(np.float64)  # column-major → row-major

    primitive = gltf["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    positions = read_accessor(gltf, binary, attributes["POSITION"]).astype(np.float64)
    joints4 = read_accessor(gltf, binary, attributes["JOINTS_0"]).astype(np.int64)
    weights4 = read_accessor(gltf, binary, attributes["WEIGHTS_0"]).astype(np.float64)

    manifest = gltf.get("asset", {}).get("extras", {})
    grounding = manifest.get("grounding")
    if grounding:
        declared_ground = float(grounding["plane_height_m"])
        # The plane describes what ships: the minimum over every skinned
        # render mesh — the body AND its shells. A shoe-wearing character
        # stands on its shoe soles (the barefoot sole is deleted).
        rest_ground = float(positions[:, 1].min())
        for node in gltf["nodes"]:
            if node.get("skin") is not None and node.get("mesh") is not None:
                prim = gltf["meshes"][node["mesh"]]["primitives"][0]
                mesh_pos = read_accessor(
                    gltf, binary, prim["attributes"]["POSITION"])
                rest_ground = min(rest_ground, float(mesh_pos[:, 1].min()))
        assert abs(rest_ground - declared_ground) < 1e-6, (
            "declared ground plane does not match the exported rest geometry"
        )
        report["rest_ground_error_mm"] = abs(rest_ground - declared_ground) * 1000

    # -- weights: sum to 1, and the skeleton root deforms nothing ------------
    sums = weights4.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), "skin weights do not sum to 1"
    root_weights = weights4[joints4 == 0]
    assert not (root_weights > 0).any(), "the skeleton root carries skin weights"

    # -- the re-parse acceptance test -----------------------------------------
    joint_mats = np.stack([globals_[n] for n in joints]) @ ibms  # (J, 4, 4)
    homogeneous = np.concatenate(
        [positions, np.ones((len(positions), 1))], axis=1
    )
    skinned = np.zeros_like(positions)
    for influence in range(4):
        mats = joint_mats[joints4[:, influence]]
        skinned += weights4[:, influence : influence + 1] * np.einsum(
            "vij,vj->vi", mats, homogeneous
        )[:, :3]
    max_error_m = float(np.abs(skinned - positions).max())
    report["reparse_max_error_mm"] = max_error_m * 1000.0
    assert max_error_m < 1e-6, (
        f"bind-pose skinning deviates from the exported mesh by "
        f"{max_error_m * 1000.0:.6f} mm"
    )

    # -- upright and co-located ------------------------------------------------
    extents = positions.max(axis=0) - positions.min(axis=0)
    assert extents[1] == extents.max(), "the character is not upright (+Y tallest)"
    joint_positions = np.stack([globals_[n][:3, 3] for n in joints])
    gap = float(np.linalg.norm(joint_positions.mean(axis=0) - positions.mean(axis=0)))
    report["mesh_skeleton_centroid_gap_m"] = gap
    assert gap < 0.5, "mesh and skeleton are not co-located"

    # -- local rotations avoid the quaternion half-turn singularity ---------
    # At 180 degrees q and -q are equally canonical (w == 0), and importers
    # disagree during handedness conversion and sign normalization. The
    # exporter has freedom to choose joint roll because IBMs cancel it, so a
    # character deliverable must stay comfortably away from that boundary.
    local_abs_w = [
        abs(float(gltf["nodes"][node].get("rotation", [0, 0, 0, 1])[3]))
        for node in joints
    ]
    report["rest_local_rotation_min_abs_w"] = min(local_abs_w)
    assert min(local_abs_w) >= 0.25 - 1e-6, (
        "a joint rest rotation is too close to 180 degrees "
        f"(minimum |w| {min(local_abs_w):.8f})"
    )

    # -- mirror consistency: left frames are reflections of right frames -------
    names = {gltf["nodes"][n].get("name", ""): n for n in joints}
    worst = 0.0
    pairs = 0
    for name, node in names.items():
        if not name.startswith("l_"):
            continue
        twin = names.get("r_" + name[2:])
        if twin is None:
            continue
        pairs += 1
        left = globals_[node][:3, :3]
        right = globals_[twin][:3, :3]
        expected = _MIRROR @ right @ _FLIP_Z
        cosine = np.clip((np.trace(left @ expected.T) - 1.0) / 2.0, -1.0, 1.0)
        worst = max(worst, float(np.degrees(np.arccos(cosine))))
    report["mirror_pairs"] = pairs
    report["mirror_worst_deviation_degrees"] = worst
    if pairs:
        assert worst < 0.5, (
            f"left/right rest frames deviate from exact reflection by {worst:.3f}°"
        )

    # -- the baked idle clip, played the way a conforming engine plays it ------
    # Animation channels REPLACE node TRS: sample every channel, substitute
    # the sampled values into the node hierarchy, recompute the joint world
    # transforms, skin the mesh through the IBMs, and compare against the
    # rest-pose skinner. Engine-free, and it catches exactly the class of
    # failure where baked values only work because a forgiving viewer
    # reconciles them with node state. Three obligations:
    #   1. at t=0 the substituted clip reproduces the rest skin exactly
    #      (the clip's contract: frame 0 IS the rest pose);
    #   2. every joint is fully driven (complete T, R, and S);
    #   3. the clip actually MOVES — some channels vary over time and the
    #      peak mesh deviation is non-zero yet bounded (a fully-driven
    #      statue fails, and so does an explosion).
    import copy

    animations = gltf.get("animations", [])
    assert animations, "no baked idle clip"
    idle = animations[0]

    channel_data = []
    driven: dict[int, set] = {}
    key_times: set[float] = set()
    for channel in idle["channels"]:
        sampler = idle["samplers"][channel["sampler"]]
        output = read_accessor(gltf, binary, sampler["output"]).astype(np.float64)
        times = read_accessor(gltf, binary, sampler["input"]).astype(np.float64)
        target = channel["target"]
        channel_data.append((target["node"], target["path"], times, output))
        driven.setdefault(target["node"], set()).add(target["path"])
        key_times.update(float(t) for t in times)
    for node_index in joints:
        assert driven.get(node_index) == {"translation", "rotation", "scale"}, (
            f"idle clip leaves node {node_index} partially driven "
            f"({sorted(driven.get(node_index, ()))}) — channels replace node "
            f"TRS, so every joint must carry its complete transform"
        )

    def sample(times: np.ndarray, output: np.ndarray, at: float) -> np.ndarray:
        upper = int(np.searchsorted(times, at, side="right"))
        upper = min(max(upper, 1), len(times) - 1)
        span = times[upper] - times[upper - 1]
        blend = 0.0 if span == 0 else (at - times[upper - 1]) / span
        blend = min(max(blend, 0.0), 1.0)
        return (1.0 - blend) * output[upper - 1] + blend * output[upper]

    def _pose(joint_mats, mesh_positions, mesh_joints4, mesh_weights4):
        mesh_homogeneous = np.concatenate(
            [mesh_positions, np.ones((len(mesh_positions), 1))], axis=1)
        skinned = np.zeros_like(mesh_positions)
        for influence in range(4):
            mats = joint_mats[mesh_joints4[:, influence]]
            skinned += mesh_weights4[:, influence : influence + 1] * np.einsum(
                "vij,vj->vi", mats, mesh_homogeneous
            )[:, :3]
        return skinned

    def _idle_joint_matrices(at: float) -> np.ndarray:
        animated_nodes = copy.deepcopy(gltf["nodes"])
        for node, path, times, output in channel_data:
            value = sample(times, output, at)
            if path == "rotation":
                value = value / np.linalg.norm(value)
            animated_nodes[node][path] = [float(v) for v in value]
        globals_animated = _global_matrices({**gltf, "nodes": animated_nodes})
        return np.stack([globals_animated[n] for n in joints]) @ ibms

    def substituted_skin(at: float) -> np.ndarray:
        return _pose(_idle_joint_matrices(at), positions, joints4, weights4)

    # Every skinned render mesh participates in grounding (the shoe sole
    # is the floor of a shoe-wearing character; the body's own sole is
    # deleted underneath it).
    skinned_meshes = []
    for node in gltf["nodes"]:
        if node.get("skin") is not None and node.get("mesh") is not None:
            prim = gltf["meshes"][node["mesh"]]["primitives"][0]
            skinned_meshes.append((
                read_accessor(gltf, binary,
                              prim["attributes"]["POSITION"]).astype(np.float64),
                read_accessor(gltf, binary,
                              prim["attributes"]["JOINTS_0"]).astype(np.int64),
                read_accessor(gltf, binary,
                              prim["attributes"]["WEIGHTS_0"]).astype(np.float64),
            ))

    # 1. t=0: the substituted clip must BE the rest pose.
    rest_error_m = float(np.abs(substituted_skin(0.0) - positions).max())
    report["idle_clip_rest_error_mm"] = rest_error_m * 1000.0
    assert rest_error_m < 1e-6, (
        f"the baked idle clip at t=0, substituted for node TRS, deviates "
        f"from the rest skin by {rest_error_m * 1000.0:.6f} mm"
    )

    # 3a. Channel-level temporal variance: some channels must vary; none by
    # much (translations in meters, rotations in quaternion components), and
    # scale never animates.
    largest_spread = 0.0
    for node, path, times, output in channel_data:
        spread = float((output.max(axis=0) - output.min(axis=0)).max())
        if path == "scale":
            assert spread < 1e-9, f"idle clip animates scale on node {node}"
        else:
            bound = 0.05 if path == "translation" else 0.1
            assert spread < bound, (
                f"idle {path} channel on node {node} spans {spread:.4f} — "
                f"far beyond a subtle idle"
            )
            largest_spread = max(largest_spread, spread)
    assert largest_spread > 1e-4, (
        "the idle clip is motionless — every sampler is constant; the clip "
        "must carry visible motion, not a statue hold"
    )

    # 3b. Mesh-level: at every key time, the substituted mesh stays near the
    # rest pose, and at some time it measurably departs from it.
    peak_m = 0.0
    ground_drift_m = 0.0
    for at in sorted(key_times):
        joint_mats = _idle_joint_matrices(at)
        posed = _pose(joint_mats, positions, joints4, weights4)
        deviation = float(np.abs(posed - positions).max())
        peak_m = max(peak_m, deviation)
        if grounding:
            posed_min = min(
                float(_pose(joint_mats, p, j, w)[:, 1].min())
                for p, j, w in skinned_meshes
            )
            ground_drift_m = max(
                ground_drift_m,
                abs(posed_min - float(grounding["plane_height_m"])),
            )
    report["idle_clip_peak_deviation_mm"] = peak_m * 1000.0
    assert peak_m < 0.05, (
        f"the idle clip displaces the mesh by {peak_m * 1000.0:.1f} mm — "
        f"far beyond a subtle idle"
    )
    assert peak_m > 5e-5, (
        f"the idle clip never displaces the mesh beyond "
        f"{peak_m * 1000.0:.4f} mm — a statue in all but channel count"
    )
    if grounding:
        tolerance = float(grounding["idle_ground_tolerance_m"])
        report["idle_ground_drift_mm"] = ground_drift_m * 1000.0
        assert ground_drift_m <= tolerance + 1e-9, (
            f"idle ground drift {ground_drift_m * 1000.0:.3f} mm exceeds "
            f"the declared {tolerance * 1000.0:.3f} mm tolerance"
        )
    report["idle_channels"] = len(idle["channels"])

    return report
