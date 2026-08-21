"""Exported-artifact validation (ARCHITECTURE.md §7.2).

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
    # Animation channels REPLACE node TRS: sample every channel mid-clip,
    # substitute the sampled values into the node hierarchy, recompute the
    # joint world transforms, skin the mesh through the IBMs, and compare
    # against the rest-pose skinner. Engine-free, and it catches exactly
    # the class of failure where baked values only work because a
    # forgiving viewer reconciles them with node state.
    import copy

    animations = gltf.get("animations", [])
    assert animations, "no baked idle clip"
    idle = animations[0]
    animated_nodes = copy.deepcopy(gltf["nodes"])
    driven: dict[int, set] = {}
    for channel in idle["channels"]:
        sampler = idle["samplers"][channel["sampler"]]
        output = read_accessor(gltf, binary, sampler["output"]).astype(np.float64)
        times = read_accessor(gltf, binary, sampler["input"]).astype(np.float64)
        # Mid-clip LINEAR sample (t = the middle of the clip's time range).
        middle = (float(times[0]) + float(times[-1])) / 2.0
        upper = int(np.searchsorted(times, middle, side="right"))
        upper = min(max(upper, 1), len(times) - 1)
        span = times[upper] - times[upper - 1]
        blend = 0.0 if span == 0 else (middle - times[upper - 1]) / span
        value = (1.0 - blend) * output[upper - 1] + blend * output[upper]
        if channel["target"]["path"] == "rotation":
            value = value / np.linalg.norm(value)
        target = channel["target"]
        animated_nodes[target["node"]][target["path"]] = [float(v) for v in value]
        driven.setdefault(target["node"], set()).add(target["path"])
    for node_index in joints:
        assert driven.get(node_index) == {"translation", "rotation", "scale"}, (
            f"idle clip leaves node {node_index} partially driven "
            f"({sorted(driven.get(node_index, ()))}) — channels replace node "
            f"TRS, so every joint must carry its complete transform"
        )
    globals_animated = _global_matrices({**gltf, "nodes": animated_nodes})
    joint_mats = np.stack([globals_animated[n] for n in joints]) @ ibms
    skinned = np.zeros_like(positions)
    for influence in range(4):
        mats = joint_mats[joints4[:, influence]]
        skinned += weights4[:, influence : influence + 1] * np.einsum(
            "vij,vj->vi", mats, homogeneous
        )[:, :3]
    idle_error_m = float(np.abs(skinned - positions).max())
    report["idle_clip_skin_max_error_mm"] = idle_error_m * 1000.0
    assert idle_error_m < 1e-6, (
        f"the baked idle clip, substituted for node TRS, deviates from the "
        f"rest skin by {idle_error_m * 1000.0:.6f} mm"
    )
    report["idle_channels"] = len(idle["channels"])

    return report
