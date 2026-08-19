"""The skinned character exporter: rig evaluation → engine-ready .glb.

Follows the exporter conventions in ARCHITECTURE.md §3.1: one 0.01 cm→m
constant and no axis flip; UV-seam unwelding with weights carried through;
re-authored mirror-invariant rest orientations; a versioned baked knee
flexion; inverse bind matrices rebuilt after all rest edits; winding
verified, not assumed; a bone-role manifest sidecar and a baked idle clip
with every export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from character_factory.assembly import restpose
from character_factory.assembly.gltf import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    GlbWriter,
)
from character_factory.assembly.rig import RigDefinition

__all__ = ["ExportResult", "SCALE", "export_character_glb"]

# The single unit-conversion constant in the exporter: rig centimeters to
# glTF meters. There is deliberately no axis flip anywhere (both are Y-up,
# +Z-forward).
SCALE = 0.01

_SAMPLER = {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}


@dataclass
class ExportResult:
    glb_path: Path
    manifest_path: Path
    vertex_count: int
    joint_count: int


def _unweld(rig: RigDefinition, vertices: np.ndarray):
    """Split UV-seam vertices: one output vertex per distinct
    (position-index, texcoord-index) corner pair, skin weights copied through.
    """
    pairs = np.stack(
        [rig.faces.reshape(-1), rig.texcoord_faces.reshape(-1)], axis=1
    )
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    position_index = unique_pairs[:, 0]
    positions = vertices[position_index]
    uvs = rig.texcoords[unique_pairs[:, 1]]
    joints = rig.vertex_joints[position_index]
    weights = rig.vertex_weights[position_index]
    indices = inverse.reshape(-1, 3).astype(np.uint32)
    return positions, uvs, joints, weights, indices, position_index


def _vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions)
    triangles = positions[indices]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    for corner in range(3):
        np.add.at(normals, indices[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def _ensure_ccw(positions, indices, normals):
    """glTF wants counter-clockwise front faces: verify against outward
    normals from the centroid, flip winding if the mesh disagrees."""
    outward = positions - positions.mean(axis=0)
    agreement = (normals * outward).sum(axis=1) > 0
    if agreement.mean() < 0.5:
        indices = indices[:, ::-1].copy()
        normals = -normals
    return indices, normals


def export_character_glb(
    rig: RigDefinition,
    identity: list[float],
    resting_expression: list[float],
    out_path: str | Path,
    *,
    albedo_png: bytes | None = None,
    name: str = "character",
    generator: str = "character-factory",
) -> ExportResult:
    out_path = Path(out_path)

    # 1. Rest evaluation and rest-pose authoring (ARCHITECTURE.md §3.1).
    evaluation = rig.evaluate(identity, resting_expression)
    skeleton = restpose.Skeleton.from_rig_state(evaluation.skeleton, rig.parents)
    knees = [rig.role_index("left_knee"), rig.role_index("right_knee")]
    vertices_cm = restpose.bake_knee_flexion(
        skeleton, evaluation.vertices, rig.vertex_joints, rig.vertex_weights,
        knees, [rig.subtree(k) for k in knees],
    )
    restpose.reauthor_orientations(skeleton)
    ibms = restpose.inverse_binds(skeleton, SCALE)          # after ALL rest edits
    local = restpose.local_transforms(skeleton, SCALE)

    # 2. Mesh: unweld seams, scale to meters, normals, winding.
    positions_cm, uvs, joints4, weights4, indices, _ = _unweld(rig, vertices_cm)
    positions = (positions_cm * SCALE).astype(np.float32)
    normals64 = _vertex_normals(positions.astype(np.float64), indices)
    indices, normals64 = _ensure_ccw(positions, indices, normals64)
    normals = normals64.astype(np.float32)

    # 3. The world-transform root must not deform anything.
    world_joint = rig.role_index("world")
    if bool((weights4[(joints4 == world_joint)] > 0).any()):
        raise ValueError(
            "the skeleton's world-transform root carries skin weights; the rig "
            "does not match the exporter's contract"
        )

    writer = GlbWriter()
    a_position = writer.add_accessor(
        positions, "VEC3", target=ARRAY_BUFFER, minmax=True
    )
    a_normal = writer.add_accessor(normals, "VEC3", target=ARRAY_BUFFER)
    a_uv = writer.add_accessor(uvs.astype(np.float32), "VEC2", target=ARRAY_BUFFER)
    a_joints = writer.add_accessor(joints4, "VEC4", target=ARRAY_BUFFER)
    a_weights = writer.add_accessor(weights4, "VEC4", target=ARRAY_BUFFER)
    a_indices = writer.add_accessor(
        indices.reshape(-1), "SCALAR", target=ELEMENT_ARRAY_BUFFER
    )
    a_ibms = writer.add_accessor(
        ibms.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32), "MAT4"
    )

    # 4. Nodes: 0 = the character (mesh + skin), 1..J = joints in rig order,
    # so glTF joint indices equal rig joint indices verbatim.
    joint_count = rig.joint_count
    children: dict[int, list[int]] = {}
    for index, parent in enumerate(rig.parents):
        if parent >= 0:
            children.setdefault(int(parent), []).append(index + 1)
    nodes: list[dict] = [
        {"name": name, "mesh": 0, "skin": 0, "children": [1]}
    ]
    joint_names = rig.joint_names
    for joint in range(joint_count):
        translation, rotation, local_scale = local[joint]
        node = {
            "name": joint_names[joint],
            "translation": [float(v) for v in translation],
            "rotation": [float(v) for v in rotation],
            "scale": [local_scale] * 3,
        }
        if joint in children:
            node["children"] = children[joint]
        nodes.append(node)

    # 5. Material: albedo texture when provided, a plain grey otherwise.
    material: dict = {
        "name": "body",
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.55},
        "doubleSided": False,
    }
    gltf_extra: dict = {}
    if albedo_png is not None:
        gltf_extra["samplers"] = [_SAMPLER]
        gltf_extra["images"] = [writer.add_image(albedo_png)]
        gltf_extra["textures"] = [{"sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    else:
        material["pbrMetallicRoughness"]["baseColorFactor"] = [0.75, 0.72, 0.68, 1.0]

    # 6. The baked idle clip: one second of every joint held at its bind-pose
    # local rotation — the integrator's retarget sanity check.
    times = writer.add_accessor(
        np.array([0.0, 1.0], dtype=np.float32), "SCALAR", minmax=True
    )
    samplers, channels = [], []
    for joint in range(joint_count):
        rotation = np.asarray(local[joint][1], dtype=np.float32)
        output = writer.add_accessor(np.stack([rotation, rotation]), "VEC4")
        samplers.append({"input": times, "output": output, "interpolation": "LINEAR"})
        channels.append(
            {"sampler": joint, "target": {"node": joint + 1, "path": "rotation"}}
        )

    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "skins": [
            {
                "inverseBindMatrices": a_ibms,
                "joints": list(range(1, joint_count + 1)),
                "skeleton": 1,
            }
        ],
        "meshes": [
            {
                "name": "body",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": a_position,
                            "NORMAL": a_normal,
                            "TEXCOORD_0": a_uv,
                            "JOINTS_0": a_joints,
                            "WEIGHTS_0": a_weights,
                        },
                        "indices": a_indices,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [material],
        "animations": [{"name": "idle", "samplers": samplers, "channels": channels}],
        **gltf_extra,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(writer.finish(gltf))

    manifest_path = out_path.with_suffix(".manifest.json")
    manifest = {
        "format": "character-factory/export-manifest",
        "generator": generator,
        "units": "meters",
        "up_axis": "+Y",
        "forward_axis": "+Z",
        "joint_count": joint_count,
        "rest_knee_flexion_degrees": restpose.KNEE_FLEXION_DEGREES,
        "skeleton_root": rig.metadata["roles"]["world"],
        "humanoid_map": rig.metadata.get("humanoid_map", {}),
        "notes": [
            "Proportions are per-character and not normalized; ground contact "
            "needs foot IK.",
            "Jaw is mappable but should default to unmapped: locomotion clips "
            "commonly inject spurious jaw curves.",
            "Finger mappings should be verified in-engine before use.",
            "The rig animates as linear-blend skinning; the generator's own "
            "renders additionally apply learned pose correctives.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return ExportResult(out_path, manifest_path, len(positions), joint_count)
