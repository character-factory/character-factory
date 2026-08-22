"""The skinned character exporter: rig evaluation → engine-ready .glb.

Follows the exporter conventions in ARCHITECTURE.md §3.1: one 0.01 cm→m
constant and no axis flip; UV-seam unwelding with weights carried through;
re-authored mirror-invariant rest orientations; a versioned baked knee
flexion; inverse bind matrices rebuilt after all rest edits; winding
verified, not assumed; the bone-role manifest embedded in the GLB's asset
extras (the file is self-describing; a sidecar is an on-request
projection, never the authority) and a baked idle clip with every export.
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

# --- the baked idle clip's motion ------------------------------------------
# A subtle breathing-and-weight-sway cycle. Every term is a sine (or raised
# cosine) with an integer number of periods over the clip, so the clip loops
# seamlessly and frame 0 is EXACTLY the rest pose — the t=0 substitution
# check in validate_glb depends on that. Amplitudes are deliberately small:
# the clip is a retarget sanity check and a sign of life, not a performance.
IDLE_SECONDS = 4.0            # one breath at ~15 breaths/minute
IDLE_KEYS = 41                # 0.1 s keys; LINEAR between
# Per-role world-frame deltas, applied only when the rig's role table names
# the joint. pitch = rotation about world +X (chest rise), roll = about
# world +Z (weight shift); sway/bob are root translations in cm.
IDLE_MOTION: dict[str, dict[str, float]] = {
    "spine_mid": {"pitch_degrees": 0.9},
    "spine_upper": {"pitch_degrees": 0.5},
    "neck": {"pitch_degrees": -0.8},
    "head": {"pitch_degrees": -0.4},
    "root": {"roll_degrees": 0.3, "sway_cm": 0.3, "bob_cm": 0.12},
}


def _idle_motion_keys(rig: RigDefinition, skeleton) -> dict[int, dict]:
    """Per-joint keyframed idle deltas: joint index → {"rotation": (K, 4),
    "translation": (K, 3)} in export units, composed onto the rest local
    transform. Deltas are authored in the world frame and conjugated into
    each joint's local frame through the parent's rest orientation; for
    these sub-degree amplitudes, treating animated ancestors as at-rest in
    the conjugation is exact to second order."""
    roles = rig.metadata.get("roles", {})
    t = np.linspace(0.0, IDLE_SECONDS, IDLE_KEYS)
    breath = np.sin(2.0 * np.pi * t / IDLE_SECONDS)          # 0 at t=0, loops
    bob = (1.0 - np.cos(4.0 * np.pi * t / IDLE_SECONDS)) / 2.0

    keys: dict[int, dict] = {}
    for role, spec in IDLE_MOTION.items():
        joint_name = roles.get(role)
        if joint_name is None:
            continue
        joint = rig.joint_index(joint_name)
        parent = int(rig.parents[joint])
        parent_rot = skeleton.rotations[parent] if parent >= 0 else np.eye(3)
        parent_pos = skeleton.positions[parent] if parent >= 0 else np.zeros(3)
        parent_scale = skeleton.scales[parent] if parent >= 0 else 1.0

        entry: dict = {}
        pitch = np.deg2rad(spec.get("pitch_degrees", 0.0)) * breath
        roll = np.deg2rad(spec.get("roll_degrees", 0.0)) * breath
        if spec.get("pitch_degrees") or spec.get("roll_degrees"):
            quats = np.empty((IDLE_KEYS, 4), dtype=np.float64)
            for k in range(IDLE_KEYS):
                cx, sx = np.cos(pitch[k]), np.sin(pitch[k])
                cz, sz = np.cos(roll[k]), np.sin(roll[k])
                d_pitch = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                d_roll = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                local = parent_rot.T @ (d_roll @ d_pitch) @ skeleton.rotations[joint]
                quats[k] = restpose.quat_from_matrix(local)
                # quat_from_matrix canonicalizes to w >= 0; near w == 0 that
                # flips hemispheres between neighboring keys. Keep the key
                # sequence on one continuous path instead.
                if k > 0 and float(quats[k] @ quats[0]) < 0.0:
                    quats[k] = -quats[k]
            entry["rotation"] = quats.astype(np.float32)
        if spec.get("sway_cm") or spec.get("bob_cm"):
            delta_world = np.zeros((IDLE_KEYS, 3))
            delta_world[:, 0] = spec.get("sway_cm", 0.0) * breath
            delta_world[:, 1] = -spec.get("bob_cm", 0.0) * bob
            translations = (
                (skeleton.positions[joint] + delta_world - parent_pos)
                @ parent_rot / parent_scale
            ) * SCALE
            entry["translation"] = translations.astype(np.float32)
        if entry:
            keys[joint] = entry
    return keys


@dataclass
class ExportResult:
    glb_path: Path
    manifest: dict          # the embedded export manifest (asset extras)
    vertex_count: int
    joint_count: int


@dataclass
class Attachment:
    """A rigid accessory: parented to one joint, not skinned.

    Vertices arrive in rig-native world coordinates (cm); the exporter
    re-expresses them in the carrier joint's local frame so they follow the
    joint under animation.
    """

    name: str
    vertices: "np.ndarray"          # (V, 3) world, cm
    faces: "np.ndarray"             # (F, 3)
    uv: "np.ndarray | None"         # (V, 2) or None
    parent_joint: int               # rig joint index
    albedo_png: bytes | None = None
    normal_png: bytes | None = None
    base_color: tuple | None = None  # rgba, used when no albedo texture
    double_sided: bool = False
    roughness: float = 0.5


def _unweld(rig: RigDefinition, vertices: np.ndarray,
            faces: np.ndarray, texcoord_faces: np.ndarray):
    """Split UV-seam vertices: one output vertex per distinct
    (position-index, texcoord-index) corner pair, skin weights copied through.
    """
    pairs = np.stack(
        [faces.reshape(-1), texcoord_faces.reshape(-1)], axis=1
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
    remove_faces: "np.ndarray | None" = None,
    attachments: list[Attachment] = (),
    evaluation=None,
) -> ExportResult:
    out_path = Path(out_path)

    # 1. Rest evaluation and rest-pose authoring (ARCHITECTURE.md §3.1).
    if evaluation is None:
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

    # 2. Mesh: optional face removal (eye sockets), unweld seams, scale to
    # meters, normals, winding.
    faces, texcoord_faces = rig.faces, rig.texcoord_faces
    if remove_faces is not None and len(remove_faces):
        keep = np.ones(len(faces), dtype=bool)
        keep[np.asarray(remove_faces, dtype=np.int64)] = False
        faces, texcoord_faces = faces[keep], texcoord_faces[keep]
    positions_cm, uvs, joints4, weights4, indices, _ = _unweld(
        rig, vertices_cm, faces, texcoord_faces
    )
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

    # 5. Materials and textures. Attachments each get their own material;
    # everything shares one sampler.
    images: list[dict] = []
    texture_defs: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []

    def add_texture(png: bytes) -> int:
        images.append(writer.add_image(png))
        texture_defs.append({"sampler": 0, "source": len(images) - 1})
        return len(texture_defs) - 1

    body_material: dict = {
        "name": "body",
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.55},
        "doubleSided": False,
    }
    if albedo_png is not None:
        body_material["pbrMetallicRoughness"]["baseColorTexture"] = {
            "index": add_texture(albedo_png)
        }
    else:
        body_material["pbrMetallicRoughness"]["baseColorFactor"] = [
            0.75, 0.72, 0.68, 1.0,
        ]
    materials.append(body_material)

    # 6. The baked idle clip: a subtle breathing-and-sway loop — the
    # integrator's retarget sanity check and a sign of life. Animation
    # channels REPLACE node TRS in a conforming player, so the clip carries
    # the complete local transform for every joint — the baked rotation is
    # the composed local rotation (pre-rotation · animated rotation), never
    # a partial value the engine must reconcile with node state, and
    # translation and scale are baked alongside so nothing is left to
    # engine defaults. Frame 0 is exactly the rest pose and the last frame
    # equals the first, so the clip loops seamlessly and validate_glb can
    # check the composition against the rest skin at t=0.
    motion = _idle_motion_keys(rig, skeleton)
    times_hold = writer.add_accessor(
        np.array([0.0, IDLE_SECONDS], dtype=np.float32), "SCALAR", minmax=True
    )
    times_motion = None
    if motion:
        times_motion = writer.add_accessor(
            np.linspace(0.0, IDLE_SECONDS, IDLE_KEYS).astype(np.float32),
            "SCALAR", minmax=True,
        )
    samplers, channels = [], []
    for joint in range(joint_count):
        translation, rotation, local_scale = local[joint]
        animated = motion.get(joint, {})
        for path, value, kind in (
            ("translation", np.asarray(translation, dtype=np.float32), "VEC3"),
            ("rotation", np.asarray(rotation, dtype=np.float32), "VEC4"),
            ("scale", np.asarray([local_scale] * 3, dtype=np.float32), "VEC3"),
        ):
            if path in animated:
                output = writer.add_accessor(animated[path], kind)
                input_times = times_motion
            else:
                output = writer.add_accessor(np.stack([value, value]), kind)
                input_times = times_hold
            channels.append(
                {"sampler": len(samplers),
                 "target": {"node": joint + 1, "path": path}}
            )
            samplers.append(
                {"input": input_times, "output": output,
                 "interpolation": "LINEAR"}
            )

    meshes.append(
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
    )

    # 7. Rigid attachments: each becomes a child node of its carrier joint,
    # re-expressed in that joint's local frame (the IBMs are exactly the
    # world-to-joint transforms in export units).
    for attachment in attachments:
        world_m = np.concatenate(
            [np.asarray(attachment.vertices, dtype=np.float64) * SCALE,
             np.ones((len(attachment.vertices), 1))],
            axis=1,
        )
        local_pos = (ibms[attachment.parent_joint] @ world_m.T).T[:, :3]
        local_pos = local_pos.astype(np.float32)
        att_faces = np.asarray(attachment.faces, dtype=np.uint32)
        att_normals = _vertex_normals(
            local_pos.astype(np.float64), att_faces.astype(np.int64)
        ).astype(np.float32)

        attributes = {
            "POSITION": writer.add_accessor(
                local_pos, "VEC3", target=ARRAY_BUFFER, minmax=True
            ),
            "NORMAL": writer.add_accessor(att_normals, "VEC3", target=ARRAY_BUFFER),
        }
        if attachment.uv is not None:
            attributes["TEXCOORD_0"] = writer.add_accessor(
                np.asarray(attachment.uv, dtype=np.float32), "VEC2",
                target=ARRAY_BUFFER,
            )
        a_att_idx = writer.add_accessor(
            att_faces.reshape(-1), "SCALAR", target=ELEMENT_ARRAY_BUFFER
        )

        pbr: dict = {"metallicFactor": 0.0,
                     "roughnessFactor": float(attachment.roughness)}
        if attachment.albedo_png is not None:
            pbr["baseColorTexture"] = {"index": add_texture(attachment.albedo_png)}
        elif attachment.base_color is not None:
            pbr["baseColorFactor"] = [float(c) for c in attachment.base_color]
        material_def: dict = {
            "name": attachment.name,
            "pbrMetallicRoughness": pbr,
            "doubleSided": bool(attachment.double_sided),
        }
        if attachment.normal_png is not None:
            material_def["normalTexture"] = {"index": add_texture(attachment.normal_png)}
        materials.append(material_def)

        meshes.append(
            {
                "name": attachment.name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": a_att_idx,
                        "material": len(materials) - 1,
                    }
                ],
            }
        )
        node_index = len(nodes)
        nodes.append({"name": attachment.name, "mesh": len(meshes) - 1})
        parent_node = nodes[attachment.parent_joint + 1]
        parent_node.setdefault("children", []).append(node_index)

    # The export manifest: facts about the GLB as an engine deliverable —
    # a pure function of rig version, exporter constants, and the
    # character's skeletal proportions (stature is measured from the
    # exported geometry, never copied from the document — one source of
    # truth per fact). It embeds in the asset's extras so
    # the file is self-describing and the manifest can never be separated
    # from the mesh it describes. Character identity, textures, hair, and
    # provenance live in the character document exclusively; nothing from
    # it is ever duplicated here — one source of truth per fact.
    manifest = {
        "format": "character-factory/export-manifest",
        "generator": generator,
        "units": "meters",
        "up_axis": "+Y",
        "forward_axis": "+Z",
        "joint_count": joint_count,
        "rest_knee_flexion_degrees": restpose.KNEE_FLEXION_DEGREES,
        "skeleton_root": rig.metadata["roles"]["world"],
        "stature_m": round(
            float(positions[:, 1].max() - positions[:, 1].min()), 4
        ),
        "humanoid_map": rig.metadata.get("humanoid_map", {}),
        "idle_clip": {
            "name": "idle",
            "seconds": IDLE_SECONDS,
            "loops": True,
            "starts_at_rest": True,
            "content": "subtle breathing and weight sway; every joint fully "
                       "driven (complete local TRS)",
        },
        "notes": [
            "Proportions vary within six semantic controls; detailed "
            "segment scales are uniform in v0.1. Ground contact needs "
            "foot IK.",
            "The rig animates as linear-blend skinning; the generator's own "
            "renders additionally apply learned pose correctives.",
        ],
    }

    gltf = {
        "asset": {"version": "2.0", "generator": generator,
                  "extras": manifest},
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
        "meshes": meshes,
        "materials": materials,
        "animations": [{"name": "idle", "samplers": samplers, "channels": channels}],
    }
    if images:
        gltf["samplers"] = [_SAMPLER]
        gltf["images"] = images
        gltf["textures"] = texture_defs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(writer.finish(gltf))
    return ExportResult(out_path, manifest, len(positions), joint_count)
