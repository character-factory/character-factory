"""Rest-pose authoring for the exported skeleton.

Two deliberate edits happen between rig evaluation and binding
(ARCHITECTURE.md §3.1):

1. **Joint rest orientations are re-authored from geometry** under one
   mirror-invariant convention, because the rig's native per-bone roll is
   not mirrored between left and right limbs — harmless to skinning, hostile
   to humanoid retargeting.
2. **A small knee flexion is baked** (KNEE_FLEXION_DEGREES, a versioned
   constant recorded in the export manifest), because a near-straight knee
   is a degenerate hinge retargeters can resolve backwards.

Joint world *positions* are only touched by the knee edit; inverse bind
matrices are always rebuilt afterward, so the bound mesh is bit-identical
to the rig's output.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KNEE_FLEXION_DEGREES",
    "Skeleton",
    "bake_knee_flexion",
    "inverse_binds",
    "local_transforms",
    "quat_from_matrix",
    "quat_to_matrix",
    "reauthor_orientations",
]

# The baked forward knee bend, in degrees. Versioned: changing it changes
# every exported rest pose, so it moves only with a documented reason.
KNEE_FLEXION_DEGREES = 5.0

_REFERENCE = np.array([0.0, 0.0, 1.0])   # +Z: invariant under the sagittal mirror
_FALLBACK = np.array([0.0, 1.0, 0.0])    # +Y: also mirror-invariant
_PARALLEL_LIMIT = 0.99


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion → 3×3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n == 0 else 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → xyzw quaternion (unit, w ≥ 0 for determinism)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
        q = np.empty(4)
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = q / np.linalg.norm(q)
    return -q if q[3] < 0 else q


class Skeleton:
    """World-space rest skeleton: positions, orientations, uniform scales."""

    def __init__(self, positions: np.ndarray, rotations: np.ndarray,
                 scales: np.ndarray, parents: np.ndarray):
        self.positions = positions.astype(np.float64)   # (J, 3)
        self.rotations = rotations.astype(np.float64)   # (J, 3, 3)
        self.scales = scales.astype(np.float64)         # (J,)
        self.parents = parents

    @classmethod
    def from_rig_state(cls, skeleton_state: np.ndarray, parents: np.ndarray) -> "Skeleton":
        rotations = np.stack([quat_to_matrix(row[3:7]) for row in skeleton_state])
        return cls(skeleton_state[:, :3], rotations, skeleton_state[:, 7], parents)

    def world_matrix(self, joint: int) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotations[joint] * self.scales[joint]
        m[:3, 3] = self.positions[joint]
        return m


def reauthor_orientations(skeleton: Skeleton) -> None:
    """Replace every joint's rest rotation with a geometry-derived frame.

    Column 0: the bone-long axis, from the joint toward the mean of its
    children (a leaf inherits the incoming bone direction from its parent).
    Column 1: the +Z reference orthogonalized against the bone axis (falling
    back to +Y when nearly parallel). Column 2: their cross product. The
    reference axes are invariant under the sagittal (X = 0) mirror, so left
    and right frames come out as exact reflections of each other. Positions
    are untouched; the IBMs cancel any orientation choice, so the bound mesh
    is unchanged.
    """
    children: dict[int, list[int]] = {}
    for index, parent in enumerate(skeleton.parents):
        if parent >= 0:
            children.setdefault(int(parent), []).append(index)

    bone_axes = np.zeros_like(skeleton.positions)
    for joint in range(len(skeleton.parents)):
        if joint in children:
            mean_child = skeleton.positions[children[joint]].mean(axis=0)
            direction = mean_child - skeleton.positions[joint]
        else:
            parent = int(skeleton.parents[joint])
            direction = skeleton.positions[joint] - skeleton.positions[parent]
        norm = np.linalg.norm(direction)
        bone_axes[joint] = (
            direction / norm if norm > 1e-9 else bone_axes[int(skeleton.parents[joint])]
        )

    for joint in range(len(skeleton.parents)):
        bone = bone_axes[joint]
        reference = _REFERENCE
        if abs(float(bone @ reference)) > _PARALLEL_LIMIT:
            reference = _FALLBACK
        side = reference - bone * float(bone @ reference)
        side /= np.linalg.norm(side)
        skeleton.rotations[joint] = np.column_stack([bone, side, np.cross(bone, side)])


def bake_knee_flexion(
    skeleton: Skeleton,
    vertices: np.ndarray,
    vertex_joints: np.ndarray,
    vertex_weights: np.ndarray,
    knee_joints: list[int],
    subtrees: list[list[int]],
    degrees: float = KNEE_FLEXION_DEGREES,
) -> np.ndarray:
    """Rotate each lower-leg subtree forward about a world-X hinge through
    the knee pivot, blending each vertex by its total weight on that subtree.

    Pure sagittal motion about one shared world axis, so left and right stay
    symmetric by construction. Returns the edited vertex array; edits the
    skeleton's subtree joints (positions and orientations) in place.
    """
    angle = np.deg2rad(degrees)
    # Forward flexion: a positive rotation about +X carries points below the
    # knee pivot backward (-Z), i.e. the foot swings behind the knee.
    c, s = np.cos(angle), np.sin(angle)
    hinge = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    vertices = vertices.copy()
    for knee, subtree in zip(knee_joints, subtrees):
        pivot = skeleton.positions[knee].copy()
        members = np.asarray(subtree)

        in_subtree = np.isin(vertex_joints, members)
        blend = np.where(in_subtree, vertex_weights, 0.0).sum(axis=1)
        moved = (vertices - pivot) @ hinge.T + pivot
        vertices = vertices + blend[:, None] * (moved - vertices)

        skeleton.positions[members] = (
            (skeleton.positions[members] - pivot) @ hinge.T + pivot
        )
        skeleton.rotations[members] = hinge @ skeleton.rotations[members]
    return vertices


def inverse_binds(skeleton: Skeleton, scale: float) -> np.ndarray:
    """Column-major-ready inverse bind matrices, in export units.

    Built from the *final* rest skeleton — always call after every rest-pose
    edit (A-rule: rebuild IBMs last), with translations scaled to meters.
    """
    joint_count = len(skeleton.parents)
    ibms = np.zeros((joint_count, 4, 4), dtype=np.float64)
    for joint in range(joint_count):
        m = skeleton.world_matrix(joint)
        m[:3, 3] *= scale
        ibms[joint] = np.linalg.inv(m)
    return ibms


def local_transforms(skeleton: Skeleton, scale: float):
    """Per-joint local (translation, rotation-quaternion, uniform-scale) in
    export units, derived from the world rest frames."""
    locals_ = []
    for joint in range(len(skeleton.parents)):
        parent = int(skeleton.parents[joint])
        if parent < 0:
            translation = skeleton.positions[joint] * scale
            rotation = quat_from_matrix(skeleton.rotations[joint])
            local_scale = skeleton.scales[joint]
        else:
            parent_rot = skeleton.rotations[parent]
            translation = (
                parent_rot.T
                @ (skeleton.positions[joint] - skeleton.positions[parent])
                / skeleton.scales[parent]
            ) * scale
            rotation = quat_from_matrix(parent_rot.T @ skeleton.rotations[joint])
            local_scale = skeleton.scales[joint] / skeleton.scales[parent]
        locals_.append((translation, rotation, float(local_scale)))
    return locals_
