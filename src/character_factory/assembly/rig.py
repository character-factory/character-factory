"""Loading and evaluating the body rig component.

The rig is a TorchScript bundle (see the `body-rig` registry component): a
differentiable forward from (identity, pose, expression) to posed vertices
and a world-space skeleton state, carrying its own topology, UV, skinning,
and hierarchy buffers. This module reads exactly those buffers — no side
files besides the component's ``rig.json`` metadata (joint names and roles).

Everything here verifies before trusting: topology counts are asserted
against the component metadata on load, because the rest of the assembler
indexes into these buffers by position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["RigDefinition", "RigEvaluation", "load_rig"]

_BUFFERS = {
    "faces": "character_torch.mesh.faces",
    "texcoords": "character_torch.mesh.texcoords",
    "texcoord_faces": "character_torch.mesh.texcoord_faces",
    "parents": "character_torch.skeleton.joint_parents",
    "skin_joints": "character_torch.linear_blend_skinning.skin_indices_flattened",
    "skin_weights": "character_torch.linear_blend_skinning.skin_weights_flattened",
    "skin_vertices": "character_torch.linear_blend_skinning.vert_indices_flattened",
}


@dataclass
class RigDefinition:
    """A loaded rig: the TorchScript module plus its static buffers/metadata.

    Array conventions: numpy, rig-native units (centimeters, Y-up, character
    facing +Z, feet at y=0).
    """

    model: object                # TorchScript module (torch.jit)
    metadata: dict               # rig.json: joints, parents, roles, topology
    faces: "np.ndarray"          # (F, 3) int64 — position indices
    texcoords: "np.ndarray"      # (T, 2) float32
    texcoord_faces: "np.ndarray" # (F, 3) int64 — texcoord indices per corner
    parents: "np.ndarray"        # (J,) int64, parents[i] < i, parents[0] == -1
    vertex_joints: "np.ndarray"  # (V, 4) uint16 — LBS joints (rig native ≤ 4)
    vertex_weights: "np.ndarray" # (V, 4) float32 — sums to 1

    @property
    def joint_names(self) -> list[str]:
        return list(self.metadata["joints"])

    @property
    def joint_count(self) -> int:
        return len(self.parents)

    def joint_index(self, name: str) -> int:
        return self.metadata["joints"].index(name)

    def role_index(self, role: str) -> int:
        return self.joint_index(self.metadata["roles"][role])

    def subtree(self, root: int) -> list[int]:
        """`root` and all its descendants, ascending."""
        members = {root}
        for index, parent in enumerate(self.parents):
            if parent in members:
                members.add(index)
        return sorted(members)

    def evaluate(
        self,
        identity: list[float],
        resting_expression: list[float],
    ) -> "RigEvaluation":
        """Rest-pose evaluation: the character's identity and resting face,
        zero body pose. Deterministic; CPU-capable."""
        import torch

        topology = self.metadata["topology"]
        with torch.no_grad():
            verts, skeleton = self.model(
                torch.tensor([identity], dtype=torch.float32),
                torch.zeros(1, topology["pose_size"]),
                torch.tensor([resting_expression], dtype=torch.float32),
            )
        return RigEvaluation(
            vertices=verts[0].numpy().astype("float64"),
            skeleton=skeleton[0].numpy().astype("float64"),
        )


@dataclass
class RigEvaluation:
    vertices: "np.ndarray"  # (V, 3) cm
    skeleton: "np.ndarray"  # (J, 8): translation xyz, quaternion xyzw, uniform scale


def _dense_weights(entries, vertex_count: int, joint_count: int):
    """Flattened (vertex, joint, weight) triples → per-vertex (4,) arrays.

    The rig's native skinning has at most 4 influences per vertex with
    weights summing to 1; both properties are asserted, not assumed, because
    the glTF export carries them verbatim (ARCHITECTURE.md §3.5).
    """
    import numpy as np

    v_idx, j_idx, w = entries
    order = np.argsort(v_idx, kind="stable")
    v_idx, j_idx, w = v_idx[order], j_idx[order], w[order]
    joints = np.zeros((vertex_count, 4), dtype=np.uint16)
    weights = np.zeros((vertex_count, 4), dtype=np.float32)
    starts = np.searchsorted(v_idx, np.arange(vertex_count))
    ends = np.searchsorted(v_idx, np.arange(vertex_count), side="right")
    for vertex in range(vertex_count):
        a, b = int(starts[vertex]), int(ends[vertex])
        count = b - a
        if count == 0 or count > 4:
            raise ValueError(
                f"vertex {vertex} has {count} skin influences; the rig contract "
                f"is 1..4"
            )
        joints[vertex, :count] = j_idx[a:b]
        weights[vertex, :count] = w[a:b]
    sums = weights.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError("rig skin weights do not sum to 1")
    weights /= sums[:, None]
    if int(j_idx.max()) >= joint_count:
        raise ValueError("skin influence references a joint outside the skeleton")
    return joints, weights


def load_rig(component_dir: str | Path, device: str = "cpu") -> RigDefinition:
    import numpy as np
    import torch

    component_dir = Path(component_dir)
    metadata_path = component_dir / "rig.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{component_dir} has no rig.json — the body-rig component metadata "
            f"(joint names, roles, topology) is required"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "character-factory/rig-metadata":
        raise ValueError(f"{metadata_path} is not rig metadata")

    model = torch.jit.load(str(component_dir / "mhr_model.pt"), map_location=device)
    model.eval()
    buffers = dict(model.named_buffers())
    arrays = {key: buffers[name].numpy() for key, name in _BUFFERS.items()}

    topology = metadata["topology"]
    rest = buffers["character_torch.mesh.rest_vertices"].numpy()
    checks = {
        "vertices": rest.shape[0],
        "triangles": arrays["faces"].shape[0],
        "joints": arrays["parents"].shape[0],
    }
    for key, actual in checks.items():
        if topology[key] != actual:
            raise ValueError(
                f"rig topology mismatch: metadata says {topology[key]} {key}, "
                f"the bundle has {actual} — refusing to index into it"
            )
    if len(metadata["joints"]) != checks["joints"]:
        raise ValueError("rig.json joint-name table does not match the skeleton size")
    parents = arrays["parents"].astype(np.int64)
    if not all(parents[i] < i for i in range(1, len(parents))) or parents[0] != -1:
        raise ValueError("rig joint hierarchy is not topologically ordered")
    if list(parents) != list(metadata["parents"]):
        raise ValueError("rig.json parents do not match the bundle's hierarchy")

    vertex_joints, vertex_weights = _dense_weights(
        (
            arrays["skin_vertices"].astype(np.int64),
            arrays["skin_joints"].astype(np.int64),
            arrays["skin_weights"].astype(np.float32),
        ),
        checks["vertices"],
        checks["joints"],
    )
    return RigDefinition(
        model=model,
        metadata=metadata,
        faces=arrays["faces"].astype(np.int64),
        texcoords=arrays["texcoords"].astype(np.float32),
        texcoord_faces=arrays["texcoord_faces"].astype(np.int64),
        parents=parents,
        vertex_joints=vertex_joints,
        vertex_weights=vertex_weights,
    )
