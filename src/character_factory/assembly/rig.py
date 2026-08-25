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

__all__ = ["RenderTopology", "RigDefinition", "RigEvaluation", "load_rig"]

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
class RenderTopology:
    """A coarser tessellation the component declares for rendering.

    The rig always *evaluates* at its source topology — identity, the
    expression basis, the proportion controls, and articulation are all
    defined there. A component may additionally declare a render LOD:
    the supplied barycentric map carries each render vertex as a fixed
    combination of one source triangle's corners, so evaluated geometry
    flows through unchanged and nothing is re-solved per character.

    Skin weights and the expression morphs are transferred through the
    same correspondence at authoring time and pinned in the component;
    aperture face lists are exact per-LOD authored data (transferring a
    selection by thresholding a mapped field over-removes — that failure
    is a permanent guard in the authoring tool, never a fallback).
    """

    lod: int
    faces: "np.ndarray"           # (F, 3) int64 — render position indices
    texcoords: "np.ndarray"       # (T, 2) float32, image-convention V
    texcoord_faces: "np.ndarray"  # (F, 3) int64
    vertex_joints: "np.ndarray"   # (V, 4) uint16 — transferred
    vertex_weights: "np.ndarray"  # (V, 4) float32 — transferred, sums to 1
    map_triangles: "np.ndarray"   # (V,) int64 — source triangle per vertex
    map_barycentric: "np.ndarray" # (V, 3) float64
    eye_faces: "np.ndarray"       # authored aperture (render indices)
    mouth_faces: "np.ndarray"     # authored aperture (render indices)
    morph_indices: list           # per unit: moved render-vertex indices
    morph_deltas: list            # per unit: (n, 3) float32 cm

    def vertices_from(self, source_vertices, source_faces):
        """Carry evaluated source geometry to the render tessellation."""
        import numpy as np

        corners = source_vertices[source_faces[self.map_triangles]]
        return np.einsum("nk,nkd->nd", self.map_barycentric, corners)

    def morph_dense(self, unit: int) -> "np.ndarray":
        import numpy as np

        dense = np.zeros((len(self.map_triangles), 3), dtype=np.float64)
        dense[self.morph_indices[unit]] = self.morph_deltas[unit]
        return dense


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
    component_dir: "Path | None" = None  # where the component was loaded from
    render: "RenderTopology | None" = None  # declared render LOD, if any

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

    def proportion_pose(self, proportions: dict) -> "np.ndarray":
        """Resolve named skeletal-proportion parameters into a full pose
        vector via the rig metadata's proportion table. The articulation
        channels stay zero — proportions are the only pose channels a
        character document may set. Unknown names are errors: a proportion
        ignored is a different skeleton than the document describes."""
        import numpy as np

        table = self.metadata.get("proportions", {}).get("parameters", {})
        pose = np.zeros(self.metadata["topology"]["pose_size"], dtype=np.float64)
        for name, value in proportions.items():
            entry = table.get(name)
            if entry is None:
                raise ValueError(
                    f"unknown proportion parameter {name!r} for this rig"
                )
            pose[int(entry["channel"])] = float(value)
        return pose

    def evaluate(
        self,
        identity: list[float],
        resting_expression: list[float],
        proportions: dict | None = None,
    ) -> "RigEvaluation":
        """Rest-pose evaluation: the character's identity, resting face, and
        skeletal proportions (template when absent), zero articulation.
        Deterministic; CPU-capable."""
        import torch

        topology = self.metadata["topology"]
        if proportions:
            pose = torch.from_numpy(
                self.proportion_pose(proportions)
            ).to(torch.float32).unsqueeze(0)
        else:
            pose = torch.zeros(1, topology["pose_size"])
        with torch.no_grad():
            verts, skeleton = self.model(
                torch.tensor([identity], dtype=torch.float32),
                pose,
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


def _load_render_topology(component_dir: "Path", metadata: dict):
    """The component's declared render LOD, hash-verified before use.

    Absent declaration means the source topology is the render topology
    (every component before this capability existed).
    """
    import hashlib

    import numpy as np

    block = metadata.get("render")
    if not block:
        return None
    path = component_dir / block["artifact"]
    if not path.is_file():
        raise FileNotFoundError(
            f"{component_dir} declares render LOD {block.get('lod')} but "
            f"{block['artifact']} is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != block["sha256"]:
        raise ValueError(
            f"{path} does not match its pinned hash — refusing to render "
            f"from unverified topology")
    data = np.load(path)
    faces = data["faces"].astype(np.int64)
    checks = {
        "triangles": len(faces),
        "vertices": len(data["map_triangles"]),
        "texcoords": len(data["texcoords"]),
    }
    for key, actual in checks.items():
        if block[key] != actual:
            raise ValueError(
                f"render topology mismatch: metadata says {block[key]} {key}, "
                f"the artifact has {actual}")
    units = int(data["unit_count"][0]) if "unit_count" in data else 0
    return RenderTopology(
        lod=int(block["lod"]),
        faces=faces,
        texcoords=data["texcoords"].astype(np.float32),
        texcoord_faces=data["texcoord_faces"].astype(np.int64),
        vertex_joints=data["vertex_joints"].astype(np.uint16),
        vertex_weights=data["vertex_weights"].astype(np.float32),
        map_triangles=data["map_triangles"].astype(np.int64),
        map_barycentric=data["map_barycentric"].astype(np.float64),
        eye_faces=data["eye_faces"].astype(np.int64),
        mouth_faces=data["mouth_faces"].astype(np.int64),
        morph_indices=[data[f"indices_{i:02d}"].astype(np.int64)
                       for i in range(units)],
        morph_deltas=[data[f"deltas_{i:02d}"].astype(np.float32)
                      for i in range(units)],
    )


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
        render=_load_render_topology(component_dir, metadata),
        faces=arrays["faces"].astype(np.int64),
        texcoords=arrays["texcoords"].astype(np.float32),
        texcoord_faces=arrays["texcoord_faces"].astype(np.int64),
        parents=parents,
        vertex_joints=vertex_joints,
        vertex_weights=vertex_weights,
        component_dir=component_dir,
    )
