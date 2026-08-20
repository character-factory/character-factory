"""Robust GLB loading for quantized/skinned files.

gltfpack-style GLBs bake each mesh's dequantization transform into its skin's
inverse bind matrices (node transforms are ignored for skinned meshes per the
glTF spec). trimesh's generic loader mangles these, so we reconstruct
bind-pose world vertices ourselves: for a skinned mesh in bind pose,
G_j @ IBM_j is the same (dequant) transform for every joint j, so applying
G_0 @ IBM_0 recovers world positions.
"""

import numpy as np
import trimesh
from pygltflib import GLTF2

_CTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_acc(g: GLTF2, blob: bytes, idx: int) -> np.ndarray:
    acc = g.accessors[idx]
    bv = g.bufferViews[acc.bufferView]
    off = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    dt = _CTYPE[acc.componentType]
    n = _NCOMP[acc.type]
    isz = np.dtype(dt).itemsize * n
    stride = bv.byteStride or isz
    if stride == isz:
        arr = np.frombuffer(blob[off : off + isz * acc.count], dtype=dt).reshape(acc.count, n)
    else:
        arr = np.stack(
            [np.frombuffer(blob[off + i * stride : off + i * stride + isz], dtype=dt) for i in range(acc.count)]
        )
    return arr


def _node_world(g: GLTF2, ni: int, parent: dict) -> np.ndarray:
    M = np.eye(4)
    cur = ni
    while cur is not None:
        n = g.nodes[cur]
        if n.matrix:
            L = np.array(n.matrix).reshape(4, 4).T
        else:
            L = np.eye(4)
            if n.scale:
                L[:3, :3] = np.diag(n.scale)
            if n.rotation:
                x, y, z, w = n.rotation
                L[:3, :3] = trimesh.transformations.quaternion_matrix([w, x, y, z])[:3, :3] @ L[:3, :3]
            if n.translation:
                L[:3, 3] = n.translation
        M = L @ M
        cur = parent.get(cur)
    return M


def load_glb_meshes(path: str) -> dict[str, trimesh.Trimesh]:
    """Return {mesh_name: world-space Trimesh} in bind pose."""
    g = GLTF2().load_binary(str(path))
    blob = g.binary_blob()
    parent = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}
    out = {}
    for ni, node in enumerate(g.nodes):
        if node.mesh is None:
            continue
        mesh = g.meshes[node.mesh]
        prim = mesh.primitives[0]
        P = _read_acc(g, blob, prim.attributes.POSITION).astype(np.float64)[:, :3]
        acc_p = g.accessors[prim.attributes.POSITION]
        if acc_p.normalized:
            P = P / np.iinfo(_CTYPE[acc_p.componentType]).max
        F = _read_acc(g, blob, prim.indices).reshape(-1, 3).astype(np.int64)
        if node.skin is not None:
            # full linear-blend skinning: correct regardless of how the
            # packer folded dequantization into the inverse bind matrices
            skin = g.skins[node.skin]
            ibm = _read_acc(g, blob, skin.inverseBindMatrices).astype(np.float64)
            J = _read_acc(g, blob, prim.attributes.JOINTS_0).astype(np.int64)
            W = _read_acc(g, blob, prim.attributes.WEIGHTS_0).astype(np.float64)
            acc_w = g.accessors[prim.attributes.WEIGHTS_0]
            if acc_w.normalized:
                W = W / np.iinfo(_CTYPE[acc_w.componentType]).max
            W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
            M = np.stack(
                [
                    _node_world(g, skin.joints[j], parent) @ ibm[j].reshape(4, 4).T
                    for j in range(len(skin.joints))
                ]
            )
            Ph = np.concatenate([P, np.ones((len(P), 1))], 1)  # (n,4)
            # (n,4,4): weighted blend of per-vertex joint matrices
            Mv = np.einsum("nk,nkab->nab", W, M[J])
            Pw = np.einsum("nab,nb->na", Mv, Ph)[:, :3]
        else:
            D = _node_world(g, ni, parent)
            Pw = P @ D[:3, :3].T + D[:3, 3]
        out[mesh.name or f"mesh{node.mesh}"] = trimesh.Trimesh(Pw, F, process=False)
    return out
