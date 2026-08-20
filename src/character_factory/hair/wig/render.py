"""Quick matplotlib preview renders (orthographic, textured-or-flat)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

VIEWS = [(0, 4, "front"), (90, 4, "right"), (180, 4, "back"), (0, 65, "top")]


def _face_shade(mesh: trimesh.Trimesh, base, light=(0.45, 0.7, 0.55)):
    n = mesh.face_normals
    lam = 0.45 + 0.55 * np.abs(n @ (np.array(light) / np.linalg.norm(light)))
    if isinstance(base, str) and base == "texture":
        img = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float64) / 255
        h, w, _ = img.shape
        uvf = mesh.visual.uv[mesh.faces].mean(1)
        px = (uvf[:, 0] % 1.0 * (w - 1)).astype(int)
        py = ((1 - uvf[:, 1]) % 1.0 * (h - 1)).astype(int)
        base = img[py, px]
    else:
        base = np.asarray(base, dtype=np.float64)
    if base.ndim == 1:
        base = np.tile(base, (len(mesh.faces), 1))
    return np.clip(base * lam[:, None], 0, 1)


def preview(meshes, colors, out_path, title="", crop_to=0, views=VIEWS):
    """meshes: list of trimesh. colors: per-mesh RGB (or per-face array).
    crop_to: index of the mesh whose bounds set the view; -1 = union."""
    used = [m.vertices[np.unique(m.faces)] for m in meshes]
    ref = used[crop_to] if crop_to >= 0 else np.vstack(used)
    ctr = ref.mean(0).copy()
    rad = np.abs(ref - ctr).max() * 1.15

    # merge into ONE collection: matplotlib depth-sorts per collection, so
    # separate meshes occlude each other wrongly. Per-face zsort is honest.
    all_tris, all_cols = [], []
    c = np.array([ctr[0], -ctr[2], ctr[1]])
    for m, col in zip(meshes, colors):
        # y-up world -> matplotlib z-up
        V = np.stack([m.vertices[:, 0], -m.vertices[:, 2], m.vertices[:, 1]], 1)
        all_tris.append((V - c)[m.faces])
        all_cols.append(_face_shade(m, col))
    tris = np.concatenate(all_tris)
    cols = np.concatenate(all_cols)

    fig = plt.figure(figsize=(4.2 * len(views), 4.6))
    for k, (az, el, name) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        pc = Poly3DCollection(tris, linewidths=0, zsort="average")
        pc.set_facecolor(cols)
        ax.add_collection3d(pc)
        ax.set_xlim(-rad, rad)
        ax.set_ylim(-rad, rad)
        ax.set_zlim(-rad, rad)
        ax.view_init(elev=el, azim=az - 90)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_title(name, fontsize=10)
    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
