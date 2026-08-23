"""Eye placement: anatomy-derived, data-driven.

The `assembly-assets` component supplies a stock eyeball mesh (its own
concentric UV layout), the eyelid free-edge loop of its companion eye-region
asset, a measured gaze axis, and the list of socket faces to remove from the
body surface. Placement is a similarity fit — eyelid margin loop onto the
body's socket rim — so position, orientation, *and* scale come from anatomy
(the fitted scale also absorbs the asset's native units).

The asset models one eye; the other side uses its mirror. A mirrored fit
flips triangle winding, and a near-planar rim cannot distinguish a
180°-flipped rotation by RMS alone, so candidates are constrained to point
the transformed gaze axis outward along the socket normal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["EyeAssets", "PlacedEye", "place_eyes", "socket_backing"]

_RESAMPLE = 64


@dataclass
class EyeAssets:
    vertices: np.ndarray      # (E, 3) eyeball, native units
    faces: np.ndarray         # (F, 3)
    uv: np.ndarray            # (E, 2), glTF v-orientation
    margin_loop: np.ndarray   # (M, 3) eyelid free edge, same frame as the eyeball
    gaze_axis: np.ndarray     # (3,) unit, same frame
    socket_faces: np.ndarray  # body-face indices to remove
    forward_cm: float
    up_cm: float
    scale: float

    @classmethod
    def load(cls, component_dir: str | Path) -> "EyeAssets":
        component_dir = Path(component_dir)
        placement = json.loads(
            (component_dir / "eye_placement.json").read_text(encoding="utf-8")
        )
        if placement.get("format") != "character-factory/eye-placement":
            raise ValueError(f"{component_dir} has no eye placement data")
        with np.load(component_dir / "eyeball.npz") as data:
            vertices = data["vertices"].astype(np.float64)
            faces = data["faces"].astype(np.int64)
            uv = data["uv"].astype(np.float32)
        return cls(
            vertices=vertices,
            faces=faces,
            uv=uv,
            margin_loop=np.asarray(placement["lid_margin_loop"], dtype=np.float64),
            gaze_axis=np.asarray(placement["gaze_axis"], dtype=np.float64),
            socket_faces=np.asarray(placement["socket_faces"], dtype=np.int64),
            forward_cm=float(placement["forward_cm"]),
            up_cm=float(placement["up_cm"]),
            scale=float(placement["scale"]),
        )


@dataclass
class PlacedEye:
    side: str                 # "left" | "right"
    vertices: np.ndarray      # (E, 3) world, cm
    faces: np.ndarray         # (F, 3), winding corrected for mirroring
    uv: np.ndarray
    gaze: np.ndarray          # (3,) world unit vector
    fit_rms_cm: float
    rim: np.ndarray = None    # (M, 3) ordered socket-rim loop, world cm


def socket_backing(rim: np.ndarray, gaze: np.ndarray):
    """A dark occluder skirt behind the eyeball: the socket rim extruded
    inward and back, closed by an apex. It exists to stop the see-through
    gap between the eyeball and the socket rim — a rigid attachment with
    its own material, like the eyeball itself (the interior-UV contract
    covers geometry stitched into the body; this is not).

    Returns (vertices, faces); winding is irrelevant (rendered
    double-sided)."""
    centroid = rim.mean(axis=0)
    inner = centroid + (rim - centroid) * 0.45 - gaze * 1.5
    apex = centroid - gaze * 2.2
    n = len(rim)
    vertices = np.vstack([rim, inner, apex])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
        faces.append((n + i, n + j, 2 * n))
    return vertices, np.asarray(faces, dtype=np.int64)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (uniform scale, proper rotation, translation)."""
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / src_c.var(axis=0).sum()
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def _resample_loop(points: np.ndarray, count: int) -> np.ndarray:
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, arc[-1], count, endpoint=False)
    out = np.empty((count, 3))
    for i, t in enumerate(targets):
        j = int(np.searchsorted(arc, t, side="right") - 1)
        j = min(j, len(seg) - 1)
        f = (t - arc[j]) / max(seg[j], 1e-12)
        out[i] = closed[j] * (1 - f) + closed[j + 1] * f
    return out


def _fit_loops(src_loop: np.ndarray, dst_loop: np.ndarray):
    """Best similarity between two closed loops with unknown start/winding."""
    dst = _resample_loop(dst_loop, _RESAMPLE)
    best = None
    for direction in (1, -1):
        src0 = _resample_loop(src_loop[::direction], _RESAMPLE)
        for offset in range(_RESAMPLE):
            src = np.roll(src0, offset, axis=0)
            s, rotation, translation = _umeyama(src, dst)
            fitted = src @ (s * rotation).T + translation
            rms = float(np.sqrt(((fitted - dst) ** 2).sum(axis=1).mean()))
            if best is None or rms < best[0]:
                best = (rms, s, rotation, translation)
    return best


def _rim_loop(faces: np.ndarray, socket_subset: np.ndarray) -> np.ndarray:
    """Ordered vertex loop of one socket hole: edges used by exactly one
    removed face are the rim."""
    from collections import Counter, defaultdict

    triangles = faces[socket_subset]
    counts = Counter(
        tuple(sorted((f[i], f[(i + 1) % 3]))) for f in triangles for i in range(3)
    )
    rim_edges = [edge for edge, count in counts.items() if count == 1]
    adjacency = defaultdict(list)
    for a, b in rim_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    start = rim_edges[0][0]
    loop, previous, current = [start], None, start
    while True:
        options = adjacency[current]
        nxt = options[0] if options[0] != previous else options[1]
        if nxt == start:
            break
        loop.append(nxt)
        previous, current = current, nxt
    return np.asarray(loop, dtype=np.int64)


def place_eyes(
    body_vertices: np.ndarray, body_faces: np.ndarray, assets: EyeAssets
) -> list[PlacedEye]:
    """Fit the eyeball into each socket of the evaluated body. `body_vertices`
    in rig-native cm; returns world-space placed eyes (cm)."""
    socket_triangles = body_faces[assets.socket_faces]
    centroids = body_vertices[socket_triangles].mean(axis=1)

    placed = []
    for side, selector in (("left", centroids[:, 0] < 0),
                           ("right", centroids[:, 0] > 0)):
        subset = assets.socket_faces[selector]
        rim_points = body_vertices[_rim_loop(body_faces, subset)]
        rim = rim_points

        # outward socket normal: mean of the removed faces' normals
        tri = body_vertices[body_faces[subset]]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        outward = normals.mean(0)
        outward /= np.linalg.norm(outward)

        candidates = []
        for mirrored in (False, True):
            source = assets.margin_loop.copy()
            gaze = assets.gaze_axis.copy()
            if mirrored:
                source[:, 0] *= -1.0
                gaze[0] *= -1.0
            rms, s, rotation, translation = _fit_loops(source, rim)
            if float(np.dot(rotation @ gaze, outward)) > 0:
                candidates.append((rms, mirrored, s, rotation, translation, gaze))
        if not candidates:
            raise ValueError(f"eye_{side}: no rim fit points the gaze outward")
        rms, mirrored, s, rotation, translation, gaze = min(candidates,
                                                            key=lambda c: c[0])

        vertices = assets.vertices.copy()
        faces = assets.faces
        if mirrored:
            vertices = vertices.copy()
            vertices[:, 0] *= -1.0
            faces = faces[:, ::-1].copy()   # a reflection inverts winding
        world = vertices @ (s * rotation).T + translation
        world_gaze = rotation @ gaze
        world_gaze /= np.linalg.norm(world_gaze)

        center = world.mean(0)
        world = center + (world - center) * assets.scale
        world = world + world_gaze * assets.forward_cm
        world = world + np.array([0.0, 1.0, 0.0]) * assets.up_cm

        placed.append(
            PlacedEye(side=side, vertices=world, faces=faces, uv=assets.uv,
                      gaze=world_gaze, fit_rms_cm=rms, rim=rim_points)
        )
    return placed
