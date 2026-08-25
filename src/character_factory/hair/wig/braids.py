"""Braid-family primitives: braids, twists, locs — as opaque swept tubes.

Core construction is Pixar's Lissajous braid sweep (Ogunseitan, "Space Rangers
with Cornrows", SIGGRAPH Talks 2022): strand i of N gets cross-section offsets
    x(t) = A sin(2pi (F t + delta_i))
    y(t) = B sin(2pi (F t/2 - delta_i)) * 2     (half frequency -> figure-eight)
with delta_i = i/N, swept along a spine curve. Twists are the N=2, equal-
frequency case; locs are a single tube with radius noise and spine wobble.
Cornrow mode replaces the sweep frame normal with the scalp normal.

Every dimension is a creation-time parameter in head.scale units.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from .density import FULL
from .head import Head
from .primitives import _mesh


# --------------------------------------------------------------- sweep core
def _frames(spine, ref):
    """Tangent + two normals along a polyline. e1 = cross(t, ref)."""
    T = np.gradient(spine, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.asarray(ref, dtype=np.float64)
    e1 = np.cross(T, ref)
    bad = np.linalg.norm(e1, axis=1) < 1e-6
    if bad.any():
        e1[bad] = np.cross(T[bad], [0.017, 0.94, 0.33])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(e1, T)
    return T, e1, e2


def _tube_along(pts, radii, n_seg=7, uv_u0=0.5, ref=(1.0, 0, 0), close_tip=True):
    """Minimal swept circle tube (own frames, capped tip)."""
    T, e1, e2 = _frames(pts, ref)
    ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    n = len(pts)
    rings = (
        pts[:, None, :]
        + radii[:, None, None] * (np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :])
    )
    verts = rings.reshape(-1, 3)
    uv = np.stack(
        [
            np.tile(uv_u0 + ang / (2 * np.pi) * 0.15, n),
            np.repeat(np.linspace(0, 1.3, n), n_seg),
        ],
        1,
    )
    faces = []
    for i in range(n - 1):
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            a, b = i * n_seg + j, i * n_seg + jn
            c, d = (i + 1) * n_seg + jn, (i + 1) * n_seg + j
            faces += [[a, b, c], [a, c, d]]
    if close_tip:
        tip_i = len(verts)
        verts = np.vstack([verts, pts[-1] + T[-1] * radii[-1] * 0.8])
        uv = np.vstack([uv, [[uv_u0, 1.3]]])
        for j in range(n_seg):
            faces.append([(n - 1) * n_seg + j, (n - 1) * n_seg + (j + 1) % n_seg, tip_i])
    return verts, np.array(faces), uv


def _concat(parts):
    verts, faces, uv = [], [], []
    off = 0
    for v, f, u in parts:
        verts.append(v)
        faces.append(f + off)
        uv.append(u)
        off += len(v)
    return _mesh(np.vstack(verts), np.vstack(faces), np.vstack(uv))


def braided_strands(
    spine: np.ndarray,
    width: float,
    depth: float,
    knot_freq: float,
    strand_radius: float,
    mode: str = "braid",
    taper: float = 0.85,
    seed: int = 0,
    ref=(1.0, 0, 0),
    scalp_normals: np.ndarray | None = None,
    n_seg: int = 7,
):
    """Sweep braid/twist/loc strands along a spine polyline.

    scalp_normals: optional per-point normals — cornrow mode (frame e2 follows
    the scalp so knots sit ON the head, per the Pixar construction)."""
    n = len(spine)
    T, e1, e2 = _frames(spine, ref)
    if scalp_normals is not None:
        e2 = scalp_normals / (np.linalg.norm(scalp_normals, axis=1, keepdims=True) + 1e-12)
        e1 = np.cross(e2, T)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    t = np.linspace(0, 1, n)
    fade = np.minimum(1.0, t / 0.08)  # gather at the root
    tap = 1 - (1 - taper) * 0  # strand radius taper handled below
    rng = np.random.default_rng(seed)

    if mode == "braid":
        deltas, fy = [0.0, 1 / 3, 2 / 3], 0.5
        n_str = 3
    elif mode == "twist":
        deltas, fy = [0.0, 0.5], 1.0
        n_str = 2
    else:  # loc
        deltas, fy = [0.0], 1.0
        n_str = 1

    parts = []
    for i, d in enumerate(deltas):
        if mode == "loc":
            wob = rng.standard_normal((4, 3))
            off = sum(
                0.35 * width * np.sin(2 * np.pi * (knot_freq * 0.35 * t * (k + 1) + rng.uniform(0, 1)))[:, None]
                * (np.cos(ph) * e1 + np.sin(ph) * e2)
                / (k + 1)
                for k, ph in enumerate(rng.uniform(0, 2 * np.pi, 2))
            )
            r = strand_radius * (1 + 0.18 * np.sin(2 * np.pi * t * knot_freq + rng.uniform(0, 6)))
        else:
            x = width * np.sin(2 * np.pi * (knot_freq * t + d))
            y = depth * np.sin(2 * np.pi * (knot_freq * fy * t - d)) * (2 if mode == "braid" else 1)
            off = x[:, None] * e1 + y[:, None] * e2
            r = np.full(n, strand_radius)
        r = r * (1 - (1 - taper) * np.clip((t - 0.75) / 0.25, 0, 1))
        r[-1] *= 0.35  # tied-off tip
        pts = spine + off * fade[:, None]
        parts.append(_tube_along(pts, r, n_seg=n_seg, uv_u0=0.4 + 0.2 * i, ref=ref))
    return parts


# ------------------------------------------------------------ hanging fields
@dataclass
class BraidFieldSpec:
    """Box braids / twists / locs: scattered roots, hanging swept strands."""

    mode: str = "braid"  # 'braid' | 'twist' | 'loc'
    count: int = 52
    length: float = 1.9  # head.scale units
    length_jitter: float = 0.12
    width: float = 0.055  # braid cross-section half-width
    strand_radius: float = 0.048
    knot_freq: float = 9.0
    taper: float = 0.8
    spread: float = 0.35  # how far tips swing outward
    hairline_margin: float = 0.06
    beads: float = 0.0  # fraction of braids wearing bead cuffs near the tips
    seed: int = 0


def generate_braid_field(head: Head, spec: BraidFieldSpec, hairline_v,
                         density=FULL) -> trimesh.Trimesh:
    s = head.scale
    rng = np.random.default_rng(spec.seed)
    # roots: sunflower spiral over the chart, kept inside the hairline
    golden = np.pi * (3 - np.sqrt(5))
    pts = []
    k = 0
    while len(pts) < spec.count and k < spec.count * 8:
        frac = (k + 0.5) / (spec.count * 1.9)
        v = np.sqrt(frac) * 1.45
        u = ((k * golden / np.pi + 1) % 2) - 1
        k += 1
        if v <= hairline_v(np.array([u]))[0] - spec.hairline_margin:
            pts.append((u, v))
    roots_uv = np.array(pts)
    n_b = len(roots_uv)
    u, v = roots_uv[:, 0], roots_uv[:, 1]
    P0 = head.scalp_point(u, v, radial_offset=np.full(n_b, -0.01 * s))
    d_out = head.radial_dir_world(u, v)
    down = np.array([0.0, -1.0, 0.0])

    L = spec.length * s * (1 + spec.length_jitter * rng.standard_normal(n_b))
    n_pts = max(4, round(16 * density.braid_spine_scale))
    tt = np.linspace(0, 1, n_pts)
    fwd = head.forward
    parts = []
    for i in range(n_b):
        # spine: out along scalp normal, then bend to gravity, swing outward.
        # Clamp the forward component of the swing so front-of-crown braids
        # fall beside the face, not over it.
        dir0 = d_out[i]
        swing = dir0.copy()
        f_comp = swing @ fwd
        if f_comp > 0.15:
            swing = swing - (f_comp - 0.15) * fwd
            swing /= np.linalg.norm(swing) + 1e-12
        if abs(u[i]) < 0.30 and v[i] < 0.55:
            # front-of-crown roots: sweep sideways so braids frame the face
            side = np.cross(np.array([0.0, 1.0, 0.0]), fwd) * np.sign(u[i] if u[i] != 0 else 1)
            swing = swing * 0.4 + side * 0.6
        ctrl = P0[i] + dir0 * 0.30 * L[i]
        bot = P0[i] + swing * spec.spread * L[i] * 0.6 + down * L[i]
        spine = (
            (1 - tt[:, None]) ** 2 * P0[i]
            + 2 * tt[:, None] * (1 - tt[:, None]) * ctrl
            + tt[:, None] ** 2 * bot
        )
        # keep outside the body/head
        r_min = head.body_clearance(spine) + 0.03 * s
        cx, cz = head.center[0], head.center[2]
        dxz = spine[:, [0, 2]] - np.array([cx, cz])
        r_now = np.linalg.norm(dxz, axis=1)
        # cap the push so shoulder-level clearance bends, never kinks, a braid
        push = np.minimum(np.maximum(r_min - r_now, 0), 0.30 * L[i]) * np.clip(tt * 3, 0, 1)
        spine[:, [0, 2]] += dxz / (r_now[:, None] + 1e-9) * push[:, None]
        parts += braided_strands(
            spine,
            width=spec.width * s,
            depth=spec.width * 0.8 * s,
            knot_freq=spec.knot_freq * L[i] / (spec.length * s + 1e-9) * spec.length,
            strand_radius=spec.strand_radius * s,
            mode=spec.mode,
            taper=spec.taper,
            seed=spec.seed + i,
            ref=head.forward,
            n_seg=density.braid_radial_segments or (6 if spec.mode != "loc" else 7),
        )
        if spec.beads > 0 and rng.random() < spec.beads:
            # bead cuff: short fat tube around the braid near the tip
            for tb in rng.uniform(0.55, 0.92, rng.integers(1, 3)):
                k0 = int(tb * (n_pts - 2))
                seg = spine[k0 : k0 + 2]
                seg = np.vstack([seg[0], seg.mean(0), seg[1]])
                br = (spec.width + spec.strand_radius) * s * 1.15
                parts.append(
                    _tube_along(seg, np.array([br, br * 1.08, br]),
                                n_seg=density.braid_radial_segments or 8,
                                uv_u0=0.93, ref=head.forward)
                )
    return _concat(parts)


# ----------------------------------------------------------------- cornrows
@dataclass
class CornrowSpec:
    rows: int = 10
    width: float = 0.055
    strand_radius: float = 0.05
    knot_freq: float = 11.0
    tail_length: float = 0.0  # >0: rows continue into hanging braids at the nape
    seed: int = 0


def generate_cornrows(head: Head, spec: CornrowSpec, hairline_v,
                      density=FULL) -> trimesh.Trimesh:
    """Braids lying ON the scalp, front hairline to nape, scalp-normal frames.

    Row paths are front-to-back arcs in laterally offset sagittal planes:
    D(theta) = normalize(sin(theta)*fwd + cos(theta)*up + k*left), theta from
    the front hairline over the crown to the nape; k = lateral offset."""
    s = head.scale
    up = np.array([0.0, 1.0, 0.0])
    fwd = head.forward
    left = np.cross(up, fwd)
    parts = []
    n_pts = max(6, round(26 * density.braid_spine_scale))
    ks = np.linspace(-0.85, 0.85, spec.rows)
    for i, k in enumerate(ks):
        v_front = min(hairline_v(np.array([abs(k) * 0.7]))[0] - 0.04, 0.86)
        th = np.linspace(v_front * np.pi / 2, -1.38 * np.pi / 2 * (1 - 0.25 * abs(k)), n_pts)
        D = (
            np.sin(th)[:, None] * fwd[None]
            + np.cos(th)[:, None] * up[None]
            + (k * (0.4 + 0.6 * np.cos(th / 1.4) ** 2))[..., None] * left[None]
        )
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        uu, vv = head._chart_of_dirs(D)
        vv = np.clip(vv, 0.01, head.V_MAX - 0.01)
        spine = head.scalp_point(uu, vv, radial_offset=np.full(n_pts, 0.028 * s))
        normals = head.radial_dir_world(uu, vv)
        parts += braided_strands(
            spine,
            width=spec.width * s,
            depth=spec.width * 0.55 * s,
            knot_freq=spec.knot_freq,
            strand_radius=spec.strand_radius * s,
            mode="braid",
            seed=spec.seed + i,
            ref=head.forward,
            scalp_normals=normals,
            n_seg=density.braid_radial_segments or 6,
        )
        if spec.tail_length > 0:
            end = spine[-1]
            d_out = normals[-1]
            down = np.array([0.0, -1.0, 0.0])
            L = spec.tail_length * s
            tt = np.linspace(0, 1, 12)[:, None]
            tail = (1 - tt) ** 2 * end + 2 * tt * (1 - tt) * (end + d_out * 0.3 * L) + tt**2 * (
                end + d_out * 0.2 * L + down * L
            )
            r_min = head.body_clearance(tail) + 0.03 * s
            cx, cz = head.center[0], head.center[2]
            dxz = tail[:, [0, 2]] - np.array([cx, cz])
            r_now = np.linalg.norm(dxz, axis=1)
            push = np.maximum(r_min - r_now, 0) * np.clip(tt[:, 0] * 3, 0, 1)
            tail[:, [0, 2]] += dxz / (r_now[:, None] + 1e-9) * push[:, None]
            parts += braided_strands(
                tail, spec.width * s, spec.width * 0.8 * s, spec.knot_freq * 0.5,
                spec.strand_radius * s, mode="braid", seed=spec.seed + 100 + i,
                ref=head.forward, n_seg=density.braid_radial_segments or 6,
            )
    return _concat(parts)


# -------------------------------------------------------------- braided tail
@dataclass
class BraidTailSpec:
    """Single braided ponytail, or twin side braids."""

    twin: bool = False
    pos_v: float = 0.55
    length: float = 2.4
    width: float = 0.14
    strand_radius: float = 0.11
    knot_freq: float = 7.0
    sag: float = 0.5
    seed: int = 0


def generate_braid_tail(head: Head, spec: BraidTailSpec,
                        density=FULL) -> trimesh.Trimesh:
    s = head.scale
    down = np.array([0.0, -1.0, 0.0])
    anchors = [(0.62, spec.pos_v * 0.85), (-0.62, spec.pos_v * 0.85)] if spec.twin else [(1.0, spec.pos_v)]
    parts = []
    for k, (u0, v0) in enumerate(anchors):
        g = head.scalp_point(np.array([u0]), np.array([v0]), radial_offset=np.array([0.05 * s]))[0]
        d_out = head.radial_dir_world(np.array([u0]), np.array([v0]))[0]
        L = spec.length * s
        n = 30
        tt = np.linspace(0, 1, n)[:, None]
        dir0 = d_out * (1 - spec.sag * 0.5) + down * spec.sag * 0.5
        dir0 /= np.linalg.norm(dir0)
        spine = (1 - tt) ** 2 * g + 2 * tt * (1 - tt) * (g + dir0 * 0.4 * L) + tt**2 * (
            g + d_out * 0.2 * L * (1 - spec.sag) + down * L * 0.95
        )
        r_min = head.body_clearance(spine) + 0.03 * s
        cx, cz = head.center[0], head.center[2]
        dxz = spine[:, [0, 2]] - np.array([cx, cz])
        r_now = np.linalg.norm(dxz, axis=1)
        push = np.maximum(r_min - r_now, 0) * np.clip(tt[:, 0] * 3, 0, 1)
        spine[:, [0, 2]] += dxz / (r_now[:, None] + 1e-9) * push[:, None]
        parts += braided_strands(
            spine, spec.width * s, spec.width * 0.8 * s, spec.knot_freq,
            spec.strand_radius * s, mode="braid", seed=spec.seed + k,
            ref=head.forward, n_seg=density.braid_radial_segments or 8,
        )
    return _concat(parts)
