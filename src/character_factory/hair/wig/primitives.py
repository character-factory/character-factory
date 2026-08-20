"""Accent primitives: bangs flap, swept tubes (ponytail), coils (bun), mohawk fin.

All lengths in multiples of head.scale. Each generator returns a Trimesh with
vertex_attributes['chart_uv'] (u, v-ish) for texturing.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from .head import Head


def _mesh(verts, faces, uv):
    m = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces), process=False)
    m.vertex_attributes["chart_uv"] = np.asarray(uv, dtype=np.float64)
    return m


def _tube(points, radii, flatten=1.0, n_seg=10, uv_u0=0.0, ref=None):
    """Sweep an ellipse profile along a polyline. `ref` picks the frame: the
    full-radius axis is cross(tangent, ref) — pass the head's forward so a
    flattened tail is wide laterally, not front-back."""
    P = np.asarray(points)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([1.0, 0.0, 0.0]) if ref is None else np.asarray(ref, dtype=np.float64)
    verts, uv = [], []
    for i in range(n):
        t = T[i]
        side = np.cross(t, ref)
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(t, [0, 0, 1.0])
        side /= np.linalg.norm(side)
        up2 = np.cross(side, t)
        ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        ring = P[i] + radii[i] * (np.outer(np.cos(ang), side) + flatten * np.outer(np.sin(ang), up2))
        verts.append(ring)
        uv.append(np.stack([uv_u0 + ang / (2 * np.pi) * 0.2, np.full(n_seg, i / max(n - 1, 1) * 1.5)], 1))
    verts = np.vstack(verts)
    uv = np.vstack(uv)
    faces = []
    for i in range(n - 1):
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            a, b = i * n_seg + j, i * n_seg + jn
            c, d = (i + 1) * n_seg + jn, (i + 1) * n_seg + j
            faces += [[a, b, c], [a, c, d]]
    # close the tip with a fan
    tip = len(verts)
    verts = np.vstack([verts, P[-1][None]])
    uv = np.vstack([uv, [[uv_u0, 1.5]]])
    for j in range(n_seg):
        jn = (j + 1) % n_seg
        faces.append([(n - 1) * n_seg + j, (n - 1) * n_seg + jn, tip])
    return _mesh(verts, faces, uv)


# ------------------------------------------------------------------- bangs
@dataclass
class BangsSpec:
    length: float = 0.62  # down the forehead, head-scale units
    u_range: float = 0.32  # matches the curtain gap
    lift: float = 0.14  # how far off the forehead the flap floats
    tips: str = "blunt"  # 'blunt' | 'wispy'
    updo: float = 0.0  # 0 = bangs fall down; 1 = swept up-and-back (pompadour)
    n_u: int = 24
    seed: int = 0


def generate_bangs(head: Head, spec: BangsSpec, hairline_v, volume: float) -> trimesh.Trimesh:
    """Flap from the front hairline sweeping down over the forehead.
    hairline_v: callable v(u); volume: cap volume at the front (for root offset)."""
    s = head.scale
    n_u, n_t = spec.n_u, 10
    u = np.linspace(-spec.u_range, spec.u_range, n_u)
    v0 = hairline_v(u) - 0.06
    P0 = head.scalp_point(u, v0, radial_offset=np.full(n_u, volume * 0.8 * s))
    d_out = head.radial_dir_world(u, v0)
    down = np.array([0.0, -1.0, 0.0])
    L = spec.length * s * (0.82 + 0.18 * np.cos(u / max(spec.u_range, 1e-6) * np.pi))  # shorter at edges
    up = np.array([0.0, 1.0, 0.0])
    # drop direction blends from gravity (bangs) to up-and-back (pompadour)
    drop = (1 - spec.updo) * down + spec.updo * (0.85 * up - 0.45 * head.forward)
    drop = drop / np.linalg.norm(drop)
    lift_eff = spec.lift * (1 + 1.6 * spec.updo)
    ctrl = P0 + d_out * (lift_eff * s) + drop[None] * (0.4 * L[:, None])
    # bottom presses toward the forehead again (or crests back, for updo)
    bot = head.scalp_point(u, v0 + 0.001, radial_offset=np.full(n_u, lift_eff * 0.6 * s)) + drop[None] * L[:, None]
    t = np.linspace(0, 1, n_t)[:, None, None]
    Pg = (1 - t) ** 2 * P0 + 2 * t * (1 - t) * ctrl + t**2 * bot  # (n_t, n_u, 3)

    if spec.tips == "wispy":
        rng = np.random.default_rng(spec.seed)
        jit = 1.0 + 0.25 * np.sin(u * 40 + rng.uniform(0, 6.28)) * rng.uniform(0.6, 1.0)
        Pg[-1] = Pg[-2] + (Pg[-1] - Pg[-2]) * jit[:, None]

    verts = Pg.reshape(-1, 3)
    uv = np.stack(
        [np.broadcast_to(u, (n_t, n_u)), np.broadcast_to(t[..., 0], (n_t, n_u)) * 0.8], -1
    ).reshape(-1, 2)
    faces = []
    idx = np.arange(n_t * n_u).reshape(n_t, n_u)
    for i in range(n_t - 1):
        for j in range(n_u - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]
            faces += [[a, b, c], [a, c, d]]
    return _mesh(verts, faces, uv)


# ---------------------------------------------------------------- ponytail
@dataclass
class TailSpec:
    pos_v: float = 0.55  # gather point: v along the anchor meridian
    pos_u: float = 1.0  # anchor meridian: 1 = back center, +-0.6 = sides
    twin: bool = False  # mirror at +-pos_u (pigtails)
    length: float = 2.2
    sag: float = 0.55  # how much the tail curve sags toward vertical
    radius: float = 0.30  # bulge radius
    tie_radius: float = 0.13
    flatten: float = 0.85
    seed: int = 0


def _one_tail(head: Head, spec: TailSpec, u0: float) -> list:
    s = head.scale
    g = head.scalp_point(np.array([u0]), np.array([spec.pos_v]), radial_offset=np.array([0.06 * s]))[0]
    d_out = head.radial_dir_world(np.array([u0]), np.array([spec.pos_v]))[0]
    down = np.array([0.0, -1.0, 0.0])
    L = spec.length * s
    # sagging spine: out, then down
    n = 24
    tt = np.linspace(0, 1, n)
    dir0 = d_out * (1 - spec.sag * 0.5) + down * spec.sag * 0.5
    dir0 /= np.linalg.norm(dir0)
    ctrl = g + dir0 * 0.45 * L
    bot = g + d_out * 0.25 * L * (1 - spec.sag) + down * L * 0.92
    spine = (1 - tt[:, None]) ** 2 * g + 2 * tt[:, None] * (1 - tt[:, None]) * ctrl + tt[:, None] ** 2 * bot
    # radius profile: tie -> bulge -> taper
    r = spec.tie_radius + (spec.radius - spec.tie_radius) * np.clip(tt / 0.25, 0, 1)
    r = r * (1 - np.clip((tt - 0.55) / 0.45, 0, 1) ** 1.3 * 0.96)
    tail = _tube(spine, r * s, flatten=spec.flatten, uv_u0=0.6, ref=head.forward)
    # tie ring
    ring_t = np.linspace(0, 2 * np.pi, 16)
    axis = dir0
    a1 = np.cross(axis, [0, 1, 0.0])
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(axis, a1)
    ring_pts = g + axis * 0.05 * L + (np.outer(np.cos(ring_t), a1) + np.outer(np.sin(ring_t), a2)) * spec.tie_radius * 1.25 * s
    tie = _tube(ring_pts, np.full(len(ring_pts), 0.045 * s), uv_u0=0.9)
    return [tail, tie]


def generate_tail(head: Head, spec: TailSpec) -> trimesh.Trimesh:
    anchors = [abs(spec.pos_u), -abs(spec.pos_u)] if spec.twin else [spec.pos_u]
    parts = []
    for u0 in anchors:
        parts += _one_tail(head, spec, u0)
    out = trimesh.util.concatenate(parts)
    out.vertex_attributes["chart_uv"] = np.vstack([p.vertex_attributes["chart_uv"] for p in parts])
    return out


# --------------------------------------------------------------------- bun
@dataclass
class BunSpec:
    pos_v: float = 0.35  # small = high (top knot)
    pos_u: float = 1.0  # anchor meridian: 1 = back center, +-0.5 = space buns
    twin: bool = False  # mirror at +-pos_u
    radius: float = 0.42  # coil loop radius
    tube: float = 0.16  # coil tube radius
    turns: float = 2.2
    count: int = 1  # >1: scatter `count` coils over the scalp (bantu knots)
    seed: int = 0


def _one_coil(head: Head, spec: BunSpec, u0: float, v0: float, scale_mul: float = 1.0):
    s = head.scale * scale_mul
    g = head.scalp_point(np.array([u0]), np.array([v0]), radial_offset=np.array([0.02 * head.scale]))[0]
    ax = head.radial_dir_world(np.array([u0]), np.array([v0]))[0]
    a1 = np.cross(ax, [0, 1, 0.0])
    if np.linalg.norm(a1) < 1e-6:
        a1 = np.cross(ax, [1.0, 0, 0])
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(ax, a1)
    n = 60
    th = np.linspace(0, spec.turns * 2 * np.pi, n)
    frac = th / th[-1]
    R = spec.radius * s * (1 - 0.55 * frac)  # spiral inward/upward
    lift = (0.3 + 0.9 * frac) * spec.tube * s
    spine = g + np.outer(np.cos(th), a1) * R[:, None] + np.outer(np.sin(th), a2) * R[:, None] + np.outer(lift, ax)
    r = spec.tube * s * (1 - 0.6 * frac**2)
    return _tube(spine, r, uv_u0=0.4)


def generate_bun(head: Head, spec: BunSpec) -> trimesh.Trimesh:
    if spec.count > 1:
        # bantu knots: sunflower-scattered small coils over the crown
        golden = np.pi * (3 - np.sqrt(5))
        anchors = []
        k = 0
        while len(anchors) < spec.count and k < spec.count * 8:
            fr = (k + 0.5) / (spec.count * 1.3)
            vv = np.sqrt(fr) * 1.05
            uu = ((k * golden / np.pi + 1) % 2) - 1
            k += 1
            # conservative hairline gate: low over the face, higher at the back
            v_lim = 0.55 + 0.5 * np.clip((abs(uu) - 0.3) / 0.7, 0, 1)
            if vv <= v_lim:
                anchors.append((uu, vv))
    elif spec.twin:
        anchors = [(abs(spec.pos_u), spec.pos_v), (-abs(spec.pos_u), spec.pos_v)]
    else:
        anchors = [(spec.pos_u, spec.pos_v)]
    parts = [_one_coil(head, spec, u0, v0) for u0, v0 in anchors]
    if len(parts) == 1:
        return parts[0]
    out = trimesh.util.concatenate(parts)
    out.vertex_attributes["chart_uv"] = np.vstack([p.vertex_attributes["chart_uv"] for p in parts])
    return out


# ------------------------------------------------------------------ mohawk
@dataclass
class MohawkSpec:
    height: float = 0.55
    thickness: float = 0.32
    front_v: float = 0.62  # where the fin starts on the front sagittal
    back_v: float = 1.25  # where it ends at the nape
    spike: float = 0.0  # 0 = smooth fin, >0 = jagged crest
    seed: int = 0


def generate_mohawk(head: Head, spec: MohawkSpec) -> trimesh.Trimesh:
    s = head.scale
    n = 36
    # sagittal path: front hairline -> crown (v=0) -> nape, via signed angle
    a = np.linspace(spec.front_v, -spec.back_v, n)  # + = front side, - = back
    u = np.where(a >= 0, 0.0, 1.0)
    v = np.abs(a)
    base = head.scalp_point(u, v, radial_offset=np.full(n, 0.01 * s))
    up_dir = head.radial_dir_world(u, v)
    left = np.cross(np.array([0.0, 1.0, 0.0]), head.forward)
    prof = np.sin(np.clip((a - a.min()) / (a.max() - a.min()), 0, 1) * np.pi) ** 0.7
    H = spec.height * s * (0.35 + 0.65 * prof)
    if spec.spike > 0:
        rng = np.random.default_rng(spec.seed)
        H = H * (1 + spec.spike * np.sin(np.arange(n) * 2.1 + rng.uniform(0, 6)) * 0.5)
    crest = base + up_dir * H[:, None]
    half = spec.thickness * 0.5 * s
    rows = [base - left * half, base - left * half * 0.85 + up_dir * H[:, None] * 0.55,
            crest, base + left * half * 0.85 + up_dir * H[:, None] * 0.55, base + left * half]
    n_r = len(rows)
    verts = np.vstack(rows)
    uv = np.vstack([np.stack([np.full(n, k / (n_r - 1)), np.linspace(0, 1.4, n)], 1) for k in range(n_r)])
    faces = []
    for k in range(n_r - 1):
        for i in range(n - 1):
            a_, b = k * n + i, k * n + i + 1
            c, d = (k + 1) * n + i + 1, (k + 1) * n + i
            faces += [[a_, b, c], [a_, c, d]]
    return _mesh(verts, faces, uv)
