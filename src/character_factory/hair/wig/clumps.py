"""Flow-guided opaque hair clumps.

The shell generator is useful as a scalp under-layer, but a connected curtain
cannot provide the overlaps, negative space, or tapered tips that make medium
and long hair read as hair.  This module builds those visible masses as
separate, flattened sweeps.  The roots follow the canonical scalp chart and
long clumps transition into gravity/body-aware world-space curves.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from .head import Head
from .guides import GuideDrapeSpec, drape_guide, hanging_normals, route_direction
from .shell import HairlineSpec


@dataclass
class ClumpFieldSpec:
    """A deterministic field of visible polygonal locks.

    All geometric lengths are multiples of ``Head.scale``.  ``short`` follows
    the scalp, ``long`` adds a hanging body-cleared section,
    and ``coily`` scatters shallow rounded curl clusters over an inflated cap.
    """

    mode: str = "short"
    count: int = 18
    part_u: float = 0.0
    part_gap: float = 0.035
    part_open: bool = False
    root_v: float = 0.15
    root_spread: float = 0.68
    volume: float = 0.10
    lift: float = 0.08
    width: float = 0.23
    thickness: float = 0.055
    lower_width: float = 0.70
    layers: int = 2
    layer_offset: float = 0.018
    hairline_margin: float = 0.055
    # Hanging section.  ``length`` is the back length, ``length_front`` the
    # face-framing length, and optional ``length_side`` adds an independently
    # controllable temple/shoulder anchor before interpolation to the nape.
    length: float = 2.8
    length_front: float | None = None
    length_side: float | None = None
    face_gap: float = 0.28
    flare: float = 0.34
    tuck: float = 0.16
    wave: float = 0.0
    wave_freq: float = 2.0
    tip_jitter: float = 0.12
    guide_segments: int = 19
    profile_segments: int = 8
    seed: int = 0
    drape: GuideDrapeSpec | None = None

    def __post_init__(self):
        if self.mode not in {"short", "long", "coily"}:
            raise ValueError(f"unsupported clump mode: {self.mode}")
        if self.count < 2:
            raise ValueError("a clump field needs at least two clumps")
        if self.guide_segments < 8:
            raise ValueError("a hanging guide needs at least eight segments")
        if self.length_front is None:
            self.length_front = self.length


def _length_at_u(spec: ClumpFieldSpec, u: float) -> float:
    """Interpolate semantic front/side/back cut lengths around the head."""

    frac = float(np.clip(
        (abs(u) - spec.face_gap) / max(1.0 - spec.face_gap, 1e-6), 0.0, 1.0
    ))
    if spec.length_side is None:
        return float(spec.length_front + (spec.length - spec.length_front) * np.sqrt(frac))

    side_frac = float(np.clip(
        (0.58 - spec.face_gap) / max(1.0 - spec.face_gap, 1e-6), 0.18, 0.82
    ))
    if frac <= side_frac:
        t = float(_smoothstep(np.array([frac / side_frac]))[0])
        return float((1.0 - t) * spec.length_front + t * spec.length_side)
    t = float(_smoothstep(np.array([(frac - side_frac) / (1.0 - side_frac)]))[0])
    return float((1.0 - t) * spec.length_side + t * spec.length)


def _mesh(vertices, faces, uv) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.vertex_attributes["chart_uv"] = np.asarray(uv, dtype=np.float64)
    return mesh


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _swept_lens(
    spine: np.ndarray,
    surface_normal: np.ndarray,
    half_width: np.ndarray,
    half_thickness: np.ndarray,
    *,
    n_seg: int,
    uv_phase: float,
) -> trimesh.Trimesh:
    """Sweep a flattened closed profile with its broad face on the scalp.

    Supplying a normal field avoids the arbitrary twisting caused by a fixed
    frame reference.  The width axis is tangent to the scalp and the thin
    axis follows the local outward normal.  Eight sides keep a sculpted,
    faceted highlight instead of a round hose silhouette.
    """

    P = np.asarray(spine, dtype=np.float64)
    N = _unit(np.asarray(surface_normal, dtype=np.float64))
    T = _unit(np.gradient(P, axis=0))
    side = _unit(np.cross(N, T))
    # Re-orthogonalize the thickness axis so waves never skew the profile.
    thick = _unit(np.cross(T, side))

    angle = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    ca, sa = np.cos(angle), np.sin(angle)
    rings = (
        P[:, None, :]
        + side[:, None, :] * (half_width[:, None, None] * ca[None, :, None])
        + thick[:, None, :] * (half_thickness[:, None, None] * sa[None, :, None])
    )

    n = len(P)
    vertices = rings.reshape(-1, 3)
    # Texture v follows the physical arclength, normalized per clump.  The
    # narrow u band prevents unrelated locks from sharing identical streaks.
    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg_len)]
    arc /= max(float(arc[-1]), 1e-9)
    uu = uv_phase + (angle / (2 * np.pi)) * 0.16
    uv = np.stack(
        [np.broadcast_to(uu, (n, n_seg)), np.broadcast_to(arc[:, None] * 1.5, (n, n_seg))],
        axis=-1,
    ).reshape(-1, 2)

    faces = []
    for i in range(n - 1):
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            a, b = i * n_seg + j, i * n_seg + jn
            c, d = (i + 1) * n_seg + jn, (i + 1) * n_seg + j
            faces.extend(((a, b, c), (a, c, d)))

    # Both ends are closed: the root cap is usually hidden by the under-cap,
    # while the tiny end fan produces a clean opaque tapered tip.
    for ring_i, reverse in ((0, True), (n - 1, False)):
        center_i = len(vertices)
        vertices = np.vstack([vertices, P[ring_i]])
        uv = np.vstack([uv, [uv_phase, 0.0 if ring_i == 0 else 1.5]])
        base = ring_i * n_seg
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            tri = (base + j, base + jn, center_i)
            faces.append(tri[::-1] if reverse else tri)
    return _mesh(vertices, faces, uv)


def _wrapped_delta(u1: float, u0: float) -> float:
    """Shortest signed delta on the periodic [-1, 1) chart."""

    return float((u1 - u0 + 1.0) % 2.0 - 1.0)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _end_us(spec: ClumpFieldSpec) -> np.ndarray:
    if spec.mode == "short":
        # Avoid duplicating the back seam while covering the whole head.
        return np.linspace(-1.0, 1.0, spec.count, endpoint=False) + 1.0 / spec.count

    # A real opening in front is essential: separate face-framing locks look
    # better than a sheet with its faces deleted after construction.
    n_left = spec.count // 2
    n_right = spec.count - n_left
    eps = 0.025
    left = np.linspace(-1.0 + eps, -spec.face_gap, n_left)
    right = np.linspace(spec.face_gap, 1.0 - eps, n_right)
    return np.r_[left, right]


def _root_chart_path(
    head: Head,
    spec: ClumpFieldSpec,
    hairline: HairlineSpec,
    u_end: float,
    layer: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a scalp-following root path and its chart/normal samples."""

    d_from_part = _wrapped_delta(u_end, spec.part_u)
    side_sign = 1.0 if d_from_part >= 0 else -1.0
    part_root = spec.part_u + side_sign * spec.part_gap * (1.0 + 0.34 * layer)
    delta_from_root = _wrapped_delta(u_end, part_root)
    # Only front/temple locks truly originate at the visible part.  Side and
    # back locks begin progressively farther around the crown; otherwise all
    # sweeps cross in an artificial X/starburst at the back of the head.
    backness = float(_smoothstep(np.array([
        np.clip((abs(delta_from_root) - 0.24) / 0.66, 0.0, 1.0)
    ]))[0])
    u0 = part_root + delta_from_root * spec.root_spread * backness
    delta = _wrapped_delta(u_end, u0)
    t = np.linspace(0.0, 1.0, n)
    ease = _smoothstep(t)
    u = u0 + delta * ease
    v_end = float(hairline.v_of_u(np.array([u_end]), rough=False)[0] - spec.hairline_margin)
    # Staggering roots avoids a pinched starburst at the crown while keeping a
    # legible part line.  Each lock still travels from root to its own rim.
    root_v = (
        spec.root_v
        + 0.025 * layer
        + 0.035 * min(abs(delta), 1.0)
        + 0.20 * spec.root_spread * backness
    )
    v = root_v + (v_end - root_v) * ease

    # Crown lift fades toward the hairline.  Alternating layers are separated
    # just enough to show overlap without creating floating strips.
    overlap = spec.layer_offset * layer
    lift = spec.lift * np.sin(np.pi * t) ** 1.2
    off = (spec.volume + overlap + lift) * head.scale
    P = head.scalp_point(u, v, radial_offset=off)
    N = head.radial_dir_world(u, v)
    return P, N, u, v


def _profile(
    n: int,
    width: float,
    thickness: float,
    scale: float,
    *,
    root_size: float = 0.42,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    # Fast root expansion, broad overlapping mid-body, then a long tapered tip.
    grow = root_size + (1.0 - root_size) * np.clip(t / 0.18, 0, 1) ** 0.65
    taper = np.clip((1.0 - t) / 0.32, 0.0, 1.0) ** 0.72
    shape = np.minimum(grow, taper)
    shape[-1] = 0.06
    return width * scale * shape, thickness * scale * np.maximum(shape, 0.18)


def _part_root_size(spec: ClumpFieldSpec, width: float, default: float) -> float:
    """Keep visible-part root profiles from bridging the cap opening."""

    if not spec.part_open:
        return default
    # Adjacent root centres are approximately +/- part_gap. Restrict each
    # root half-width to a fraction of that gap so opaque lenses cannot cover
    # the physical scalp strip immediately after the compiler opens it.
    return float(np.clip(0.62 * spec.part_gap / max(width, 1e-6), 0.08, default))


def _short_clump(
    head: Head,
    spec: ClumpFieldSpec,
    hairline: HairlineSpec,
    u_end: float,
    index: int,
) -> trimesh.Trimesh:
    rng = np.random.default_rng(spec.seed + 7919 * (index + 1))
    layer = index % max(spec.layers, 1)
    P, N, _u, _v = _root_chart_path(head, spec, hairline, u_end, layer, 13)

    # Low-frequency lateral deviation breaks the combed radial regularity but
    # stays zero at both ends, so the part and hairline remain controlled.
    T = _unit(np.gradient(P, axis=0))
    side = _unit(np.cross(N, T))
    tt = np.linspace(0.0, 1.0, len(P))
    sway = rng.uniform(-0.045, 0.045) * head.scale
    P = P + side * (sway * np.sin(np.pi * tt) ** 1.5)[:, None]

    width_jitter = rng.uniform(0.88, 1.12)
    if spec.layers >= 3:
        width_jitter *= (1.18, 0.88, 0.68)[layer % 3]
    width = spec.width * width_jitter
    W, H = _profile(
        len(P), width, spec.thickness, head.scale,
        root_size=_part_root_size(spec, width, 0.42),
    )
    return _swept_lens(
        P,
        N,
        W,
        H,
        n_seg=spec.profile_segments,
        uv_phase=(index * 0.173) % 0.84,
    )


def _clear_body(head: Head, P: np.ndarray, margin: float, blend_start: int) -> np.ndarray:
    """Push hanging samples outside the torso without changing their height."""

    out = P.copy()
    cx, cz = head.center[0], head.center[2]
    r_min = head.body_clearance(out) + margin * head.scale
    dxz = out[:, [0, 2]] - np.array([cx, cz])
    r_now = np.linalg.norm(dxz, axis=1)
    push = np.maximum(r_min - r_now, 0.0)
    blend = np.clip((np.arange(len(out)) - blend_start + 1) / 4.0, 0.0, 1.0)
    factor = 1.0 + push * blend / (r_now + 1e-9)
    out[:, 0] = cx + dxz[:, 0] * factor
    out[:, 2] = cz + dxz[:, 1] * factor
    return out


def _long_clump(
    head: Head,
    spec: ClumpFieldSpec,
    hairline: HairlineSpec,
    u_end: float,
    index: int,
) -> trimesh.Trimesh:
    rng = np.random.default_rng(spec.seed + 104729 * (index + 1))
    layer = index % max(spec.layers, 1)
    root, root_n, _u, _v = _root_chart_path(head, spec, hairline, u_end, layer, 8)

    rim = root[-1]
    cx, cz = head.center[0], head.center[2]
    rvec = rim - np.array([cx, rim[1], cz])
    rhat = rvec / (np.linalg.norm(rvec) + 1e-12)
    tangent_side = np.cross(np.array([0.0, 1.0, 0.0]), rhat)
    tangent_side /= np.linalg.norm(tangent_side) + 1e-12

    L = _length_at_u(spec, u_end) * head.scale
    L *= 1.0 + spec.tip_jitter * rng.uniform(-1.0, 1.0)
    down = np.array([0.0, -1.0, 0.0])
    route_hat = route_direction(head, u_end, spec.drape) if spec.drape is not None else None
    if route_hat is None:
        hang_hat = rhat
        shoulder_hat = rhat
    else:
        # Stay tangent-continuous at the hairline, then enter a deliberate
        # front/back shoulder lane lower down.
        is_back_lane = float(route_hat @ head.forward) < 0.0
        route_mix = 0.60 if is_back_lane else 0.38
        shoulder_hat = _unit(((1.0 - route_mix) * rhat + route_mix * route_hat)[None])[0]
        end_mix = 0.84 if is_back_lane else 0.72
        hang_hat = _unit(((1.0 - end_mix) * rhat + end_mix * route_hat)[None])[0]
    ctrl = rim + shoulder_hat * (spec.flare * head.scale) + down * (0.40 * L)
    radial_now = np.linalg.norm(rvec)
    bot = (
        np.array([cx, rim[1], cz])
        + hang_hat * radial_now * (1.0 - 0.42 * spec.tuck)
        + down * L
    )
    tt = np.linspace(0.0, 1.0, spec.guide_segments)[1:]
    hang = (
        (1.0 - tt[:, None]) ** 2 * rim
        + 2.0 * tt[:, None] * (1.0 - tt[:, None]) * ctrl
        + tt[:, None] ** 2 * bot
    )

    if spec.wave > 0:
        # Neighboring locks share a low-frequency phase trend so their waves
        # overlap as one groomed mass instead of reading as unrelated ribbons.
        phase = 1.70 * np.pi * u_end + 0.52 * layer + rng.uniform(-0.34, 0.34)
        envelope = np.sin(np.pi * tt) ** 0.75
        wave = np.sin(2 * np.pi * spec.wave_freq * tt + phase)
        hang += tangent_side * (spec.wave * 0.22 * head.scale * envelope * wave)[:, None]
        # A weaker radial wave gives each lock changing highlights in profile.
        hang += rhat * (spec.wave * 0.070 * head.scale * envelope * np.cos(2 * np.pi * spec.wave_freq * tt + phase))[:, None]

    if spec.drape is not None and spec.drape.enabled:
        # Put the authored curve into one coherent azimuthal clearance lane
        # before nearest-triangle projection.  Starting deep inside a shoulder
        # lets adjacent samples choose opposite body surfaces, folding a guide
        # into a loop before the solver has a chance to settle it.
        guide_seed = np.vstack([rim, hang])
        for _ in range(3):
            smooth = guide_seed.copy()
            smooth[1:-1] = (
                0.18 * guide_seed[:-2]
                + 0.64 * guide_seed[1:-1]
                + 0.18 * guide_seed[2:]
            )
            smooth[1:] = _clear_body(
                head, smooth[1:], margin=spec.drape.collision_margin, blend_start=0
            )
            guide_seed = smooth
        guide = drape_guide(head, guide_seed, u_end, spec.drape)
        hang = guide[1:]
    else:
        hang = _clear_body(head, hang, margin=0.11 + spec.thickness, blend_start=0)
    P = np.vstack([root, hang])

    # Use scalp normals over the root and a stable outward field below it.  A
    # gentle blend avoids a hard frame rotation at the hairline.
    hang_n = hanging_normals(head, hang)
    N = np.vstack([root_n, hang_n])
    if len(root_n) >= 3:
        N[len(root_n) - 2] = _unit((root_n[-2] + rhat)[None])[0]
        N[len(root_n) - 1] = _unit((root_n[-1] + 2.0 * rhat)[None])[0]

    width_jitter = rng.uniform(0.93, 1.07)
    if spec.layers >= 3:
        width_jitter *= (1.18, 0.88, 0.68)[layer % 3]
    width = spec.width * width_jitter
    W, H = _profile(
        len(P), width, spec.thickness, head.scale,
        root_size=_part_root_size(spec, width, 0.30),
    )
    # Long clumps should stay broad through the shoulder before tapering.
    t_all = np.linspace(0.0, 1.0, len(P))
    W *= 0.94 + 0.14 * np.sin(np.pi * t_all) ** 2
    lower_blend = _smoothstep(np.clip((t_all - 0.40) / 0.34, 0.0, 1.0))
    W *= 1.0 - (1.0 - spec.lower_width) * lower_blend
    return _swept_lens(
        P,
        N,
        W,
        H,
        n_seg=spec.profile_segments,
        uv_phase=(index * 0.137) % 0.84,
    )


def _coily_field(
    head: Head,
    spec: ClumpFieldSpec,
    hairline: HairlineSpec,
) -> trimesh.Trimesh:
    """Shallow, interlocking curl clusters for dense coily silhouettes."""

    rng = np.random.default_rng(spec.seed + 2309)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    order = rng.permutation(spec.count)
    parts = []
    world_up = np.array([0.0, 1.0, 0.0])
    for out_i, k in enumerate(order):
        # Low-discrepancy chart scatter, slightly asymmetric through phase and
        # size jitter.  sqrt(q) approximates equal-area coverage from crown to
        # the irregular hairline.
        u = float(((k * golden / np.pi + 0.37) % 2.0) - 1.0)
        hv = float(hairline.v_of_u(np.array([u]), rough=False)[0])
        q = (k + 0.55) / (spec.count + 0.1)
        v = float(0.88 * np.sqrt(q) * (hv - 0.035))
        edge = np.clip(v / max(hv, 1e-6), 0.0, 1.0)
        close_t = np.clip((1.0 - edge) / 0.48, 0.0, 1.0)
        close = float(_smoothstep(np.array([close_t]))[0])
        off = spec.volume * (0.08 + 0.92 * close) * head.scale
        center = head.scalp_point(np.array([u]), np.array([v]), radial_offset=np.array([off]))[0]
        normal = head.radial_dir_world(np.array([u]), np.array([v]))[0]

        tangent_a = np.cross(normal, world_up)
        if np.linalg.norm(tangent_a) < 1e-5:
            tangent_a = np.cross(normal, head.forward)
        tangent_a /= np.linalg.norm(tangent_a) + 1e-12
        tangent_b = np.cross(normal, tangent_a)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        flow = np.cos(angle) * tangent_a + np.sin(angle) * tangent_b
        cross_flow = np.cross(normal, flow)
        cross_flow /= np.linalg.norm(cross_flow) + 1e-12

        tt = np.linspace(-1.0, 1.0, 7)
        length = spec.length * head.scale * rng.uniform(0.72, 1.28)
        arch = (1.0 - tt**2) * spec.lift * head.scale
        kink = np.sin(np.pi * tt) * spec.wave * 0.035 * head.scale
        P = center + flow * (0.5 * length * tt)[:, None]
        P += normal * arch[:, None] + cross_flow * kink[:, None]
        N = np.broadcast_to(normal, P.shape).copy()

        size = rng.uniform(0.78, 1.22) * (0.48 + 0.52 * np.sqrt(close))
        shape = 0.72 + 0.28 * np.cos(0.5 * np.pi * tt) ** 2
        W = spec.width * head.scale * size * shape
        H = spec.thickness * head.scale * size * shape
        parts.append(_swept_lens(
            P, N, W, H, n_seg=spec.profile_segments,
            uv_phase=(out_i * 0.193) % 0.84,
        ))

    out = trimesh.util.concatenate(parts)
    out.vertex_attributes["chart_uv"] = np.vstack([p.vertex_attributes["chart_uv"] for p in parts])
    return out


def generate_clump_field(
    head: Head,
    spec: ClumpFieldSpec,
    hairline: HairlineSpec | None = None,
) -> trimesh.Trimesh:
    """Generate and concatenate a visible field of separate hair clumps."""

    hairline = hairline or HairlineSpec()
    if spec.mode == "coily":
        return _coily_field(head, spec, hairline)
    generator = _short_clump if spec.mode == "short" else _long_clump
    parts = [generator(head, spec, hairline, float(u), i) for i, u in enumerate(_end_us(spec))]
    out = trimesh.util.concatenate(parts)
    out.vertex_attributes["chart_uv"] = np.vstack([p.vertex_attributes["chart_uv"] for p in parts])
    return out
