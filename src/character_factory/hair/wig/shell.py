"""Shell primitives: the 'cap' (scalp-hugging hair mass).

The cap is a radially inflated patch of the scalp chart bounded by a hairline
curve, closed back down to the scalp with a rim skirt. All lengths are in
multiples of head.scale (mean cranium radius) so styles are head-agnostic.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from .head import Head


@dataclass
class HairlineSpec:
    """Hairline as v(u): where hair meets skin, in chart coords.

    Anchored on the anatomy the hair-restoration literature parameterizes
    (frontal area -> frontotemporal recess -> temporal point -> infratemple /
    sideburn -> post-auricular -> nape). The defining feature is the KNEE at
    the temporal point (|u| ~ 0.44, just in front of the ear) where the line
    turns and plunges to the sideburn. Interpolating straight from temple to
    nape instead — as v1 did — draws a hard diagonal strap across the side of
    the head, which was the single strongest "swim cap" tell.
    """

    front: float = 0.77       # trichion, u=0
    mid: float = 0.85         # |u|=0.20, frontal area
    recess: float = 1.00      # |u|=0.33, frontotemporal recess
    knee: float = 1.29        # |u|=0.44, temporal point — the turn downward
    sideburn: float = 1.40    # |u|=0.52, infratemple, in front of the ear
    post_ear: float = 1.30    # |u|=0.66, behind the ear
    nape: float = 1.40        # |u|=1, back of the neck
    # a hairline is never a clean curve: band-limited roughness in chart v
    jitter: float = 0.022
    jitter_freq: float = 15.0
    seed: int = 0

    _XS = (-0.33, -0.20, 0.0, 0.20, 0.33, 0.44, 0.52, 0.66, 1.0)

    def v_of_u(self, u: np.ndarray, rough: bool = True) -> np.ndarray:
        au = np.abs(np.asarray(u, dtype=np.float64))
        # monotone cubic through the anchors, reflected across u=0 so the
        # forehead is a rounded arc, not a linear-interp widow's-peak wedge
        from scipy.interpolate import PchipInterpolator
        ys = np.array([self.recess, self.mid, self.front, self.mid, self.recess,
                       self.knee, self.sideburn, self.post_ear, self.nape])
        v = PchipInterpolator(np.array(self._XS), ys)(au)
        if rough and self.jitter > 0:
            v = v + self.jitter * _hairline_noise(u, self.jitter_freq, self.seed)
        return v


def _hairline_noise(u, freq: float, seed: int) -> np.ndarray:
    """Periodic band-limited roughness in [-1, 1] along the hairline."""
    rng = np.random.default_rng(seed + 977)
    u = np.asarray(u, dtype=np.float64)
    out = np.zeros_like(u)
    for k, amp in ((1.0, 1.0), (2.3, 0.55), (4.7, 0.3)):
        ph = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(np.pi * u * np.round(freq * k) + ph)
    return out / 1.85


@dataclass
class CapSpec:
    hairline: HairlineSpec = None
    volume: float = 0.16  # radial inflation at the crown, in head-scale units
    volume_back: float = None  # optional different volume toward the back
    edge_frac: float = 0.16  # fraction of rows over which the edge closes down
    taper: float = 0.25  # gentle thinning crown -> edge (0 = none; negative = bell)
    lift: float = 0.0  # extra crown-weighted volume (gravity-defying body at the top)
    lump_amp: float = 0.0  # curl-lump noise amplitude (head-scale units)
    lump_freq: float = 9.0  # lumps around the head
    # --- lock structure (the difference between hair and a helmet) ---
    clump: float = 0.45  # 0 = smooth shell; 1 = grooves cut the full volume
    clump_count: int = 16  # locks around the head
    clump_sharp: float = 2.4  # >1 = wide locks, narrow grooves
    clump_warp: float = 0.30  # irregularity of lock widths
    clump_start: float = 0.30  # fraction of the way to the hairline where
    #                            grooves open up (hair leaves the crown as one
    #                            mass and separates toward the tips)
    wisps: int = 26  # loose strands along the hairline (0 = none)
    wisp_len: float = 0.20  # head-scale units
    tip_jag: float = 0.10  # per-lock curtain length variation (0-1)
    part_u: float = None  # chart-u of a part line (None = no part)
    part_depth: float = 0.35  # fraction of local volume removed at the part
    part_width: float = 0.06  # gaussian width of the part valley in u
    part_start_v: float = 0.0  # chart-v where the visible part begins toward the hairline
    part_open: bool = False  # omit a narrow face strip so real scalp shows
    part_open_width: float = 0.014
    # --- curtain (hanging hair below the hairline; 0 = cap only) ---
    curtain_len: float = 0.0  # length at the back, in head-scale units
    curtain_len_front: float = None  # length at the face gap edge (None = same)
    curtain_gap: float = 0.30  # |u| below which no curtain hangs (face opening)
    curtain_flare: float = 0.30  # outward bulge of the hanging curve
    curtain_in: float = 0.35  # how much the bottom tucks toward the neck axis
    # --- length gradient (fades/tapers) ---
    fade_start_v: float = None  # v where volume starts ramping down (None = off)
    fade_end_v: float = 1.05  # v where the ramp reaches fade_floor
    fade_floor: float = 0.10  # remaining volume fraction below the fade
    # --- balding ---
    crown_gap_v: float = None  # if set, cap is annular: bare scalp above this v
    # --- asymmetry ---
    side_bias: float = 0.0  # +right/-left volume & curtain-length skew
    seed: int = 0

    def __post_init__(self):
        if self.hairline is None:
            self.hairline = HairlineSpec()
        if self.volume_back is None:
            self.volume_back = self.volume
        if self.curtain_len_front is None:
            self.curtain_len_front = self.curtain_len


def _lock_field(u, count: int, sharp: float, warp: float, seed: int,
                swirl: np.ndarray | float = 0.0):
    """Lock/clump field on the chart: 0 at a lock centre, 1 in a groove.

    Real hair leaves the scalp in clumps separated by partings, and the
    negative space between clumps is what stops a shell reading as a helmet
    (every stylized-hair reference says the same thing). `swirl` rotates the
    lock phase with v so locks fan out from the crown instead of running as
    perfect meridians. Periodic in u by construction (integer harmonics)."""
    rng = np.random.default_rng(seed + 31)
    p = (np.asarray(u) * 0.5 + 1.0) * count + rng.uniform(0, 1) + swirl * count * 0.5
    p = p + warp * np.sin(2 * np.pi * (3.0 * p / count + rng.uniform(0, 1)))
    return ((1 - np.cos(2 * np.pi * p)) / 2) ** sharp


def _lump_noise(u, v, freq, seed):
    """Cheap band-limited bump field on the chart (periodic in u)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(u)
    for k in range(1, 4):
        f = freq * k / 2.0
        ph1, ph2, ph3 = rng.uniform(0, 2 * np.pi, 3)
        amp = 1.0 / k
        out += amp * (
            np.sin(np.pi * u * np.round(f) + ph1)
            * np.cos(2 * np.pi * v * f / 2.4 + ph2)
        )
        out += 0.6 * amp * np.sin(np.pi * u * np.round(f * 1.7) + 2 * np.pi * v * f + ph3)
    return out / 2.2


def generate_cap(head: Head, spec: CapSpec, n_u: int = 152, n_v: int = 56) -> trimesh.Trimesh:
    """Build the cap mesh on a head. Returns a trimesh with per-vertex chart
    coords stashed in vertex_attributes['chart_uv'] for texturing."""
    s = head.scale

    u = np.linspace(-1, 1, n_u, endpoint=False)
    hair_v = spec.hairline.v_of_u(u)
    # rows: v from the crown (or the balding gap edge) to the hairline
    t = np.linspace(0, 1, n_v)[:, None]  # 0 at inner edge, 1 at hairline
    annular = spec.crown_gap_v is not None
    v0 = spec.crown_gap_v if annular else 0.0
    V_grid = v0 + t * np.maximum(hair_v[None, :] - v0, 0.05)
    U_grid = np.broadcast_to(u[None, :], V_grid.shape).copy()

    # volume profile: hair keeps its thickness over the skull, then the edge
    # closes down steeply over the last edge_frac rows (a wig-like rim), with
    # optional gentle thinning from crown to edge.
    e = np.clip((1 - t) / spec.edge_frac, 0, 1)
    w = (e * e * (3 - 2 * e)) * (1 - spec.taper * t) * np.ones_like(V_grid)
    if annular:  # close down at the inner (bald-crown) edge too
        ei = np.clip(t / (spec.edge_frac * 0.7), 0, 1)
        w = w * (ei * ei * (3 - 2 * ei))
    # front/back volume blend over chart u
    vol = spec.volume + (spec.volume_back - spec.volume) * (np.abs(U_grid) ** 1.5)
    off = vol * w
    if spec.lift > 0:
        off = off * (1 + spec.lift * (1 - t) ** 1.5)   # crown-weighted body
    if spec.lump_amp > 0:
        off = off + spec.lump_amp * w * _lump_noise(U_grid, V_grid, spec.lump_freq, spec.seed)
    if spec.clump > 0:
        # grooves open up away from the crown (Yuksel's clump falls off as w^b)
        depth = spec.clump * np.clip((t - spec.clump_start) / max(1 - spec.clump_start, 1e-6), 0, 1) ** 1.1
        off = off * (1 - depth * _lock_field(U_grid, spec.clump_count, spec.clump_sharp,
                                             spec.clump_warp, spec.seed,
                                             swirl=0.10 * t))
    if spec.part_u is not None:
        valley = np.exp(-0.5 * ((U_grid - spec.part_u) / spec.part_width) ** 2)
        extent = np.clip((V_grid - spec.part_start_v) / 0.08, 0.0, 1.0)
        extent = extent * extent * (3.0 - 2.0 * extent)
        valley *= extent
        off = off * (1 - spec.part_depth * valley)
    if spec.fade_start_v is not None:
        # length-gradient: full volume above fade_start_v, smoothstep down to
        # fade_floor at fade_end_v (fades, tapers, slick sides)
        m = np.clip((V_grid - spec.fade_start_v) / max(spec.fade_end_v - spec.fade_start_v, 1e-6), 0, 1)
        m = m * m * (3 - 2 * m)
        off = off * (1 - (1 - spec.fade_floor) * m)
    if spec.side_bias != 0.0:
        off = off * np.clip(1 + spec.side_bias * np.sin(U_grid * np.pi / 2), 0.15, 2.0)
    # Every (u, v=0) is the same physical point, so any u-varying modulation
    # (part valley, side bias, lumps) would tear the crown apart. Fade them
    # into their u-mean as the rows approach the pole.
    pw = np.clip(1.0 - t / 0.12, 0, 1) ** 2
    off = off * (1 - pw) + off.mean(axis=1, keepdims=True) * pw
    off = np.maximum(off, 0.0) * s

    P_out = head.scalp_point(U_grid, V_grid, radial_offset=off)

    curtain = spec.curtain_len > 0
    if not curtain:
        # rim: sink slightly below the scalp so the edge is guaranteed hidden
        P_out[-1] = head.scalp_point(U_grid[-1], V_grid[-1], radial_offset=np.full(n_u, -0.03 * s))
    if annular:
        # inner rim likewise sinks below the bare crown
        P_out[0] = head.scalp_point(U_grid[0], V_grid[0], radial_offset=np.full(n_u, -0.03 * s))

    # duplicate the seam column so UVs stay continuous across u = +-1
    # (faces never straddle the wrap; the duplicate carries u = u0 + 2)
    P_out = np.concatenate([P_out, P_out[:, :1]], axis=1)
    U_grid = np.concatenate([U_grid, U_grid[:, :1] + 2.0], axis=1)
    V_grid = np.concatenate([V_grid, V_grid[:, :1]], axis=1)
    hair_v = np.append(hair_v, hair_v[0])
    u = np.append(u, u[0] + 2.0)
    n_u = n_u + 1

    # ---------------------------------------------------------------- curtain
    UV_extra = None
    P_curt = None
    cmask = None
    n_c = 0
    if curtain:
        n_c = 22
        au = np.abs(u)
        cmask = au >= spec.curtain_gap
        frac = np.clip((au - spec.curtain_gap) / max(1e-6, 1 - spec.curtain_gap), 0, 1)
        L = (spec.curtain_len_front + (spec.curtain_len - spec.curtain_len_front) * np.sqrt(frac)) * s
        lock_u = _lock_field(u, spec.clump_count, spec.clump_sharp, spec.clump_warp,
                             spec.seed, swirl=0.10) if spec.clump > 0 else None
        if spec.tip_jag > 0 and lock_u is not None:
            # locks end at different lengths — a curtain cut to one clean curve
            # is the second-strongest helmet tell after the hairline
            L = L * (1 - spec.tip_jag * lock_u)
        if spec.side_bias != 0.0:
            L = L * np.clip(1 + spec.side_bias * np.sin(np.clip(u, -1, 1) * np.pi / 2), 0.25, 1.9)
        P0 = P_out[-1]  # rim ring, hair volume still on
        cx, cz = head.center[0], head.center[2]
        r_vec = P0 - np.stack([np.full(n_u, cx), P0[:, 1], np.full(n_u, cz)], 1)
        r_len = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_hat = r_vec / (r_len + 1e-12)
        down = np.array([0.0, -1.0, 0.0])
        ctrl = P0 + r_hat * (spec.curtain_flare * s) + down * (0.45 * L[:, None])
        bot = (
            np.stack([np.full(n_u, cx), P0[:, 1], np.full(n_u, cz)], 1)
            + r_hat * (r_len * (1 - 0.4 * spec.curtain_in))
            + down * L[:, None]
        )
        t = np.linspace(0, 1, n_c + 1)[1:][:, None, None]
        P_curt = (1 - t) ** 2 * P0 + 2 * t * (1 - t) * ctrl + t**2 * bot  # (n_c, n_u, 3)
        if lock_u is not None:
            # carve the locks BEFORE the drape: grooved columns pushed inward
            # afterwards end up inside the torso and read as holes in the hair
            groove = (spec.clump * 0.22 * s) * lock_u[None, :] * np.clip(t[..., 0] * 2.0, 0, 1)
            P_curt = P_curt - r_hat[None] * groove[..., None]
        # drape over the body: push points out to the body clearance radius
        flat = P_curt.reshape(-1, 3)
        # 0.05 left the sheet grazing the shoulders — the 64x64 clearance
        # field under-reads between bins and the curtain tore into the back
        r_min = head.body_clearance(flat) + 0.14 * s
        dxz = flat[:, [0, 2]] - np.array([cx, cz])
        r_now = np.linalg.norm(dxz, axis=1)
        push = np.maximum(r_min - r_now, 0.0)
        # ramp the clearance push in fast: at 2.5 the first ~9 rows stayed
        # unpushed and sank into the neck, opening a hole at the nape
        blend = np.repeat(np.clip(t[:, 0, 0] * 9.0, 0, 1), n_u)
        scale_f = 1 + (push * blend) / (r_now + 1e-9)
        flat[:, 0] = cx + dxz[:, 0] * scale_f
        flat[:, 2] = cz + dxz[:, 1] * scale_f
        P_curt = flat.reshape(n_c, n_u, 3)
        if spec.lump_amp > 0:
            # gentle waviness continuing down the curtain
            tv = t[..., 0]
            wav = _lump_noise(
                np.broadcast_to(u, (n_c, n_u)),
                hair_v[None, :] + tv * 0.8,
                spec.lump_freq,
                spec.seed + 1,
            )
            P_curt = P_curt + r_hat[None] * (spec.lump_amp * s * wav * (0.3 + 0.7 * tv))[..., None]
        v_curt = hair_v[None, :] + t[..., 0] * (L[None, :] / (0.5 * np.pi * s))
        UV_extra = np.stack([np.broadcast_to(u, (n_c, n_u)), v_curt], -1)

    # ---------------------------------------------------------------- assemble
    faces = []
    if annular:
        # no pole: the annulus is open at the (sunk) inner rim
        verts = [P_out.reshape(-1, 3)]
        idx = np.arange(n_u * n_v).reshape(n_v, n_u)
        uv = [np.stack([U_grid, V_grid], -1).reshape(-1, 2)]
        n_extra = 0
    else:
        # crown pole vertex (top center, offset by crown volume)
        pole = head.scalp_point(np.array([0.0]), np.array([0.0]),
                                radial_offset=np.array([float(off[0].mean())]))[0]
        verts = [pole[None], P_out.reshape(-1, 3)]
        idx = np.arange(n_u * n_v).reshape(n_v, n_u) + 1
        uv = [np.array([[0.0, 0.0]]), np.stack([U_grid, V_grid], -1).reshape(-1, 2)]
        n_extra = 1
        for j in range(n_u - 1):  # pole fan (seam column duplicated; no wrap)
            if spec.part_open and spec.part_u is not None and spec.part_start_v <= 0.0:
                um = 0.5 * (u[j] + u[j + 1])
                du = (um - spec.part_u + 1.0) % 2.0 - 1.0
                if abs(du) < spec.part_open_width:
                    continue
            faces.append([0, idx[0, j], idx[0, j + 1]])
    for i in range(n_v - 1):  # cap quads
        for j in range(n_u - 1):
            if spec.part_open and spec.part_u is not None:
                um = 0.5 * (u[j] + u[j + 1])
                du = (um - spec.part_u + 1.0) % 2.0 - 1.0
                vm = 0.25 * (
                    V_grid[i, j] + V_grid[i, j + 1]
                    + V_grid[i + 1, j] + V_grid[i + 1, j + 1]
                )
                if abs(du) < spec.part_open_width and vm >= spec.part_start_v:
                    continue
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]
            faces.append([a, b, c])
            faces.append([a, c, d])

    if curtain:
        base = n_extra + n_u * n_v
        cidx = np.arange(n_u * n_c).reshape(n_c, n_u) + base
        verts.append(P_curt.reshape(-1, 3))
        uv.append(UV_extra.reshape(-1, 2))
        rows = np.vstack([idx[-1][None], cidx])  # rim row + curtain rows
        for i in range(n_c):
            for j in range(n_u - 1):
                if not (cmask[j] and cmask[j + 1]):
                    continue
                a, b, c, d = rows[i, j], rows[i, j + 1], rows[i + 1, j + 1], rows[i + 1, j]
                faces.append([a, b, c])
                faces.append([a, c, d])

    mesh = trimesh.Trimesh(vertices=np.vstack(verts), faces=np.array(faces), process=False)
    mesh.vertex_attributes["chart_uv"] = np.vstack(uv)
    return mesh


def generate_wisps(head: Head, spec: CapSpec, n: int | None = None) -> trimesh.Trimesh | None:
    """Loose strands escaping the hairline.

    Every real-time-hair reference lands on the same finishing touch: sparse
    thin strands breaking the outline ("flyaways layer... breaks the
    silhouette nicely"), and short transitional wisps at the hairline to
    soften the skin-to-hair edge. A handful of tapered tubes is the cheapest
    realism in the whole system.
    """
    from .primitives import _tube

    n = spec.wisps if n is None else n
    if n <= 0:
        return None
    s = head.scale
    rng = np.random.default_rng(spec.seed + 4211)
    # bias roots toward the sides and nape; the frontal hairline gets a few
    u = np.sort(rng.uniform(-1, 1, n))
    v_h = spec.hairline.v_of_u(u)
    parts = []
    for i in range(n):
        L = spec.wisp_len * s * rng.uniform(0.45, 1.5)
        v0 = v_h[i] - rng.uniform(0.02, 0.10)
        root = head.scalp_point(np.array([u[i]]), np.array([v0]),
                                radial_offset=np.array([spec.volume * 0.5 * s]))[0]
        out = head.radial_dir_world(np.array([u[i]]), np.array([v0]))[0]
        down = np.array([0.0, -1.0, 0.0])
        # curl away from the scalp, then fall
        tip_dir = out * rng.uniform(0.25, 0.9) + down * rng.uniform(0.4, 1.0)
        tip_dir /= np.linalg.norm(tip_dir)
        t = np.linspace(0, 1, 6)[:, None]
        spine = ((1 - t) ** 2 * root
                 + 2 * t * (1 - t) * (root + out * 0.35 * L)
                 + t ** 2 * (root + tip_dir * L))
        r = np.full(6, 0.016 * s) * (1 - 0.85 * np.linspace(0, 1, 6) ** 1.5)
        parts.append(_tube(spine, r, n_seg=4, uv_u0=float(rng.uniform(0, 0.8)),
                           ref=head.forward))
    out_m = trimesh.util.concatenate(parts)
    out_m.vertex_attributes["chart_uv"] = np.vstack(
        [p.vertex_attributes["chart_uv"] for p in parts])
    return out_m
