"""Head ingestion and canonical scalp chart.

The chart is the head-agnostic foundation: every style is defined in chart
coordinates (u, v) and evaluated against a specific head's radial scalp field.

Chart convention (right-handed, y-up):
  u in [-1, 1]  : azimuth around the vertical axis through the cranium center.
                  0 = front (nose direction), +/-1 = back, +u = wearer's left.
  v in [0, 1+]  : polar angle from straight up, scaled so v=1 is the ellipsoid
                  equator (roughly ear/eyebrow level). v can exceed 1 slightly
                  (nape, sideburns).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import trimesh


def _load_mesh(path: str) -> tuple[trimesh.Trimesh, np.ndarray | None]:
    """Returns (body mesh, eye midpoint or None)."""
    if str(path).lower().endswith((".glb", ".gltf")):
        from .gltf_load import load_glb_meshes

        meshes = load_glb_meshes(path)
        # exclude hair meshes (a wig-wearing head is our bald canvas);
        # then the body/head is the largest remaining mesh
        candidates = {k: v for k, v in meshes.items() if "wig" not in k.lower() and "hair" not in k.lower()}
        body = max(candidates.values(), key=lambda m: len(m.vertices))
        eyes = [v for k, v in meshes.items() if "eye" in k.lower() and "brow" not in k.lower()]
        eye_mid = None
        if eyes:
            eye_mid = np.mean([e.vertices.mean(0) for e in eyes], axis=0)
        return body, eye_mid
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):  # pragma: no cover
        m = m.to_mesh()
    return m, None


@dataclass
class Head:
    """A head (or full body) mesh with a fitted cranium chart."""

    mesh: trimesh.Trimesh
    center: np.ndarray = field(default=None)  # cranium ellipsoid center
    radii: np.ndarray = field(default=None)  # ellipsoid semi-axes (x, y, z)
    forward: np.ndarray = field(default=None)  # unit horizontal vector, nose direction
    head_height: float = field(default=None)  # chin-to-crown estimate
    # radial scalp field, indexed [iv, iu] over the chart grid
    grid_u: np.ndarray = field(default=None)
    grid_v: np.ndarray = field(default=None)
    grid_r: np.ndarray = field(default=None)  # radius in normalized-sphere space

    N_U, N_V = 128, 64
    V_MAX = 1.55  # chart lower limit (below ears / into nape)
    R_MAX = 1.55  # loose scalp-field backstop, normalized-sphere units
    HORIZ_MARGIN = 1.12  # reject beyond this x skull half-width (shoulders)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_file(
        cls,
        path: str,
        forward: np.ndarray | None = None,
        head_height: float | None = None,
        eye_level: float | None = None,
    ) -> "Head":
        """Load a head (or full-body) mesh and fit the scalp chart.

        Auto-estimation prefers anatomical anchors: eyeball sub-meshes when the
        file has them (eye level sits ~45% down a human head and pins the
        forward direction); otherwise slice-profile heuristics. For unusual
        heads pass `forward` / `head_height` / `eye_level` (world y) directly.
        """
        mesh, eye_mid = _load_mesh(path)
        head = cls(mesh=mesh)
        y_hi = mesh.vertices[:, 1].max()
        if eye_level is None and eye_mid is not None:
            eye_level = float(eye_mid[1])
        if head_height is not None:
            head.head_height = float(head_height)
        elif eye_level is not None:
            head.head_height = (y_hi - eye_level) / 0.45
        else:
            head._estimate_head_height()
        head._head_mask = mesh.vertices[:, 1] > y_hi - head.head_height

        if forward is not None:
            f = np.asarray(forward, dtype=np.float64)
            head.forward = f / np.linalg.norm(f)
        elif eye_mid is not None:
            band = mesh.vertices[head._head_mask]
            band = band[band[:, 1] > (eye_level if eye_level is not None else eye_mid[1])]
            c = np.array([np.median(band[:, 0]), 0.0, np.median(band[:, 2])])
            f = np.array([eye_mid[0] - c[0], 0.0, eye_mid[2] - c[2]])
            head.forward = f / np.linalg.norm(f)
        else:
            head._detect_forward()
        head._fit_cranium()
        head._build_radial_field()
        return head

    def _estimate_head_height(self) -> None:
        """Fallback estimate when no eye anchor is available (approximate;
        pass head_height/eye_level explicitly for unusual heads)."""
        V = self.mesh.vertices
        ext_y = V[:, 1].max() - V[:, 1].min()
        y_hi = V[:, 1].max()
        # width profile in horizontal slices, walking down from the crown
        n_sl = 60
        depth = 0.30 * ext_y
        edges = y_hi - depth * np.arange(n_sl + 1) / n_sl
        w = np.full(n_sl, np.nan)
        for i in range(n_sl):
            sl = V[(V[:, 1] <= edges[i]) & (V[:, 1] > edges[i + 1])]
            if len(sl) >= 8:
                wx = np.percentile(sl[:, 0], 98) - np.percentile(sl[:, 0], 2)
                wz = np.percentile(sl[:, 2], 98) - np.percentile(sl[:, 2], 2)
                # lateral (ear-to-ear) width: the narrower horizontal span.
                # The front-back span is polluted by the nose and chin.
                w[i] = min(wx, wz)
        w = np.where(np.isnan(w), np.nanmax(w), w)
        for _ in range(2):  # smooth
            w[1:-1] = 0.5 * w[1:-1] + 0.25 * (w[:-2] + w[2:])
        # anchor on the FIRST width peak walking down from the crown: maximum
        # skull width sits ~42% down a human head. (The neck minimum is
        # unreliable — the jaw/chin masks it.)
        i_peak = n_sl - 3
        for i in range(2, n_sl - 2):
            if w[i + 2] < 0.985 * w[i]:
                i_peak = i
                break
        h = (y_hi - edges[i_peak]) / 0.42
        self.head_height = float(np.clip(h, 0.05 * ext_y, ext_y))

    def _detect_forward(self) -> None:
        """Assumes an axis-aligned head (y up). The forward axis is the longer
        horizontal skull axis (front-back beats ear-to-ear); the sign is where
        the chin/nose band protrudes farther from the skull center."""
        V = self.mesh.vertices[self._head_mask]
        y_hi = V[:, 1].max()
        h = self.head_height
        skull = V[V[:, 1] > y_hi - 0.55 * h]
        span = lambda a: np.percentile(skull[:, a], 98) - np.percentile(skull[:, a], 2)
        axis = 0 if span(0) > span(2) else 2
        c = np.array([np.percentile(skull[:, 0], 50), 0, np.percentile(skull[:, 2], 50)])
        chin = V[(V[:, 1] < y_hi - 0.65 * h) & (V[:, 1] > y_hi - 1.05 * h)]
        d = np.zeros(3)
        d[axis] = 1.0
        pos = np.percentile((chin - c) @ d, 99)
        neg = np.percentile((chin - c) @ -d, 99)
        self.forward = d if pos >= neg else -d

    def _fit_cranium(self, drop_face: bool = True) -> None:
        """Axis-aligned ellipsoid least-squares fit on the cranium band
        (upper part of the head, excluding face/jaw)."""
        V = self.mesh.vertices[self._head_mask]
        y_hi = V[:, 1].max()
        band = V[V[:, 1] > y_hi - 0.55 * self.head_height]
        if drop_face:
            # keep points not strongly forward of the band centroid
            c0 = band.mean(0)
            fwd_amount = (band - c0) @ self.forward
            band = band[fwd_amount < np.percentile(fwd_amount, 80)]
        # Pin the center anatomically: mid-skull horizontally, 45% down the
        # head vertically — this puts the chart equator (v=1) near ear height
        # by construction. Then fit radii only: a x'^2 + b y'^2 + c z'^2 = 1
        # is linear LSQ with a guaranteed positive solution on real data.
        mid = lambda a: 0.5 * (np.percentile(band[:, a], 2) + np.percentile(band[:, a], 98))
        ctr = np.array([mid(0), y_hi - 0.45 * self.head_height, mid(2)])
        from scipy.optimize import nnls

        Q = (band - ctr) ** 2
        coef, _ = nnls(Q, np.ones(len(band)))
        h = self.head_height
        radii = 1.0 / np.sqrt(np.maximum(coef, 1e-12))
        self.center = ctr
        self.radii = np.clip(radii, 0.30 * h, 0.85 * h)
        # widest the skull itself gets from the vertical cranium axis — the
        # reference the radial field uses to reject shoulder geometry
        self._skull_horiz = float(np.percentile(
            np.hypot(band[:, 0] - ctr[0], band[:, 2] - ctr[2]), 99))

    # ------------------------------------------------------ chart <-> space
    def _to_sphere(self, P: np.ndarray) -> np.ndarray:
        """Map points to the normalized (unit-sphere) frame of the ellipsoid."""
        return (P - self.center) / self.radii

    def _chart_of_dirs(self, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) of unit directions in the normalized frame."""
        up = np.array([0.0, 1.0, 0.0])
        f = self.forward
        left = np.cross(up, f)  # +u side
        phi = np.arccos(np.clip(D[:, 1], -1, 1))
        theta = np.arctan2(D @ left, D @ f)
        return theta / np.pi, phi / (np.pi / 2)

    def dirs_of_chart(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Unit directions (normalized frame) for chart coords."""
        up = np.array([0.0, 1.0, 0.0])
        f = self.forward
        left = np.cross(up, f)
        theta = np.asarray(u) * np.pi
        phi = np.asarray(v) * (np.pi / 2)
        sin_p = np.sin(phi)
        D = (
            sin_p[..., None] * (np.cos(theta)[..., None] * f + np.sin(theta)[..., None] * left)
            + np.cos(phi)[..., None] * up
        )
        return D

    def _build_radial_field(self) -> None:
        """Outermost surface radius r(u, v): bin every head vertex into the
        chart grid, keep the max radius per cell, fill holes, smooth."""
        V = self.mesh.vertices[self._head_mask]
        S = self._to_sphere(V)
        r = np.linalg.norm(S, axis=1)
        D = S / r[:, None]
        u, v = self._chart_of_dirs(D)
        keep = v <= self.V_MAX
        # exclude the face sector (front, below the hairline zone): otherwise
        # nose/brow/chin protrusions pollute the scalp field and hairline rim
        # vertices land on the face. Holes are filled from skull neighbors.
        face = (np.abs(u) < 0.42) & (v > 0.68)
        keep &= ~face
        # ... and anything that plainly isn't scalp. On a full body the head
        # mask reaches the shoulders, and max-per-cell then pushed the field
        # out to r ~ 2, hanging a flap of "hair" off the trapezius. A plain
        # radius ceiling can't do this (a forward-set ellipsoid centre puts
        # the occiput at r ~ 1.45 too) — cull on horizontal distance from the
        # cranium axis instead, where skull (~10 cm) and shoulder (~13 cm)
        # separate cleanly on every body measured.
        horiz = np.hypot(V[:, 0] - self.center[0], V[:, 2] - self.center[2])
        keep &= horiz <= self.HORIZ_MARGIN * self._skull_horiz
        u, v, r = u[keep], v[keep], r[keep]

        nu, nv = self.N_U, self.N_V
        iu = np.clip(((u + 1) / 2 * nu).astype(int), 0, nu - 1)
        iv = np.clip((v / self.V_MAX * nv).astype(int), 0, nv - 1)
        R = np.full((nv, nu), -np.inf)
        np.maximum.at(R, (iv, iu), r)
        R[np.isinf(R)] = np.nan

        # fill holes by repeated neighbor averaging (u wraps around)
        for _ in range(60):
            nanmask = np.isnan(R)
            if not nanmask.any():
                break
            Rp = np.pad(R, ((1, 1), (0, 0)), mode="edge")
            neigh = np.stack(
                [
                    np.roll(R, 1, axis=1),
                    np.roll(R, -1, axis=1),
                    Rp[:-2, :],
                    Rp[2:, :],
                ]
            )
            # Cells whose four neighbours are all still holes average to
            # NaN by design and fill on a later pass; numpy's warning for
            # that case is noise here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fill = np.nanmean(neigh, axis=0)
            R[nanmask] = fill[nanmask]
        # light smoothing (keeps hairline features, kills bin noise)
        for _ in range(2):
            Rp = np.pad(R, ((1, 1), (0, 0)), mode="edge")
            R = 0.5 * R + 0.125 * (
                np.roll(R, 1, axis=1) + np.roll(R, -1, axis=1) + Rp[:-2, :] + Rp[2:, :]
            )

        self.grid_u = np.linspace(-1, 1, nu, endpoint=False) + 1.0 / nu
        self.grid_v = (np.arange(nv) + 0.5) / nv * self.V_MAX

        # The chart pole is ONE physical point: every (u, v=0) maps to straight
        # up. If r still varies with u there, the crown row becomes a column of
        # coincident vertices at different heights — a ragged 1.5 cm spike on
        # the top of every cap. Blend r toward its u-mean as v -> 0.
        w = np.clip(1.0 - (self.grid_v - self.grid_v[0]) / 0.22, 0, 1)[:, None] ** 2
        R = R * (1 - w) + R.mean(axis=1, keepdims=True) * w
        self.grid_r = np.clip(R, 0.70, self.R_MAX)   # backstop only

    # ------------------------------------------------------------- sampling
    def sample_radius(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Bilinear sample of the radial field (u wraps)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        nu, nv = self.N_U, self.N_V
        gu = (np.mod(u + 1, 2) / 2) * nu - 0.5
        gv = np.clip(v / self.V_MAX * nv - 0.5, 0, nv - 1 - 1e-6)
        iu0 = np.floor(gu).astype(int) % nu
        iu1 = (iu0 + 1) % nu
        iv0 = np.clip(np.floor(gv).astype(int), 0, nv - 1)
        iv1 = np.clip(iv0 + 1, 0, nv - 1)
        fu = gu - np.floor(gu)
        fv = gv - iv0
        R = self.grid_r
        return (
            R[iv0, iu0] * (1 - fu) * (1 - fv)
            + R[iv0, iu1] * fu * (1 - fv)
            + R[iv1, iu0] * (1 - fu) * fv
            + R[iv1, iu1] * fu * fv
        )

    def scalp_point(self, u: np.ndarray, v: np.ndarray, radial_offset: np.ndarray = 0.0) -> np.ndarray:
        """World-space point on (or radially offset from) the scalp surface.

        radial_offset is in world units, applied along the (world) radial
        direction of the chart."""
        D = self.dirs_of_chart(np.asarray(u), np.asarray(v))
        r = self.sample_radius(u, v)
        P_sphere = D * r[..., None]
        P = P_sphere * self.radii + self.center
        Dw = D * self.radii
        Dw = Dw / np.linalg.norm(Dw, axis=-1, keepdims=True)
        return P + np.asarray(radial_offset)[..., None] * Dw

    def radial_dir_world(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        D = self.dirs_of_chart(np.asarray(u), np.asarray(v))
        Dw = D * self.radii
        return Dw / np.linalg.norm(Dw, axis=-1, keepdims=True)

    # ------------------------------------------------------- body clearance
    def _build_body_field(self) -> None:
        """Cylindrical clearance field around the vertical neck axis: max
        horizontal body radius per (y-slice, azimuth) bin, for draping hair
        over shoulders/back. Arm geometry (far from the axis) is excluded."""
        V = self.mesh.vertices
        y_hi = V[:, 1].max()
        depth = 3.4 * self.head_height
        cx, cz = self.center[0], self.center[2]
        dx, dz = V[:, 0] - cx, V[:, 2] - cz
        r = np.hypot(dx, dz)
        band = (V[:, 1] > y_hi - depth) & (r < 1.05 * self.head_height)
        n_az, n_y = 64, 64
        az = np.arctan2(dx[band], dz[band])  # 0 toward +z
        iy = np.clip(((y_hi - V[band, 1]) / depth * n_y).astype(int), 0, n_y - 1)
        ia = np.clip(((az + np.pi) / (2 * np.pi) * n_az).astype(int), 0, n_az - 1)
        F = np.zeros((n_y, n_az))
        np.maximum.at(F, (iy, ia), r[band])
        # fill empty bins from vertical neighbors, then smooth in azimuth
        for _ in range(n_y):
            empty = F == 0
            if not empty.any():
                break
            Fp = np.pad(F, ((1, 1), (0, 0)), mode="edge")
            fill = np.maximum(Fp[:-2], Fp[2:])
            F[empty] = fill[empty]
        for _ in range(2):
            F = 0.5 * F + 0.25 * (np.roll(F, 1, axis=1) + np.roll(F, -1, axis=1))
        self._body_field = F
        self._body_depth = depth
        self._body_y_hi = y_hi

    def body_clearance(self, P: np.ndarray) -> np.ndarray:
        """Minimum horizontal radius (from the neck axis) the body occupies at
        each point's (y, azimuth). Points are (n,3) world."""
        if not hasattr(self, "_body_field"):
            self._build_body_field()
        F = self._body_field
        n_y, n_az = F.shape
        cx, cz = self.center[0], self.center[2]
        az = np.arctan2(P[:, 0] - cx, P[:, 2] - cz)
        iy = np.clip((self._body_y_hi - P[:, 1]) / self._body_depth * n_y, 0, n_y - 1 - 1e-6)
        ia = ((az + np.pi) / (2 * np.pi) * n_az) % n_az
        iy0 = iy.astype(int)
        ia0 = ia.astype(int) % n_az
        ia1 = (ia0 + 1) % n_az
        fy = iy - iy0
        fa = ia - ia.astype(int)
        iy1 = np.clip(iy0 + 1, 0, n_y - 1)
        return (
            F[iy0, ia0] * (1 - fa) * (1 - fy)
            + F[iy0, ia1] * fa * (1 - fy)
            + F[iy1, ia0] * (1 - fa) * fy
            + F[iy1, ia1] * fa * fy
        )

    @property
    def scale(self) -> float:
        """A length scale: mean horizontal cranium radius (world units).

        Style lengths are expressed in multiples of this, which is what makes
        styles transfer between heads of different absolute size."""
        return float(np.sqrt(self.radii[0] * self.radii[2]))
