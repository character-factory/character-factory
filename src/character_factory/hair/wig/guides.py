"""Deterministic quasi-static guide draping for opaque hair meshes.

The solver acts only on a small number of style guides.  Those guides are
subsequently swept into closed polygonal clumps by :mod:`character_factory.hair.wig.clumps`; no
curves, particles, alpha cards, or simulation state appear in the exported
asset.

This is deliberately a modest position-based solver rather than a claim at
full strand dynamics.  Stretch constraints preserve authored length, a rest
curvature term preserves the haircut/waves, gravity settles the free section,
and nearest-surface constraints conform it to the actual body mesh.  A weak
front/back routing field supplies the styling decision that gravity alone
cannot make at the shoulders.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from .head import Head


@dataclass
class GuideDrapeSpec:
    """Controls the creation-time guide relaxation.

    Distances are in ``Head.scale`` units.  ``routing`` is a styling choice,
    not a force simulation mode: loose hair must be told whether locks pass in
    front of or behind the shoulders.
    """

    enabled: bool = False
    routing: str = "split"
    steps: int = 22
    iterations: int = 7
    gravity: float = 1.0
    stiffness: float = 0.55
    collision_margin: float = 0.10
    route_strength: float = 0.055
    route_boundary: float = 0.50
    damping: float = 0.86

    def __post_init__(self):
        if self.routing not in {"natural", "split", "mostly_back", "front", "back"}:
            raise ValueError(f"unsupported shoulder routing: {self.routing}")
        if self.steps < 1 or self.iterations < 1:
            raise ValueError("guide drape steps and iterations must be positive")
        if not 0.0 <= self.stiffness <= 1.0:
            raise ValueError("guide stiffness must be in [0, 1]")
        if not 0.0 <= self.damping <= 1.0:
            raise ValueError("guide damping must be in [0, 1]")
        if self.collision_margin < 0.0:
            raise ValueError("collision margin must be non-negative")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def route_side(spec: GuideDrapeSpec, u: float) -> int:
    """Return +1 for front, -1 for back, and 0 for unstyled routing."""

    if spec.routing == "natural":
        return 0
    if spec.routing == "front":
        return 1
    if spec.routing == "back":
        return -1
    boundary = spec.route_boundary
    if spec.routing == "mostly_back":
        boundary = min(boundary, 0.43)
    return 1 if abs(u) <= boundary else -1


def route_direction(head: Head, u: float, spec: GuideDrapeSpec) -> np.ndarray | None:
    """A broad front/back lane for the free end of one guide.

    The lateral component keeps left and right locks distinct.  Consolidating
    the remainder into front/back quadrants is what prevents side roots from
    expanding into a continuous shoulder mantle.
    """

    lane = route_side(spec, u)
    if lane == 0:
        return None
    up = np.array([0.0, 1.0, 0.0])
    left = np.cross(up, head.forward)
    lateral = 1.0 if u >= 0.0 else -1.0
    a = abs(u)
    if lane > 0:
        # Face-framing lanes fan from near the cheek toward the clavicle.
        q = np.clip((a - 0.28) / 0.30, 0.0, 1.0)
        lateral_weight = 0.20 + 0.28 * q
    else:
        # Side-back roots stay broad over the shoulder blade while true nape
        # roots converge toward the middle.  A constant lateral term made all
        # rear guides overlap into one narrow V.
        q = np.clip((1.0 - a) / max(1.0 - spec.route_boundary, 1e-6), 0.0, 1.0)
        lateral_weight = 0.08 + 0.42 * q
    forward_weight = np.sqrt(max(1.0 - lateral_weight * lateral_weight, 0.05))
    return _unit((
        lateral_weight * lateral * left + forward_weight * lane * head.forward
    )[None])[0]


def _query(head: Head) -> trimesh.proximity.ProximityQuery:
    query = getattr(head, "_body_query_cache", None)
    if query is None:
        query = trimesh.proximity.ProximityQuery(head.mesh)
        head._body_query_cache = query
    return query


def project_outside_body(head: Head, points: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Project points to the exterior side of the closest body triangles.

    Returns the corrected points and a mask of contacts.  The loaded body is
    expected to have outward, consistent winding (the normal make-wig input).
    For imperfect source meshes, a radial check flips obviously inward-facing
    side normals before projection.
    """

    P = np.asarray(points, dtype=np.float64).copy()
    if len(P) == 0:
        return P, np.zeros(0, dtype=bool)
    closest, _distance, triangle_id = _query(head).on_surface(P)
    normals = head.mesh.face_normals[triangle_id].copy()

    # Repair an obviously reversed side normal without disturbing upward
    # shoulder normals, for which the horizontal dot is intentionally small.
    radial = closest - np.column_stack([
        np.full(len(P), head.center[0]),
        closest[:, 1],
        np.full(len(P), head.center[2]),
    ])
    radial_dot = np.einsum("ij,ij->i", normals, radial)
    flip = radial_dot < -0.15 * np.linalg.norm(radial, axis=1)
    normals[flip] *= -1.0

    clearance = np.einsum("ij,ij->i", P - closest, normals)
    contact = clearance < margin * head.scale
    P[contact] += normals[contact] * (
        margin * head.scale - clearance[contact]
    )[:, None]
    return P, contact


def _solve_stretch(
    x: np.ndarray,
    rest_length: np.ndarray,
    lambdas: np.ndarray,
    compliance: float,
    dt: float,
) -> None:
    """One colored XPBD pass over segment-length constraints."""

    for parity in (0, 1):
        for i in range(parity, len(rest_length), 2):
            j = i + 1
            delta = x[j] - x[i]
            distance = float(np.linalg.norm(delta))
            if distance < 1e-10:
                continue
            direction = delta / distance
            c = distance - rest_length[i]
            wi = 0.0 if i == 0 else 1.0
            wj = 1.0
            alpha = compliance / (dt * dt)
            dlambda = (-c - alpha * lambdas[i]) / (wi + wj + alpha)
            if wi:
                x[i] -= wi * dlambda * direction
            x[j] += wj * dlambda * direction
            lambdas[i] += dlambda


def _solve_rest_curvature(x: np.ndarray, rest_laplacian: np.ndarray, amount: float) -> None:
    """Shape-matching bend pass that preserves authored waves and flare."""

    if amount <= 0.0 or len(x) < 3:
        return
    # Alternating centers avoid applying overlapping triplets simultaneously.
    for parity in (0, 1):
        for i in range(1 + parity, len(x) - 1, 2):
            error = x[i - 1] - 2.0 * x[i] + x[i + 1] - rest_laplacian[i - 1]
            k = amount / 6.0
            if i - 1 != 0:
                x[i - 1] -= k * error
            x[i] += 2.0 * k * error
            x[i + 1] -= k * error


def drape_guide(head: Head, curve: np.ndarray, u: float, spec: GuideDrapeSpec) -> np.ndarray:
    """Relax a rooted guide against the actual body, deterministically.

    ``curve[0]`` is the pinned hairline/rim point.  Simulation is performed in
    head-scale units so the same intent transfers across centimeters, meters,
    and differently sized heads.
    """

    source = np.asarray(curve, dtype=np.float64)
    if not spec.enabled or len(source) < 3:
        return source.copy()

    scale = head.scale
    x = source / scale
    previous = x.copy()
    root = x[0].copy()
    rest_length = np.linalg.norm(np.diff(x, axis=0), axis=1)
    rest_laplacian = x[:-2] - 2.0 * x[1:-1] + x[2:]
    lane = route_side(spec, u)

    # The compliance is intentionally tiny: a lock can bend but should not
    # gain length as solver iterations or avatar scale change.
    compliance = 2.0e-6
    dt = 1.0 / 24.0
    free_t = np.linspace(0.0, 1.0, len(x))
    route_envelope = np.clip((free_t - 0.12) / 0.55, 0.0, 1.0) ** 1.5

    for _step in range(spec.steps):
        old = x.copy()
        velocity = (x - previous) * spec.damping
        previous = old
        x[1:] += velocity[1:]
        x[1:, 1] -= 11.0 * spec.gravity * dt * dt

        # A weak lane field makes the haircut decision; collision and gravity
        # determine the final surface-conforming path.
        if lane:
            local_forward = (x * scale - head.center) @ head.forward / scale
            target = lane * (0.94 if lane > 0 else 1.27)
            shift = (target - local_forward) * spec.route_strength * route_envelope
            x[1:] += head.forward * shift[1:, None]

        lambdas = np.zeros(len(rest_length), dtype=np.float64)
        for _iteration in range(spec.iterations):
            _solve_stretch(x, rest_length, lambdas, compliance, dt)
            # High semantic stiffness retains more of the rest curve.  Even a
            # "soft" lock keeps enough curvature to preserve authored waves.
            bend = 0.025 + 0.115 * spec.stiffness
            _solve_rest_curvature(x, rest_laplacian, bend)
            x[0] = root
            corrected, contact = project_outside_body(
                head, x[1:] * scale, spec.collision_margin
            )
            x[1:] = corrected / scale
            if np.any(contact):
                # Simple contact damping approximates friction and suppresses
                # perpetual shoulder sliding during this creation-time settle.
                ids = np.flatnonzero(contact) + 1
                previous[ids] = 0.35 * previous[ids] + 0.65 * x[ids]
            x[0] = root

    return x * scale


def hanging_normals(head: Head, points: np.ndarray) -> np.ndarray:
    """Stable outward frame field for body-conforming swept guide meshes."""

    P = np.asarray(points, dtype=np.float64)
    axis = np.column_stack([
        np.full(len(P), head.center[0]),
        P[:, 1],
        np.full(len(P), head.center[2]),
    ])
    radial = P - axis
    small = np.linalg.norm(radial, axis=1) < 1e-8
    radial[small] = head.forward
    return _unit(radial)
