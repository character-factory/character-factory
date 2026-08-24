"""Mouth-interior assembly (SPEC.md §4.2, §9 step 4).

The body-rig component's `mouth` metadata block carries every constant this
module consumes: the portal removal set (a fixed topology component of the
rig — identical for every character), the inner-lip paths and entrance-ring
sampling spec, the identity anchors with their canonical-neutral positions,
the socket layer tables, and the expression-morph derived artifact. Nothing
here is discovered at runtime; everything is verified against the metadata
before it is trusted (counts, hashes), because index-based data from the
wrong rig version would assemble a silently different character.

Interior geometry stitched into the skinned body (the socket strip) obeys
the interior-UV contract: original vertex UVs are untouched, new UVs map
into the removed patch's own atlas region (already interior-adjacent skin,
so generated textures shade it with zero texture-side changes), density is
arc-length-even, and nothing overlaps existing charts. Teeth, gums, and
tongue are separate meshes with their own UVs and materials.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AnatomyPiece",
    "ExportStrip",
    "MouthData",
    "SocketBuild",
    "build_socket",
    "export_strip",
    "place_anatomy",
    "socket_uvs",
]


@dataclass
class MouthData:
    """The rig version's mouth constants, loaded and verified."""

    portal_faces: np.ndarray        # (288,) int64 — rig face indices to remove
    lip_upper: np.ndarray           # (U,) int64 — position-vertex path
    lip_lower: np.ndarray           # (L,) int64
    upper_portal: np.ndarray        # (U+2,) int64 — incl. seam duplicates
    samples_per_lip: int            # 32 → a 62-point entrance ring
    anchors_upper: np.ndarray       # (5,) int64
    anchors_lower: np.ndarray       # (6,) int64
    canonical_upper: np.ndarray     # (5, 3) — anchor positions, canonical neutral
    canonical_lower: np.ndarray     # (6, 3)
    cuff_layers: list[tuple]        # (dz, sx, sy, min_x, min_up, min_down)
    cavity_layers: list[tuple]
    morph_names: list[str]          # facs_00 … facs_71
    morph_indices: list[np.ndarray]  # per unit: moved rig-vertex indices
    morph_deltas: list[np.ndarray]   # per unit: (n, 3) float32 cm
    jaw: dict                       # certified-control guidance (manifest)
    semantics: dict                 # provisional-measured naming table
    limitations: dict               # measured animation-limitation table

    @classmethod
    def load(cls, component_dir: str | Path, metadata: dict) -> "MouthData":
        block = metadata.get("mouth")
        if block is None:
            raise ValueError(
                "this body-rig component version declares no mouth data"
            )
        portal = np.asarray(block["portal_faces"], dtype=np.int64)
        if len(portal) != 288 or len(np.unique(portal)) != 288:
            raise ValueError(
                f"mouth portal removal set must be 288 unique faces, got "
                f"{len(portal)}"
            )
        morphs_path = Path(component_dir) / block["expression_morphs"]["artifact"]
        data = morphs_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != block["expression_morphs"]["sha256"]:
            raise ValueError(
                "expression-morph artifact does not match its pinned hash — "
                "refusing to bake unverified morph targets"
            )
        archive = np.load(morphs_path)
        units = int(archive["unit_count"][0])
        indices = [archive[f"indices_{i:02d}"].astype(np.int64) for i in range(units)]
        deltas = [archive[f"deltas_{i:02d}"].astype(np.float32) for i in range(units)]
        names = list(block["expression_morphs"]["names"])
        if len(names) != units:
            raise ValueError("morph name table does not match the unit count")
        return cls(
            portal_faces=portal,
            lip_upper=np.asarray(block["lip_paths"]["upper"], dtype=np.int64),
            lip_lower=np.asarray(block["lip_paths"]["lower"], dtype=np.int64),
            upper_portal=np.asarray(block["lip_paths"]["upper_portal"], dtype=np.int64),
            samples_per_lip=int(block["entrance_ring"]["samples_per_lip"]),
            anchors_upper=np.asarray(block["anchors"]["upper"], dtype=np.int64),
            anchors_lower=np.asarray(block["anchors"]["lower"], dtype=np.int64),
            canonical_upper=np.asarray(block["anchors"]["canonical_upper_cm"], dtype=np.float64),
            canonical_lower=np.asarray(block["anchors"]["canonical_lower_cm"], dtype=np.float64),
            cuff_layers=[tuple(s) for s in block["socket"]["cuff_layers"]],
            cavity_layers=[tuple(s) for s in block["socket"]["cavity_layers"]],
            morph_names=names,
            morph_indices=indices,
            morph_deltas=deltas,
            jaw=dict(block["jaw"]),
            semantics=dict(block["expression_semantics"]),
            limitations=dict(block["animation_limitations"]),
        )

    def morph_dense(self, unit: int, vertex_count: int) -> np.ndarray:
        """One unit's delta as a dense (V, 3) float64 array in cm."""
        dense = np.zeros((vertex_count, 3), dtype=np.float64)
        dense[self.morph_indices[unit]] = self.morph_deltas[unit]
        return dense


# -- entrance ring ------------------------------------------------------------

@dataclass
class _Resampled:
    points: np.ndarray      # (n, 3)
    segment: np.ndarray     # (n,) int — source segment index
    fraction: np.ndarray    # (n,) float — position within the segment


def _resample(points: np.ndarray, n: int) -> _Resampled:
    """Arc-length resampling that keeps the interpolation structure, so
    companion per-vertex attributes (UVs, skin weights) resample through
    exactly the same parameterization as the positions."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, arc[-1], n)
    segment = np.clip(
        np.searchsorted(arc, targets, side="right") - 1, 0, len(seg) - 1
    )
    fraction = (targets - arc[segment]) / np.maximum(seg[segment], 1e-12)
    fraction = np.clip(fraction, 0.0, 1.0)
    out = points[segment] * (1.0 - fraction[:, None]) + points[segment + 1] * fraction[:, None]
    return _Resampled(points=out, segment=segment, fraction=fraction)


@dataclass
class RingParam:
    """The 62-point entrance ring plus the resampling parameterization that
    produced it: upper path left→right, then the lower path's interior
    reversed."""

    points: np.ndarray          # (62, 3)
    upper: _Resampled
    lower: _Resampled

    def interpolate(self, upper_values: np.ndarray, lower_values: np.ndarray) -> np.ndarray:
        """Resample per-path-vertex attribute rows through the ring's own
        parameterization (positions, UVs, weights — anything linear)."""
        def lerp(values, r):
            return (values[r.segment] * (1.0 - r.fraction[:, None])
                    + values[r.segment + 1] * r.fraction[:, None])
        up = lerp(np.asarray(upper_values, dtype=np.float64), self.upper)
        low = lerp(np.asarray(lower_values, dtype=np.float64), self.lower)
        return np.concatenate([up, low[-2:0:-1]], axis=0)


def entrance_ring(vertices: np.ndarray, data: MouthData) -> RingParam:
    n = data.samples_per_lip
    upper = _resample(vertices[data.lip_upper], n)
    lower = _resample(vertices[data.lip_lower], n)
    points = np.concatenate([upper.points, lower.points[-2:0:-1]], axis=0)
    return RingParam(points=points, upper=upper, lower=lower)


# -- socket -------------------------------------------------------------------

@dataclass
class SocketBuild:
    vertices: np.ndarray        # (S, 3) cm — one welded strip, seam ring first
    faces: np.ndarray           # (F, 3) local indices
    ring_size: int              # 62
    layer_count: int            # rings before the cap vertex
    layer_depths: np.ndarray    # (layer_count,) mean 3D distance from the seam


def _circular_smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Circular moving average along the ring."""
    half = window // 2
    stacked = np.stack([np.roll(values, k) for k in range(-half, half + 1)])
    return stacked.mean(axis=0)


def _shaped_layer(entrance, center, dz, sx, sy, min_x, min_up, min_down):
    p = entrance.copy()
    dx = p[:, 0] - center[0]
    dy = p[:, 1] - center[1]
    x_scale = max(sx, min_x / max(float(np.abs(dx).max()), 1e-4))
    p[:, 0] = center[0] + dx * x_scale
    # The seam is not symmetric about its mean; the palate needs more
    # clearance than the floor, so each side scales independently. At a
    # nearly closed aperture the minimum-clearance clamps amplify the
    # ring's vertical offsets by an order of magnitude, which would turn
    # millimeter wiggles of the resting lip curve into centimeter ridges —
    # so the AMPLIFIED component follows a smoothed profile while the raw
    # offsets pass through unscaled (a scale of 1 reproduces the entrance
    # exactly, which keeps the seam ring untouched).
    dy_smooth = _circular_smooth(dy)
    # Scales measured against the smoothed profile, so the minimum
    # roof/floor clearances are actually reached by the smoothed component.
    up_scale = max(sy, min_up / max(float(dy_smooth.max()), 1e-4))
    down_scale = max(sy, min_down / max(float(-dy_smooth.min()), 1e-4))
    scale = np.where(dy >= 0, up_scale, down_scale)
    p[:, 1] = center[1] + dy + (scale - 1.0) * np.where(
        dy_smooth * dy > 0, dy_smooth, dy
    )
    p[:, 2] += dz
    return p


def build_socket(vertices: np.ndarray, data: MouthData) -> tuple[SocketBuild, RingParam]:
    """The interior strip for a posed vertex buffer: posterior-lip cuff rings
    flowing into the cavity rings, one welded strip closed by a rear cap.
    Topology is constant; only vertex positions depend on the pose."""
    ring = entrance_ring(vertices, data)
    center = ring.points.mean(axis=0)
    specs = list(data.cuff_layers) + list(data.cavity_layers)
    layers = [_shaped_layer(ring.points, center, *spec) for spec in specs]
    n = len(ring.points)
    verts = np.concatenate(layers, axis=0)
    faces: list[tuple[int, int, int]] = []
    for layer in range(len(layers) - 1):
        a0, b0 = layer * n, (layer + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a0 + i, a0 + j, b0 + j))
            faces.append((a0 + i, b0 + j, b0 + i))
    cap = len(verts)
    verts = np.vstack([verts, layers[-1].mean(axis=0)])
    base = (len(layers) - 1) * n
    for i in range(n):
        faces.append((base + i, base + (i + 1) % n, cap))
    depths = np.zeros(len(layers))
    for k in range(1, len(layers)):
        depths[k] = depths[k - 1] + float(
            np.linalg.norm(layers[k] - layers[k - 1], axis=1).mean()
        )

    # Interior winding: the strip is seen from inside the mouth and the body
    # material is single-sided, so face normals must point into the void.
    # The construction above is coherently wound (every quad follows the
    # same ring order), so orientation is ONE global decision — a per-face
    # flip would break coherence and scramble UV winding with it. Majority
    # vote against the nearest layer center decides the flip.
    face_array = np.asarray(faces, dtype=np.int64)
    centers = np.stack([layer.mean(axis=0) for layer in layers])
    tri = verts[face_array]
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    # The vote uses only the rear cap: its facing (toward the mouth
    # entrance) is unambiguous, while side-wall faces sit nearly tangent to
    # their layer center and would dilute the signal.
    cap_normals = normals[-n:]
    toward_entrance = centers[0] - centroids[-n:]
    if float((cap_normals * toward_entrance).sum()) < 0:
        face_array = face_array[:, ::-1].copy()

    return (
        SocketBuild(
            vertices=verts,
            faces=face_array,
            ring_size=n,
            layer_count=len(layers),
            layer_depths=depths,
        ),
        ring,
    )


# -- interior UVs -------------------------------------------------------------

def _patch_texcoord_for(position_ids: np.ndarray, faces: np.ndarray,
                        texcoord_faces: np.ndarray, portal: np.ndarray) -> np.ndarray:
    """For each position vertex on the portal boundary, the texcoord index
    the patch's own faces use for it. Exactly one per vertex — the patch
    side of the seam."""
    lookup: dict[int, int] = {}
    for face, tface in zip(faces[portal], texcoord_faces[portal]):
        for p, t in zip(face, tface):
            previous = lookup.setdefault(int(p), int(t))
            if previous != int(t):
                raise ValueError(
                    f"portal vertex {int(p)} maps to multiple texcoords "
                    f"inside the patch — the atlas does not match the rig"
                )
    try:
        return np.asarray([lookup[int(p)] for p in position_ids], dtype=np.int64)
    except KeyError as missing:
        raise ValueError(
            f"lip-path vertex {missing} is not on the portal patch boundary"
        ) from None


def socket_uvs(rig, data: MouthData, socket: SocketBuild, ring: RingParam) -> np.ndarray:
    """UVs for the socket strip, inside the removed patch's own atlas region.

    The seam ring lands exactly on the patch's UV boundary (resampled through
    the ring's own arc-length parameterization), deeper rings contract toward
    the patch's UV centroid in proportion to their 3D depth — so texel
    density stays even along the interior — and the cap takes the centroid.
    No other chart is touched: the region belonged to the removed faces.
    """
    upper_t = _patch_texcoord_for(data.lip_upper, rig.faces, rig.texcoord_faces,
                                  data.portal_faces)
    lower_t = _patch_texcoord_for(data.lip_lower, rig.faces, rig.texcoord_faces,
                                  data.portal_faces)
    ring_uv = ring.interpolate(rig.texcoords[upper_t], rig.texcoords[lower_t])

    # Inset the base ring a hair along its local inward normal: resampled
    # ring edges are chords of the patch's boundary polygon, and at concave
    # boundary vertices a chord cuts outside the patch — into a neighboring
    # lip chart. Two texels of inset keep every band strictly inside.
    tangents = np.roll(ring_uv, -1, axis=0) - np.roll(ring_uv, 1, axis=0)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    nxt = np.roll(ring_uv, -1, axis=0)
    signed_area = float(
        (ring_uv[:, 0] * nxt[:, 1] - ring_uv[:, 1] * nxt[:, 0]).sum()
    ) / 2
    if signed_area < 0:
        normals = -normals
    ring_uv = ring_uv + normals * 0.002

    patch_t = np.unique(rig.texcoord_faces[data.portal_faces])
    centroid = rig.texcoords[patch_t].astype(np.float64).mean(axis=0)

    # Radial ring scales chosen so each band's UV area is proportional to
    # its 3D area (an annulus between scales a and b covers a²−b² of the
    # region): texel density stays even from the seam to the cap, even
    # though the cavity widens in 3D while the UV region narrows.
    n = socket.ring_size
    tri = socket.vertices[socket.faces]
    face_areas = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ) / 2.0
    band_faces = 2 * n
    band_areas = [
        float(face_areas[k * band_faces:(k + 1) * band_faces].sum())
        for k in range(socket.layer_count - 1)
    ]
    band_areas.append(float(face_areas[-n:].sum()))     # the cap fan
    cumulative = np.concatenate([[0.0], np.cumsum(band_areas)])
    scales = np.sqrt(1.0 - cumulative[:-1] / cumulative[-1])

    uvs = np.empty((len(socket.vertices), 2), dtype=np.float64)
    for k in range(socket.layer_count):
        uvs[k * n:(k + 1) * n] = centroid + (ring_uv - centroid) * scales[k]
    uvs[-1] = centroid
    return uvs


# -- skinning for the strip ---------------------------------------------------

def socket_skin(rig, data: MouthData, socket: SocketBuild, ring: RingParam):
    """Skin weights for the strip: each ring point interpolates its source
    lip vertices' influences (so the floor follows the jaw and the roof the
    skull, exactly like the lips it extends), every deeper ring inherits its
    ring point's weights, and the cap averages the last ring."""
    joint_ids = np.unique(rig.vertex_joints[np.r_[data.lip_upper, data.lip_lower]])
    dense_upper = np.zeros((len(data.lip_upper), len(joint_ids)))
    dense_lower = np.zeros((len(data.lip_lower), len(joint_ids)))
    column = {int(j): c for c, j in enumerate(joint_ids)}
    for dense, path in ((dense_upper, data.lip_upper), (dense_lower, data.lip_lower)):
        for row, vertex in enumerate(path):
            for j, w in zip(rig.vertex_joints[vertex], rig.vertex_weights[vertex]):
                if w > 0:
                    dense[row, column[int(j)]] += float(w)
    ring_dense = ring.interpolate(dense_upper, dense_lower)

    strip = np.tile(ring_dense, (socket.layer_count, 1))
    strip = np.vstack([strip, ring_dense[..., :].mean(axis=0, keepdims=True)])

    if strip.shape[1] < 4:   # the lips may use fewer than 4 distinct joints
        pad = 4 - strip.shape[1]
        strip = np.hstack([strip, np.zeros((len(strip), pad))])
        joint_ids = np.concatenate([joint_ids, np.zeros(pad, joint_ids.dtype)])
    order = np.argsort(-strip, axis=1)[:, :4]
    joints4 = joint_ids[order].astype(rig.vertex_joints.dtype)
    weights4 = np.take_along_axis(strip, order, axis=1).astype(np.float32)
    sums = weights4.sum(axis=1, keepdims=True)
    weights4 = weights4 / np.maximum(sums, 1e-12)
    weights4[weights4 < 0] = 0.0
    # Zero-weight slots must not reference arbitrary joints.
    joints4[weights4 == 0] = 0
    return joints4, weights4


# -- the baked export strip ---------------------------------------------------

@dataclass
class ExportStrip:
    """The socket strip as baked into a GLB: rest positions authored so
    that SKINNING the strip open lands on the pose-correct socket.

    The socket construction is aperture-adaptive (its clamps are
    non-linear in the lip positions), so a strip built at rest and merely
    skinned cannot land where the open-mouth socket belongs — measured
    4 cm of deviation at full open, standing as ridges through the
    jaw-following anatomy. The interior rings are therefore built at the
    certified full-open reference pose and inverse-skinned back to rest:
    correct where the interior is visible (open), slack only where it is
    hidden (closed). The seam ring stays the exact rest entrance ring."""

    vertices: np.ndarray        # (S, 3) rest cm
    faces: np.ndarray
    uv: np.ndarray
    joints: np.ndarray          # (S, 4)
    weights: np.ndarray         # (S, 4)
    morph_deltas: list          # per unit: (S, 3) vs these rest vertices
    weld_pairs: list            # [(lip corner vertex, seam duplicate)] to weld


def _jaw_rotation(data: MouthData, pivot: np.ndarray, level: float):
    axis = np.asarray(data.jaw["world_axis"], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(float(data.jaw["full_open_degrees"]) * level)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * k @ k
    return rotation, pivot


def jaw_subtree_weights(rig, vertex_joints, vertex_weights):
    """Per-vertex total influence of the c_jaw subtree — the fraction of
    the jaw rotation a skinned vertex follows."""
    subtree = set(rig.subtree(rig.joint_index("c_jaw")))
    total = np.zeros(len(vertex_joints))
    for joint in subtree:
        total += np.where(
            (vertex_joints == joint) & (vertex_weights > 0), vertex_weights, 0
        ).sum(axis=1)
    return np.clip(total, 0.0, 1.0)


def skin_jaw(points, jaw_weight, rotation, pivot):
    """LBS under a pure c_jaw rotation (head static)."""
    moved = (points - pivot) @ rotation.T + pivot
    return points * (1.0 - jaw_weight[:, None]) + moved * jaw_weight[:, None]


def export_strip(rig, data: MouthData, evaluation) -> ExportStrip:
    rest = evaluation.vertices
    pivot = evaluation.skeleton[rig.joint_index("c_jaw"), :3]
    rotation, pivot = _jaw_rotation(data, pivot, 1.0)

    body_jaw = jaw_subtree_weights(rig, rig.vertex_joints, rig.vertex_weights)
    posed = skin_jaw(rest, body_jaw, rotation, pivot)

    rest_build, ring = build_socket(rest, data)
    target, target_ring = build_socket(posed, data)
    joints4, weights4 = socket_skin(rig, data, rest_build, ring)

    # MHR's inner-lip seam has near-coincident duplicate vertices at each
    # mouth corner with slightly different skin weights; with the portal
    # removed they tear visibly apart under the jaw (measured 1.5 mm at
    # full open). Both are welded to the pair's average — the body copies
    # at export, and the strip's corner columns here, so every party at
    # each corner shares one set of weights.
    weld_pairs = []
    for pair, ring_position in (
        ((int(data.lip_upper[0]), int(data.upper_portal[1])), 0),
        ((int(data.lip_upper[-1]), int(data.upper_portal[-2])),
         data.samples_per_lip - 1),
    ):
        merged: dict[int, float] = {}
        for vertex in pair:
            for joint, weight in zip(rig.vertex_joints[vertex],
                                     rig.vertex_weights[vertex]):
                if weight > 0:
                    merged[int(joint)] = merged.get(int(joint), 0.0) + 0.5 * float(weight)
        top = sorted(merged.items(), key=lambda kv: -kv[1])[:4]
        total = sum(w for _, w in top)
        joint_row = np.zeros(4, dtype=joints4.dtype)
        weight_row = np.zeros(4, dtype=np.float32)
        for slot, (joint, weight) in enumerate(top):
            joint_row[slot] = joint
            weight_row[slot] = weight / total
        rows = (np.arange(rest_build.layer_count) * rest_build.ring_size
                + ring_position)
        joints4[rows] = joint_row
        weights4[rows] = weight_row
        weld_pairs.append((pair, joint_row, weight_row))

    strip_jaw = jaw_subtree_weights(rig, joints4, weights4)

    # Inverse LBS per vertex: y = [(1-w)I + wR](x) + w(p - Rp)
    systems = ((1.0 - strip_jaw)[:, None, None] * np.eye(3)[None]
               + strip_jaw[:, None, None] * rotation[None])
    offsets = target.vertices - strip_jaw[:, None] * (pivot - rotation @ pivot)
    inverse_skinned = np.linalg.solve(systems, offsets[..., None])[..., 0]

    # Blend per layer between the rest build and the open-referenced
    # shape: the seam ring must BE the rest lips; the cuff sits right
    # behind them (visible at rest, small open-pose error), so it stays
    # near its rest shape; the deep cavity — where a rest-built strip
    # lands centimeters wrong when skinned open — follows the open
    # reference fully. The ramp keeps the rest state clear of the closed
    # lips (the open reference inverse-rotates slightly in front of them).
    n = rest_build.ring_size
    ramp = np.concatenate([
        np.array([0.0, 0.0, 0.5] + [1.0] * (rest_build.layer_count - 3)
                 ).repeat(n),
        [1.0],                                   # the cap
    ])
    vertices = (rest_build.vertices * (1.0 - ramp[:, None])
                + inverse_skinned * ramp[:, None])
    vertices[:n] = ring.points        # the seam ring is the rest lips, exactly

    uv = socket_uvs(rig, data, target, ring)

    morph_deltas = []
    for unit in range(len(data.morph_names)):
        morphed, _ = build_socket(
            rest + data.morph_dense(unit, len(rest)), data
        )
        morph_deltas.append(morphed.vertices - vertices)

    return ExportStrip(
        vertices=vertices,
        faces=rest_build.faces,
        uv=uv,
        joints=joints4,
        weights=weights4,
        morph_deltas=morph_deltas,
        weld_pairs=weld_pairs,
    )


# -- anatomy ------------------------------------------------------------------

@dataclass
class AnatomyPiece:
    name: str
    vertices: np.ndarray    # (V, 3) world cm, identity-placed
    faces: np.ndarray
    uv: np.ndarray
    parent_role: str        # rig joint name to parent under
    base_color: tuple
    roughness: float


def _similarity_fit(src: np.ndarray, dst: np.ndarray):
    """Proper similarity (uniform scale, rotation, translation); reflection
    is forbidden — a mirrored dental arch is an error, not a fit."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - cs, dst - cd
    u, singular, vt = np.linalg.svd(xs.T @ xd)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
        singular[-1] *= -1
    scale = singular.sum() / max(float(np.sum(xs * xs)), 1e-12)
    a = scale * r
    return a, cd - a @ cs


_PIECES = (
    # (asset file stem, parent joint, base color, roughness)
    ("mouth_upper", "c_head", (0.92, 0.88, 0.80, 1.0), 0.35),
    ("mouth_lower", "c_teeth", (0.90, 0.86, 0.78, 1.0), 0.35),
    ("mouth_tongue", "c_tongue0", (0.62, 0.23, 0.25, 1.0), 0.42),
)


def place_anatomy(assets_dir: str | Path, data: MouthData,
                  identity_neutral: np.ndarray) -> list[AnatomyPiece]:
    """Identity placement per the integration contract: independent upper
    and lower similarity fits from the canonical-neutral anchors to this
    identity's anchors. Upper anatomy is skull-locked; lower anatomy and the
    tongue ride the jaw joint chain at runtime (they are placed here at
    rest, exactly like the eyes)."""
    assets_dir = Path(assets_dir)
    placement = json.loads(
        (assets_dir / "mouth_placement.json").read_text(encoding="utf-8")
    )
    if placement.get("format") != "character-factory/mouth-assets":
        raise ValueError(f"{assets_dir} has no mouth asset data")

    a_up, t_up = _similarity_fit(
        data.canonical_upper, identity_neutral[data.anchors_upper]
    )
    a_low, t_low = _similarity_fit(
        data.canonical_lower, identity_neutral[data.anchors_lower]
    )
    transforms = {"mouth_upper": (a_up, t_up), "mouth_lower": (a_low, t_low),
                  "mouth_tongue": (a_low, t_low)}
    pieces = []
    for stem, parent, color, roughness in _PIECES:
        with np.load(assets_dir / f"{stem}.npz") as archive:
            vertices = archive["vertices"].astype(np.float64)
            faces = archive["faces"].astype(np.int64)
            uv = archive["uv"].astype(np.float32)
        expected = placement["components"][stem]
        if len(vertices) != expected["vertices"] or len(faces) != expected["faces"]:
            raise ValueError(
                f"{stem} does not match its declared topology "
                f"({len(vertices)}/{len(faces)} vs {expected})"
            )
        a, t = transforms[stem]
        pieces.append(AnatomyPiece(
            name=stem, vertices=vertices @ a.T + t, faces=faces, uv=uv,
            parent_role=parent, base_color=color, roughness=roughness,
        ))
    return pieces
