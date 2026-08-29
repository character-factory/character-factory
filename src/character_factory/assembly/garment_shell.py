"""Generated-alpha garment shells: body-following mesh garments.

Turns a character's own baked garment albedo into a closed, skinned,
body-following garment solid — the cut topology comes from *that
texture's* keyed coverage, never from a canonical or per-style cut. The
painted composite remains underneath (a conservative covered-body face
set is omitted at export); a character whose extraction fails any gate
falls back to the painted composite, silently.

Contract highlights (SPEC-independent — assembly behavior, like turbo):

- **Source-driven**: the black-keyed alpha of the baked garment texture
  is the only cut authority. Nonconforming masks fail closed; nothing is
  snapped to a known silhouette, filled, extended, or repaired.
- **Pure function**: extraction consumes the published baked asset bytes,
  the rig buffers, and versioned constants. No inference, no RNG — the
  same inputs produce the same solid, byte for byte.
- **Fail-closed ladder**: alpha gates → cut/topology audits → closed-
  solid audits → weight audits → pose-envelope gate. Validators
  validate; nothing repairs.

All geometry is rig-native: centimeters, Y-up. The exporter applies its
usual unit scale; nothing here converts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ShellConstants",
    "ShellRejected",
    "PreparedShell",
    "prepare_shell",
    "pose_gate",
    "POSE_SET",
    "PROPORTION_EXTREMES",
]


class ShellRejected(Exception):
    """This character's garment cannot ship as a shell; the painted
    composite is the correct result. `reason` is a stable machine-
    readable code (recorded in the manifest)."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class ShellConstants:
    """Versioned extraction constants (the accepted R&D values). These are
    data, not judgment: changing any of them is a new constants version
    and re-certification, never a per-character tweak."""

    version: str = "1"
    key_cutoff: int = 20              # max(RGB) > cutoff, over valid atlas
    key_cutoff_check: int = 16        # stability partner for the IoU gate
    opening_size: int = 3             # binary opening structure (square)
    alpha_sigma_px: float = 1.0       # gaussian feather of the hard key
    smooth_band: float = 0.22         # |raw - 0.5| < band gets smoothed
    smooth_iterations: int = 5
    smooth_keep: float = 0.55         # value = keep*raw + (1-keep)*ring mean
    min_component_faces: int = 8      # drop only if BOTH small
    min_component_area_cm2: float = 0.05
    base_lift_cm: float = 0.24
    boundary_extra_cm: float = 0.32
    boundary_falloff_cm: float = 3.0
    fair_iterations: int = 18
    fair_amount: float = 0.22
    fair_boundary_factor: float = 0.14
    fair_tether: float = 0.11
    fair_boundary_tether: float = 0.58
    normal_clamp_extra_cm: float = 0.32
    tangent_clamp_cm: float = 0.12
    cut_t_clamp: float = 0.02         # keep refined crossings off the exact
                                      # corners: no sliver faces, no
                                      # degenerate sidewalls
    inner_min_cm: float = 0.12        # inner = max(min, outer_offset - backoff)
    inner_backoff_cm: float = 0.12
    rim_uv_inset: float = 0.35        # sidewall UVs pulled toward face centroid
    erode_rings: int = 2              # covered-body erosion before face hiding
    coverage_min: float = 0.005       # keyed fraction of the valid atlas
    coverage_max: float = 0.90
    cutoff_stability_iou: float = 0.99      # measured 0.9977+ across seeds
    excluded_removed_max: float = 0.25  # keyed fraction the excluded-region
                                        # masking may remove before the mask
                                        # itself is judged untrustworthy
    poke_mm: float = 0.5              # pose-gate visible-intersection threshold
    # Seam-disagreement budget: D5 ruling — the value arrives as config/
    # registry data once its derivation evidence lands. None = the
    # detector runs report-only and never fails a character.
    seam_disagreement_budget: float | None = None


# The R&D pose-envelope set: MHR 204-dim articulation vectors. Indices are
# rig-native pose channels; the set is rest, walk-mid, arms-raised, crouch.
def _pose_set() -> dict[str, np.ndarray]:
    poses = {"rest": np.zeros(204, dtype=np.float32)}
    walk = np.zeros(204, dtype=np.float32)
    for index, value in ((52, 0.55), (53, 0.30), (61, -0.55), (62, 0.82),
                         (35, 0.32), (36, 0.22), (45, -0.32), (46, 0.25),
                         (7, 0.10)):
        walk[index] = value
    poses["walk_mid"] = walk
    raised = np.zeros(204, dtype=np.float32)
    for index, value in ((31, 0.24), (41, 0.24), (35, 1.42), (45, 1.42),
                         (36, 0.20), (46, 0.20), (17, -0.18)):
        raised[index] = value
    poses["arms_raised"] = raised
    crouch = np.zeros(204, dtype=np.float32)
    for index, value in ((1, -8.0), (11, 0.42), (17, 0.38), (52, -0.92),
                         (61, -0.92), (53, 1.62), (62, 1.62), (55, -0.34),
                         (64, -0.34), (35, 0.36), (45, 0.36), (36, 0.58),
                         (46, 0.58)):
        crouch[index] = value
    poses["crouch"] = crouch
    return poses


POSE_SET = _pose_set()

# The six proportion controls at their envelope extremes (±0.40), in the
# schema's control order — the certification sweep runs the pose set
# against these in addition to each character's own proportions.
PROPORTION_EXTREMES = (
    {"spine_length": 0.4, "neck_length": 0.4, "shoulder_width": 0.4,
     "arm_length": 0.4, "hip_width": 0.4, "leg_length": 0.4},
    {"spine_length": -0.4, "neck_length": -0.4, "shoulder_width": -0.4,
     "arm_length": -0.4, "hip_width": -0.4, "leg_length": -0.4},
)


# --------------------------------------------------------------------------
# alpha preparation and gates
# --------------------------------------------------------------------------

def _hard_key(rgb: np.ndarray, atlas_valid: np.ndarray, cutoff: int,
              opening: int) -> np.ndarray:
    from scipy import ndimage

    hard = (rgb.max(axis=2) > cutoff) & atlas_valid
    structure = np.ones((opening, opening), dtype=bool)
    return ndimage.binary_opening(hard, structure=structure)


def prepare_alpha(rgb: np.ndarray, atlas_valid: np.ndarray,
                  constants: ShellConstants,
                  excluded_regions: np.ndarray | None = None) -> dict:
    """The normative key: hard cutoff over the valid atlas, opened, then
    gaussian-feathered. `excluded_regions` (the atlas's declared
    garment-never-paints-here region — the head mask) subtracts from the
    key exactly as the compositor's region contract does for paint, so
    the shell keys the same effective garment the painted path
    composites. Region masking is atlas contract, never content repair.
    Returns hard/soft masks plus audit metrics; raises ShellRejected on
    any alpha-quality gate."""
    from scipy import ndimage

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ShellRejected("alpha-bad-input", f"texture shape {rgb.shape}")
    if rgb.shape[:2] != atlas_valid.shape:
        raise ShellRejected(
            "alpha-bad-input",
            f"texture {rgb.shape[:2]} vs atlas {atlas_valid.shape}")
    rgb = rgb[:, :, :3]
    hard = _hard_key(rgb, atlas_valid, constants.key_cutoff,
                     constants.opening_size)
    excluded_removed = 0.0
    if excluded_regions is not None:
        keyed_total = int(hard.sum())
        hard = hard & ~excluded_regions
        if keyed_total:
            excluded_removed = 1.0 - hard.sum() / keyed_total
        if excluded_removed > constants.excluded_removed_max:
            # The garment substantially lives in the head/feet regions:
            # the mask is not trustworthy as a garment.
            raise ShellRejected("alpha-excluded-region",
                                f"{excluded_removed:.3f} of the key removed")
        atlas_valid = atlas_valid & ~excluded_regions
    check = _hard_key(rgb, atlas_valid, constants.key_cutoff_check,
                      constants.opening_size)
    if excluded_regions is not None:
        check = check & ~excluded_regions
    valid_area = int(atlas_valid.sum())
    coverage = float(hard.sum()) / max(valid_area, 1)
    if coverage < constants.coverage_min:
        raise ShellRejected("alpha-coverage-small", f"{coverage:.4f}")
    if coverage > constants.coverage_max:
        raise ShellRejected("alpha-coverage-large", f"{coverage:.4f}")
    union = int((hard | check).sum())
    intersection = int((hard & check).sum())
    stability = intersection / max(union, 1)
    if stability < constants.cutoff_stability_iou:
        raise ShellRejected("alpha-cutoff-unstable", f"IoU {stability:.6f}")
    soft = ndimage.gaussian_filter(
        hard.astype(np.float32), sigma=constants.alpha_sigma_px)
    np.clip(soft, 0.0, 1.0, out=soft)
    return {
        "hard": hard,
        "soft": soft,
        "coverage": coverage,
        "cutoff_stability_iou": stability,
        "excluded_removed": excluded_removed,
    }


def dilate_garment_colors(rgb: np.ndarray, keyed: np.ndarray) -> np.ndarray:
    """Atlas hygiene for the shell's texture: every non-keyed texel takes
    the color of its nearest keyed texel (Voronoi bleed), so boundary
    faces and rim insets sample cloth color instead of the keyed-out
    background. Deterministic; the keyed region itself is untouched, and
    the *key* is always derived from the original image — dilation can
    never grow coverage."""
    from scipy import ndimage

    if not keyed.any():
        return rgb
    nearest = ndimage.distance_transform_edt(
        ~keyed, return_distances=False, return_indices=True)
    return rgb[nearest[0], nearest[1]]


# --------------------------------------------------------------------------
# welded scalar field
# --------------------------------------------------------------------------

def _bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample image (H, W) at uv in [0,1]², image-convention V (v down)."""
    height, width = image.shape
    x = np.clip(uv[..., 0], 0.0, 1.0) * (width - 1)
    y = np.clip(uv[..., 1], 0.0, 1.0) * (height - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, width - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, height - 2)
    fx, fy = x - x0, y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def welded_field(soft: np.ndarray, surface, canonical_vertices: np.ndarray,
                 constants: ShellConstants) -> dict:
    """Area-reconcile per-corner alpha samples onto welded body vertices,
    then smooth only the cutoff-ambiguous band on the surface graph."""
    faces = surface.faces
    corner_uv = surface.texcoords[surface.texcoord_faces]          # (F, 3, 2)
    corner_alpha = _bilinear(soft, corner_uv)              # (F, 3)

    v0 = canonical_vertices[faces[:, 0]]
    area2 = np.linalg.norm(
        np.cross(canonical_vertices[faces[:, 1]] - v0,
                 canonical_vertices[faces[:, 2]] - v0), axis=1)
    vertex_count = len(canonical_vertices)
    weight_sum = np.zeros(vertex_count)
    value_sum = np.zeros(vertex_count)
    low = np.full(vertex_count, np.inf)
    high = np.full(vertex_count, -np.inf)
    for corner in range(3):
        index = faces[:, corner]
        np.add.at(weight_sum, index, area2)
        np.add.at(value_sum, index, area2 * corner_alpha[:, corner])
        np.minimum.at(low, index, corner_alpha[:, corner])
        np.maximum.at(high, index, corner_alpha[:, corner])
    referenced = weight_sum > 0
    raw = np.zeros(vertex_count)
    raw[referenced] = value_sum[referenced] / weight_sum[referenced]
    disagreement = np.where(referenced, high - low, 0.0)

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                            faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    from scipy import sparse

    adjacency = sparse.coo_matrix(
        (np.ones(len(edges) * 2),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(vertex_count, vertex_count)).tocsr()
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)

    ambiguous = np.abs(raw - 0.5) < constants.smooth_band
    values = raw.copy()
    for _ in range(constants.smooth_iterations):
        ring_mean = adjacency.dot(values) / degree
        values[ambiguous] = (constants.smooth_keep * raw[ambiguous]
                             + (1 - constants.smooth_keep) * ring_mean[ambiguous])
    return {"values": values, "raw": raw, "disagreement": disagreement,
            "edges": edges, "adjacency": adjacency}


# --------------------------------------------------------------------------
# marching the per-character cut
# --------------------------------------------------------------------------

def _refine_crossing(soft: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray,
                     t_linear: float, constants: ShellConstants) -> float:
    """Slide a cut vertex onto the mask curve: bisect the feathered alpha
    along the edge's UV segment for its 0.5 crossing. The welded field
    decides *whether* an edge is crossed (topology); this decides *where*
    — at texture resolution, so the cut follows the learned contour
    instead of quantizing to edge midpoints. Falls back to the linear
    field estimate when the alpha does not straddle 0.5 along this edge
    (the band smoothing moved the decision)."""
    def sample(t: float) -> float:
        uv = (1.0 - t) * uv_a + t * uv_b
        return float(_bilinear(soft, uv[None])[0])

    low_t, high_t = 0.0, 1.0
    low_value = sample(low_t) - 0.5
    high_value = sample(high_t) - 0.5
    clamp = constants.cut_t_clamp
    if low_value * high_value >= 0.0:
        return float(np.clip(t_linear, clamp, 1.0 - clamp))
    for _ in range(24):
        mid = 0.5 * (low_t + high_t)
        value = sample(mid) - 0.5
        if low_value * value <= 0.0:
            high_t = mid
        else:
            low_t, low_value = mid, value
    return float(np.clip(0.5 * (low_t + high_t), clamp, 1.0 - clamp))


def march_cut(values: np.ndarray, soft: np.ndarray, surface,
              canonical_vertices: np.ndarray,
              constants: ShellConstants) -> dict:
    """Clip the body surface at the 0.5 level set. Shared geometry at cut
    edges (their crossing refined against the alpha in UV space),
    per-corner UVs interpolated in the owning source face, exact
    source-face/barycentric correspondence for every shell vertex."""
    faces = surface.faces
    corner_uv = surface.texcoords[surface.texcoord_faces]
    covered = values >= 0.5

    body_to_shell: dict[int, int] = {}
    edge_cut: dict[tuple[int, int], int] = {}
    positions: list[np.ndarray] = []
    source_face: list[int] = []
    source_bary: list[np.ndarray] = []
    out_faces: list[tuple[int, int, int]] = []
    out_uv: list[np.ndarray] = []

    def body_vertex(index: int) -> int:
        shell = body_to_shell.get(index)
        if shell is None:
            shell = len(positions)
            body_to_shell[index] = shell
            positions.append(canonical_vertices[index])
            source_face.append(-1)          # one-hot row assigned later
            source_bary.append(np.zeros(3))
        return shell

    def cut_vertex(a: int, b: int, uv_a: np.ndarray,
                   uv_b: np.ndarray) -> tuple[int, float]:
        """The shared cut vertex on undirected edge (a, b) and its
        crossing parameter in a→b order. The first face to reach the
        edge fixes the refined crossing; adjacent faces reuse it, so
        both sides share one vertex at one position."""
        key = (a, b) if a < b else (b, a)
        cached = edge_cut.get(key)
        if cached is not None:
            shell, t_key = cached
            return shell, (t_key if key == (a, b) else 1.0 - t_key)
        t_linear = float(np.clip(
            (0.5 - values[a]) / (values[b] - values[a]), 0.0, 1.0))
        t = _refine_crossing(soft, uv_a, uv_b, t_linear, constants)
        shell = len(positions)
        positions.append((1 - t) * canonical_vertices[a]
                         + t * canonical_vertices[b])
        source_face.append(-1)
        source_bary.append(np.zeros(3))
        edge_cut[key] = (shell, t if key == (a, b) else 1.0 - t)
        return shell, t

    for face_index in range(len(faces)):
        corners = faces[face_index]
        states = covered[corners]
        if not states.any():
            continue
        uvs = corner_uv[face_index]
        if states.all():
            polygon = [(body_vertex(int(corners[c])), uvs[c],
                        _one_hot(c)) for c in range(3)]
        else:
            polygon = []
            for c in range(3):
                nxt = (c + 1) % 3
                a, b = int(corners[c]), int(corners[nxt])
                if states[c]:
                    polygon.append((body_vertex(a), uvs[c], _one_hot(c)))
                if states[c] != states[nxt]:
                    shell, t = cut_vertex(a, b, uvs[c], uvs[nxt])
                    uv_point = (1 - t) * uvs[c] + t * uvs[nxt]
                    bary = (1 - t) * _one_hot(c) + t * _one_hot(nxt)
                    polygon.append((shell, uv_point, bary))
            if len(polygon) < 3:
                continue
        for third in range(1, len(polygon) - 1):
            triangle = (polygon[0], polygon[third], polygon[third + 1])
            out_faces.append(tuple(p[0] for p in triangle))
            out_uv.append(np.stack([p[1] for p in triangle]))
        # Correspondence: assign this face as owner where not yet owned.
        for shell, _uv, bary in polygon:
            if source_face[shell] < 0:
                source_face[shell] = face_index
                source_bary[shell] = bary

    if not out_faces:
        raise ShellRejected("cut-empty", "no covered surface at threshold")

    result = {
        "positions": np.asarray(positions),
        "faces": np.asarray(out_faces, dtype=np.int64),
        "corner_uv": np.asarray(out_uv, dtype=np.float64),
        "source_face": np.asarray(source_face, dtype=np.int64),
        "source_bary": np.asarray(source_bary, dtype=np.float64),
    }
    _filter_components(result, constants)
    _audit_correspondence(result, surface, canonical_vertices)
    result["boundary"] = _boundary(result["faces"])
    return result


def _one_hot(corner: int) -> np.ndarray:
    row = np.zeros(3)
    row[corner] = 1.0
    return row


def _filter_components(cut: dict, constants: ShellConstants) -> None:
    """Drop connected components that are tiny by BOTH face count and
    canonical area — versioned noise thresholds, not semantic filters."""
    from scipy import sparse
    from scipy.sparse import csgraph

    faces = cut["faces"]
    count = len(cut["positions"])
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(count, count))
    _, labels = csgraph.connected_components(graph, directed=False)
    face_label = labels[faces[:, 0]]
    v0 = cut["positions"][faces[:, 0]]
    areas = 0.5 * np.linalg.norm(
        np.cross(cut["positions"][faces[:, 1]] - v0,
                 cut["positions"][faces[:, 2]] - v0), axis=1)
    keep_face = np.ones(len(faces), dtype=bool)
    for label in np.unique(face_label):
        members = face_label == label
        if (members.sum() < constants.min_component_faces
                and areas[members].sum() < constants.min_component_area_cm2):
            keep_face[members] = False
    if not keep_face.any():
        raise ShellRejected("cut-empty", "every component below noise floor")
    if not keep_face.all():
        _compact(cut, keep_face)


def _compact(cut: dict, keep_face: np.ndarray) -> None:
    faces = cut["faces"][keep_face]
    used = np.unique(faces)
    remap = np.full(len(cut["positions"]), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    cut["faces"] = remap[faces]
    cut["corner_uv"] = cut["corner_uv"][keep_face]
    cut["positions"] = cut["positions"][used]
    cut["source_face"] = cut["source_face"][used]
    cut["source_bary"] = cut["source_bary"][used]


def _audit_correspondence(cut: dict, surface, canonical_vertices: np.ndarray) -> None:
    reconstructed = _from_bary(cut, canonical_vertices, surface)
    error = np.linalg.norm(reconstructed - cut["positions"], axis=1)
    if error.max() > 2e-5:
        raise ShellRejected("correspondence-error",
                            f"max reconstruction {error.max():.2e} cm")
    if cut["source_bary"].min() < -2e-4:
        raise ShellRejected("correspondence-error",
                            f"barycentric {cut['source_bary'].min():.2e}")
    sums = cut["source_bary"].sum(axis=1)
    if np.abs(sums - 1.0).max() > 1e-5:
        raise ShellRejected("correspondence-error", "barycentric sum")


def _from_bary(cut: dict, body_vertices: np.ndarray, surface) -> np.ndarray:
    triangles = surface.faces[cut["source_face"]]
    corners = body_vertices[triangles]                     # (N, 3, 3)
    return np.einsum("nk,nkd->nd", cut["source_bary"], corners)


def _boundary(faces: np.ndarray) -> dict:
    """Open-boundary edges (single incidence). Verifies degree-two
    boundary vertices and no non-manifold (over-shared) edges."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                            faces[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    if (counts > 2).any():
        raise ShellRejected("topology-nonmanifold",
                            f"{int((counts > 2).sum())} over-shared edges")
    boundary = unique[counts == 1]
    if len(boundary):
        degree = np.bincount(boundary.ravel())
        vertices = np.unique(boundary)
        if (degree[vertices] != 2).any():
            raise ShellRejected("topology-boundary-branch",
                                "boundary vertex degree != 2")
    # Oriented boundary edges in original winding (for sidewall building).
    oriented = []
    boundary_set = {tuple(edge) for edge in boundary}
    for a, b in edges:
        key = (a, b) if a < b else (b, a)
        if key in boundary_set:
            oriented.append((int(a), int(b)))
    return {"edges": boundary, "oriented": oriented,
            "vertices": np.unique(boundary) if len(boundary) else
            np.zeros(0, dtype=np.int64)}


# --------------------------------------------------------------------------
# identity construction, lift, fairing, closed solid
# --------------------------------------------------------------------------

def _body_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    face_normal = np.cross(vertices[faces[:, 1]] - v0,
                           vertices[faces[:, 2]] - v0)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normal)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def _geodesic_from_boundary(positions: np.ndarray, faces: np.ndarray,
                            boundary_vertices: np.ndarray) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse import csgraph

    if not len(boundary_vertices):
        return np.full(len(positions), np.inf)
    edges = np.unique(np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0)
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]],
                             axis=1)
    graph = sparse.coo_matrix(
        (np.concatenate([lengths, lengths]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(len(positions), len(positions))).tocsr()
    distances = csgraph.dijkstra(graph, directed=False,
                                 indices=boundary_vertices, min_only=True)
    return distances


def build_solid(cut: dict, identity_vertices: np.ndarray, surface,
                constants: ShellConstants) -> dict:
    """Reconstruct on the character's identity, lift, fair, and close into
    one watertight solid (outer + inner + sidewalls)."""
    source = _from_bary(cut, identity_vertices, surface)
    body_normal = _body_normals(identity_vertices, surface.faces)
    triangles = surface.faces[cut["source_face"]]
    normal = np.einsum("nk,nkd->nd", cut["source_bary"],
                       body_normal[triangles])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)

    distance = _geodesic_from_boundary(source, cut["faces"],
                                       cut["boundary"]["vertices"])
    q = np.clip(1.0 - distance / constants.boundary_falloff_cm, 0.0, 1.0)
    ease = q * q * (3.0 - 2.0 * q)
    target_lift = constants.base_lift_cm + constants.boundary_extra_cm * ease
    base = source + normal * target_lift[:, None]

    is_boundary = np.zeros(len(source), dtype=bool)
    is_boundary[cut["boundary"]["vertices"]] = True
    from scipy import sparse

    edges = np.unique(np.sort(np.concatenate(
        [cut["faces"][:, [0, 1]], cut["faces"][:, [1, 2]],
         cut["faces"][:, [2, 0]]]), axis=1), axis=0)
    adjacency = sparse.coo_matrix(
        (np.ones(len(edges) * 2),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(len(source), len(source))).tocsr()
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)

    amount = np.full(len(source), constants.fair_amount)
    amount[is_boundary] *= constants.fair_boundary_factor
    tether = np.full(len(source), constants.fair_tether)
    tether[is_boundary] = constants.fair_boundary_tether

    current = base.copy()
    for _ in range(constants.fair_iterations):
        ring_mean = adjacency.dot(current) / degree[:, None]
        proposed = current + amount[:, None] * (ring_mean - current)
        proposed = proposed * (1 - tether[:, None]) + base * tether[:, None]
        delta = proposed - source
        normal_offset = np.einsum("nd,nd->n", delta, normal)
        tangent = delta - normal * normal_offset[:, None]
        tangent_length = np.linalg.norm(tangent, axis=1)
        scale = np.minimum(1.0, constants.tangent_clamp_cm
                           / np.maximum(tangent_length, 1e-12))
        tangent *= scale[:, None]
        normal_offset = np.clip(
            normal_offset, target_lift,
            target_lift + constants.normal_clamp_extra_cm)
        current = source + normal * normal_offset[:, None] + tangent
    outer = current

    # Closed solid: inner surface + sidewalls along every boundary edge.
    delta = outer - source
    normal_offset = np.einsum("nd,nd->n", delta, normal)
    tangent = delta - normal * normal_offset[:, None]
    inner_offset = np.maximum(constants.inner_min_cm,
                              normal_offset - constants.inner_backoff_cm)
    inner = source + normal * inner_offset[:, None] + tangent

    count = len(outer)
    vertices = np.concatenate([outer, inner])
    outer_faces = cut["faces"]
    inner_faces = outer_faces[:, ::-1] + count
    oriented = cut["boundary"]["oriented"]
    side_faces = []
    for a, b in oriented:
        side_faces.append((a, b, b + count))
        side_faces.append((a, b + count, a + count))
    faces = np.concatenate([
        outer_faces, inner_faces,
        np.asarray(side_faces, dtype=np.int64).reshape(-1, 3)])

    _audit_solid(vertices, faces, count, len(outer_faces), len(oriented))

    # UVs: outer corner UVs as cut; inner mirrors with reversed corner
    # order; sidewalls sample inside the owning covered face (inset toward
    # its UV centroid so the rim shows cloth, not the keyed background).
    outer_uv = cut["corner_uv"]
    inner_uv = outer_uv[:, ::-1, :]
    edge_uv = _boundary_edge_uv(cut, constants)
    side_uv = []
    for a, b in oriented:
        ua, ub = edge_uv[(a, b)]
        side_uv.append(np.stack([ua, ub, ub]))
        side_uv.append(np.stack([ua, ub, ua]))
    corner_uv = np.concatenate([
        outer_uv, inner_uv,
        np.asarray(side_uv).reshape(-1, 3, 2)]) if side_uv else \
        np.concatenate([outer_uv, inner_uv])

    return {
        "vertices": vertices,
        "faces": faces,
        "corner_uv": corner_uv,
        "outer_count": count,
        "outer_face_count": len(outer_faces),
        "outer_vertices": outer,
        "lift_cm": target_lift,
    }


def _boundary_edge_uv(cut: dict, constants: ShellConstants) -> dict:
    """Per oriented boundary edge, cloth-side UVs for the sidewall: the
    edge corners' UVs pulled toward the owning face's UV centroid."""
    mapping = {}
    faces = cut["faces"]
    for face_index in range(len(faces)):
        corners = faces[face_index]
        uvs = cut["corner_uv"][face_index]
        centroid = uvs.mean(axis=0)
        for c in range(3):
            a, b = int(corners[c]), int(corners[(c + 1) % 3])
            inset = constants.rim_uv_inset
            mapping[(a, b)] = (
                uvs[c] * (1 - inset) + centroid * inset,
                uvs[(c + 1) % 3] * (1 - inset) + centroid * inset,
            )
    return mapping


def _audit_solid(vertices: np.ndarray, faces: np.ndarray, outer_count: int,
                 outer_face_count: int, boundary_edge_count: int) -> None:
    if len(vertices) != 2 * outer_count:
        raise ShellRejected("solid-audit", "vertex count")
    if len(faces) != 2 * outer_face_count + 2 * boundary_edge_count:
        raise ShellRejected("solid-audit", "face count")
    edges = np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if (counts != 2).any():
        raise ShellRejected(
            "solid-not-watertight",
            f"{int((counts != 2).sum())} edges without exactly two faces")
    v0 = vertices[faces[:, 0]]
    areas = 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - v0, vertices[faces[:, 2]] - v0),
        axis=1)
    side = areas[2 * outer_face_count:]
    if len(side) and side.min() <= 1e-9:
        raise ShellRejected("solid-degenerate-sidewall",
                            f"minimum area {side.min():.2e}")


# --------------------------------------------------------------------------
# skin weights and covered-body faces
# --------------------------------------------------------------------------

def transfer_weights(cut: dict, surface, outer_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric combination of the source triangle's body influences;
    deterministic top-4 truncation; inner vertices copy their outer."""
    triangles = surface.faces[cut["source_face"]]              # (N, 3)
    joints = np.zeros((outer_count, 4), dtype=np.uint16)
    weights = np.zeros((outer_count, 4), dtype=np.float64)
    for index in range(outer_count):
        combined: dict[int, float] = {}
        for k in range(3):
            bary = cut["source_bary"][index, k]
            if bary <= 0:
                continue
            body_vertex = triangles[index, k]
            for slot in range(surface.vertex_joints.shape[1]):
                weight = float(surface.vertex_weights[body_vertex, slot])
                if weight <= 0:
                    continue
                joint = int(surface.vertex_joints[body_vertex, slot])
                combined[joint] = combined.get(joint, 0.0) + bary * weight
        rows = [(joint, weight) for joint, weight in combined.items()
                if weight > 1e-7]
        rows.sort(key=lambda item: (-item[1], item[0]))
        rows = rows[:4]
        total = sum(weight for _, weight in rows)
        if total <= 0:
            raise ShellRejected("weights-empty", f"vertex {index}")
        for slot, (joint, weight) in enumerate(rows):
            joints[index, slot] = joint
            weights[index, slot] = weight / total
    sums = weights.sum(axis=1)
    if np.abs(sums - 1.0).max() > 1e-5:
        raise ShellRejected("weights-audit", "weight sums")
    return (np.concatenate([joints, joints]),
            np.concatenate([weights, weights]).astype(np.float32))


def covered_body_faces(values: np.ndarray, surface, adjacency,
                       constants: ShellConstants) -> np.ndarray:
    """Conservative hide set: coverage eroded by welded rings; a face hides
    only when all three vertices remain covered."""
    covered = values >= 0.5
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)
    for _ in range(constants.erode_rings):
        covered_neighbors = adjacency.dot(covered.astype(np.float64))
        covered = covered & (covered_neighbors >= degree - 1e-9)
    hide = covered[surface.faces].all(axis=1)
    return np.where(hide)[0].astype(np.int64)


# --------------------------------------------------------------------------
# the prepared shell
# --------------------------------------------------------------------------

@dataclass
class PreparedShell:
    vertices: np.ndarray          # (2N, 3) float64 cm — closed solid, rest
    faces: np.ndarray             # (2F + 2B, 3) int64
    corner_uv: np.ndarray         # (2F + 2B, 3, 2) float64
    joints4: np.ndarray           # (2N, 4) uint16
    weights4: np.ndarray          # (2N, 4) float32
    covered_body_faces: np.ndarray  # (K,) int64 — body faces to omit
    outer_count: int
    outer_face_count: int
    source_face: np.ndarray
    source_bary: np.ndarray
    hard_key: np.ndarray | None = None   # (H, W) bool — for color dilation
    audit: dict = field(default_factory=dict)


def prepare_shell(surface, garment_rgb: np.ndarray, identity_vertices: np.ndarray,
                  canonical_vertices: np.ndarray, atlas_valid: np.ndarray,
                  excluded_regions: np.ndarray | None = None,
                  constants: ShellConstants | None = None) -> PreparedShell:
    """The full extraction: baked garment bytes → closed skinned solid.

    Raises ShellRejected (with a stable reason code) on any gate; the
    caller falls back to the painted composite for this character.
    """
    constants = constants or ShellConstants()
    alpha = prepare_alpha(garment_rgb, atlas_valid, constants,
                          excluded_regions=excluded_regions)
    fields = welded_field(alpha["soft"], surface, canonical_vertices, constants)
    seam = _seam_diagnostic(fields, constants)
    cut = march_cut(fields["values"], alpha["soft"], surface,
                    canonical_vertices, constants)
    solid = build_solid(cut, identity_vertices, surface, constants)
    joints4, weights4 = transfer_weights(cut, surface, solid["outer_count"])
    hidden = covered_body_faces(fields["values"], surface, fields["adjacency"],
                                constants)

    return PreparedShell(
        vertices=solid["vertices"],
        faces=solid["faces"],
        corner_uv=solid["corner_uv"],
        joints4=joints4,
        weights4=weights4,
        covered_body_faces=hidden,
        outer_count=solid["outer_count"],
        outer_face_count=solid["outer_face_count"],
        source_face=cut["source_face"],
        source_bary=cut["source_bary"],
        hard_key=alpha["hard"],
        audit={
            "constants_version": constants.version,
            "coverage": alpha["coverage"],
            "cutoff_stability_iou": alpha["cutoff_stability_iou"],
            "excluded_removed": alpha["excluded_removed"],
            "components": _component_count(cut),
            "boundary_loops": _loop_count(cut["boundary"]),
            "seam_disagreement": seam,
            "lift_cm": {
                "min": float(solid["lift_cm"].min()),
                "max": float(solid["lift_cm"].max()),
            },
        },
    )


def _seam_diagnostic(fields: dict, constants: ShellConstants) -> dict:
    """The D5 seam detector, report-only until its budget value lands as
    config/registry data: measures welded-vertex corner disagreement in
    the decision band (a paired-seam crack shows as high disagreement at
    near-threshold vertices)."""
    band = np.abs(fields["raw"] - 0.5) < constants.smooth_band
    in_band = fields["disagreement"][band]
    measured = float(in_band.max()) if len(in_band) else 0.0
    result = {"max_band_disagreement": measured,
              "budget": constants.seam_disagreement_budget,
              "enforced": constants.seam_disagreement_budget is not None}
    if (constants.seam_disagreement_budget is not None
            and measured > constants.seam_disagreement_budget):
        raise ShellRejected("alpha-seam-disagreement",
                            f"{measured:.4f} > budget")
    return result


def _component_count(cut: dict) -> int:
    from scipy import sparse
    from scipy.sparse import csgraph

    faces = cut["faces"]
    graph = sparse.coo_matrix(
        (np.ones(len(faces) * 3),
         (np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]]),
          np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]]))),
        shape=(len(cut["positions"]),) * 2)
    count, labels = csgraph.connected_components(graph, directed=False)
    return int(len(np.unique(labels[faces[:, 0]])))


def _loop_count(boundary: dict) -> int:
    edges = boundary["edges"]
    if not len(edges):
        return 0
    from scipy import sparse
    from scipy.sparse import csgraph

    count = int(edges.max()) + 1
    graph = sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(count, count))
    components, labels = csgraph.connected_components(graph, directed=False)
    return int(len(np.unique(labels[edges[:, 0]])))


# --------------------------------------------------------------------------
# the pose-envelope gate
# --------------------------------------------------------------------------

def _world_matrices(skeleton_state: np.ndarray, parents: np.ndarray) -> np.ndarray:
    from character_factory.assembly.restpose import Skeleton

    skeleton = Skeleton.from_rig_state(skeleton_state, parents)
    return np.stack([skeleton.world_matrix(j) for j in range(len(parents))])


def _joint_transforms(rest_state: np.ndarray, posed_state: np.ndarray,
                      parents: np.ndarray) -> np.ndarray:
    rest = _world_matrices(rest_state, parents)
    posed = _world_matrices(posed_state, parents)
    return np.einsum("jab,jbc->jac", posed, np.linalg.inv(rest))


def _lbs(vertices: np.ndarray, joints4: np.ndarray, weights4: np.ndarray,
         transforms: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [vertices, np.ones((len(vertices), 1))], axis=1)
    result = np.zeros_like(vertices)
    for slot in range(joints4.shape[1]):
        weight = weights4[:, slot].astype(np.float64)
        active = weight > 0
        if not active.any():
            continue
        matrices = transforms[joints4[active, slot].astype(np.int64)]
        moved = np.einsum("nab,nb->na", matrices, homogeneous[active])[:, :3]
        result[active] += weight[active, None] * moved
    return result


def evaluate_posed_skeleton(rig, identity, resting_expression, pose_vector,
                            proportions=None) -> np.ndarray:
    """The rig's skeleton state at an articulation vector (adds the
    character's proportion pose when present)."""
    import torch

    pose = np.asarray(pose_vector, dtype=np.float32).copy()
    if proportions:
        pose = pose + rig.proportion_pose(proportions).astype(np.float32)
    with torch.no_grad():
        _, skeleton = rig.model(
            torch.tensor([list(identity)], dtype=torch.float32),
            torch.from_numpy(pose).unsqueeze(0),
            torch.tensor([list(resting_expression)], dtype=torch.float32),
        )
    return skeleton[0].numpy().astype(np.float64)


def pose_gate(rig, shell: PreparedShell, evaluation, identity,
              resting_expression, proportions=None,
              constants: ShellConstants | None = None,
              poses: dict[str, np.ndarray] | None = None,
              surface=None, body_rest=None) -> dict:
    """Certification: across the pose set, the LBS-posed shell must show
    zero visible body poke — measured against the *retained* body faces
    (covered faces are omitted at export) and confirmed by an ID-buffer
    render test. Raises ShellRejected on failure; returns diagnostics.

    The gate poses body and shell with the same exact-LBS deformation the
    exported GLB uses (this product ships no pose correctives — the
    stack's ruled export contract), so what passes here is what a
    consumer's engine actually renders.
    """
    constants = constants or ShellConstants()
    poses = poses or POSE_SET
    # The body under test is whatever ships: the rig's own surface, or
    # the render LOD it declares. The skeleton always comes from the rig.
    surface = surface if surface is not None else rig
    rest_state = evaluation.skeleton
    body_rest = body_rest if body_rest is not None else evaluation.vertices
    retained = np.ones(len(surface.faces), dtype=bool)
    retained[shell.covered_body_faces] = False
    diagnostics = {}
    for name, vector in poses.items():
        posed_state = evaluate_posed_skeleton(
            rig, identity, resting_expression, vector, proportions)
        transforms = _joint_transforms(rest_state, posed_state, rig.parents)
        body = _lbs(body_rest, surface.vertex_joints, surface.vertex_weights,
                    transforms)
        posed = _lbs(shell.vertices, shell.joints4, shell.weights4,
                     transforms)
        if not np.isfinite(posed).all():
            raise ShellRejected("pose-nonfinite", name)
        _audit_posed_sidewalls(posed, shell)
        outer = posed[:shell.outer_count]
        visible_depth, hidden_depth = _poke_depths(outer, body, surface.faces,
                                                   retained)
        threshold_cm = constants.poke_mm / 10.0
        diagnostics[name] = {
            "visible_max_mm": round(float(visible_depth.max()) * 10.0, 3)
            if len(visible_depth) else 0.0,
            "covered_contact_max_mm": round(float(hidden_depth.max()) * 10.0, 3)
            if len(hidden_depth) else 0.0,
            "covered_contact_over": int((hidden_depth > threshold_cm).sum()),
        }
        if len(visible_depth) and visible_depth.max() > threshold_cm:
            raise ShellRejected(
                "pose-visible-poke",
                f"{name}: {visible_depth.max() * 10.0:.3f} mm")
        render = _render_poke_check(posed, shell.faces, body,
                                    surface.faces[retained],
                                    surface.faces[~retained])
        diagnostics[name]["render"] = render
        if render["skin_in_silhouette"] or render["body_holes"]:
            raise ShellRejected(
                "pose-render-poke",
                f"{name}: skin {render['skin_in_silhouette']}, "
                f"holes {render['body_holes']}")
    return diagnostics


def _audit_posed_sidewalls(posed: np.ndarray, shell: PreparedShell) -> None:
    side = shell.faces[2 * shell.outer_face_count:]
    if not len(side):
        return
    v0 = posed[side[:, 0]]
    areas = 0.5 * np.linalg.norm(
        np.cross(posed[side[:, 1]] - v0, posed[side[:, 2]] - v0), axis=1)
    if areas.min() <= 1e-9:
        raise ShellRejected("pose-degenerate-sidewall",
                            f"minimum area {areas.min():.2e}")


def _poke_depths(points: np.ndarray, body_vertices: np.ndarray,
                 body_faces: np.ndarray,
                 retained: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Penetration depths (cm) of outer-shell points into the complete
    body, split by *what they penetrate through*: the reliable sign comes
    from the watertight complete body; each penetrating point is
    attributed to its nearest body face — a retained (visible) face is a
    real poke, a hidden face is covered-cloth contact (diagnostic)."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=body_vertices, faces=body_faces,
                           process=False)
    signed = trimesh.proximity.signed_distance(mesh, points)
    depth = np.maximum(0.0, signed)   # trimesh: positive = inside
    inside = depth > 0
    visible = np.zeros(0)
    hidden = np.zeros(0)
    if inside.any():
        _, _, triangle = trimesh.proximity.closest_point(mesh, points[inside])
        through_visible = retained[triangle]
        visible = depth[inside][through_visible]
        hidden = depth[inside][~through_visible]
    return visible, hidden


_VIEWS = {"front": 0.0, "three_quarter": 35.0, "side": 90.0, "back": 180.0}


# A pixel counts as skin-through-cloth only when the visible body surface
# sits just in front of the shell (within this window). A true poke
# renders skin within roughly the lift distance of the pierced cloth;
# body parts legitimately passing in front (a raised wrist before the
# torso, measured at ~9.5 mm in the R&D-pose sweep) sit farther out and
# are occlusion, not poke. This is a measurement-classification constant
# of the render check, not a clearance: the geometric 0.5 mm gate against
# retained faces is unaffected by it.
_POKE_DEPTH_WINDOW_CM = 0.3


def _render_poke_check(shell_vertices: np.ndarray, shell_faces: np.ndarray,
                       body_vertices: np.ndarray, retained_faces: np.ndarray,
                       hidden_faces: np.ndarray,
                       resolution: int = 512) -> dict:
    """Object-ID + depth rasterization at four views: zero skin pixels
    poking through the opaque garment silhouette (visible body within the
    poke window in front of cloth), zero body holes outside it (pixels
    only hidden faces would have covered)."""
    skin_pixels = 0
    hole_pixels = 0
    for yaw in _VIEWS.values():
        angle = np.radians(yaw)
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)]])
        shell_view = shell_vertices @ rotation.T
        body_view = body_vertices @ rotation.T
        everything = np.concatenate([shell_view, body_view])
        low = everything.min(axis=0)
        high = everything.max(axis=0)
        scale = (resolution - 1) / max(high[0] - low[0], high[1] - low[1], 1e-6)

        def raster(vertices, faces):
            depth = np.full((resolution, resolution), -np.inf)
            xy = (vertices[:, :2] - low[:2]) * scale
            z = vertices[:, 2]
            for a, b, c in faces:
                _fill(depth, xy[a], xy[b], xy[c], z[a], z[b], z[c])
            return depth

        shell_depth = raster(shell_view, shell_faces)
        retained_depth = raster(body_view, retained_faces)
        hidden_depth = raster(body_view, hidden_faces)
        silhouette = shell_depth > -np.inf
        in_front = retained_depth > shell_depth
        with np.errstate(invalid="ignore"):
            # -inf minus -inf (empty vs empty pixels) is NaN; NaN compares
            # False, which is the correct "not a poke" answer.
            near = (retained_depth - shell_depth) < _POKE_DEPTH_WINDOW_CM
        skin_pixels += int((silhouette & in_front & near).sum())
        hole_pixels += int(
            (~silhouette & (hidden_depth > retained_depth)).sum())
    return {"skin_in_silhouette": skin_pixels, "body_holes": hole_pixels}


def _fill(depth: np.ndarray, p0, p1, p2, z0, z1, z2) -> None:
    minimum = np.floor(np.minimum(np.minimum(p0, p1), p2)).astype(int)
    maximum = np.ceil(np.maximum(np.maximum(p0, p1), p2)).astype(int)
    minimum = np.maximum(minimum, 0)
    maximum = np.minimum(maximum, np.array(depth.shape)[::-1] - 1)
    if (maximum < minimum).any():
        return
    xs = np.arange(minimum[0], maximum[0] + 1)
    ys = np.arange(minimum[1], maximum[1] + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    d = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(d) < 1e-12:
        return
    w0 = ((p1[1] - p2[1]) * (grid_x - p2[0])
          + (p2[0] - p1[0]) * (grid_y - p2[1])) / d
    w1 = ((p2[1] - p0[1]) * (grid_x - p2[0])
          + (p0[0] - p2[0]) * (grid_y - p2[1])) / d
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return
    z = w0 * z0 + w1 * z1 + w2 * z2
    region = depth[minimum[1]:maximum[1] + 1, minimum[0]:maximum[0] + 1]
    update = inside & (z > region)
    region[update] = z[update]


# --------------------------------------------------------------------------
# data-supplied constants
# --------------------------------------------------------------------------

ENV_SEAM_BUDGET = "CHARACTER_FACTORY_GARMENT_SEAM_BUDGET"


def _config_section() -> dict:
    import json
    import os  # noqa: F401 — parallel shape with the env readers

    from character_factory.registry.store import cache_dir

    path = cache_dir() / "config.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    section = document.get("assembly")
    return section if isinstance(section, dict) else {}


def configured_constants() -> ShellConstants:
    """Extraction constants with their data-supplied overrides. The seam-
    disagreement budget arrives as data (environment or
    `assembly.garment_shell_seam_budget`) once its derivation evidence
    lands — until then the seam detector reports and never rejects."""
    import os

    budget = os.environ.get(ENV_SEAM_BUDGET)
    if budget is None:
        budget = _config_section().get("garment_shell_seam_budget")
    if budget is None:
        return ShellConstants()
    return ShellConstants(seam_disagreement_budget=float(budget))


# --------------------------------------------------------------------------
# atlas helpers
# --------------------------------------------------------------------------

def valid_atlas_mask(surface, resolution: int) -> np.ndarray:
    """Rasterize the rig's UV triangles: the valid-atlas mask (True where
    a body face owns the texel). Deterministic per rig + resolution."""
    mask = np.full((resolution, resolution), -np.inf)
    uv = surface.texcoords[surface.texcoord_faces]                 # (F, 3, 2)
    xy = uv * (resolution - 1)
    for face in xy:
        _fill(mask, face[0], face[1], face[2], 1.0, 1.0, 1.0)
    return mask > -np.inf
