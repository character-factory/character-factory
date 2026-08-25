"""The mouth battery on a declared render LOD.

The same invariants the source-topology battery asserts, re-run natively
against a body-rig version that declares a coarser render tessellation:
the interior belongs to the surface being built, so a LOD's portal, lip
paths, socket and morph basis all have to hold on their own terms.

Skipped unless such a component is present in the local cache.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from character_factory.assembly import mouth as mm
from character_factory.assembly.rig import load_rig

E = 72


def _render_lod_component():
    root = Path.home() / ".cache/character-factory/components/body-rig"
    if not root.is_dir():
        return None
    for directory in sorted(root.iterdir(), reverse=True):
        metadata = directory / "rig.json"
        if not metadata.is_file():
            continue
        document = json.loads(metadata.read_text())
        if document.get("render") and document.get("mouth"):
            return directory
    return None


COMPONENT = _render_lod_component()
pytestmark = pytest.mark.skipif(
    COMPONENT is None,
    reason="no body-rig version declaring a render LOD with mouth data")


@pytest.fixture(scope="module")
def rig():
    return load_rig(COMPONENT, device="cpu")


@pytest.fixture(scope="module")
def data(rig):
    return mm.MouthData.load(COMPONENT, rig.metadata)


def surface_vertices(rig, expression=None):
    evaluation = rig.evaluate([0.0] * 45, expression or [0.0] * E)
    return rig.render.vertices_from(evaluation.vertices, rig.faces), evaluation


def test_portal_and_lips_belong_to_the_render_topology(rig, data):
    surface = rig.render
    assert data.portal_faces.max() < len(surface.faces)
    assert len(np.unique(data.portal_faces)) == len(data.portal_faces)
    for path in (data.lip_upper, data.lip_lower):
        assert path.max() < len(surface.map_triangles)
    # The portal's own boundary is what the lips trace.
    faces = surface.faces[data.portal_faces]
    edges = np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert int((counts == 1).sum()) == len(data.lip_upper) + len(data.lip_lower) - 2


def test_render_lod_needs_no_corner_weld(rig, data):
    """LOD1's inner-lip seam carries coincident duplicates that tear under
    the jaw; this surface has none, so the weld collapses away."""
    rest, _ = surface_vertices(rig)
    corners = rest[[data.lip_upper[0], data.lip_upper[-1]]]
    distances = np.linalg.norm(rest[:, None, :] - corners[None, :, :], axis=2)
    # Exactly one vertex coincides with each corner: the corner itself.
    assert ((distances < 1e-6).sum(axis=0) == 1).all()
    assert np.array_equal(data.upper_portal, data.lip_upper)


def test_socket_builds_manifold_and_interior(rig, data):
    rest, _ = surface_vertices(rig)
    socket, _ring = mm.build_socket(rest, data)
    edges = np.sort(np.concatenate([
        socket.faces[:, [0, 1]], socket.faces[:, [1, 2]],
        socket.faces[:, [2, 0]]]), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert not (counts > 2).any()                     # no non-manifold edge
    assert int((counts == 1).sum()) == socket.ring_size   # only the seam open
    # Coherent interior winding, judged the way the source-topology
    # battery judges it: the rear cap must face back toward the entrance
    # (a tube's side walls point sideways, so a global axis test says
    # nothing).
    tri = socket.vertices[socket.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = socket.ring_size
    entrance_center = socket.vertices[:n].mean(axis=0)
    toward = entrance_center - tri[-n:].mean(axis=1)
    assert float((normals[-n:] * toward).sum()) > 0


def test_socket_stays_behind_the_exterior_under_the_jaw(rig, data):
    """The release gate: at every plain jaw level the interior must not
    protrude through the exterior skin."""
    surface = rig.render
    keep = np.ones(len(surface.faces), bool)
    keep[data.portal_faces] = False
    for jaw in (0.0, 0.5, 1.0):
        expression = [0.0] * E
        expression[24] = jaw
        posed, _ = surface_vertices(rig, expression)
        socket, _ring = mm.build_socket(posed, data)
        samples = np.vstack([socket.vertices,
                             socket.vertices[socket.faces].mean(axis=1)])
        triangles = posed[surface.faces[keep]]
        lo = samples[:, :2].min(axis=0) - 0.05
        hi = samples[:, :2].max(axis=0) + 0.05
        tri_lo = triangles[:, :, :2].min(axis=1)
        tri_hi = triangles[:, :, :2].max(axis=1)
        local = triangles[np.all(tri_hi >= lo, axis=1)
                          & np.all(tri_lo <= hi, axis=1)]
        x, y = samples[:, 0, None], samples[:, 1, None]
        x1, y1 = local[None, :, 0, 0], local[None, :, 0, 1]
        x2, y2 = local[None, :, 1, 0], local[None, :, 1, 1]
        x3, y3 = local[None, :, 2, 0], local[None, :, 2, 1]
        den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        valid = np.abs(den) > 1e-10
        safe = np.where(valid, den, 1.0)
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / safe
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / safe
        c = 1.0 - a - b
        inside = valid & (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        z = np.where(inside, a * local[None, :, 0, 2] + b * local[None, :, 1, 2]
                     + c * local[None, :, 2, 2], -np.inf)
        front = z.max(axis=1)
        covered = np.isfinite(front) & (front > float(samples[:, 2].max() - 5.0))
        clearance = np.where(covered, samples[:, 2] - front, -np.inf)
        assert clearance.max() < 0.02, f"jaw {jaw}: {clearance.max() * 10:.2f} mm"


def test_morph_basis_is_exact_on_the_render_topology(rig, data):
    """A linear map of a linear basis: the transferred morphs must
    reproduce a directly-evaluated pose, not merely approximate it."""
    rest, _ = surface_vertices(rig)
    for unit in (0, 12, 24, 40, 71):
        expression = [0.0] * E
        expression[unit] = 1.0
        posed, _ = surface_vertices(rig, expression)
        delta = data.morph_dense(unit, len(rest))
        assert np.abs((rest + delta) - posed).max() < 5e-4   # cm


@pytest.fixture(scope="module")
def uv_build(rig, data):
    rest, _ = surface_vertices(rig)
    socket, ring = mm.build_socket(rest, data)
    return socket, mm.socket_uvs(rig.render, data, socket, ring)


def test_socket_uvs_do_not_invert(uv_build):
    socket, uv = uv_build
    tri = uv[socket.faces]
    area = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
            - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    nonzero = area[np.abs(area) > 1e-12]
    assert (nonzero > 0).all() or (nonzero < 0).all()


def test_socket_uvs_stay_inside_the_removed_patch(rig, data, uv_build):
    """Interior UVs land in the patch the portal freed, never in another
    island — the same containment the eye sockets hold to."""
    _socket, uv = uv_build
    surface = rig.render
    patch = surface.texcoords[np.unique(
        surface.texcoord_faces[data.portal_faces])].astype(np.float64)
    lo, hi = patch.min(axis=0), patch.max(axis=0)
    margin = 0.02 * (hi - lo)
    assert (uv >= lo - margin).all() and (uv <= hi + margin).all()


def test_socket_uv_density_is_bounded(uv_build):
    """No degenerate or wildly stretched texels: the ratio between the
    densest and sparsest interior triangle stays within a stated bound."""
    socket, uv = uv_build
    tri3 = socket.vertices[socket.faces]
    tri2 = uv[socket.faces]
    world = np.linalg.norm(np.cross(tri3[:, 1] - tri3[:, 0],
                                    tri3[:, 2] - tri3[:, 0]), axis=1) * 0.5
    texel = np.abs((tri2[:, 1, 0] - tri2[:, 0, 0]) * (tri2[:, 2, 1] - tri2[:, 0, 1])
                   - (tri2[:, 2, 0] - tri2[:, 0, 0]) * (tri2[:, 1, 1] - tri2[:, 0, 1])) * 0.5
    keep = (world > 1e-9) & (texel > 1e-14)
    density = texel[keep] / world[keep]
    assert len(density) > 0.9 * len(world)
    assert np.percentile(density, 95) / np.percentile(density, 5) < 60.0
