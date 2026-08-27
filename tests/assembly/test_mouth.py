"""The mouth-interior regression surface (SPEC.md §4.2, §9 step 4).

Interior geometry is a tested surface, not a one-time construction: portal
topology, seam attachment, anatomy containment (template and the synthetic
stress-envelope identities), the interior-UV contract (bit-exact originals,
no inversion, no chart overlap, measured even density), morph exactness
against the rig's own forward, jaw clearance on the exported deliverable,
and byte determinism. Runs only with the real body-rig component cached.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.assembly.stress_identities import NARROW_IDENTITY, WIDE_IDENTITY
from tests.assembly.test_export import _real_rig_dir

pytestmark = pytest.mark.skipif(
    _real_rig_dir() is None
    or "mouth" not in json.loads(
        (_real_rig_dir() / "rig.json").read_text(encoding="utf-8")
    ),
    reason="mouth-capable body-rig component not present in the local cache",
)

E = 72


@pytest.fixture(scope="module")
def rig():
    from character_factory.assembly.rig import load_rig

    return load_rig(_real_rig_dir())


@pytest.fixture(scope="module")
def mouth_data(rig):
    from character_factory.assembly.mouth import MouthData

    return MouthData.load(rig.component_dir, rig.metadata)


def _evaluate(rig, identity, expression=None):
    import torch

    e = torch.zeros(1, E)
    if expression is not None:
        e = torch.tensor([expression], dtype=torch.float32)
    with torch.no_grad():
        v, _ = rig.model(
            torch.tensor([identity], dtype=torch.float32), torch.zeros(1, 204), e
        )
    return v[0].numpy().astype(np.float64)


# -- portal topology ----------------------------------------------------------

def test_portal_set_is_the_topology_component(rig, mouth_data):
    """The stored 288 faces are exactly the component bounded by the
    52-edge inner-lip loop — re-derived from the rig's own buffers."""
    faces = rig.faces
    upper_portal = np.asarray(
        rig.metadata["mouth"]["lip_paths"]["upper_portal"], dtype=np.int64
    )
    loop = np.r_[upper_portal, mouth_data.lip_lower[-2:0:-1]]
    boundary = {tuple(sorted((int(loop[i]), int(loop[(i + 1) % len(loop)]))))
                for i in range(len(loop))}
    assert len(boundary) == 52
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(fi)
    assert not boundary.difference(edge_faces)
    adjacency: list[list[int]] = [[] for _ in range(len(faces))]
    for edge, linked in edge_faces.items():
        if edge in boundary or len(linked) != 2:
            continue
        a, b = linked
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = np.zeros(len(faces), bool)
    stack, patch = [int(mouth_data.portal_faces[0])], []
    seen[stack[0]] = True
    while stack:
        f = stack.pop()
        patch.append(f)
        for g in adjacency[f]:
            if not seen[g]:
                seen[g] = True
                stack.append(g)
    assert sorted(patch) == sorted(int(x) for x in mouth_data.portal_faces)
    assert len(patch) == 288


# -- socket construction ------------------------------------------------------

@pytest.mark.parametrize(("identity", "margin_cm"), [
    ([0.0] * 45, 0.05),
    (NARROW_IDENTITY, 1.5),
    (WIDE_IDENTITY, 1.5),
], ids=["template", "narrow-envelope", "wide-envelope"])
def test_socket_seam_and_containment(rig, mouth_data, identity, margin_cm):
    """Template anatomy sits inside the cavity envelope almost exactly; at
    the synthetic stress-envelope extremes the coarse similarity-fit
    placement (the spec's flagged v0 limitation) leaves gum edges up to a
    measured ~1.3 cm outside the socket bounding box — inside the head,
    guarded against visible protrusion by the clearance test below."""
    from character_factory.assembly.mouth import (
        build_socket,
        entrance_ring,
        place_anatomy,
    )
    from character_factory.registry import Registry

    rest = _evaluate(rig, identity)
    socket, ring = build_socket(rest, mouth_data)
    n = socket.ring_size
    assert n == 62
    # Seam attachment: the first ring IS the derived entrance ring.
    assert np.abs(socket.vertices[:n] - entrance_ring(rest, mouth_data).points).max() == 0.0
    # Constant topology.
    assert socket.layer_count == 7
    assert len(socket.vertices) == 7 * n + 1
    assert len(socket.faces) == 2 * n * 6 + n

    assets = Registry.default().ensure("assembly-assets")
    if not (Path(assets) / "mouth_placement.json").is_file():
        pytest.skip("assembly-assets has no mouth data")
    lo = socket.vertices.min(axis=0) - margin_cm
    hi = socket.vertices.max(axis=0) + margin_cm
    for piece in place_anatomy(assets, mouth_data, rest):
        assert (piece.vertices.min(axis=0) >= lo).all(), piece.name
        assert (piece.vertices.max(axis=0) <= hi).all(), piece.name


@pytest.mark.parametrize("identity", [NARROW_IDENTITY, WIDE_IDENTITY],
                         ids=["narrow-envelope", "wide-envelope"])
def test_envelope_plain_jaw_clearance(rig, mouth_data, identity):
    """Front-surface clearance at the plain jaw levels for the envelope
    extremes: the interior must never protrude through the exterior skin,
    however far the identity is from the library's cluster."""
    from character_factory.assembly.mouth import build_socket

    keep = np.ones(len(rig.faces), bool)
    keep[mouth_data.portal_faces] = False
    for jaw in (0.0, 0.5, 1.0):
        expression = [0.0] * E
        expression[24] = jaw
        posed = _evaluate(rig, identity, expression)
        socket, _ = build_socket(posed, mouth_data)
        samples = np.vstack([
            socket.vertices,
            socket.vertices[socket.faces].mean(axis=1),
        ])
        triangles = posed[rig.faces[keep]]
        lo = samples[:, :2].min(axis=0) - 0.05
        hi = samples[:, :2].max(axis=0) + 0.05
        tri_lo = triangles[:, :, :2].min(axis=1)
        tri_hi = triangles[:, :, :2].max(axis=1)
        local = triangles[np.all(tri_hi >= lo, axis=1) & np.all(tri_lo <= hi, axis=1)]
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


def test_socket_winding_is_coherent_and_interior(rig, mouth_data):
    from character_factory.assembly.mouth import build_socket

    rest = _evaluate(rig, [0.0] * 45)
    socket, _ = build_socket(rest, mouth_data)
    tri = socket.vertices[socket.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    # The rear cap must face the entrance (the interior viewpoint).
    n = socket.ring_size
    entrance_center = socket.vertices[:n].mean(axis=0)
    toward = entrance_center - tri[-n:].mean(axis=1)
    assert float((normals[-n:] * toward).sum()) > 0


# -- the interior-UV contract -------------------------------------------------

@pytest.fixture(scope="module")
def socket_uv_build(rig, mouth_data):
    from character_factory.assembly.mouth import build_socket, socket_uvs

    rest = _evaluate(rig, [0.0] * 45)
    socket, ring = build_socket(rest, mouth_data)
    uv = socket_uvs(rig, mouth_data, socket, ring)
    return socket, uv


def test_socket_uv_no_inversion(socket_uv_build):
    socket, uv = socket_uv_build
    tri = uv[socket.faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    assert (signed != 0).all()
    assert len(np.unique(np.sign(signed))) == 1


def test_socket_uv_no_chart_overlap(rig, mouth_data, socket_uv_build):
    socket, uv = socket_uv_build
    kept = np.ones(len(rig.faces), bool)
    kept[mouth_data.portal_faces] = False
    kept_uv = rig.texcoords[rig.texcoord_faces[kept]].astype(np.float64)
    socket_uv_tris = uv[socket.faces].astype(np.float64)

    def separated(t1, t2):
        for t in (t1, t2):
            edges = np.roll(t, -1, axis=0) - t
            for ax in np.stack([-edges[:, 1], edges[:, 0]], axis=1):
                p1, p2 = t1 @ ax, t2 @ ax
                if p1.max() <= p2.min() + 1e-10 or p2.max() <= p1.min() + 1e-10:
                    return True
        return False

    klo = kept_uv.min(axis=1)
    khi = kept_uv.max(axis=1)
    overlaps = 0
    for tri in socket_uv_tris:
        lo, hi = tri.min(axis=0), tri.max(axis=0)
        candidates = np.where(
            (khi[:, 0] > lo[0]) & (klo[:, 0] < hi[0])
            & (khi[:, 1] > lo[1]) & (klo[:, 1] < hi[1])
        )[0]
        if any(not separated(tri, kept_uv[c]) for c in candidates):
            overlaps += 1
    assert overlaps == 0


def test_socket_uv_density_even_and_bounded(rig, mouth_data, socket_uv_build):
    """The stated density bounds (reported in the proof escalation): even
    along the interior — band medians within 1.5x of each other, per-face
    p5-p95 spread within 15x — and no worse than 2% of the neighboring lip
    faces' texel density (the whole interior shares the small atlas region
    that belonged to the removed patch)."""
    socket, uv = socket_uv_build
    rest = _evaluate(rig, [0.0] * 45)
    tri3 = socket.vertices[socket.faces]
    area3 = np.linalg.norm(
        np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]), axis=1
    ) / 2
    triuv = uv[socket.faces]
    u1 = triuv[:, 1] - triuv[:, 0]
    u2 = triuv[:, 2] - triuv[:, 0]
    areauv = np.abs(u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]) / 2
    density = areauv / np.maximum(area3, 1e-12)
    n = socket.ring_size
    medians = [np.median(density[k * 2 * n:(k + 1) * 2 * n])
               for k in range(socket.layer_count - 1)]
    medians.append(np.median(density[-n:]))
    assert max(medians) / min(medians) < 1.5
    assert np.percentile(density, 95) / np.percentile(density, 5) < 15

    lips = np.r_[mouth_data.lip_upper, mouth_data.lip_lower]
    lip_faces = np.where(np.isin(rig.faces, lips).any(axis=1))[0]
    lip_faces = lip_faces[~np.isin(lip_faces, mouth_data.portal_faces)]
    t3 = rest[rig.faces[lip_faces]]
    la3 = np.linalg.norm(np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0]), axis=1) / 2
    tu = rig.texcoords[rig.texcoord_faces[lip_faces]].astype(np.float64)
    v1 = tu[:, 1] - tu[:, 0]
    v2 = tu[:, 2] - tu[:, 0]
    lip_density = (np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2) / np.maximum(la3, 1e-12)
    assert np.median(density) / np.median(lip_density) > 0.02


# -- exported artifact --------------------------------------------------------

EXAMPLES = Path(__file__).parents[2] / "examples" / "characters"


def _assemble_example(tmp_path):
    import json as jsonlib

    from PIL import Image

    from character_factory.api import assemble
    from character_factory.schema import Character

    from tests.assembly.test_export import source_topology_registry

    # This battery asserts source-topology facts (portal size, the corner
    # seam duplicates, exact vertex indices), so it pins the tier it was
    # written against rather than following what the index serves today.
    registry = source_topology_registry()

    assets = tmp_path / "assets-mouth-interior"
    assets.mkdir(exist_ok=True)
    for slot, color in (("skin", (170, 132, 105)), ("eye", (90, 60, 40)),
                        ("garment", (0, 0, 0))):
        Image.new("RGB", (64, 64), color).save(assets / f"{slot}.png")
    document = jsonlib.loads((EXAMPLES / "storyteller.char.json").read_text())
    resolved = registry.resolve_slots(sorted(document["textures"]))
    for slot, recipe in document["textures"].items():
        recipe["component_version"] = str(resolved[slot].version)
    out = tmp_path / "mouth-interior.glb"
    assemble(Character.from_document(document), assets, out, registry=registry)
    return out.read_bytes()


def _export_body_oracle(tmp_path, rig):
    """A low-level body-only export used solely as the UV preservation
    oracle. It is not a valid character tier or a public assembly path."""
    from character_factory.assembly import export_character_glb
    from character_factory.schema import Character

    character = Character.load(EXAMPLES / "storyteller.char.json")
    result = export_character_glb(
        rig,
        character.identity,
        character.resting_expression,
        tmp_path / "body-oracle.glb",
        generator="character-factory/test",
        _body_only_test=True,
        evaluation=rig.evaluate(
            character.identity,
            character.resting_expression,
            proportions=character.proportions,
        ),
    )
    return result.glb_path.read_bytes()


@pytest.fixture(scope="module")
def exported(tmp_path_factory, rig):
    tmp = tmp_path_factory.mktemp("mouth-glb")
    return {
        "mouth-interior": _assemble_example(tmp),
        "body-oracle": _export_body_oracle(tmp, rig),
    }


def _body_prim(data):
    from character_factory.assembly.gltf import parse_glb

    gltf, binary = parse_glb(data)
    body = next(m for m in gltf["meshes"] if m["name"] == "body")
    return gltf, binary, body


def test_original_uvs_bit_exact_after_interior(exported):
    from character_factory.assembly.gltf import read_accessor

    gltf_m, bin_m, body_m = _body_prim(exported["mouth-interior"])
    gltf_c, bin_c, body_c = _body_prim(exported["body-oracle"])
    from character_factory.assembly.gltf import read_accessor as _read

    def pairs(gltf, binary, body, drop_tail=0):
        attrs = body["primitives"][0]["attributes"]
        pos = _read(gltf, binary, attrs["POSITION"])
        uv = _read(gltf, binary, attrs["TEXCOORD_0"])
        count = len(pos) - drop_tail
        rows = np.concatenate([pos[:count], uv[:count]], axis=1)
        return {row.tobytes() for row in rows}

    # Every original (position, uv) vertex of the character export must
    # exist byte-identically in a low-level body-only oracle — the interior
    # only removes portal faces and appends; it never touches an original
    # vertex's texcoords. (Unweld order differs between the two face sets,
    # so this is a set comparison, not a prefix comparison.)
    mouthed = pairs(gltf_m, bin_m, body_m, drop_tail=7 * 62 + 1)
    body_only = pairs(gltf_c, bin_c, body_c)
    assert mouthed <= body_only


def test_export_has_exact_named_morphs(rig, mouth_data, exported):
    from character_factory.assembly.gltf import read_accessor

    gltf, binary, body = _body_prim(exported["mouth-interior"])
    prim = body["primitives"][0]
    targets = prim.get("targets")
    assert targets is not None and len(targets) == E
    assert body["extras"]["targetNames"] == [f"facs_{i:02d}" for i in range(E)]
    assert body["weights"] == [0.0] * E

    # Exactness: applying the jaw morph to the exported rest equals the
    # rig's own forward at that expression, on the original vertices.
    positions = read_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(np.float64)
    delta = read_accessor(gltf, binary, targets[24]["POSITION"]).astype(np.float64)
    manifest = gltf["asset"]["extras"]
    assert manifest["topology"] == "mouth-interior"

    document = json.loads((EXAMPLES / "storyteller.char.json").read_text())
    body_block = document["body"]
    rest = rig.evaluate(
        body_block["identity"], body_block["resting_expression"],
        proportions=body_block.get("proportions"),
    ).vertices
    expr = list(body_block["resting_expression"])
    expr[24] += 1.0
    truth_delta_cm = rig.evaluate(
        body_block["identity"], expr,
        proportions=body_block.get("proportions"),
    ).vertices - rest

    # Match a sample of moved rig vertices to their unwelded export copies
    # by rest position (knee baking moves legs, never the face).
    moved = mouth_data.morph_indices[24][:200]
    export_cm = positions * 100.0
    gaps = np.linalg.norm(
        export_cm[None, :, :] - rest[moved][:, None, :], axis=2
    )
    located = gaps.argmin(axis=1)
    close = gaps[np.arange(len(moved)), located] < 0.05
    assert close.sum() > 100
    got = delta[located[close]] * 100.0
    want = truth_delta_cm[moved[close]]
    assert np.abs(got - want).max() < 5e-3   # float32 storage noise, cm


def test_manifest_carries_tables_and_jaw_guidance(exported):
    gltf, _, _ = _body_prim(exported["mouth-interior"])
    manifest = gltf["asset"]["extras"]
    assert manifest["expression_morphs"]["count"] == E
    assert manifest["expression_morphs"]["semantics"]["semantic_source"] == "provisional-measured"
    assert "0..1" in manifest["expression_morphs"]["weights"]
    limitations = manifest["animation_limitations"]
    entries = limitations["entries"]
    assert any(e["kind"] == "neutral-seating" for e in entries)
    assert any(e["kind"] == "socket-clearance" for e in entries)
    # Structured params beside every case string, and a table-level
    # statement of the parameterization — consumers never parse the case
    # strings by inference.
    assert "expression_playback" in limitations["parameterization"]
    for entry in entries:
        assert "facs_24" in entry["params"]
    compound = next(e for e in entries if "expr" in e["case"])
    parts = compound["case"].split("_")
    assert compound["params"] == {
        "facs_24": float(parts[1]), "unit": int(parts[3]),
        "weight": float(parts[4]),
    }
    jaw = manifest["jaw"]
    assert "rotation_axis_local" in jaw and "full_open_degrees" in jaw
    # Sign and composition are contract, not consumer inference (the jaw
    # opens under POSITIVE rotation in the file's own right-handed frame;
    # joint_only and expression_playback are alternatives, never summed).
    assert "positive rotation" in jaw["rotation_sign"]
    assert set(jaw["composition"]) == {
        "joint_only", "expression_playback", "rule"}
    assert "never their sum" in jaw["composition"]["rule"]


def test_jaw_rotation_keeps_interior_behind_exterior(exported):
    """The certified jaw path, on the actual deliverable: rotate c_jaw per
    the manifest guidance and verify the socket strip stays behind the
    exterior surface at every plain jaw level (the release gate that was
    clean in the construction-space sweep holds for the skinned export)."""
    from character_factory.assembly.gltf import parse_glb, read_accessor

    gltf, binary = parse_glb(exported["mouth-interior"])
    nodes = gltf["nodes"]
    jaw_manifest = gltf["asset"]["extras"]["jaw"]
    axis = np.asarray(jaw_manifest["rotation_axis_local"], dtype=np.float64)
    body = next(m for m in gltf["meshes"] if m["name"] == "body")
    prim = body["primitives"][0]
    positions = read_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(np.float64)
    joints4 = read_accessor(gltf, binary, prim["attributes"]["JOINTS_0"]).astype(np.int64)
    weights4 = read_accessor(gltf, binary, prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    indices = read_accessor(gltf, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
    skin = gltf["skins"][0]
    ibms = read_accessor(gltf, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    jaw_node = next(i for i, n in enumerate(nodes) if n.get("name") == "c_jaw")

    def quat_matrix(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def quat_mul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    def skinned_at(level):
        theta = np.radians(jaw_manifest["full_open_degrees"] * level)
        extra = np.concatenate([axis * np.sin(theta / 2), [np.cos(theta / 2)]])
        local = {}
        for i, node in enumerate(nodes):
            t = np.asarray(node.get("translation", [0, 0, 0]), float)
            q = np.asarray(node.get("rotation", [0, 0, 0, 1]), float)
            s = np.asarray(node.get("scale", [1, 1, 1]), float)
            if i == jaw_node:
                q = quat_mul(q, extra)
            m = np.eye(4)
            m[:3, :3] = quat_matrix(q) * s
            m[:3, 3] = t
            local[i] = m
        world = {}
        stack = [(0, np.eye(4))]
        while stack:
            i, pm = stack.pop()
            world[i] = pm @ local[i]
            for child in nodes[i].get("children", []):
                stack.append((child, world[i]))
        joint_mats = np.stack([world[skin["joints"][j]] @ ibms[j]
                               for j in range(len(skin["joints"]))])
        homo = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        out = np.zeros((len(positions), 3))
        for k in range(4):
            m = joint_mats[joints4[:, k]]
            out += weights4[:, k, None] * np.einsum("vij,vj->vi", m[:, :3, :], homo)
        return out

    # The socket strip is the appended tail of the body vertex buffer. The
    # seam ring (its first 62 vertices) lies BY CONSTRUCTION on the exterior
    # lip surface — exactly on triangle edges, where float32 barycentric
    # inclusion is unstable — and its attachment is asserted separately, so
    # clearance measures the interior rings and the cap.
    strip = np.arange(len(positions))[-(7 * 62 + 1) + 62:]
    exterior_faces = indices[~np.isin(indices, strip).any(axis=1)]

    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        posed = skinned_at(level) * 100.0    # meters -> cm
        samples = posed[strip]
        triangles = posed[exterior_faces]
        lo = samples[:, :2].min(axis=0) - 0.05
        hi = samples[:, :2].max(axis=0) + 0.05
        tri_lo = triangles[:, :, :2].min(axis=1)
        tri_hi = triangles[:, :, :2].max(axis=1)
        local_tris = triangles[
            np.all(tri_hi >= lo, axis=1) & np.all(tri_lo <= hi, axis=1)
        ]
        x, y = samples[:, 0, None], samples[:, 1, None]
        x1, y1 = local_tris[None, :, 0, 0], local_tris[None, :, 0, 1]
        x2, y2 = local_tris[None, :, 1, 0], local_tris[None, :, 1, 1]
        x3, y3 = local_tris[None, :, 2, 0], local_tris[None, :, 2, 1]
        den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        valid = np.abs(den) > 1e-10
        safe = np.where(valid, den, 1.0)
        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / safe
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / safe
        c = 1.0 - a - b
        inside = valid & (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        z = (a * local_tris[None, :, 0, 2] + b * local_tris[None, :, 1, 2]
             + c * local_tris[None, :, 2, 2])
        z = np.where(inside, z, -np.inf)
        front = z.max(axis=1)
        cutoff = float(samples[:, 2].max() - 5.0)
        covered = np.isfinite(front) & (front > cutoff)
        clearance = np.where(covered, samples[:, 2] - front, -np.inf)
        assert clearance.max() < 0.02, f"socket protrudes at jaw {level}"


def test_mouthed_assembly_is_deterministic(tmp_path):
    first = _assemble_example(tmp_path)
    second = _assemble_example(tmp_path)
    assert first == second


def test_strip_skins_to_the_pose_correct_socket(rig, mouth_data):
    """The 2d-review artifact fix, held permanently: skinning the baked
    strip open must land near the per-pose rebuilt socket (a rest-built
    strip measured 4 cm of deviation, standing as ridges through the
    jaw-following anatomy). Deep interior exact at the full-open
    reference; the rest-shaped cuff accounts for the small residual."""
    import json as jsonlib

    from character_factory.assembly.mouth import (
        _jaw_rotation,
        build_socket,
        export_strip,
        jaw_subtree_weights,
        skin_jaw,
    )

    document = jsonlib.loads((EXAMPLES / "storyteller.char.json").read_text())
    body = document["body"]
    evaluation = rig.evaluate(body["identity"], body["resting_expression"],
                              proportions=body.get("proportions"))
    strip = export_strip(rig, mouth_data, evaluation)
    pivot = evaluation.skeleton[rig.joint_index("c_jaw"), :3]
    strip_jaw = jaw_subtree_weights(rig, strip.joints, strip.weights)
    body_jaw = jaw_subtree_weights(rig, rig.vertex_joints, rig.vertex_weights)
    for level, bound_cm in ((0.5, 1.6), (1.0, 1.2)):
        rotation, _ = _jaw_rotation(mouth_data, pivot, level)
        skinned = skin_jaw(strip.vertices, strip_jaw, rotation, pivot)
        posed = skin_jaw(evaluation.vertices, body_jaw, rotation, pivot)
        rebuilt, _ = build_socket(posed, mouth_data)
        deviation = np.linalg.norm(skinned - rebuilt.vertices, axis=1)
        assert deviation.max() < bound_cm, f"jaw {level}: {deviation.max():.2f} cm"


def test_corner_seam_duplicates_are_welded(exported):
    """The rig's corner seam-duplicate vertices carry different weights and
    tear ~1.5 mm apart under the jaw once the portal is removed; the export
    welds each pair (and the strip's corner columns) to one averaged set."""
    from character_factory.assembly.gltf import read_accessor

    gltf, binary, body = _body_prim(exported["mouth-interior"])
    prim = body["primitives"][0]
    positions = read_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(np.float64)
    joints4 = read_accessor(gltf, binary, prim["attributes"]["JOINTS_0"]).astype(np.int64)
    weights4 = read_accessor(gltf, binary, prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    # Every exported copy of each pair (the lip corner and its seam
    # duplicate) must share one influence set.
    document = json.loads((EXAMPLES / "storyteller.char.json").read_text())
    body_block = document["body"]
    rest = rig_module_rest(body_block)
    for pair in ((5463, 5462), (2577, 2576)):
        cluster = np.concatenate([
            np.where(np.linalg.norm(positions - rest[v] * 0.01, axis=1) < 1e-6)[0]
            for v in pair
        ])
        assert len(cluster) >= 2
        influence = {(int(j), round(float(w), 5))
                     for j, w in zip(joints4[cluster[0]], weights4[cluster[0]]) if w > 0}
        for vertex in cluster[1:]:
            got = {(int(j), round(float(w), 5))
                   for j, w in zip(joints4[vertex], weights4[vertex]) if w > 0}
            assert got == influence


_REST_CACHE = {}


def rig_module_rest(body_block):
    key = json.dumps(body_block, sort_keys=True)[:64]
    if key not in _REST_CACHE:
        from character_factory.assembly.rig import load_rig

        rig = load_rig(_real_rig_dir())
        _REST_CACHE[key] = rig.evaluate(
            body_block["identity"], body_block["resting_expression"],
            proportions=body_block.get("proportions"),
        ).vertices
    return _REST_CACHE[key]
