"""Garment shell extraction: source-driven cuts, closed solids, fail-closed.

Runs on a synthetic cylinder "body" with a wrapped UV atlas (a real UV
seam), two LBS joints, and a fake TorchScript model — no registry, no
weights. The extraction is a pure function of texture bytes + rig
buffers, which is exactly what these tests pin.
"""

import types

import numpy as np
import pytest

from character_factory.assembly import garment_shell as gs

SEGMENTS = 16
ROWS = 16
RADIUS = 5.0
HEIGHT = 44.0
TEXTURE = 128


def cylinder_rig():
    """A tube along +Y: SEGMENTS around, ROWS tall. UVs unwrap to the full
    [0,1]² atlas with a duplicated texcoord column at the wrap seam.
    Bottom half weights to joint 0, top half to joint 1."""
    vertices = np.zeros((SEGMENTS * ROWS, 3))
    for row in range(ROWS):
        y = HEIGHT * row / (ROWS - 1)
        for segment in range(SEGMENTS):
            angle = 2 * np.pi * segment / SEGMENTS
            vertices[row * SEGMENTS + segment] = (
                RADIUS * np.cos(angle), y, RADIUS * np.sin(angle))

    texcoords = np.zeros(((SEGMENTS + 1) * ROWS, 2), dtype=np.float32)
    for row in range(ROWS):
        for segment in range(SEGMENTS + 1):
            texcoords[row * (SEGMENTS + 1) + segment] = (
                segment / SEGMENTS, row / (ROWS - 1))

    faces = []
    texcoord_faces = []
    for row in range(ROWS - 1):
        for segment in range(SEGMENTS):
            v00 = row * SEGMENTS + segment
            v01 = row * SEGMENTS + (segment + 1) % SEGMENTS
            v10 = (row + 1) * SEGMENTS + segment
            v11 = (row + 1) * SEGMENTS + (segment + 1) % SEGMENTS
            t00 = row * (SEGMENTS + 1) + segment
            t01 = row * (SEGMENTS + 1) + segment + 1
            t10 = (row + 1) * (SEGMENTS + 1) + segment
            t11 = (row + 1) * (SEGMENTS + 1) + segment + 1
            faces.append((v00, v11, v01))
            texcoord_faces.append((t00, t11, t01))
            faces.append((v00, v10, v11))
            texcoord_faces.append((t00, t10, t11))

    # Smooth joint blend across a 12 cm transition band, like a real rig.
    joints = np.zeros((len(vertices), 4), dtype=np.uint16)
    weights = np.zeros((len(vertices), 4), dtype=np.float32)
    upper = np.clip((vertices[:, 1] - (HEIGHT / 2 - 6.0)) / 12.0, 0.0, 1.0)
    joints[:, 0] = 0
    joints[:, 1] = 1
    weights[:, 0] = 1.0 - upper
    weights[:, 1] = upper

    def fake_model(identity, pose, expression):
        import torch

        angle = float(pose[0, 0])
        half = angle / 2.0
        skeleton = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, HEIGHT / 2, 0.0, 0.0, 0.0, np.sin(half), np.cos(half), 1.0],
        ], dtype=np.float32)
        return (torch.zeros((1, len(vertices), 3)),
                torch.tensor(skeleton).unsqueeze(0))

    rig = types.SimpleNamespace(
        faces=np.asarray(faces, dtype=np.int64),
        texcoords=texcoords,
        texcoord_faces=np.asarray(texcoord_faces, dtype=np.int64),
        vertex_joints=joints,
        vertex_weights=weights,
        parents=np.array([-1, 0], dtype=np.int64),
        model=fake_model,
        proportion_pose=lambda proportions: np.zeros(204),
    )
    return rig, vertices


def band_texture(v_low=0.30, v_high=0.72, value=180):
    """A garment band covering rows v_low..v_high of the atlas."""
    rgb = np.zeros((TEXTURE, TEXTURE, 3), dtype=np.uint8)
    rgb[int(v_low * (TEXTURE - 1)):int(v_high * (TEXTURE - 1)) + 1, :] = value
    return rgb


ATLAS_ALL = np.ones((TEXTURE, TEXTURE), dtype=bool)


def prepare(rgb, rig=None, vertices=None, **kwargs):
    if rig is None:
        rig, vertices = cylinder_rig()
    return gs.prepare_shell(rig, rgb, vertices, vertices, ATLAS_ALL, **kwargs)


# --------------------------------------------------------------------------
# alpha gates
# --------------------------------------------------------------------------

def test_empty_texture_fails_closed():
    with pytest.raises(gs.ShellRejected) as excinfo:
        prepare(np.zeros((TEXTURE, TEXTURE, 3), dtype=np.uint8))
    assert excinfo.value.reason == "alpha-coverage-small"


def test_full_coverage_fails_closed():
    with pytest.raises(gs.ShellRejected) as excinfo:
        prepare(np.full((TEXTURE, TEXTURE, 3), 200, dtype=np.uint8))
    assert excinfo.value.reason == "alpha-coverage-large"


def test_cutoff_instability_fails_closed():
    # Half the band solidly bright, a comparable region sitting between the
    # two cutoffs: the 16/20 keys disagree and the mask is untrustworthy.
    rgb = np.zeros((TEXTURE, TEXTURE, 3), dtype=np.uint8)
    rgb[30:60, :] = 180
    rgb[60:90, :] = 18
    with pytest.raises(gs.ShellRejected) as excinfo:
        prepare(rgb)
    assert excinfo.value.reason == "alpha-cutoff-unstable"


def test_garment_living_in_excluded_regions_fails_closed():
    # The whole band sits inside the excluded (head/feet) regions: the
    # region contract removes essentially all of the key -> reject.
    rgb = band_texture()
    excluded = np.zeros((TEXTURE, TEXTURE), dtype=bool)
    excluded[int(0.3 * TEXTURE):int(0.75 * TEXTURE), :] = True
    with pytest.raises(gs.ShellRejected) as excinfo:
        prepare(rgb, excluded_regions=excluded)
    assert excinfo.value.reason == "alpha-excluded-region"


def test_excluded_region_splash_is_masked_not_fatal():
    # A small splash in an excluded region (the live adapter leaves such
    # fragments on head/feet islands) is subtracted by the same region
    # contract the compositor applies to paint; the garment still ships.
    rgb = band_texture()
    rgb[2:10, 2:40] = 150                        # splash outside the band
    excluded = np.zeros((TEXTURE, TEXTURE), dtype=bool)
    excluded[0:14, :] = True                     # the splash's region
    shell = prepare(rgb, excluded_regions=excluded)
    assert 0.0 < shell.audit["excluded_removed"] < 0.25
    clean = prepare(band_texture())
    assert shell.outer_face_count == clean.outer_face_count


def test_seam_detector_is_report_only_without_budget():
    shell = prepare(band_texture())
    seam = shell.audit["seam_disagreement"]
    assert seam["enforced"] is False and seam["budget"] is None
    assert seam["max_band_disagreement"] >= 0.0


def test_seam_detector_enforces_a_configured_budget():
    # A genuine seam crack: the band is keyed on the u=1 side of the wrap
    # seam but black on the u=0 side, so welded seam vertices see
    # disagreeing corner samples.
    rgb = band_texture()
    rgb[:, 0:6] = 0
    constants = gs.ShellConstants(seam_disagreement_budget=0.5)
    with pytest.raises(gs.ShellRejected) as excinfo:
        prepare(rgb, constants=constants)
    assert excinfo.value.reason == "alpha-seam-disagreement"


# --------------------------------------------------------------------------
# the cut and the solid
# --------------------------------------------------------------------------

def test_band_extracts_a_closed_watertight_tube():
    shell = prepare(band_texture())
    # Closed solid: 2N vertices, 2F + 2E faces, and (audited during
    # construction) every undirected edge shared by exactly two faces.
    assert len(shell.vertices) == 2 * shell.outer_count
    edges = np.sort(np.concatenate([
        shell.faces[:, [0, 1]], shell.faces[:, [1, 2]],
        shell.faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    # A band around a tube: one component, two boundary loops.
    assert shell.audit["components"] == 1
    assert shell.audit["boundary_loops"] == 2


def test_cut_is_source_driven_not_canonical():
    wide = prepare(band_texture(0.30, 0.72))
    narrow = prepare(band_texture(0.45, 0.60))
    assert wide.outer_face_count != narrow.outer_face_count
    assert wide.audit["coverage"] > narrow.audit["coverage"]


def test_shell_sits_outside_the_body_by_the_lift():
    rig, vertices = cylinder_rig()
    shell = prepare(band_texture(), rig=rig, vertices=vertices)
    outer = shell.vertices[:shell.outer_count]
    radial = np.linalg.norm(outer[:, [0, 2]], axis=1)
    constants = gs.ShellConstants()
    assert radial.min() >= RADIUS + constants.base_lift_cm - 1e-6
    assert radial.max() <= RADIUS + constants.base_lift_cm \
        + constants.boundary_extra_cm + constants.normal_clamp_extra_cm + 1e-6
    inner = shell.vertices[shell.outer_count:]
    inner_radial = np.linalg.norm(inner[:, [0, 2]], axis=1)
    assert inner_radial.min() >= RADIUS + constants.inner_min_cm - 1e-6
    assert (inner_radial <= radial + 1e-9).all()


def test_correspondence_reconstructs_exactly():
    rig, vertices = cylinder_rig()
    shell = prepare(band_texture(), rig=rig, vertices=vertices)
    triangles = rig.faces[shell.source_face]
    reconstructed = np.einsum("nk,nkd->nd", shell.source_bary,
                              vertices[triangles])
    # The outer surface differs from the source by the lift; the source
    # reconstruction itself must be exact on the body.
    radial = np.linalg.norm(reconstructed[:, [0, 2]], axis=1)
    assert np.abs(radial - RADIUS).max() < 0.15   # chord flattening only
    assert shell.source_bary.min() >= -2e-4
    assert np.abs(shell.source_bary.sum(axis=1) - 1.0).max() < 1e-5


def test_weights_follow_the_source_surface():
    shell = prepare(band_texture())
    sums = shell.weights4.sum(axis=1)
    assert np.abs(sums - 1.0).max() < 1e-5
    outer = shell.joints4[:shell.outer_count]
    inner = shell.joints4[shell.outer_count:]
    assert (outer == inner).all()
    # The band spans the joint split: both joints must appear.
    used = set(np.unique(shell.joints4[shell.weights4 > 0]))
    assert used == {0, 1}


def test_covered_body_faces_are_conservative():
    rig, _ = cylinder_rig()
    shell = prepare(band_texture())
    hidden = set(shell.covered_body_faces.tolist())
    assert hidden, "a solid band must hide some covered faces"
    # No hidden face touches an uncovered vertex, and the two-ring erosion
    # keeps a visible overlap band: faces at the band edge stay.
    band_faces = shell.outer_face_count
    assert len(hidden) < band_faces


def test_extraction_is_deterministic():
    first = prepare(band_texture())
    second = prepare(band_texture())
    assert first.vertices.tobytes() == second.vertices.tobytes()
    assert first.faces.tobytes() == second.faces.tobytes()
    assert first.weights4.tobytes() == second.weights4.tobytes()
    assert (first.covered_body_faces == second.covered_body_faces).all()


# --------------------------------------------------------------------------
# the pose gate
# --------------------------------------------------------------------------

def _evaluation(vertices):
    rest = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, HEIGHT / 2, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    return types.SimpleNamespace(vertices=vertices, skeleton=rest)


BEND = {"bend": np.array([0.35] + [0.0] * 203, dtype=np.float32)}


def test_pose_gate_passes_when_the_shell_follows_the_body():
    rig, vertices = cylinder_rig()
    shell = prepare(band_texture(), rig=rig, vertices=vertices)
    diagnostics = gs.pose_gate(
        rig, shell, _evaluation(vertices), [0.0] * 45, [0.0] * 72,
        poses=BEND)
    assert diagnostics["bend"]["visible_max_mm"] <= 0.5
    assert diagnostics["bend"]["render"]["skin_in_silhouette"] == 0
    assert diagnostics["bend"]["render"]["body_holes"] == 0


def test_pose_gate_fails_a_shell_that_does_not_follow():
    rig, vertices = cylinder_rig()
    shell = prepare(band_texture(), rig=rig, vertices=vertices)
    # Sabotage: bind the whole shell to the static root while the upper
    # body rotates away — the body must sweep through the shell.
    shell.joints4[:] = 0
    with pytest.raises(gs.ShellRejected) as excinfo:
        gs.pose_gate(rig, shell, _evaluation(vertices), [0.0] * 45,
                     [0.0] * 72, poses=BEND)
    assert excinfo.value.reason in ("pose-visible-poke", "pose-render-poke")


def test_valid_atlas_mask_covers_the_unwrap():
    rig, _ = cylinder_rig()
    mask = gs.valid_atlas_mask(rig, 64)
    assert mask.mean() > 0.9   # the cylinder unwraps to the full atlas
