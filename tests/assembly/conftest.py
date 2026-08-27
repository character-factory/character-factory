"""A tiny synthetic rig so exporter/validator tests run without the real
700 MB component. Seven joints (world root, pelvis, two legs with knees,
head), ten vertices with one deliberate UV seam."""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from character_factory.assembly.rig import RigDefinition  # noqa: E402


class FakeRigModel:
    """Stands in for the TorchScript module: fixed rest output."""

    def __init__(self, vertices, skeleton):
        self.vertices = vertices
        self.skeleton = skeleton

    def __call__(self, identity, pose, expression):
        return (
            torch.tensor(self.vertices[None, ...], dtype=torch.float32),
            torch.tensor(self.skeleton[None, ...], dtype=torch.float32),
        )


JOINTS = ["body_world", "root", "l_upleg", "l_lowleg", "r_upleg", "r_lowleg", "c_head"]
PARENTS = np.array([-1, 0, 1, 2, 1, 4, 1], dtype=np.int64)

# World rest positions (cm): pelvis at 90, hips at 70, knees at 50, head at
# 160. The deliberately diagonal thigh links exercise mirror handling
# without creating an artificial exact half-turn between synthetic joints.
POSITIONS = np.array(
    [
        [0, 0, 0],       # body_world
        [0, 90, 0],      # root
        [10, 70, 0],     # l_upleg
        [30, 50, 0],     # l_lowleg (knee)
        [-10, 70, 0],    # r_upleg
        [-30, 50, 0],    # r_lowleg (knee)
        [0, 160, 0],     # c_head
    ],
    dtype=np.float64,
)

# Vertices: feet (below the knees), hips, head — plus a UV seam on vertex 8.
VERTICES = np.array(
    [
        [10, 5, 2], [12, 5, -2],     # 0,1 left foot
        [-10, 5, 2], [-12, 5, -2],   # 2,3 right foot
        [10, 88, 3], [-10, 88, 3],   # 4,5 hips
        [0, 158, 4], [3, 162, 0],    # 6,7 head
        [0, 120, 5],                 # 8 chest (UV seam vertex)
        [0, 100, 6],                 # 9 belly
    ],
    dtype=np.float64,
)

FACES = np.array(
    [
        [0, 1, 3], [0, 3, 2],        # feet strip
        [4, 5, 9],                   # hips-belly
        [8, 9, 4], [8, 9, 5],        # chest-belly (vertex 8 appears twice…)
        [6, 7, 8],                   # head-chest
    ],
    dtype=np.int64,
)

TEXCOORDS = np.array(
    [
        [0.1, 0.1], [0.2, 0.1], [0.1, 0.2], [0.2, 0.2],
        [0.5, 0.5], [0.6, 0.5], [0.7, 0.7], [0.8, 0.7],
        [0.4, 0.4],   # vertex 8, chart A
        [0.9, 0.9],   # vertex 8, chart B (…with two different texcoords)
        [0.5, 0.6],
    ],
    dtype=np.float32,
)

TEXCOORD_FACES = np.array(
    [
        [0, 1, 3], [0, 3, 2],
        [4, 5, 10],
        [8, 10, 4],   # vertex 8 via chart A
        [9, 10, 5],   # vertex 8 via chart B → seam split
        [6, 7, 8],
    ],
    dtype=np.int64,
)


def make_rig(vertex_joints=None, vertex_weights=None) -> RigDefinition:
    if vertex_joints is None:
        vertex_joints = np.array(
            [
                [3, 0, 0, 0], [3, 0, 0, 0],       # left foot → left knee joint
                [5, 0, 0, 0], [5, 0, 0, 0],       # right foot → right knee joint
                [2, 4, 0, 0], [4, 2, 0, 0],       # hips blend both uplegs
                [6, 0, 0, 0], [6, 0, 0, 0],       # head
                [1, 6, 0, 0], [1, 0, 0, 0],       # chest/belly on root (+head)
            ],
            dtype=np.uint16,
        )
        vertex_weights = np.array(
            [
                [1, 0, 0, 0], [1, 0, 0, 0],
                [1, 0, 0, 0], [1, 0, 0, 0],
                [0.6, 0.4, 0, 0], [0.6, 0.4, 0, 0],
                [1, 0, 0, 0], [1, 0, 0, 0],
                [0.7, 0.3, 0, 0], [1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    skeleton = np.zeros((7, 8))
    skeleton[:, :3] = POSITIONS
    skeleton[:, 6] = 1.0  # identity quaternion (w)
    skeleton[:, 7] = 1.0  # unit scale
    metadata = {
        "format": "character-factory/rig-metadata",
        "topology": {
            "vertices": len(VERTICES),
            "triangles": len(FACES),
            "joints": len(JOINTS),
            "identity_size": 2,
            "pose_size": 3,
            "expression_size": 2,
        },
        "joints": JOINTS,
        "parents": [int(p) for p in PARENTS],
        "roles": {
            "world": "body_world",
            "root": "root",
            "left_knee": "l_lowleg",
            "right_knee": "r_lowleg",
            "head": "c_head",
        },
        "proportions": {
            "parameters": {
                "leg_length": {"channel": 1, "range": [-2.0, 2.0]},
                "spine_length": {"channel": 2, "range": [-2.0, 2.0]},
            },
        },
    }
    return RigDefinition(
        model=FakeRigModel(VERTICES, skeleton),
        metadata=metadata,
        faces=FACES,
        texcoords=TEXCOORDS,
        texcoord_faces=TEXCOORD_FACES,
        parents=PARENTS,
        vertex_joints=vertex_joints,
        vertex_weights=vertex_weights,
    )


@pytest.fixture
def rig() -> RigDefinition:
    return make_rig()
