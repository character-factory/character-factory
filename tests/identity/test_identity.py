"""Identity generation against a synthetic fixture component.

Real weights are registry data and unpublished during development, so these
tests build a component with the documented artifact format and random
(seeded) weights: they prove the loader, the sampling contract (seeded
reproducibility, cross-seed variation, the center-only diagnostic), and
the composition into full parameter vectors — everything except the
learned mapping itself.
"""

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from character_factory.identity import (  # noqa: E402  (after importorskip)
    COMPONENT_FORMAT,
    IdentityComponent,
    IdentityGenerator,
)

EMBED_DIM = 64  # small for tests; the config drives everything
BODY_INDICES = [0, 1, 2, 3, 4]
FACE_INDICES = [20, 21, 22, 23]
PROPORTION_NAMES = (
    "spine_length", "neck_length", "shoulder_width",
    "arm_length", "hip_width", "leg_length",
)


def write_component(directory, *, seed=0):
    """A synthetic joint-flow component, small enough to sample in
    milliseconds."""
    from character_factory.identity.model import (
        CenterNetwork,
        IdentityFlowNetwork,
    )

    config = {
        "format": COMPONENT_FORMAT,
        "component_version": "0.0.0+test",
        "embedding": {"pooling": "masked_mean", "normalize": "l2",
                      "max_tokens": 128, "dimensions": EMBED_DIM},
        "architecture": {
            "kind": "joint-rectified-flow",
            "hidden": 32,
            "blocks": 2,
            "output_order": ["body", "proportions", "face"],
            "center": {"hidden": 16, "blocks": 1},
            "sampling": {"steps": 4, "guidance": 1.25, "temperature": 0.75},
        },
        "heads": {
            "body": {"identity_indices": BODY_INDICES},
            "face": {"identity_indices": FACE_INDICES},
            "proportions": {"parameters": list(PROPORTION_NAMES), "bound": 0.40},
        },
        "identity_size": 45,
        "expression_size": 72,
        "base_identity": None,
    }
    output_dim = len(BODY_INDICES) + 6 + len(FACE_INDICES)
    torch.manual_seed(seed)
    flow = IdentityFlowNetwork(EMBED_DIM, output_dim, hidden=32, blocks=2)
    center = CenterNetwork(EMBED_DIM, hidden=16, blocks=1,
                           body_size=len(BODY_INDICES),
                           proportion_size=6,
                           face_size=len(FACE_INDICES))
    tensors = {}
    tensors.update({f"flow.{k}": v for k, v in flow.state_dict().items()})
    tensors.update({f"center.{k}": v for k, v in center.state_dict().items()})
    generator = torch.Generator().manual_seed(seed + 1)
    for head, size in (("body", len(BODY_INDICES)), ("proportions", 6),
                       ("face", len(FACE_INDICES))):
        tensors[f"stats.{head}.mean"] = torch.randn(size, generator=generator) * 0.1
        tensors[f"stats.{head}.std"] = torch.rand(size, generator=generator) * 0.1 + 0.05
    (directory / "config.json").write_text(json.dumps(config))
    safetensors_torch.save_file(tensors, str(directory / "weights.safetensors"))
    return directory


def fake_embedder(prompt: str):
    seed = sum(prompt.encode())
    return torch.nn.functional.normalize(
        torch.randn(EMBED_DIM, generator=torch.Generator().manual_seed(seed)), dim=0
    )


@pytest.fixture
def generator(tmp_path):
    component = IdentityComponent.load(write_component(tmp_path))
    return IdentityGenerator(component, fake_embedder)


def test_shapes_types_and_owned_positions(generator):
    result = generator.generate("a lean marathon runner", seed=3)
    identity, expression = result.identity, result.resting_expression
    assert len(identity) == 45 and len(expression) == 72
    assert all(isinstance(v, float) for v in identity + expression)
    assert expression == [0.0] * 72
    owned = set(BODY_INDICES) | set(FACE_INDICES)
    assert all(identity[i] == 0.0 for i in range(45) if i not in owned)
    assert set(result.proportions) == set(PROPORTION_NAMES)
    assert all(abs(v) <= 0.40 for v in result.proportions.values())


def test_same_seed_reproduces(generator):
    a = generator.generate("a lean marathon runner", seed=41)
    b = generator.generate("a lean marathon runner", seed=41)
    assert a.identity == b.identity
    assert a.proportions == b.proportions


def test_different_seeds_differ(generator):
    a = generator.generate("a lean marathon runner", seed=41)
    b = generator.generate("a lean marathon runner", seed=42)
    assert a.identity != b.identity


def test_different_prompts_differ(generator):
    a = generator.generate("a lean marathon runner", seed=0).identity
    b = generator.generate("a heavyset dockworker", seed=0).identity
    assert a != b


def test_default_seed_is_zero(generator):
    explicit = generator.generate("a lean marathon runner", seed=0)
    default = generator.generate("a lean marathon runner")
    assert explicit.identity == default.identity


def test_seed_none_is_center_only(generator):
    # No generator → the model's semantic center: deterministic, and
    # distinct from any sampled draw.
    a = generator.generate("a lean marathon runner", seed=None)
    b = generator.generate("a lean marathon runner", seed=None)
    sampled = generator.generate("a lean marathon runner", seed=0)
    assert a.identity == b.identity
    assert a.identity != sampled.identity


def test_base_identity_survives_unowned_positions(tmp_path):
    directory = write_component(tmp_path)
    config = json.loads((directory / "config.json").read_text())
    config["base_identity"] = [3.25] * 45
    (directory / "config.json").write_text(json.dumps(config))
    generator = IdentityGenerator(IdentityComponent.load(directory), fake_embedder)
    identity = generator.generate("anyone", seed=5).identity
    owned = set(BODY_INDICES) | set(FACE_INDICES)
    assert all(identity[i] == 3.25 for i in range(45) if i not in owned)
    assert all(identity[i] != 3.25 for i in owned)


def test_destandardization_applied(tmp_path):
    """With std=1 and a large mean, body outputs must sit near the mean."""
    directory = write_component(tmp_path)
    tensors = safetensors_torch.load_file(str(directory / "weights.safetensors"))
    tensors["stats.body.mean"] = torch.full((len(BODY_INDICES),), 100.0)
    tensors["stats.body.std"] = torch.ones(len(BODY_INDICES))
    safetensors_torch.save_file(tensors, str(directory / "weights.safetensors"))
    generator = IdentityGenerator(IdentityComponent.load(directory), fake_embedder)
    identity = generator.generate("anyone", seed=0).identity
    assert all(50.0 < identity[i] < 150.0 for i in BODY_INDICES)


def test_proportions_bound_clamps_extreme_outputs(tmp_path):
    # A destandardization mean far outside the bound clamps to it.
    directory = write_component(tmp_path)
    tensors = safetensors_torch.load_file(str(directory / "weights.safetensors"))
    tensors["stats.proportions.mean"] = torch.full((6,), 5.0)
    tensors["stats.proportions.std"] = torch.full((6,), 0.01)
    safetensors_torch.save_file(tensors, str(directory / "weights.safetensors"))
    generator = IdentityGenerator(IdentityComponent.load(directory), fake_embedder)
    result = generator.generate("anyone", seed=0)
    assert all(v == 0.40 for v in result.proportions.values())


def test_wrong_format_rejected(tmp_path):
    write_component(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["format"] = "something-else"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        IdentityComponent.load(tmp_path)


def test_missing_stats_rejected(tmp_path):
    write_component(tmp_path)
    tensors = safetensors_torch.load_file(str(tmp_path / "weights.safetensors"))
    del tensors["stats.body.mean"]
    safetensors_torch.save_file(tensors, str(tmp_path / "weights.safetensors"))
    with pytest.raises(ValueError):
        IdentityComponent.load(tmp_path)


def test_wrong_output_order_rejected(tmp_path):
    directory = write_component(tmp_path)
    config = json.loads((directory / "config.json").read_text())
    config["architecture"]["output_order"] = ["face", "proportions", "body"]
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        IdentityComponent.load(directory)


def test_unknown_kind_rejected(tmp_path):
    directory = write_component(tmp_path)
    config = json.loads((directory / "config.json").read_text())
    config["architecture"]["kind"] = "dual-expert-residual"
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported identity architecture"):
        IdentityComponent.load(directory)
