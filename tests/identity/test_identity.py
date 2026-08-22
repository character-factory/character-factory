"""Identity generation against a synthetic fixture component.

Real weights are registry data and unpublished during development, so these
tests build a component with the documented artifact format and random
(seeded) weights: they prove the loader, the composition into full parameter
vectors, and determinism — everything except the learned mapping itself.
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
from character_factory.identity.model import IdentityNetwork  # noqa: E402

BODY_INDICES = list(range(20)) + list(range(40, 45))
FACE_INDICES = list(range(20, 40))
EYELID_INDICES = [12, 13, 14, 15, 20, 21]
EMBED_DIM = 64  # small for tests; the config drives everything


def write_component(directory, *, base_identity=None, seed=0):
    config = {
        "format": COMPONENT_FORMAT,
        "component_version": "0.0.0+test",
        "embedding": {
            "pooling": "masked_mean",
            "normalize": "l2",
            "max_tokens": 128,
            "dimensions": EMBED_DIM,
        },
        "architecture": {"kind": "dual-expert-residual", "hidden": 16, "blocks": 2},
        "heads": {
            "body": {"identity_indices": BODY_INDICES},
            "face": {"identity_indices": FACE_INDICES},
            "eyelid": {"expression_indices": EYELID_INDICES},
        },
        "identity_size": 45,
        "expression_size": 72,
        "base_identity": base_identity,
    }
    torch.manual_seed(seed)
    model = IdentityNetwork(EMBED_DIM, 16, 2, 25, 20, 6)
    tensors = dict(model.state_dict())
    generator = torch.Generator().manual_seed(seed + 1)
    for head, size in (("body", 25), ("face", 20), ("eyelid", 6)):
        tensors[f"stats.{head}.mean"] = torch.randn(size, generator=generator) * 0.1
        tensors[f"stats.{head}.std"] = torch.rand(size, generator=generator) + 0.5
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


def test_shapes_and_types(generator):
    result = generator.generate("a tall broad-shouldered person")
    identity, expression = result.identity, result.resting_expression
    assert len(identity) == 45 and len(expression) == 72
    assert all(isinstance(v, float) for v in identity + expression)
    assert result.proportions == {}  # this component does not own them


def test_deterministic_no_seed_anywhere(generator):
    a = generator.generate("a lean marathon runner")
    b = generator.generate("a lean marathon runner")
    assert a.identity == b.identity
    assert a.resting_expression == b.resting_expression
    assert a.proportions == b.proportions


def test_different_prompts_differ(generator):
    a = generator.generate("a lean marathon runner").identity
    b = generator.generate("a heavyset dockworker").identity
    assert a != b


def test_owned_positions_only(tmp_path):
    base = [7.5] * 45
    component = IdentityComponent.load(write_component(tmp_path, base_identity=base))
    generator = IdentityGenerator(component, fake_embedder)
    result = generator.generate("anyone")
    identity, expression = result.identity, result.resting_expression
    # Body and face heads own all 45 identity positions between them, so no
    # base value survives — but expression positions outside the eyelid head
    # must remain exactly zero.
    owned = set(EYELID_INDICES)
    assert all(
        expression[i] == 0.0 for i in range(72) if i not in owned
    )
    assert any(expression[i] != 0.0 for i in owned)
    assert all(v != 7.5 for v in identity)  # every position was written


def test_base_identity_survives_unowned_positions(tmp_path):
    # A hypothetical narrower component: body head owns only index 0.
    directory = write_component(tmp_path)
    config = json.loads((directory / "config.json").read_text())
    config["heads"]["body"]["identity_indices"] = [0]
    config["heads"]["face"]["identity_indices"] = [1]
    config["base_identity"] = [3.25] * 45
    (directory / "config.json").write_text(json.dumps(config))
    # Rebuild weights to match the narrower heads.
    torch.manual_seed(3)
    model = IdentityNetwork(EMBED_DIM, 16, 2, 1, 1, 6)
    tensors = dict(model.state_dict())
    for head, size in (("body", 1), ("face", 1), ("eyelid", 6)):
        tensors[f"stats.{head}.mean"] = torch.zeros(size)
        tensors[f"stats.{head}.std"] = torch.ones(size)
    safetensors_torch.save_file(tensors, str(directory / "weights.safetensors"))

    generator = IdentityGenerator(IdentityComponent.load(directory), fake_embedder)
    identity = generator.generate("anyone").identity
    assert identity[2:] == [3.25] * 43
    assert identity[0] != 3.25 and identity[1] != 3.25


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


def test_destandardization_applied(tmp_path):
    """With std=1 and a large mean, outputs must sit near the mean."""
    directory = write_component(tmp_path)
    tensors = safetensors_torch.load_file(str(directory / "weights.safetensors"))
    tensors["stats.body.mean"] = torch.full((25,), 100.0)
    tensors["stats.body.std"] = torch.ones(25)
    safetensors_torch.save_file(tensors, str(directory / "weights.safetensors"))
    generator = IdentityGenerator(IdentityComponent.load(directory), fake_embedder)
    identity = generator.generate("anyone").identity
    body_values = [identity[i] for i in BODY_INDICES]
    assert all(50.0 < v < 150.0 for v in body_values)


PROPORTION_NAMES = (
    "spine_length", "neck_length", "shoulder_width",
    "arm_length", "hip_width", "leg_length",
)


def write_proportions_component(directory, *, seed=0, prop_mean=0.0, prop_std=1.0):
    """A component of the newer generation: proportions head on the body
    trunk, no eyelid head (resting expression is exact zeros)."""
    config = {
        "format": COMPONENT_FORMAT,
        "component_version": "0.0.0+test",
        "embedding": {"pooling": "masked_mean", "normalize": "l2",
                      "max_tokens": 128, "dimensions": EMBED_DIM},
        "architecture": {"kind": "dual-expert-residual", "hidden": 16, "blocks": 2},
        "heads": {
            "body": {"identity_indices": BODY_INDICES},
            "face": {"identity_indices": FACE_INDICES},
            "proportions": {"parameters": list(PROPORTION_NAMES), "bound": 0.40},
        },
        "identity_size": 45,
        "expression_size": 72,
        "base_identity": None,
    }
    torch.manual_seed(seed)
    model = IdentityNetwork(EMBED_DIM, 16, 2, 25, 20,
                            eyelid_size=0, proportion_size=6)
    tensors = dict(model.state_dict())
    generator = torch.Generator().manual_seed(seed + 1)
    for head, size in (("body", 25), ("face", 20)):
        tensors[f"stats.{head}.mean"] = torch.randn(size, generator=generator) * 0.1
        tensors[f"stats.{head}.std"] = torch.rand(size, generator=generator) + 0.5
    tensors["stats.proportions.mean"] = torch.full((6,), float(prop_mean))
    tensors["stats.proportions.std"] = torch.full((6,), float(prop_std))
    (directory / "config.json").write_text(json.dumps(config))
    safetensors_torch.save_file(tensors, str(directory / "weights.safetensors"))
    return directory


def test_proportions_component_emits_named_bounded_values(tmp_path):
    component = IdentityComponent.load(write_proportions_component(tmp_path))
    generator = IdentityGenerator(component, fake_embedder)
    result = generator.generate("a towering broad-shouldered smith")
    assert len(result.identity) == 45
    # No eyelid head: resting expression is exact zeros, by construction.
    assert result.resting_expression == [0.0] * 72
    assert set(result.proportions) == set(PROPORTION_NAMES)
    assert all(abs(v) <= 0.40 for v in result.proportions.values())


def test_proportions_bound_clamps_extreme_outputs(tmp_path):
    # A destandardization mean of 5.0 pushes every raw output far outside
    # the bound; the generator clamps to the component's declared bound.
    component = IdentityComponent.load(
        write_proportions_component(tmp_path, prop_mean=5.0)
    )
    generator = IdentityGenerator(component, fake_embedder)
    result = generator.generate("anyone")
    assert all(v == 0.40 for v in result.proportions.values())
