"""Texture bake orchestration against a fake pipeline (no GPU, no weights):
per-slot seeds, template merge, override precedence, asset pinning."""

import hashlib
import json

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from character_factory.registry import Registry, RegistryIndex  # noqa: E402
from character_factory.schema import Character  # noqa: E402
from character_factory.textures import bake  # noqa: E402


class FakePipeline:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["resolution"], kwargs["resolution"]))
        # Make pixel content depend on the seed so hashes differ per slot.
        image.putpixel((0, 0), (kwargs["seed"] % 256, 0, 0))
        return image


def make_registry(tmp_path):
    def adapter(name, slot, template):
        return {
            "name": name, "version": "0.1.0", "kind": "texture-adapter",
            "slot": slot,
            "requires": {"base_model": "test-base", "schema": ">=0.1 <1.0"},
            "inference": {"prompt_template": template, "steps": 7,
                          "guidance": 3.0, "resolution": 64},
            "artifacts": [], "source": None,
        }

    index = {
        "format": "character-factory/registry",
        "registry_version": "0.1",
        "components": [
            adapter("make-skin", "skin", "skin sheet: {prompt}"),
            adapter("make-eye", "eye", "eye sheet: {prompt}"),
            adapter("make-garment", "garment", "garment sheet: {prompt}"),
            {
                "name": "test-base", "version": "1.0.0", "kind": "base-model",
                "requires": {"schema": ">=0.1 <1.0"}, "artifacts": [], "source": None,
            },
        ],
    }
    registry = Registry(RegistryIndex(index))

    # Pre-place empty component dirs so ensure() succeeds without sources.
    from character_factory.registry.store import component_dir

    for entry in registry.index.entries:
        component_dir(entry).mkdir(parents=True, exist_ok=True)
    return registry


@pytest.fixture
def environment(tmp_path, monkeypatch, doc):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path / "cache"))
    fake = FakePipeline()
    registry = make_registry(tmp_path)
    character = Character.from_document(doc)
    return fake, registry, character




def test_bake_writes_assets_and_pins_hashes(environment, tmp_path):
    fake, registry, character = environment
    result = bake(
        character, tmp_path / "out", registry=registry, device="cpu",
        pipeline_factory=lambda base_dir, device: fake,
    )
    assert result.baked_slots == ["eye", "garment", "skin"]
    for slot in result.baked_slots:
        path = result.assets_dir / f"{slot}.png"
        assert path.is_file()
        pinned = result.character.assets[slot]
        assert pinned["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert pinned["width"] == pinned["height"] == 64
    # The returned character is still valid, strictly.
    assert result.character.validate(strict=True).ok


def test_bake_uses_each_recipes_seed_and_template(environment, tmp_path):
    fake, registry, character = environment
    bake(character, tmp_path / "out", registry=registry,
         pipeline_factory=lambda *a: fake)
    by_prompt = {call["prompt"]: call for call in fake.calls}
    assert by_prompt["skin sheet: medium skin tone, adult"]["seed"] == 1
    assert by_prompt["eye sheet: brown iris"]["seed"] == 2
    assert by_prompt["garment sheet: plain grey t-shirt and jeans"]["seed"] == 3
    assert all(call["steps"] == 7 and call["guidance"] == 3.0 for call in fake.calls)


def test_recipe_overrides_beat_component_defaults(environment, tmp_path):
    fake, registry, character = environment
    document = character.to_document()
    document["textures"]["skin"]["overrides"] = {"steps": 40, "resolution": 128}
    character = Character.from_document(document)
    bake(character, tmp_path / "out", registry=registry,
         pipeline_factory=lambda *a: fake)
    skin_call = next(c for c in fake.calls if c["prompt"].startswith("skin"))
    assert skin_call["steps"] == 40 and skin_call["resolution"] == 128
    assert skin_call["guidance"] == 3.0  # un-overridden default survives


def test_base_model_loaded_once(environment, tmp_path):
    fake, registry, character = environment
    factory_calls = []

    def factory(base_dir, device):
        factory_calls.append(base_dir)
        return fake

    bake(character, tmp_path / "out", registry=registry, pipeline_factory=factory)
    assert len(factory_calls) == 1


def test_assemble_verifies_pinned_hashes(environment, tmp_path):
    """The api.assemble path refuses assets that fail the character's pins."""
    from character_factory.api import AssetError, assemble

    fake, registry, character = environment
    result = bake(character, tmp_path / "assets", registry=registry,
                  pipeline_factory=lambda *a: fake)
    # Tamper with one asset after pinning.
    (tmp_path / "assets" / "skin.png").write_bytes(b"\x89PNG tampered")
    with pytest.raises(AssetError):
        assemble(result.character, tmp_path / "assets", tmp_path / "out.glb",
                 registry=registry)


def test_bake_result_json_is_loadable(environment, tmp_path):
    fake, registry, character = environment
    result = bake(character, tmp_path / "out", registry=registry,
                  pipeline_factory=lambda *a: fake)
    saved = result.character.save(tmp_path / "baked.char.json")
    assert Character.load(saved).assets.keys() == {"skin", "eye", "garment"}
    json.loads(saved.read_text())


def test_quantization_config_env_overrides_file(monkeypatch, tmp_path):
    import json

    from character_factory import textures

    (tmp_path / "config.json").write_text(
        json.dumps({"textures": {"quantization": "int8"}})
    )
    monkeypatch.setattr(
        "character_factory.registry.store.cache_dir", lambda: tmp_path
    )
    assert textures.configured_quantization() == "int8"
    monkeypatch.setenv(textures.ENV_QUANTIZATION, "nf4")
    assert textures.configured_quantization() == "nf4"


def test_quantization_unconfigured_is_full_precision(monkeypatch, tmp_path):
    from character_factory import textures

    monkeypatch.delenv(textures.ENV_QUANTIZATION, raising=False)
    monkeypatch.setattr(
        "character_factory.registry.store.cache_dir", lambda: tmp_path
    )
    assert textures.configured_quantization() is None


def test_unknown_quantization_mode_is_an_error(monkeypatch):
    import pytest

    from character_factory import textures

    monkeypatch.setenv(textures.ENV_QUANTIZATION, "fp3")
    with pytest.raises(ValueError, match="unknown texture quantization"):
        textures.configured_quantization()
