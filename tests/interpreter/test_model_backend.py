"""The interpreter's model backend, exercised through an injected
generator: parsing, validation, fallback, guidance, and configuration —
everything except the actual language model."""

import json

import pytest

from character_factory.interpreter import Interpretation, interpret
from character_factory.interpreter.backend import (
    InterpreterError,
    ModelInterpreter,
    build_instruction,
    _parse_json,
)
from character_factory.interpreter.config import InterpreterConfig


def good_document() -> dict:
    return {
        "textures": {
            "skin": {"prompt": "fair light skin, MST 2, 19 years old"},
            "eye": {"prompt": "dark brown iris, off-white sclera"},
            "garment": {"prompt": "white crop top, denim shorts"},
            "shoe": {"prompt": "simple rubber flip flops"},
        },
        "hair": {"schema_version": 1, "seed": 0, "family": "loose_long"},
    }


def backend_with(output, config=None) -> ModelInterpreter:
    return ModelInterpreter(
        config or InterpreterConfig(model="test"),
        generate=lambda instruction, description, schema: output,
    )


def test_valid_output_becomes_an_interpretation():
    result = backend_with(json.dumps(good_document())).interpret(
        "a 19 year old wearing flip flops"
    )
    assert isinstance(result, Interpretation)
    assert result.slot_prompts["shoe"] == "simple rubber flip flops"
    assert result.hair["family"] == "loose_long"
    assert result.backend == "injected"


def test_code_fenced_and_prefixed_output_still_parses():
    fenced = "Sure!\n```json\n" + json.dumps(good_document()) + "\n```"
    assert _parse_json(fenced)["textures"]["skin"]["prompt"].startswith("fair")


def test_empty_optional_slot_is_dropped_missing_required_is_an_error():
    document = good_document()
    document["textures"]["shoe"] = {"prompt": "  "}
    result = backend_with(json.dumps(document)).interpret("barefoot person")
    assert "shoe" not in result.slot_prompts

    document = good_document()
    del document["textures"]["eye"]
    with pytest.raises(InterpreterError, match="missing"):
        backend_with(json.dumps(document)).interpret("someone")


def test_unknown_or_plural_slot_key_is_an_error():
    document = good_document()
    document["textures"]["eyes"] = document["textures"].pop("eye")
    with pytest.raises(InterpreterError, match="unknown texture slot"):
        backend_with(json.dumps(document)).interpret("someone")


def test_bald_description_nulls_the_hair_block():
    result = backend_with(json.dumps(good_document())).interpret(
        "a bald middle-aged man"
    )
    assert result.hair is None
    assert any("bald" in note for note in result.notes)


def test_non_json_output_is_an_interpreter_error():
    with pytest.raises(InterpreterError, match="not JSON"):
        backend_with("I could not do that.").interpret("someone")


def test_interpret_falls_back_to_rules_when_the_model_fails(monkeypatch):
    calls = {}

    def broken_interpret(self, prompt, guidance=None):
        calls["ran"] = True
        raise InterpreterError("synthetic failure")

    monkeypatch.setattr(ModelInterpreter, "interpret", broken_interpret)
    interpretation, metrics = interpret(
        "a tall dockworker wearing a wool coat",
        config=InterpreterConfig(model="test"),
        device="cpu",
    )
    assert calls["ran"]
    assert interpretation.backend == "rules-fallback"
    assert any("rules fallback" in note for note in interpretation.notes)
    assert metrics["error"] == "synthetic failure"


def test_interpret_without_configuration_uses_rules_mode():
    interpretation, metrics = interpret(
        "a lean runner", config=InterpreterConfig(), device="cpu"
    )
    assert interpretation.backend == "rules-fallback"
    assert "wall_seconds" in metrics


def test_instruction_carries_slot_guidance_and_slot_hygiene():
    instruction = build_instruction(
        {"skin": "tone words plus a numbered tone code",
         "shoe": "supported styles: below_ankle — stay within this vocabulary"}
    )
    assert "tone words plus a numbered tone code" in instruction
    assert "below_ankle" in instruction
    assert "never mention" in instruction        # cross-slot hygiene
    assert "barefoot" in instruction             # optional-slot omission rule


def test_config_precedence_env_over_file(monkeypatch, tmp_path):
    from character_factory.interpreter import config as configuration

    (tmp_path / "config.json").write_text(
        json.dumps({"interpreter": {"model": "from-file", "max_new_tokens": 99}})
    )
    monkeypatch.setattr(
        "character_factory.interpreter.config.cache_dir", lambda: tmp_path
    )
    loaded = configuration.load_interpreter_config()
    assert loaded.model == "from-file"
    assert loaded.max_new_tokens == 99

    monkeypatch.setenv(configuration.ENV_MODEL, "from-env")
    assert configuration.load_interpreter_config().model == "from-env"


def test_endpoint_backend_speaks_openai_chat(monkeypatch):
    """The optional endpoint backend: request shape out, content in."""
    seen = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(good_document())}}]}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode())
        seen["auth"] = request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = ModelInterpreter(
        InterpreterConfig(model="served-name", endpoint="http://localhost:1",
                          api_key="k"),
    )
    result = backend.interpret("a 19 year old wearing flip flops")
    assert result.slot_prompts["skin"].startswith("fair")
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["model"] == "served-name"
    assert seen["body"]["temperature"] == 0
    assert seen["auth"] == "Bearer k"