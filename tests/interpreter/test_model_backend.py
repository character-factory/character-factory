"""The interpreter's model backend, exercised through an injected
generator: parsing, validation, fallback, guidance, and configuration —
everything except the actual language model."""

import json
import stat

import pytest

from character_factory.interpreter import Interpretation, interpret
from character_factory.interpreter.backend import (
    InterpreterError,
    ModelInterpreter,
    build_instruction,
    _parse_json,
)
from character_factory.interpreter.config import InterpreterConfig


def full_hair(**overrides) -> dict:
    hair = {
        "schema_version": 1, "seed": 0, "family": "loose_long",
        "part": {"kind": "center", "side": "wearer_left",
                 "position": "moderate", "extent": "to_crown",
                 "width": "narrow"},
        "hairline": {"height": "natural", "shape": "rounded",
                     "temple_recession": "natural", "sideburns": "natural",
                     "nape": "natural", "irregularity": "natural"},
        "length": {"overall": "mid_back", "cut_line": "soft"},
        "shape": {"volume": "medium", "density": "medium",
                  "texture": "straight", "wave_size": "medium",
                  "wave_strength": "medium", "root_lift": "medium"},
        "drape": {"gravity": "natural", "stiffness": "natural",
                  "shoulder_routing": "split", "body_clearance": "natural"},
        "color": {"family": "black"},
    }
    hair.update(overrides)
    return hair


def good_document() -> dict:
    return {
        "textures": {
            "skin": {"prompt": "fair light skin, MST 2, 19 years old"},
            "eye": {"prompt": "dark brown iris, off-white sclera"},
            "garment": {"prompt": "white crop top, denim shorts"},
            "shoe": {"prompt": "simple rubber flip flops"},
        },
        "hair": full_hair(),
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


def test_repair_drops_rgb_when_a_named_color_family_is_given():
    # The field failure: gray rgb triple alongside a named family — the
    # grammar admits it, the validator rejects it, the repair drops it.
    document = good_document()
    document["hair"]["color"] = {"family": "gray", "rgb": [128, 128, 128]}
    result = backend_with(json.dumps(document)).interpret("an older man")
    assert result.hair["color"] == {"family": "gray"}
    assert any("rgb dropped" in note for note in result.notes)


def test_repair_rescales_custom_rgb_from_255_to_unit_range():
    document = good_document()
    document["hair"]["color"] = {"family": "custom", "rgb": [255, 128, 0]}
    result = backend_with(json.dumps(document)).interpret("someone")
    assert result.hair["color"]["rgb"] == [1.0, 128 / 255, 0.0]


def test_unrepairable_hair_is_an_interpreter_error():
    # An unconstrained backend (the endpoint) can omit whole groups; that
    # must surface as InterpreterError — interpretation never degrades.
    document = good_document()
    document["hair"] = {"schema_version": 1, "seed": 0, "family": "loose_long"}
    with pytest.raises(InterpreterError, match="hair block invalid"):
        backend_with(json.dumps(document)).interpret("someone")


def test_interpret_model_failure_fails_closed_by_default(monkeypatch):
    calls = {}

    def broken_interpret(self, prompt, guidance=None):
        calls["ran"] = True
        raise InterpreterError("synthetic failure")

    monkeypatch.setattr(ModelInterpreter, "interpret", broken_interpret)
    with pytest.raises(InterpreterError, match="synthetic failure"):
        interpret(
            "a tall dockworker wearing a wool coat",
            config=InterpreterConfig(model="test"),
            device="cpu",
        )
    assert calls["ran"]


def test_interpret_without_configuration_is_a_named_error():
    with pytest.raises(InterpreterError, match="no interpreter model"):
        interpret("a lean runner", config=InterpreterConfig(), device="cpu")


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
        json.dumps({"interpreter": {"default": "local-a", "backends": {
            "local-a": {"model": "from-file", "max_new_tokens": 99}}}})
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
        status = 200
        headers = {"x-request-id": "request-1"}

        def read(self):
            return json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(good_document())
                }}]}
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
    response_format = seen["body"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    strict_schema = response_format["json_schema"]["schema"]
    assert set(strict_schema["required"]) == {"textures", "hair", "proportions"}
    assert seen["body"]["max_completion_tokens"] >= 1800
    assert "temperature" not in seen["body"]   # hosted models reject overrides
    assert seen["auth"] == "Bearer k"


def test_endpoint_retries_one_empty_truncated_response_and_audits(
    monkeypatch, tmp_path
):
    calls = 0

    class FakeResponse:
        status = 200
        headers = {"x-request-id": "request-1"}

        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse({
                "model": "endpoint-version",
                "usage": {"completion_tokens": 1800,
                          "completion_tokens_details": {"reasoning_tokens": 1800}},
                "choices": [{"finish_reason": "length",
                             "message": {"content": ""}}],
            })
        return FakeResponse({
            "model": "endpoint-version",
            "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(good_document())
            }}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    audit = tmp_path / "protected" / "interpreter.jsonl"
    backend = ModelInterpreter(InterpreterConfig(
        model="served-name", endpoint="http://localhost:1",
        audit_log=str(audit),
    ))
    result = backend.interpret("the raw prompt")

    assert result.slot_prompts["skin"].startswith("fair")
    assert calls == 2
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["classification"] for row in rows] == [
        "truncated_response", "valid",
    ]
    assert rows[0]["trace_id"] == rows[1]["trace_id"]
    assert rows[0]["attempt"] == 1 and rows[1]["attempt"] == 2
    assert rows[0]["description"] == "the raw prompt"
    assert rows[0]["finish_reason"] == "length"
    assert stat.S_IMODE(audit.stat().st_mode) == 0o600


def test_endpoint_invalid_json_is_precise_safe_and_not_retried(monkeypatch):
    calls = 0

    class FakeResponse:
        status = 200
        headers = {}

        def read(self):
            return json.dumps({
                "choices": [{"finish_reason": "stop", "message": {
                    "content": "private malformed output"
                }}],
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = ModelInterpreter(InterpreterConfig(
        model="served-name", endpoint="http://localhost:1",
    ))
    with pytest.raises(InterpreterError) as excinfo:
        backend.interpret("private prompt")
    error = excinfo.value
    assert calls == 1
    assert error.code == "interpreter_invalid_output"
    assert error.classification == "invalid_json"
    assert error.retryable is True
    assert error.trace_id
    assert "private" not in error.public_message

def test_backend_aliases_resolve_and_list(monkeypatch, tmp_path):
    from character_factory.interpreter import config as configuration

    (tmp_path / "config.json").write_text(json.dumps({"interpreter": {
        "default": "local-a",
        "instruction": "custom header",
        "backends": {
            "local-a": {"model": "some/weights"},
            "cloud": {"endpoint": "http://host/v1", "model": "tier-1",
                      "api_key": "k"},
        },
    }}))
    monkeypatch.setattr(
        "character_factory.interpreter.config.cache_dir", lambda: tmp_path
    )
    for env in (configuration.ENV_MODEL, configuration.ENV_ENDPOINT,
                configuration.ENV_API_KEY):
        monkeypatch.delenv(env, raising=False)

    default = configuration.load_interpreter_config()
    assert default.model == "some/weights"
    assert default.instruction == "custom header"

    cloud = configuration.load_interpreter_config(alias="cloud")
    assert cloud.endpoint == "http://host/v1"

    with pytest.raises(ValueError, match="unknown interpreter backend"):
        configuration.load_interpreter_config(alias="nope")

    listed = configuration.available_backends()
    assert [row["alias"] for row in listed] == ["cloud", "local-a"]
    assert {row["kind"] for row in listed} == {"endpoint", "local-model"}


def test_configured_instruction_replaces_the_header():
    text = build_instruction({}, header="my custom header")
    assert text.startswith("my custom header")
    assert "Texture slots" in text        # the structural surface remains


def test_proportion_fields_convert_from_hundredths_and_drop_zeros():
    document = good_document()
    document["proportions"] = {"leg_length": 25, "shoulder_width": -12,
                               "neck_length": 0}
    result = backend_with(json.dumps(document)).interpret("a towering figure")
    assert result.proportions == {"leg_length": 0.25, "shoulder_width": -0.12}


def test_absent_or_empty_proportions_mean_none():
    assert backend_with(json.dumps(good_document())).interpret("x").proportions is None
    document = good_document()
    document["proportions"] = {}
    assert backend_with(json.dumps(document)).interpret("x").proportions is None


def test_unknown_proportion_field_is_an_interpreter_error():
    document = good_document()
    document["proportions"] = {"leg_lenght": 20}
    with pytest.raises(InterpreterError):
        backend_with(json.dumps(document)).interpret("x")


def test_out_of_range_proportion_is_clamped_with_a_note():
    # An unconstrained backend (the endpoint) can exceed the grammar's
    # range; the repair loop clamps and notes rather than dying.
    document = good_document()
    document["proportions"] = {"leg_length": 90}
    result = backend_with(json.dumps(document)).interpret("x")
    assert result.proportions == {"leg_length": 0.40}
    assert any("clamped" in note for note in result.notes)


def test_interpretation_schema_carries_bounded_integer_proportions():
    from character_factory.interpreter.schema import interpretation_schema

    schema = interpretation_schema()
    proportions = schema["properties"]["proportions"]
    assert "proportions" not in schema["required"]
    field = proportions["properties"]["leg_length"]
    assert field == {"type": "integer", "minimum": -40, "maximum": 40}
    assert set(proportions["properties"]) == {
        "spine_length", "neck_length", "shoulder_width",
        "arm_length", "hip_width", "leg_length",
    }


def test_endpoint_schema_makes_optional_fields_nullable_and_required():
    from character_factory.interpreter.schema import endpoint_interpretation_schema

    schema = endpoint_interpretation_schema()
    assert set(schema["required"]) == {"textures", "hair", "proportions"}
    textures = schema["properties"]["textures"]
    assert set(textures["required"]) == {"skin", "eye", "garment", "shoe"}
    assert textures["properties"]["shoe"]["anyOf"][1] == {"type": "null"}
    proportions = schema["properties"]["proportions"]["anyOf"][0]
    assert set(proportions["required"]) == set(proportions["properties"])
