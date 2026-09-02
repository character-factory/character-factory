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
        "figure": {"prompt": "a slight young woman, five foot three, "
                             "slim build, soft oval face"},
        "textures": {
            "skin": {"prompt": "fair light skin, MST 2, 19 years old"},
            "eye": {"prompt": "dark brown iris, off-white sclera"},
            "garment": {"prompt": "white crop top, denim shorts"},
            "shoe": {"prompt": "simple rubber flip flops"},
        },
        "hair": full_hair(),
    }


def backend_with(output, config=None) -> ModelInterpreter:
    # Single mode: one instruction, one whole document from the generator.
    return ModelInterpreter(
        config or InterpreterConfig(model="test", mode="single"),
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

    def broken_interpret(self, prompt, *args, **kwargs):
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
    assert set(strict_schema["required"]) == {
        "figure", "textures", "hair", "proportions"}
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
    # The file's default leads and is marked; no synthetic "default" row
    # when a configured alias is the default.
    assert [row["alias"] for row in listed] == ["local-a", "cloud"]
    assert [row["default"] for row in listed] == [True, False]
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
    assert set(schema["required"]) == {
        "figure", "textures", "hair", "proportions"}
    textures = schema["properties"]["textures"]
    assert set(textures["required"]) == {"skin", "eye", "garment", "shoe"}
    assert textures["properties"]["shoe"]["anyOf"][1] == {"type": "null"}
    proportions = schema["properties"]["proportions"]["anyOf"][0]
    assert set(proportions["required"]) == set(proportions["properties"])


# --- multi-call mode ------------------------------------------------------------


def multi_answers(**overrides) -> dict:
    """What each call of the plan answers, keyed by call name."""
    answers = {
        "figure": {"prompt": "a slight young woman, five foot three, slim build"},
        "skin": {"prompt": "fair cool-toned skin, 19 years old"},
        "eye": {"prompt": "dark brown iris, ivory sclera"},
        "garment": {"prompt": "white cotton crop top, blue denim shorts"},
        "shoe": {"prompt": "flip flops, thin straps, rubber, black"},
        "hair": full_hair(),
        "proportions": {},
    }
    answers.update(overrides)
    return answers


def multi_backend(answers, config=None) -> ModelInterpreter:
    """A generator that answers by the call's instruction: the plan's
    calls are told apart by the words that open them."""
    seen = []

    def generate(instruction, description, schema):
        head = instruction.split("\n", 1)[0].lower()
        for name, cue in (("figure", "body and face shape"), ("skin", "skin"),
                          ("eye", "eyes"), ("garment", "wears on their body"),
                          ("shoe", "footwear"), ("hair", "hairstyle"),
                          ("proportions", "skeletal build")):
            if cue in head:
                seen.append((name, description, schema))
                return json.dumps(answers[name])
        raise AssertionError(f"unrecognized call: {head}")

    backend = ModelInterpreter(
        config or InterpreterConfig(model="test"), generate=generate)
    backend.seen = seen
    return backend


def test_local_model_defaults_to_multi_call_and_assembles_the_document():
    backend = multi_backend(multi_answers(proportions={"leg_length": 25}))
    result = backend.interpret("a 19 year old wearing flip flops")
    assert backend.metrics.mode == "multi"
    assert [name for name, _, _ in backend.seen] == [
        "figure", "skin", "eye", "garment", "shoe", "hair", "proportions"]
    assert all(desc == "a 19 year old wearing flip flops" for _, desc, _ in backend.seen)
    assert result.figure.startswith("a slight young woman")
    assert result.slot_prompts == {
        "skin": "fair cool-toned skin, 19 years old",
        "eye": "dark brown iris, ivory sclera",
        "garment": "white cotton crop top, blue denim shorts",
        "shoe": "flip flops, thin straps, rubber, black",
    }
    assert result.hair["family"] == "loose_long"
    assert result.proportions == {"leg_length": 0.25}
    assert set(backend.metrics.calls) == {
        "figure", "skin", "eye", "garment", "shoe", "hair", "proportions"}


def test_multi_call_schemas_are_per_call_and_empty_shoe_is_dropped():
    backend = multi_backend(multi_answers(shoe={"prompt": ""}))
    result = backend.interpret("a barefoot swimmer")
    schemas = dict((name, schema) for name, _, schema in backend.seen)
    assert schemas["figure"]["required"] == ["prompt"]
    assert "family" in schemas["hair"]["properties"]
    assert "leg_length" in schemas["proportions"]["properties"]
    assert "shoe" not in result.slot_prompts
    assert result.proportions is None


def test_multi_call_skips_the_hair_call_for_a_bald_description():
    backend = multi_backend(multi_answers())
    result = backend.interpret("a bald middle-aged man")
    assert "hair" not in [name for name, _, _ in backend.seen]
    assert result.hair is None
    assert any("bald" in note for note in result.notes)


def test_multi_call_validates_the_assembled_document():
    backend = multi_backend(multi_answers(eye={"prompt": "   "}))
    with pytest.raises(InterpreterError, match="'eye' has no prompt"):
        backend.interpret("someone")


def test_multi_call_plan_carries_the_installed_vocabulary():
    from character_factory.interpreter.multi import build_calls, hair_vocabulary_lines

    calls = {call.name: call for call in build_calls(
        {"shoe": {"styles": ["below_ankle", "tall_boot"]}})}
    assert "flip flops" in calls["shoe"].instruction
    assert "tall boots" in calls["shoe"].instruction
    assert "ankle boots" not in calls["shoe"].instruction
    assert "family: " in hair_vocabulary_lines()
    assert "loose_long" in calls["hair"].instruction
    assert "Never footwear" in calls["garment"].instruction
    # a call that never sees the vocabulary shows every launch style
    assert "mid boots" in {c.name: c for c in build_calls()}["shoe"].instruction


def test_mode_resolution_auto_by_backend_and_env(monkeypatch, tmp_path):
    from character_factory.interpreter import config as configuration

    assert InterpreterConfig(model="x").effective_mode == "multi"
    assert InterpreterConfig(model="x", endpoint="http://h/v1").effective_mode == "single"
    assert InterpreterConfig(model="x", mode="single").effective_mode == "single"
    with pytest.raises(ValueError, match="mode"):
        InterpreterConfig(model="x", mode="triple")

    (tmp_path / "config.json").write_text(json.dumps({"interpreter": {
        "default": "cloud", "mode": "multi",
        "backends": {
            "cloud": {"endpoint": "http://host/v1", "model": "tier-1"},
            "local-a": {"model": "some/weights", "mode": "single"},
        },
    }}))
    monkeypatch.setattr(
        "character_factory.interpreter.config.cache_dir", lambda: tmp_path
    )
    for env in (configuration.ENV_MODEL, configuration.ENV_ENDPOINT,
                configuration.ENV_MODE):
        monkeypatch.delenv(env, raising=False)
    assert configuration.load_interpreter_config().effective_mode == "multi"
    assert configuration.load_interpreter_config("local-a").effective_mode == "single"
    monkeypatch.setenv(configuration.ENV_MODE, "single")
    assert configuration.load_interpreter_config().effective_mode == "single"


def test_unconfigured_installation_resolves_to_the_registry_component(
    monkeypatch, tmp_path
):
    from character_factory.interpreter import config as configuration

    monkeypatch.setattr(
        "character_factory.interpreter.config.cache_dir", lambda: tmp_path
    )
    for env in (configuration.ENV_MODEL, configuration.ENV_ENDPOINT,
                configuration.ENV_MODE):
        monkeypatch.delenv(env, raising=False)
    alias, config = configuration.resolve_interpreter_config()
    assert alias == "default"
    assert config.model == configuration.DEFAULT_MODEL_COMPONENT
    assert config.endpoint is None
    assert config.effective_mode == "multi"


def test_endpoint_multi_mode_uses_strict_per_call_schemas_and_drops_nulls(
    monkeypatch
):
    answers = multi_answers()
    answers["hair"] = {**full_hair(), "color": {"family": "black", "rgb": None}}
    answers["proportions"] = {name: None for name in (
        "spine_length", "neck_length", "shoulder_width", "arm_length",
        "hip_width", "leg_length")}
    requests = []

    class FakeResponse:
        status = 200
        headers = {}

        def __init__(self, content):
            self.content = content

        def read(self):
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(self.content)}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode())
        requests.append(body)
        head = body["messages"][0]["content"].split("\n", 1)[0].lower()
        for name, cue in (("figure", "body and face shape"), ("skin", "skin"),
                          ("eye", "eyes"), ("garment", "wears on their body"),
                          ("shoe", "footwear"), ("hair", "hairstyle"),
                          ("proportions", "skeletal build")):
            if cue in head:
                return FakeResponse(answers[name])
        raise AssertionError(head)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = ModelInterpreter(InterpreterConfig(
        model="served-name", endpoint="http://localhost:1", mode="multi"))
    result = backend.interpret("a 19 year old wearing flip flops")
    assert backend.metrics.mode == "multi"
    assert len(requests) == 7
    schemas = [r["response_format"]["json_schema"]["schema"] for r in requests]
    assert schemas[0]["required"] == ["prompt"]
    assert schemas[0]["additionalProperties"] is False
    hair_schema = schemas[5]
    assert set(hair_schema["required"]) == set(hair_schema["properties"])
    assert result.hair["color"] == {"family": "black"}
    assert result.proportions is None
    assert result.slot_prompts["shoe"].startswith("flip flops")


@pytest.mark.parametrize(
    ("configured", "end_of_turn", "expected"),
    [
        (7, 9, [7, 9]),          # model stops on end-of-text only
        ([7, 9], 9, [7, 9]),     # already listed: no duplicate
        (None, 9, [9]),          # no generation config at all
        (7, None, [7]),          # tokenizer without an EOS
    ],
)
def test_local_generation_stops_on_the_end_of_turn_token(
    configured, end_of_turn, expected
):
    # A model whose generation config stops only on end-of-text while
    # its chat template ends turns with a different token would otherwise
    # emit that token after the closing brace and pad to the budget.
    class Model:
        class generation_config:
            eos_token_id = configured

    class Tokenizer:
        eos_token_id = end_of_turn

    backend = ModelInterpreter(InterpreterConfig(model="x", mode="multi"))
    backend._model, backend._tokenizer = Model(), Tokenizer()
    assert backend._stop_ids() == expected


def test_local_model_load_is_not_charged_to_the_first_call(monkeypatch):
    # A cold start (fetch + load) happens before the generation clock
    # starts: it is reported once as load time, not folded into the first
    # call's latency and the generation total.
    import time as _time

    answers = multi_answers()
    backend = ModelInterpreter(InterpreterConfig(model="x", mode="multi"))
    clock = {"now": 0.0}
    monkeypatch.setattr(_time, "monotonic", lambda: clock["now"])

    def load():
        clock["now"] += 300.0
        backend._model, backend._tokenizer = object(), object()
        backend.metrics.load_seconds = 300.0

    def generate(instruction, description, schema):
        assert backend._model is not None
        clock["now"] += 1.0
        return multi_backend(answers).generate(instruction, description, schema)

    monkeypatch.setattr(backend, "_ensure_loaded", load)
    monkeypatch.setattr(backend, "_generate_local", generate)
    backend.interpret("a slight young woman", vocabulary={})
    assert backend.metrics.load_seconds == 300.0
    assert backend.metrics.calls["figure"] == 1.0
    assert backend.metrics.generate_seconds == 7.0


# --- backend readiness and configuration ------------------------------------


def _isolate_config(monkeypatch, tmp_path, document=None):
    from character_factory.interpreter import config as configuration

    if document is not None:
        (tmp_path / "config.json").write_text(json.dumps(document))
    monkeypatch.setattr(
        "character_factory.interpreter.config.cache_dir", lambda: tmp_path
    )
    for env in (configuration.ENV_MODEL, configuration.ENV_ENDPOINT,
                configuration.ENV_API_KEY, configuration.ENV_MODE):
        monkeypatch.delenv(env, raising=False)
    return configuration


def test_readiness_of_a_local_weights_path(monkeypatch, tmp_path):
    configuration = _isolate_config(monkeypatch, tmp_path, {"interpreter": {
        "backends": {"here": {"model": str(tmp_path / "weights"),
                              "label": "workstation model"},
                     "empty": {"model": str(tmp_path / "empty")}},
    }})
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "config.json").write_text("{}")
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(
        "character_factory.preflight.device_memory", lambda device: 24 * 10**9
    )
    rows = {row["alias"]: row for row in configuration.available_backends()}
    here = rows["here"]
    assert here["ready"] is True and here["reason"] is None
    assert here["download_bytes"] == 0 and here["label"] == "workstation model"
    assert here["vram_bytes"] is None and here["fits"] is None   # undeclared
    assert here["description"] is None                           # a path, not a component
    assert rows["empty"]["ready"] is False
    assert "no model config" in rows["empty"]["reason"]
    # Nothing configured as default: the registry component leads.
    assert rows["default"]["default"] is True and rows["default"]["kind"] == "local-model"
    assert list(rows)[0] == "default"


def test_readiness_of_the_registry_component(monkeypatch, tmp_path):
    configuration = _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path / "cache"))
    from character_factory.registry import Registry

    registry = Registry.default()
    entry = registry.get("interpreter")
    size = sum(artifact["bytes"] for artifact in entry.artifacts)

    monkeypatch.setattr(
        "character_factory.preflight.device_memory", lambda device: 24 * 10**9
    )
    (row,) = configuration.available_backends(registry=registry)
    assert row["alias"] == "default" and row["default"] is True
    assert row["ready"] is False and row["download_bytes"] == size
    assert row["reason"].startswith("weights not downloaded (")
    assert row["vram_bytes"] == entry.inference["peak_vram_bytes"]
    assert row["fits"] is True and row["device_bytes"] == 24 * 10**9
    assert row["description"] == entry.document["description"]

    # A card too small for the model is the more important reason, and one
    # a download would not fix.
    monkeypatch.setattr(
        "character_factory.preflight.device_memory", lambda device: 12 * 10**9
    )
    (row,) = configuration.available_backends(registry=registry)
    assert row["fits"] is False and "of VRAM; 12.0 GB detected" in row["reason"]

    # No CUDA at all; and a CPU device asks no VRAM question.
    monkeypatch.setattr("character_factory.preflight.device_memory", lambda device: None)
    (row,) = configuration.available_backends(registry=registry)
    assert row["fits"] is False and "no CUDA device" in row["reason"]
    (row,) = configuration.available_backends(registry=registry, device="cpu")
    assert row["fits"] is None and "CUDA" not in row["reason"]


def test_endpoint_rows_expose_host_and_key_presence_only(monkeypatch, tmp_path):
    configuration = _isolate_config(monkeypatch, tmp_path, {"interpreter": {
        "default": "cloud",
        "backends": {"cloud": {"endpoint": "https://api.example.test/v1",
                               "model": "tier-1", "api_key": "sk-secret"},
                     "lan": {"endpoint": "http://box.local:8000/v1", "model": "m"}},
    }})
    rows = configuration.available_backends()
    assert [row["alias"] for row in rows] == ["cloud", "lan"]
    cloud, lan = rows
    assert cloud["default"] is True and cloud["ready"] is True
    assert cloud["endpoint_host"] == "api.example.test" and cloud["has_key"] is True
    assert lan["has_key"] is False and lan["mode"] == "single"
    assert "sk-secret" not in json.dumps(rows)
    assert "tier-1" not in json.dumps(rows)


def test_save_backend_writes_a_private_file_and_keeps_the_key(monkeypatch, tmp_path):
    import os
    import stat

    configuration = _isolate_config(monkeypatch, tmp_path)
    configuration.save_backend(
        "cloud", {"endpoint": "https://api.example.test/v1", "model": "m",
                  "api_key": "sk-secret"}, default=True,
    )
    path = tmp_path / "config.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    stored = json.loads(path.read_text())["interpreter"]
    assert stored["default"] == "cloud"
    assert stored["backends"]["cloud"]["api_key"] == "sk-secret"

    # Editing the URL without resubmitting the key keeps it.
    configuration.save_backend(
        "cloud", {"endpoint": "https://api.example.test/v2", "model": "m"}
    )
    stored = json.loads(path.read_text())["interpreter"]
    assert stored["backends"]["cloud"]["endpoint"] == "https://api.example.test/v2"
    assert stored["backends"]["cloud"]["api_key"] == "sk-secret"
    assert stored["default"] == "cloud"          # untouched when not given

    # An empty key removes it; default=False clears the pointer.
    configuration.save_backend(
        "cloud", {"endpoint": "https://api.example.test/v2", "model": "m",
                  "api_key": ""}, default=False,
    )
    stored = json.loads(path.read_text())["interpreter"]
    assert "api_key" not in stored["backends"]["cloud"]
    assert "default" not in stored

    # Unrelated sections of the file survive a write.
    document = json.loads(path.read_text())
    document["registry"] = {"index_url": "https://example.test/index.json"}
    path.write_text(json.dumps(document))
    configuration.save_backend("local", {"model": "interpreter"})
    assert json.loads(path.read_text())["registry"]["index_url"].endswith("index.json")

    configuration.delete_backend("cloud")
    assert "cloud" not in json.loads(path.read_text())["interpreter"]["backends"]
    with pytest.raises(KeyError):
        configuration.delete_backend("cloud")


@pytest.mark.parametrize("alias, values, message", [
    ("default", {"model": "x"}, "reserved"),
    ("Bad Alias", {"model": "x"}, "alias must be"),
    ("c", {"endpoint": "ftp://h/v1"}, "http\\(s\\) URL"),
    ("c", {"endpoint": "not a url"}, "http\\(s\\) URL"),
    ("c", {"api_key": "k"}, "needs an endpoint URL or a model"),
    ("c", {"model": "x", "mode": "fast"}, "mode must be"),
    ("c", {"model": "x", "max_new_tokens": 0}, "positive integer"),
    ("c", {"model": "x", "colour": "red"}, "unknown backend field"),
])
def test_backend_validation_names_the_problem(alias, values, message):
    from character_factory.interpreter import config as configuration

    with pytest.raises(ValueError, match=message):
        configuration.validate_backend(alias, values)
