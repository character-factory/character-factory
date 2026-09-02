"""The command line: validate exit codes and messages, assemble errors."""

import json
from pathlib import Path

import pytest

from character_factory.cli import main

EXAMPLES = sorted(
    (Path(__file__).parents[1] / "examples" / "characters").glob("*.char.json")
)


def test_validate_examples_pass(capsys):
    assert main(["validate", *map(str, EXAMPLES)]) == 0
    out = capsys.readouterr().out
    assert out.count("ok   ") == len(EXAMPLES)


def test_validate_broken_file_fails(tmp_path, capsys):
    document = json.loads(EXAMPLES[0].read_text())
    document["body"]["identity"] = [0.0] * 3
    bad = tmp_path / "bad.char.json"
    bad.write_text(json.dumps(document))
    assert main(["validate", str(bad)]) == 1
    assert "identity" in capsys.readouterr().out


def test_validate_strict_flags_unknown_fields(tmp_path, capsys):
    document = json.loads(EXAMPLES[0].read_text())
    document["future_field"] = 1
    document["schema_version"] = "0.2"
    path = tmp_path / "future.char.json"
    path.write_text(json.dumps(document))
    assert main(["validate", str(path)]) == 0        # default: warns
    assert "warning" in capsys.readouterr().out
    assert main(["validate", "--strict", str(path)]) == 1


def test_validate_missing_file(capsys):
    assert main(["validate", "does-not-exist.char.json"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_assemble_missing_assets_is_a_clean_error(tmp_path, capsys):
    code = main(
        [
            "assemble", str(EXAMPLES[0]),
            "--assets", str(tmp_path),
            "-o", str(tmp_path / "out.glb"),
        ]
    )
    assert code == 1
    assert "missing asset" in capsys.readouterr().err


def test_interpret_without_configuration_uses_the_registry_default(
    monkeypatch, tmp_path
):
    # Nothing configured: the registry's `interpreter` component is the
    # model. Here its fetch fails (no network in tests), and the error
    # names the component and the way out — never a silent downgrade.
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path))
    for env in ("CHARACTER_FACTORY_INTERPRETER_MODEL",
                "CHARACTER_FACTORY_INTERPRETER_ENDPOINT"):
        monkeypatch.delenv(env, raising=False)
    from character_factory.interpreter.backend import InterpreterError
    from character_factory.registry import Registry

    def refuse(self, *args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(Registry, "ensure", refuse)
    with pytest.raises(InterpreterError, match="'interpreter' is not available") as excinfo:
        main(["interpret", "a lean runner"])
    assert "CHARACTER_FACTORY_INTERPRETER_MODEL" in str(excinfo.value)
    assert excinfo.value.retryable is False


def test_preflight_reports_named_causes(capsys, monkeypatch):
    import character_factory.preflight as preflight_module

    monkeypatch.setattr(
        preflight_module, "GENERATION_IMPORTS",
        (("json", "json"), ("cf_absent_module", "cf-absent-dist")),
    )
    assert main(["preflight", "--device", "cpu"]) == 1
    out = capsys.readouterr().out
    assert "ok   [import:json]" in out
    assert "FAIL [missing-dependency]" in out
    assert "cf-absent-dist" in out


def test_preflight_passes_on_a_working_stack(capsys, monkeypatch):
    import character_factory.preflight as preflight_module

    monkeypatch.setattr(
        preflight_module, "GENERATION_IMPORTS", (("json", "json"),))
    assert main(["preflight", "--device", "cpu"]) == 0
    assert "ok  " in capsys.readouterr().out


def test_make_composes_the_stages_and_prints_the_two_paths(
    tmp_path, capsys, monkeypatch
):
    # `make` is the one-shot path: create → bake → assemble under one
    # directory. The stages are stubbed; what is under test is the wiring —
    # options reach the stages, stage timings go to stderr, and stdout is
    # exactly the two output paths.
    import character_factory.api as api
    from character_factory.textures import BakeResult

    from character_factory import Character

    calls = []
    character = Character.load(EXAMPLES[0])

    def create(prompt, *, seed, registry, device, interpreter, **_):
        calls.append(("create", prompt, seed, device, interpreter))
        return character

    def bake(character, out_dir, *, registry, device, turbo, **_):
        calls.append(("bake", Path(out_dir).name, turbo))
        return BakeResult(character, Path(out_dir), ["skin"])

    def assemble(character, assets_dir, out_path, *, registry, **_):
        calls.append(("assemble", Path(out_path).name))
        return Path(out_path)

    class Registry:
        @staticmethod
        def default():
            return object()

    monkeypatch.setattr(api, "create", create)
    monkeypatch.setattr("character_factory.textures.bake", bake)
    monkeypatch.setattr(api, "assemble", assemble)
    monkeypatch.setattr("character_factory.registry.Registry", Registry)

    out_dir = tmp_path / "professor"
    code = main([
        "make", "a retired astronomy professor", "-o", str(out_dir),
        "--seed", "7", "--backend", "fast", "--turbo",
    ])
    captured = capsys.readouterr()
    assert code == 0
    assert calls == [
        ("create", "a retired astronomy professor", 7, "cuda", "fast"),
        ("bake", "assets", True),
        ("assemble", "scene.glb"),
    ]
    assert captured.out.splitlines() == [
        str(out_dir / "character.char.json"), str(out_dir / "scene.glb"),
    ]
    assert [line.split()[0] for line in captured.err.splitlines()] == [
        "create", "bake", "assemble",
    ]
    assert (out_dir / "character.char.json").exists()


def test_make_on_an_unsuitable_machine_is_a_clean_error(capsys, monkeypatch, tmp_path):
    from character_factory.preflight import PreflightCheck, PreflightError

    def refuse(prompt, **_):
        raise PreflightError([PreflightCheck("torch", False, "CPU-only build")])

    monkeypatch.setattr("character_factory.api.create", refuse)
    monkeypatch.setattr(
        "character_factory.registry.Registry.default", staticmethod(object))
    code = main(["make", "anyone", "-o", str(tmp_path / "out")])
    assert code == 1
    assert "CPU-only" in capsys.readouterr().err
