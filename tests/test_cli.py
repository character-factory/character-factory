"""The command line: validate exit codes and messages, assemble errors."""

import json
from pathlib import Path

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


def test_interpret_rules_prints_decomposition_json_with_metrics(capsys):
    assert main([
        "interpret", "--rules",
        "a 19 year old japanese girl wearing a croptop, denim shorts, "
        "and flip flops",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "rules-fallback"
    assert "flip" in payload["textures"]["shoe"]["prompt"]
    assert "flip" not in payload["textures"]["eye"]["prompt"]
    assert payload["hair"]["family"]
    assert payload["metrics"]["wall_seconds"] >= 0


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
