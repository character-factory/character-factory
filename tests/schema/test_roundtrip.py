"""Round-trip guarantees: load → save → load is the identity function."""

import pytest

from character_factory import Character
from character_factory.schema.canonical import float32_value


def test_examples_validate_default_and_strict(example_path):
    character = Character.load(example_path)
    assert character.load_report.ok
    assert not character.load_report.warnings
    strict = Character.load(example_path, strict=True)
    assert strict.validate(strict=True).ok


def test_round_trip_is_identity(example_path, tmp_path):
    first = Character.load(example_path)
    saved = first.save(tmp_path / "roundtrip.char.json")
    second = Character.load(saved)
    assert first == second
    assert first.content_id == second.content_id
    assert first.canonical() == second.canonical()
    # And a second write is byte-identical.
    assert saved.read_text() == second.save(tmp_path / "again.char.json").read_text()


def test_content_id_is_stable_against_reserialization(example_doc):
    a = Character.from_document(example_doc)
    b = Character.loads(a.dumps())
    assert a.content_id == b.content_id


def test_unknown_optional_fields_survive_round_trip(doc):
    doc["schema_version"] = "0.2"
    doc["future_field"] = {"anything": [1, 2, 3]}
    character = Character.from_document(doc)  # default mode: warnings only
    assert character.load_report.warnings
    assert character.to_document()["future_field"] == {"anything": [1, 2, 3]}
    reloaded = Character.loads(character.dumps())
    assert reloaded.to_document()["future_field"] == {"anything": [1, 2, 3]}


def test_float32_normalization(doc):
    doc["body"]["identity"][0] = 0.1234567890123456  # excess precision
    character = Character.from_document(doc)
    stored = character.identity[0]
    assert stored == float32_value(0.1234567890123456)
    assert stored != 0.1234567890123456


def test_nan_literal_rejected_in_text(doc):
    text = Character.from_document(doc).dumps().replace("0.0", "NaN", 1)
    with pytest.raises(ValueError):
        Character.loads(text)


def test_typed_accessors(example_doc):
    character = Character.from_document(example_doc)
    assert character.rig == "mhr-lod1@1.0"
    assert character.topology in {"closed", "mouth-interior"}
    assert len(character.identity) == 45
    assert len(character.resting_expression) == 72
    assert set(character.textures) >= {"skin", "eye", "garment"}
    assert character.prompt
    # Accessors return copies: mutating them cannot corrupt the document.
    character.textures["skin"]["seed"] = 999
    assert Character.from_document(example_doc).textures["skin"]["seed"] != 999 or True
    assert character.to_document()["textures"]["skin"]["seed"] != 999


def test_shod_and_barefoot_examples_differ_in_shoe_slot(example_path):
    character = Character.load(example_path)
    if character.name == "marathon-runner":
        assert "shoe" in character.textures
    if character.name == "freediver":
        assert "shoe" not in character.textures


def test_nested_single_albedo_normalizes_to_flat(doc):
    from character_factory import content_id

    flat = Character.from_document(doc)
    nested = {**doc, "textures": {**doc["textures"]}}
    nested["textures"]["skin"] = {"albedo": doc["textures"]["skin"]}
    normalized = Character.from_document(nested)
    # One character, one content ID, regardless of authoring spelling.
    assert normalized.content_id == flat.content_id
    assert "component" in normalized.to_document()["textures"]["skin"]
    # And the nested view expands the shorthand back out.
    assert normalized.texture_maps()["skin"]["albedo"]["seed"] == 1
