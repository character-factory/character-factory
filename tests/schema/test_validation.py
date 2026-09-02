"""The broken-document corpus (ARCHITECTURE §7): every deliberately broken
document fails with the documented error, at the documented severity."""

import pytest

from character_factory import Character, validate_document
from character_factory.schema.model import CharacterError


def errors_at(report, path_prefix):
    return [e for e in report.errors if e.path.startswith(path_prefix)]


def warnings_at(report, path_prefix):
    return [w for w in report.warnings if w.path.startswith(path_prefix)]


# --- top level -------------------------------------------------------------


def test_valid_fixture_is_strictly_valid(doc):
    report = validate_document(doc, strict=True)
    assert report.ok and not report.warnings


def test_not_an_object():
    assert not validate_document([1, 2, 3]).ok


def test_wrong_format(doc):
    doc["format"] = "something-else"
    assert errors_at(validate_document(doc), "format")


@pytest.mark.parametrize("version", ["1.0", "0.0", "zero.one", 0.1, None])
def test_unsupported_schema_versions(doc, version):
    doc["schema_version"] = version
    assert errors_at(validate_document(doc), "schema_version")


def test_newer_minor_warns_default_errors_strict(doc):
    doc["schema_version"] = "0.2"
    default = validate_document(doc)
    assert default.ok and warnings_at(default, "schema_version")
    assert not validate_document(doc, strict=True).ok


def test_unknown_top_level_field(doc):
    doc["mystery"] = True
    default = validate_document(doc)
    assert default.ok and warnings_at(default, "$.mystery")
    assert errors_at(validate_document(doc, strict=True), "$.mystery")


def test_name_must_be_string(doc):
    doc["name"] = 7
    assert errors_at(validate_document(doc), "name")


@pytest.mark.parametrize("block", ["body", "textures", "hair", "provenance"])
def test_missing_required_block(doc, block):
    del doc[block]
    assert errors_at(validate_document(doc), block)


# --- body -------------------------------------------------------------------


def test_unknown_rig_is_hard_error_in_both_modes(doc):
    doc["body"]["rig"] = "someone-elses-rig@9.9"
    assert errors_at(validate_document(doc), "body.rig")
    assert errors_at(validate_document(doc, strict=True), "body.rig")


def test_required_topology_is_valid_in_both_modes(doc):
    assert validate_document(doc).ok
    assert validate_document(doc, strict=True).ok


def test_body_only_topology_is_not_a_character(doc):
    doc["body"]["topology"] = "closed"
    assert errors_at(validate_document(doc), "body.topology")
    assert errors_at(validate_document(doc, strict=True), "body.topology")


def test_unknown_topology_is_hard_error_in_both_modes(doc):
    doc["body"]["topology"] = "hollow"
    assert errors_at(validate_document(doc), "body.topology")
    assert errors_at(validate_document(doc, strict=True), "body.topology")


def test_near_miss_topology_names_the_correction(doc):
    # A misread topology assembles a different surface; the error must both
    # refuse and point at the intended value.
    doc["body"]["topology"] = "mouth_interior"
    (error,) = errors_at(validate_document(doc), "body.topology")
    assert "did you mean 'mouth-interior'" in error.message


@pytest.mark.parametrize(
    ("key", "length"), [("identity", 44), ("identity", 46), ("resting_expression", 71)]
)
def test_wrong_parameter_array_lengths(doc, key, length):
    doc["body"][key] = [0.0] * length
    assert errors_at(validate_document(doc), f"body.{key}")


@pytest.mark.parametrize("bad", [True, "0.5", None, float("nan"), float("inf")])
def test_bad_parameter_values(doc, bad):
    doc["body"]["identity"][3] = bad
    assert errors_at(validate_document(doc), "body.identity[3]")


# --- body.proportions (§4.3) -------------------------------------------------


def test_absent_proportions_block_is_valid_and_strict(doc):
    # The pre-proportions corpus: absent block = template skeleton, valid
    # forever in both modes.
    assert "proportions" not in doc["body"]
    assert validate_document(doc, strict=True).ok


def test_proportioned_document_is_valid(doc):
    doc["body"]["proportions"] = {"leg_length": 0.25, "shoulder_width": -0.1}
    report = validate_document(doc, strict=True)
    assert report.ok and not report.warnings


def test_empty_proportions_block_is_valid(doc):
    # Same meaning as absent (writers should omit it; readers accept it).
    doc["body"]["proportions"] = {}
    assert validate_document(doc, strict=True).ok


@pytest.mark.parametrize("value", [0.41, -0.41, 1.0, -7.5])
def test_out_of_range_proportion_is_hard_error_in_both_modes(doc, value):
    doc["body"]["proportions"] = {"leg_length": value}
    assert errors_at(validate_document(doc), "body.proportions.leg_length")
    assert errors_at(
        validate_document(doc, strict=True), "body.proportions.leg_length"
    )


def test_unknown_proportion_key_is_hard_error_with_suggestion(doc):
    # Topology-class: a proportion ignored is a different skeleton, so this
    # errors in BOTH modes (unlike ordinary unknown optional fields).
    doc["body"]["proportions"] = {"leg_lenght": 0.2}
    default = errors_at(validate_document(doc), "body.proportions.leg_lenght")
    assert default and "leg_length" in str(default[0])
    assert errors_at(
        validate_document(doc, strict=True), "body.proportions.leg_lenght"
    )


@pytest.mark.parametrize("bad", [True, "0.2", None, float("nan"), float("inf")])
def test_bad_proportion_values(doc, bad):
    doc["body"]["proportions"] = {"hip_width": bad}
    assert errors_at(validate_document(doc), "body.proportions.hip_width")


def test_proportions_must_be_an_object(doc):
    doc["body"]["proportions"] = [0.1] * 6
    assert errors_at(validate_document(doc), "body.proportions")


def test_boundary_proportion_values_are_valid(doc):
    doc["body"]["proportions"] = {"arm_length": 0.4, "neck_length": -0.4}
    assert validate_document(doc, strict=True).ok


# --- textures ------------------------------------------------------------------


@pytest.mark.parametrize("slot", ["skin", "eye", "garment"])
def test_missing_required_slot(doc, slot):
    del doc["textures"][slot]
    assert errors_at(validate_document(doc), f"textures.{slot}")


def test_shoe_null_is_invalid(doc):
    doc["textures"]["shoe"] = None
    assert errors_at(validate_document(doc), "textures.shoe")


def test_shoe_recipe_is_valid(doc):
    doc["textures"]["shoe"] = {
        "component": "make-shoe",
        "component_version": "0.1.0",
        "prompt": "low canvas sneakers",
        "seed": 4,
    }
    assert validate_document(doc, strict=True).ok


@pytest.mark.parametrize("wrong", ["eyes", "garments", "footwear", "shoes", "skins"])
def test_plural_slot_keys_are_hard_errors_in_both_modes(doc, wrong):
    doc["textures"][wrong] = dict(doc["textures"]["skin"])
    for strict in (False, True):
        report = validate_document(doc, strict=strict)
        issues = errors_at(report, f"textures.{wrong}")
        assert issues, f"{wrong} must be a hard error (strict={strict})"
        assert "singular" in issues[0].message and "did you mean" in issues[0].message


def test_plural_asset_slot_keys_are_hard_errors(doc):
    doc["assets"] = {
        "eyes": {"sha256": "ab" * 32, "media_type": "image/png",
                 "width": 4, "height": 4}
    }
    assert errors_at(validate_document(doc), "assets.eyes")


def test_nested_albedo_map_is_valid(doc):
    doc["textures"]["skin"] = {"albedo": doc["textures"]["skin"]}
    assert validate_document(doc, strict=True).ok


def test_map_dict_without_albedo_is_invalid(doc):
    recipe = doc["textures"]["skin"]
    doc["textures"]["skin"] = {"gloss": recipe}
    report = validate_document(doc)
    assert errors_at(report, "textures.skin.albedo")


def test_unknown_map_warns_default_errors_strict(doc):
    recipe = doc["textures"]["skin"]
    doc["textures"]["skin"] = {"albedo": recipe, "normal": dict(recipe)}
    default = validate_document(doc)
    assert default.ok and warnings_at(default, "textures.skin.normal")
    assert errors_at(validate_document(doc, strict=True), "textures.skin.normal")


def test_reserved_inputs_field_is_hard_error(doc):
    doc["textures"]["skin"]["inputs"] = {"reference": {"slot": "skin"}}
    for strict in (False, True):
        issues = errors_at(validate_document(doc, strict=strict), "textures.skin.inputs")
        assert issues and "reserved" in issues[0].message


def test_unknown_slot_warns_default_errors_strict(doc):
    doc["textures"]["cape"] = dict(doc["textures"]["garment"])
    default = validate_document(doc)
    assert default.ok and warnings_at(default, "textures.cape")
    assert errors_at(validate_document(doc, strict=True), "textures.cape")


@pytest.mark.parametrize("seed", [-1, 2**31, True, 1.5, "7", None])
def test_bad_seeds(doc, seed):
    doc["textures"]["skin"]["seed"] = seed
    assert errors_at(validate_document(doc), "textures.skin.seed")


def test_missing_recipe_fields(doc):
    del doc["textures"]["eye"]["prompt"]
    assert errors_at(validate_document(doc), "textures.eye.prompt")


def test_bad_overrides(doc):
    doc["textures"]["skin"]["overrides"] = {"steps": 0, "guidance": -1, "banana": 1}
    report = validate_document(doc, strict=True)
    assert errors_at(report, "textures.skin.overrides.steps")
    assert errors_at(report, "textures.skin.overrides.guidance")
    assert errors_at(report, "textures.skin.overrides.banana")


# --- hair -------------------------------------------------------------------


def test_hair_null_is_valid(doc):
    doc["hair"] = None
    assert validate_document(doc, strict=True).ok


def test_hair_wrong_schema_version(doc):
    doc["hair"]["schema_version"] = 2
    assert errors_at(validate_document(doc), "hair.schema_version")


def test_hair_unknown_enum_value_is_error_in_both_modes(doc):
    doc["hair"]["family"] = "tonsure"
    assert errors_at(validate_document(doc), "hair.family")
    doc["hair"]["family"] = "crop"
    doc["hair"]["length"]["cut_line"] = "razored"
    assert errors_at(validate_document(doc), "hair.length.cut_line")
    assert errors_at(validate_document(doc, strict=True), "hair.length.cut_line")


def test_hair_missing_group_and_field(doc):
    del doc["hair"]["drape"]
    assert errors_at(validate_document(doc), "hair.drape")


def test_hair_missing_required_field(doc):
    del doc["hair"]["shape"]["volume"]
    assert errors_at(validate_document(doc), "hair.shape.volume")


def test_hair_optional_length_fields(doc):
    doc["hair"]["length"]["front"] = "ear"
    doc["hair"]["length"]["back"] = "jaw"
    assert validate_document(doc, strict=True).ok


def test_hair_unknown_field_warns_default_errors_strict(doc):
    doc["hair"]["shape"]["sheen"] = "high"
    default = validate_document(doc)
    assert default.ok and warnings_at(default, "hair.shape.sheen")
    assert errors_at(validate_document(doc, strict=True), "hair.shape.sheen")


def test_custom_color_requires_rgb(doc):
    doc["hair"]["color"] = {"family": "custom"}
    assert errors_at(validate_document(doc), "hair.color.rgb")


def test_rgb_forbidden_without_custom(doc):
    doc["hair"]["color"] = {"family": "black", "rgb": [0.1, 0.1, 0.1]}
    assert errors_at(validate_document(doc), "hair.color.rgb")


@pytest.mark.parametrize("rgb", [[0.1, 0.2], [0.1, 0.2, 1.5], [0.1, 0.2, -0.1], "dark"])
def test_bad_rgb(doc, rgb):
    doc["hair"]["color"] = {"family": "custom", "rgb": rgb}
    assert errors_at(validate_document(doc), "hair.color.rgb")


# --- provenance / assets ----------------------------------------------------------


def test_provenance_prompt_key_required_but_nullable(doc):
    doc["provenance"]["prompt"] = None
    assert validate_document(doc, strict=True).ok
    del doc["provenance"]["prompt"]
    assert errors_at(validate_document(doc), "provenance.prompt")


def test_provenance_missing_components(doc):
    del doc["provenance"]["components"]
    assert errors_at(validate_document(doc), "provenance.components")


def test_bad_component_sha(doc):
    doc["provenance"]["components"]["make-skin"]["sha256"] = "XYZ"
    assert errors_at(validate_document(doc), "provenance.components.make-skin.sha256")


def test_bad_created_timestamp(doc):
    doc["provenance"]["created"] = "yesterday"
    assert errors_at(validate_document(doc), "provenance.created")


def test_assets_block(doc):
    doc["assets"] = {
        "skin": {"sha256": "ab" * 32, "media_type": "image/png", "width": 1024, "height": 1024}
    }
    assert validate_document(doc, strict=True).ok
    doc["assets"]["skin"]["sha256"] = "not-hex"
    assert errors_at(validate_document(doc), "assets.skin.sha256")
    doc["assets"]["skin"]["sha256"] = "ab" * 32
    doc["assets"]["skin"]["width"] = 0
    assert errors_at(validate_document(doc), "assets.skin.width")


def test_unknown_asset_slot_warns_default(doc):
    doc["assets"] = {
        "cape": {"sha256": "ab" * 32, "media_type": "image/png", "width": 1, "height": 1}
    }
    default = validate_document(doc)
    assert default.ok and warnings_at(default, "assets.cape")


# --- constructor behavior ------------------------------------------------------


def test_character_constructor_raises_with_report(doc):
    doc["body"]["identity"] = [0.0] * 44
    with pytest.raises(CharacterError) as excinfo:
        Character.from_document(doc)
    assert any("identity" in issue.path for issue in excinfo.value.report.errors)


def test_boundary_proportion_round_trips_through_canonicalization(doc):
    # float32(0.40) is a hair above decimal 0.40; a canonicalized document
    # carrying it must reload as valid (the bound compares at float32, the
    # format's canonical parameter precision).
    import json as _json

    doc["body"]["proportions"] = {"arm_length": 0.4}
    first = Character.from_document(doc)
    reloaded = _json.loads(_json.dumps(first.to_document()))
    assert validate_document(reloaded, strict=True).ok
    assert Character.from_document(reloaded).content_id == first.content_id
