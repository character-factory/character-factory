"""The rules-fallback decomposer: clause routing, with the field-reported
cross-slot leak as a permanent regression case (footwear words in the eye
prompt painted flip-flops onto an eyeball)."""

from character_factory.interpreter import rules_interpret


def test_field_regression_flip_flops_never_reach_the_eye_slot():
    result = rules_interpret(
        "a 19 year old japanese girl wearing a croptop, denim shorts, and flip flops"
    )
    assert "flip" not in result.slot_prompts["eye"]
    assert "denim" not in result.slot_prompts["eye"]
    assert "flip" not in result.slot_prompts["skin"]
    assert "denim" not in result.slot_prompts["skin"]
    assert "croptop" in result.slot_prompts["garment"]
    assert "denim shorts" in result.slot_prompts["garment"]
    assert "flip" not in result.slot_prompts["garment"]
    assert "flip flops" in result.slot_prompts["shoe"]
    assert "19 year old japanese girl" in result.slot_prompts["skin"]


def test_eye_color_clause_routes_to_the_eye_slot():
    result = rules_interpret(
        "a lean marathon runner with cropped dark hair and green eyes, "
        "wearing a teal running vest and black shorts"
    )
    assert result.slot_prompts["eye"].startswith("green iris")
    assert "green" not in result.slot_prompts["skin"]
    assert "vest" in result.slot_prompts["garment"]
    assert "shoe" not in result.slot_prompts


def test_no_eye_description_uses_neutral_default():
    result = rules_interpret("a tall dockworker wearing a wool coat")
    assert "iris" in result.slot_prompts["eye"]
    assert "coat" not in result.slot_prompts["eye"]
    assert any("neutral default" in note for note in result.notes)


def test_no_clothing_gets_plain_default_garment():
    result = rules_interpret("a bald middle-aged man with kind features")
    assert result.slot_prompts["garment"].startswith("plain")
    assert result.hair is None


def test_clothing_nouns_without_wearing_verb_still_route():
    result = rules_interpret("a freediver with deep brown skin, matte black "
                             "one-piece swimsuit, barefoot")
    assert "swimsuit" in result.slot_prompts["garment"]
    assert "swimsuit" not in result.slot_prompts["skin"]
    assert "deep brown skin" in result.slot_prompts["skin"]
    assert "shoe" not in result.slot_prompts


def test_deterministic():
    prompt = "a 19 year old japanese girl wearing a croptop and flip flops"
    assert rules_interpret(prompt).slot_prompts == rules_interpret(prompt).slot_prompts
