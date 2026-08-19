"""The published JSON Schema: no drift from the vocabularies, and third-party
validation (the `jsonschema` library) agrees with the reference validator."""

import pytest

from character_factory import character_json_schema
from character_factory.schema._schema_gen import build_json_schema

jsonschema = pytest.importorskip("jsonschema")


def test_committed_schema_matches_generator():
    assert character_json_schema() == build_json_schema()


def test_examples_pass_third_party_validation(example_doc):
    jsonschema.validate(example_doc, character_json_schema())


def test_fixture_passes_third_party_validation(doc):
    jsonschema.validate(doc, character_json_schema())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["body"].__setitem__("topology", "mouth-interior"),
        lambda d: d["body"].__setitem__("identity", [0.0] * 44),
        lambda d: d["textures"].pop("skin"),
        lambda d: d["textures"].__setitem__("footwear", None),
        lambda d: d["hair"]["color"].__setitem__("rgb", [0.1, 0.1, 0.1]),
        lambda d: d["hair"].__setitem__("family", "tonsure"),
        lambda d: d.__setitem__("extra", 1),
    ],
    ids=[
        "unknown-topology",
        "short-identity",
        "missing-skin",
        "null-footwear",
        "rgb-without-custom",
        "bad-family",
        "unknown-top-level",
    ],
)
def test_broken_documents_fail_third_party_validation(doc, mutate):
    mutate(doc)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, character_json_schema())
