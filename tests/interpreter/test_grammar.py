"""The constrained-decoding grammar accepts the authoring surface.

The interpreter emits per-slot prompts plus the hair block under
grammar-constrained decoding (ARCHITECTURE §2.2). This exercises the actual
enforcer library against a schema derived from the format vocabularies: a
valid interpretation walks the grammar character by character to completion,
and the likeliest authoring mistake — a plural slot key — is structurally
unreachable, not merely caught later by validation.

The grammar is deliberately slightly looser than the validator (e.g. the
color.rgb/custom co-constraint stays in the validation repair loop); what it
must guarantee is the invariant shape: four singular slot keys, closed hair
enums.
"""

import json

import pytest

lmformatenforcer = pytest.importorskip("lmformatenforcer")
from lmformatenforcer import JsonSchemaParser  # noqa: E402

from character_factory.interpreter.schema import interpretation_schema  # noqa: E402


def walk(parser, text: str) -> tuple[bool, int]:
    """Feed text through the grammar; returns (accepted fully, chars consumed)."""
    for position, char in enumerate(text):
        if char not in parser.get_allowed_characters():
            return False, position
        parser = parser.add_character(char)
    return parser.can_end(), len(text)


def valid_interpretation() -> dict:
    return {
        "textures": {
            "skin": {"prompt": "medium skin tone, adult"},
            "eye": {"prompt": "brown iris"},
            "garment": {"prompt": "plain grey t-shirt and jeans"},
            "shoe": {"prompt": "low canvas sneakers"},
        },
        "hair": {
            "schema_version": 1, "seed": 3, "family": "crop",
            "part": {"kind": "none", "side": "wearer_left", "position": "moderate",
                     "extent": "to_crown", "width": "narrow"},
            "hairline": {"height": "natural", "shape": "rounded",
                         "temple_recession": "natural", "sideburns": "natural",
                         "nape": "natural", "irregularity": "natural"},
            "length": {"overall": "cropped", "cut_line": "soft"},
            "shape": {"volume": "low", "density": "medium", "texture": "straight",
                      "wave_size": "medium", "wave_strength": "medium",
                      "root_lift": "medium"},
            "drape": {"gravity": "natural", "stiffness": "natural",
                      "shoulder_routing": "split", "body_clearance": "natural"},
            "color": {"family": "black"},
        },
    }


def test_grammar_accepts_a_valid_interpretation():
    parser = JsonSchemaParser(interpretation_schema())
    accepted, consumed = walk(parser, json.dumps(valid_interpretation()))
    assert accepted, f"grammar rejected valid interpretation at char {consumed}"


def test_grammar_rejects_plural_slot_keys():
    document = valid_interpretation()
    document["textures"] = {"eyes": {"prompt": "brown iris"},
                            **{k: v for k, v in document["textures"].items()
                               if k != "eye"}}
    parser = JsonSchemaParser(interpretation_schema())
    accepted, consumed = walk(parser, json.dumps(document))
    assert not accepted
    # The walk must fail inside the "eyes" key, before any recipe content.
    assert consumed < len(json.dumps(document)) // 2


def test_grammar_rejects_out_of_vocabulary_hair_enum():
    document = valid_interpretation()
    document["hair"]["family"] = "tonsure"
    parser = JsonSchemaParser(interpretation_schema())
    accepted, _ = walk(parser, json.dumps(document))
    assert not accepted
