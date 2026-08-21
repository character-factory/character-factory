"""The interpreter's output schema, derived from the format vocabulary.

The model backend decodes against this schema (ARCHITECTURE §2.2): a prompt
per texture slot — singular keys, closed set, `shoe` optional — plus the
hair block. Deriving it from the same vocabulary module the validator uses
means a plural slot key or an out-of-vocabulary hair value is structurally
unreachable during decoding, not merely caught later.

The grammar is deliberately slightly looser than the validator (e.g. the
color rgb/custom co-constraint stays in the validation repair loop); what
it guarantees is the invariant shape.
"""

from __future__ import annotations

import json

__all__ = ["interpretation_schema"]


def interpretation_schema() -> dict:
    from character_factory import character_json_schema
    from character_factory.schema import vocab

    slot_prompt = {
        "type": "object",
        "additionalProperties": False,
        "required": ["prompt"],
        "properties": {"prompt": {"type": "string"}},
    }
    hair = json.loads(json.dumps(character_json_schema()["properties"]["hair"]))
    hair.pop("allOf", None)          # co-constraint lives in the repair loop
    hair["type"] = "object"          # the grammar itself never emits null;
    #                                  baldness is decided after decoding
    # Enforcer limitation: non-string `const` values crash its enum path;
    # an equivalent closed integer range expresses the same constraint.
    hair["properties"]["schema_version"] = {
        "type": "integer",
        "minimum": vocab.HAIR_SCHEMA_VERSION,
        "maximum": vocab.HAIR_SCHEMA_VERSION,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["textures", "hair"],
        "properties": {
            "textures": {
                "type": "object",
                "additionalProperties": False,
                "required": list(vocab.REQUIRED_SLOTS),
                "properties": {slot: slot_prompt for slot in vocab.ALL_SLOTS},
            },
            "hair": hair,
        },
    }
