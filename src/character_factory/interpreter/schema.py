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

__all__ = ["endpoint_interpretation_schema", "endpoint_schema", "interpretation_schema"]


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
    # Skeletal-proportion overrides (§4.3), optional: integers in
    # HUNDREDTHS of the document unit (the enforcer constrains closed
    # integer ranges reliably; float bounds it does not), so 25 here means
    # 0.25 in the document, and the ±0.40 format bound is exactly ±40.
    proportions = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            name: {"type": "integer", "minimum": -40, "maximum": 40}
            for name in vocab.PROPORTION_NAMES
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["figure", "textures", "hair"],
        "properties": {
            # The body-generation prompt: the interpreter writes it in the
            # figure component's trained format, exactly like every texture
            # slot gets its component's format. Identity conditions on
            # THIS, never on the raw description.
            "figure": slot_prompt,
            "textures": {
                "type": "object",
                "additionalProperties": False,
                "required": list(vocab.REQUIRED_SLOTS),
                "properties": {slot: slot_prompt for slot in vocab.ALL_SLOTS},
            },
            "hair": hair,
            "proportions": proportions,
        },
    }


def endpoint_interpretation_schema() -> dict:
    """A strict-output form of :func:`interpretation_schema`.

    Strict chat-completions endpoints require every object property to be
    listed in ``required``. Optional fields therefore become required but
    nullable at this transport boundary; nulls are removed before the normal
    interpretation validator runs. The character-facing interpretation
    contract remains unchanged.
    """
    return endpoint_schema(interpretation_schema())


def endpoint_schema(schema: dict) -> dict:
    """Any output schema of ours in the strict form an OpenAI-compatible
    endpoint enforces (see `endpoint_interpretation_schema`): every
    property required, optionals nullable, no additional properties."""
    return _strict_object_schema(schema)


def _strict_object_schema(value: dict) -> dict:
    value = json.loads(json.dumps(value))
    if value.get("type") == "object":
        properties = value.get("properties", {})
        originally_required = set(value.get("required", []))
        converted = {}
        for name, child in properties.items():
            child = _strict_object_schema(child)
            if name not in originally_required:
                child = {"anyOf": [child, {"type": "null"}]}
            converted[name] = child
        value["properties"] = converted
        value["required"] = list(properties)
        value["additionalProperties"] = False
    elif value.get("type") == "array" and isinstance(value.get("items"), dict):
        value["items"] = _strict_object_schema(value["items"])
    for keyword in ("anyOf", "oneOf"):
        if keyword in value:
            value[keyword] = [_strict_object_schema(item) for item in value[keyword]]
    return value
