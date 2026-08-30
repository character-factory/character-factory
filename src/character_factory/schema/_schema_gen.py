"""Generator for the published JSON Schema (data/character-0.1.schema.json).

The committed schema file is generated from :mod:`character_factory.schema.vocab`
so the two can never drift; a test regenerates and compares. Regenerate with:

    python -m character_factory.schema._schema_gen
"""

from __future__ import annotations

import json
from pathlib import Path

from character_factory.schema import vocab

_SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}


def _enum(values: frozenset[str]) -> dict:
    return {"type": "string", "enum": sorted(values)}


def _seed() -> dict:
    return {"type": "integer", "minimum": vocab.SEED_MIN, "maximum": vocab.SEED_MAX}


def _float_array(length: int) -> dict:
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": length,
        "maxItems": length,
    }


def _recipe() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["component", "component_version", "prompt", "seed"],
        "properties": {
            "component": {"type": "string"},
            "component_version": {"type": "string"},
            "prompt": {"type": "string"},
            "seed": _seed(),
            "overrides": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "steps": {"type": "integer", "exclusiveMinimum": 0},
                    "guidance": {"type": "number", "exclusiveMinimum": 0},
                    "resolution": {"type": "integer", "exclusiveMinimum": 0},
                },
            },
        },
    }


def _hair() -> dict:
    groups = {}
    for group, fields in vocab.HAIR_GROUPS.items():
        properties = {name: _enum(values) for name, values in fields.items()}
        required = [
            name
            for name in fields
            if (group, name) not in vocab.HAIR_OPTIONAL_FIELDS
        ]
        if group == "color":
            properties["rgb"] = {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 3,
                "maxItems": 3,
            }
        groups[group] = {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }
    return {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["schema_version", "seed", "family"] + sorted(vocab.HAIR_GROUPS),
        "properties": {
            "schema_version": {"const": vocab.HAIR_SCHEMA_VERSION},
            "seed": _seed(),
            "family": _enum(vocab.HAIR_FAMILIES),
            **groups,
        },
        # The color.rgb/custom co-constraint (SPEC.md §6). JSON Schema can
        # express it; the Python validator enforces it identically.
        "allOf": [
            {
                "if": {
                    "type": "object",
                    "properties": {
                        "color": {"properties": {"family": {"const": "custom"}}}
                    },
                },
                "then": {"properties": {"color": {"required": ["family", "rgb"]}}},
                "else": {
                    "properties": {"color": {"not": {"required": ["rgb"]}}}
                },
            }
        ],
    }


def _slot_value() -> dict:
    """A slot: the flat albedo-recipe shorthand, or named maps (v0.1: albedo
    only). See SPEC.md §5.2."""
    return {
        "anyOf": [
            _recipe(),
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["albedo"],
                "properties": {name: _recipe() for name in vocab.MAPS},
            },
        ]
    }


def build_json_schema() -> dict:
    """The strict-flavor JSON Schema for character format v0.1."""
    slots = {slot: _slot_value() for slot in vocab.ALL_SLOTS}
    descriptor = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sha256", "media_type", "width", "height"],
        "properties": {
            "sha256": _SHA256,
            "media_type": {"type": "string"},
            "width": {"type": "integer", "exclusiveMinimum": 0},
            "height": {"type": "integer", "exclusiveMinimum": 0},
        },
    }
    asset_entry = {
        "anyOf": [
            descriptor,
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["albedo"],
                "properties": {name: descriptor for name in vocab.MAPS},
            },
        ]
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Character Factory character format v{vocab.SCHEMA_VERSION}",
        "description": (
            "Strict-flavor schema: unknown fields are rejected. Default-mode "
            "readers instead warn on unknown optional fields; see SPEC.md §10."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": ["format", "schema_version", "body", "textures", "hair", "provenance"],
        "properties": {
            "format": {"const": vocab.FORMAT},
            "schema_version": {"const": vocab.SCHEMA_VERSION},
            "name": {"type": "string"},
            "body": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rig", "topology", "identity", "resting_expression"],
                "properties": {
                    "rig": _enum(vocab.RIGS),
                    "topology": _enum(vocab.TOPOLOGIES),
                    "identity": _float_array(vocab.IDENTITY_LENGTH),
                    "proportions": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            name: {
                                "type": "number",
                                "minimum": -vocab.PROPORTION_LIMIT,
                                "maximum": vocab.PROPORTION_LIMIT,
                            }
                            for name in vocab.PROPORTION_NAMES
                        },
                    },
                    "resting_expression": _float_array(vocab.RESTING_EXPRESSION_LENGTH),
                },
            },
            "textures": {
                "type": "object",
                "additionalProperties": False,
                "required": list(vocab.REQUIRED_SLOTS),
                "properties": slots,
            },
            "hair": _hair(),
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prompt", "generator", "components"],
                "properties": {
                    "prompt": {"type": ["string", "null"]},
                    "figure_prompt": {"type": "string"},
                    "generator": {"type": "string"},
                    "components": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["version"],
                            "properties": {
                                "version": {"type": "string"},
                                "sha256": _SHA256,
                            },
                        },
                    },
                    "created": {"type": "string"},
                    "notes": {"type": "string"},
                },
            },
            "assets": {
                "type": "object",
                "additionalProperties": False,
                "properties": {slot: asset_entry for slot in vocab.ALL_SLOTS},
            },
        },
    }


def main() -> None:
    target = Path(__file__).parent / "data" / "character-0.1.schema.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(build_json_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
