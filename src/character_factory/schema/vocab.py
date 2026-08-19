"""Closed vocabularies of character format v0.1 (SPEC.md).

Single source of truth for every enum, slot name, and structural constant the
validator and model enforce. The published JSON Schema mirrors these values.
"""

from __future__ import annotations

FORMAT = "character-factory/character"
SCHEMA_MAJOR = 0
SCHEMA_MINOR = 1
SCHEMA_VERSION = f"{SCHEMA_MAJOR}.{SCHEMA_MINOR}"

RIGS = frozenset({"mhr-lod1@1.0"})
TOPOLOGIES = frozenset({"closed"})

IDENTITY_LENGTH = 45
RESTING_EXPRESSION_LENGTH = 72

REQUIRED_SLOTS = ("skin", "eyes", "garments")
OPTIONAL_SLOTS = ("footwear",)
ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS

SEED_MIN = 0
SEED_MAX = 2**31 - 1

HAIR_SCHEMA_VERSION = 1

_LENGTHS = frozenset(
    {
        "cropped", "ear", "jaw", "chin", "shoulder", "collarbone",
        "below_shoulder", "chest", "mid_back", "waist",
    }
)

HAIR_FAMILIES = frozenset(
    {
        "buzz", "crop", "pixie", "side_part", "bob", "loose_long",
        "coily", "ponytail", "bun", "braids", "locs",
    }
)

# Field vocabularies per hair block group. Every listed field is required in a
# stored character file except those in HAIR_OPTIONAL_FIELDS (SPEC.md §6).
HAIR_GROUPS: dict[str, dict[str, frozenset[str]]] = {
    "part": {
        "kind": frozenset({"none", "center", "side"}),
        "side": frozenset({"wearer_left", "wearer_right"}),
        "position": frozenset({"subtle", "moderate", "deep"}),
        "extent": frozenset({"short", "to_crown", "through_crown"}),
        "width": frozenset({"narrow", "medium", "wide"}),
    },
    "hairline": {
        "height": frozenset({"low", "natural", "high"}),
        "shape": frozenset({"rounded", "straight", "widows_peak"}),
        "temple_recession": frozenset({"none", "natural", "pronounced"}),
        "sideburns": frozenset({"short", "natural", "long"}),
        "nape": frozenset({"high", "natural", "low"}),
        "irregularity": frozenset({"clean", "natural", "textured"}),
    },
    "length": {
        "overall": _LENGTHS,
        "front": _LENGTHS,
        "side": _LENGTHS,
        "back": _LENGTHS,
        "cut_line": frozenset({"blunt", "soft", "layered"}),
    },
    "shape": {
        "volume": frozenset({"low", "medium", "high"}),
        "density": frozenset({"light", "medium", "full"}),
        "texture": frozenset({"straight", "wavy", "curly", "coily"}),
        "wave_size": frozenset({"small", "medium", "large"}),
        "wave_strength": frozenset({"subtle", "medium", "strong"}),
        "root_lift": frozenset({"low", "medium", "high"}),
    },
    "drape": {
        "gravity": frozenset({"light", "natural", "heavy"}),
        "stiffness": frozenset({"soft", "natural", "firm"}),
        "shoulder_routing": frozenset(
            {"natural", "split", "mostly_behind", "all_front", "all_behind"}
        ),
        "body_clearance": frozenset({"close", "natural", "loose"}),
    },
    "color": {
        "family": frozenset(
            {
                "black", "dark_brown", "brown", "auburn", "copper",
                "blonde", "platinum", "gray", "white", "custom",
            }
        ),
    },
}

# (group, field) pairs a stored file may omit.
HAIR_OPTIONAL_FIELDS = frozenset(
    {("length", "front"), ("length", "side"), ("length", "back"), ("color", "rgb")}
)
