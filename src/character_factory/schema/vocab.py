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
# The one v0.1 character topology (§4.2). Every character is facially
# animatable: the rig-version-fixed mouth patch is replaced by the socket
# and anatomy, and all 72 expression controls export as named morph targets.
# The field remains explicit in the document because it is build-defining
# recipe data, not a caller-selectable quality tier.
TOPOLOGIES = frozenset({"mouth-interior"})

IDENTITY_LENGTH = 45
RESTING_EXPRESSION_LENGTH = 72

# Skeletal proportions (§4.3): named semantic controls, 0.0 = the rig's
# template, valid range ±PROPORTION_LIMIT (the generator's calibrated range
# becomes the format's validity bound — out of range is an error, never a
# clamp). The mapping to rig parameters is registry metadata, not spec.
PROPORTION_NAMES = (
    "spine_length",
    "neck_length",
    "shoulder_width",
    "arm_length",
    "hip_width",
    "leg_length",
)
PROPORTION_LIMIT = 0.40

REQUIRED_SLOTS = ("skin", "eye", "garment")
OPTIONAL_SLOTS = ("shoe",)
ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS

# Texture slot keys are singular, always. These are the plural (or otherwise
# near-miss) spellings an author is most likely to write instead; they fail
# validation with a pointed message in every mode — never a silent warning.
SLOT_MISTAKES = {
    "skins": "skin",
    "eyes": "eye",
    "garments": "garment",
    "footwear": "shoe",
    "shoes": "shoe",
}

# Named maps a texture slot can hold. v0.1 defines exactly one; secondary
# maps (and recipes conditioned on other maps' outputs) are the anticipated
# additive path — see SPEC.md §5 and §10.
MAPS = ("albedo",)

# Recipe field reserved for a future schema minor version (conditioning
# inputs, SPEC.md §5.3). v0.1 writers must not emit it.
RESERVED_RECIPE_FIELDS = ("inputs",)

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
