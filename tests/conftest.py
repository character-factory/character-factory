"""Shared across all test packages: a minimal valid character document."""

from __future__ import annotations

import copy

import pytest

_BASE: dict = {
    "format": "character-factory/character",
    "schema_version": "0.1",
    "name": "fixture",
    "body": {
        "rig": "mhr-lod1@1.0",
        "topology": "closed",
        "identity": [0.0] * 45,
        "resting_expression": [0.0] * 72,
    },
    "textures": {
        "skin": {
            "component": "skin",
            "component_version": "0.1.0",
            "prompt": "medium skin tone, adult",
            "seed": 1,
        },
        "eyes": {
            "component": "eyes",
            "component_version": "0.1.0",
            "prompt": "brown iris",
            "seed": 2,
        },
        "garments": {
            "component": "garments",
            "component_version": "0.1.0",
            "prompt": "plain grey t-shirt and jeans",
            "seed": 3,
        },
    },
    "hair": {
        "schema_version": 1,
        "seed": 0,
        "family": "crop",
        "part": {
            "kind": "none",
            "side": "wearer_left",
            "position": "moderate",
            "extent": "to_crown",
            "width": "narrow",
        },
        "hairline": {
            "height": "natural",
            "shape": "rounded",
            "temple_recession": "natural",
            "sideburns": "natural",
            "nape": "natural",
            "irregularity": "natural",
        },
        "length": {"overall": "cropped", "cut_line": "soft"},
        "shape": {
            "volume": "low",
            "density": "medium",
            "texture": "straight",
            "wave_size": "medium",
            "wave_strength": "medium",
            "root_lift": "medium",
        },
        "drape": {
            "gravity": "natural",
            "stiffness": "natural",
            "shoulder_routing": "split",
            "body_clearance": "natural",
        },
        "color": {"family": "black"},
    },
    "provenance": {
        "prompt": "an unremarkable person in a grey t-shirt",
        "generator": "character-factory/0.1.0.dev0",
        "components": {
            "interpreter": {"version": "0.1.0"},
            "identity": {"version": "0.1.0"},
            "skin": {"version": "0.1.0"},
            "eyes": {"version": "0.1.0"},
            "garments": {"version": "0.1.0"},
            "hair": {"version": "0.1.0"},
        },
    },
}


@pytest.fixture
def doc() -> dict:
    """A fresh, minimal, strictly valid character document (barefoot)."""
    return copy.deepcopy(_BASE)
