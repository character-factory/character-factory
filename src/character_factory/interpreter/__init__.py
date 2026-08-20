"""Interpretation: free text → per-slot prompts + the semantic hair block.

The default backend is a small local language model with grammar-constrained
decoding (ARCHITECTURE §2.2); its component is pending the model bench. What
lives here today is the **rules fallback** — the documented degraded mode:
deterministic slot-prompt splitting plus a conservative hair block. It is
deliberately simple; quality interpretation is the model backend's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from character_factory.schema import vocab

__all__ = ["Interpretation", "rules_interpret"]


@dataclass
class Interpretation:
    slot_prompts: dict[str, str]
    hair: dict | None
    backend: str = "rules-fallback"
    notes: list[str] = field(default_factory=list)


_HAIR_FAMILIES = {
    "buzz": ("buzz", "buzzcut", "buzzed"),
    "crop": ("crop", "cropped", "short hair", "crew cut"),
    "pixie": ("pixie",),
    "bob": ("bob",),
    "ponytail": ("ponytail",),
    "bun": ("bun",),
    "braids": ("braids", "braided", "cornrows"),
    "locs": ("locs", "dreadlocks", "dreads"),
    "coily": ("afro", "coily",),
    "loose_long": ("long hair", "flowing hair", "loose hair"),
}
_HAIR_COLORS = {
    "black": ("black hair",),
    "dark_brown": ("dark brown hair", "dark hair", "brunette"),
    "brown": ("brown hair",),
    "auburn": ("auburn",),
    "copper": ("red hair", "copper hair", "ginger", "redhead"),
    "blonde": ("blond", "blonde"),
    "platinum": ("platinum",),
    "gray": ("gray hair", "grey hair", "graying", "greying"),
    "white": ("white hair",),
}
_TEXTURES = {
    "coily": ("coily", "afro", "kinky"),
    "curly": ("curly", "curls"),
    "wavy": ("wavy", "waves"),
}
_BALD = ("bald", "shaved head", "hairless")
_SHOE_WORDS = ("shoe", "shoes", "sneaker", "boot", "footwear", "trainers",
               "loafers", "heels", "sandal")


def _default_hair(family: str, color: str, texture: str | None) -> dict:
    lengths = {"buzz": "cropped", "crop": "cropped", "pixie": "ear",
               "bob": "chin", "ponytail": "below_shoulder", "bun": "shoulder",
               "braids": "below_shoulder", "locs": "below_shoulder",
               "coily": "ear", "loose_long": "mid_back"}
    return {
        "schema_version": vocab.HAIR_SCHEMA_VERSION,
        "seed": 0,
        "family": family,
        "part": {"kind": "center" if family in ("bob", "loose_long") else "none",
                 "side": "wearer_left", "position": "moderate",
                 "extent": "to_crown", "width": "narrow"},
        "hairline": {"height": "natural", "shape": "rounded",
                     "temple_recession": "natural", "sideburns": "natural",
                     "nape": "natural", "irregularity": "natural"},
        "length": {"overall": lengths[family], "cut_line": "soft"},
        "shape": {"volume": "medium", "density": "medium",
                  "texture": texture or ("coily" if family == "coily" else "straight"),
                  "wave_size": "medium", "wave_strength": "medium",
                  "root_lift": "medium"},
        "drape": {"gravity": "natural", "stiffness": "natural",
                  "shoulder_routing": "split", "body_clearance": "natural"},
        "color": {"family": color},
    }


def _first_match(text: str, table: dict[str, tuple]) -> str | None:
    for value, keywords in table.items():
        for keyword in keywords:
            if keyword in text:
                return value
    return None


def rules_interpret(prompt: str) -> Interpretation:
    """Deterministic degraded-mode interpretation.

    Slot prompts reuse the full description (each component's conditioning
    template frames it for its own canvas); the hair block is keyword-mapped
    with conservative defaults. The model backend replaces all of this.
    """
    text = re.sub(r"\s+", " ", prompt.strip().lower())
    notes = ["rules-fallback interpretation: slot prompts are the full "
             "description; hair is keyword-mapped with conservative defaults"]

    slots = {
        "skin": prompt.strip(),
        "eye": prompt.strip(),
        "garment": prompt.strip(),
    }
    if any(word in text for word in _SHOE_WORDS):
        slots["shoe"] = prompt.strip()

    if any(word in text for word in _BALD):
        hair = None
    else:
        family = _first_match(text, _HAIR_FAMILIES) or "crop"
        color = _first_match(text, _HAIR_COLORS) or "dark_brown"
        texture = _first_match(text, _TEXTURES)
        hair = _default_hair(family, color, texture)

    return Interpretation(slot_prompts=slots, hair=hair, notes=notes)
