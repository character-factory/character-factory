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
_SHOE_WORDS = ("shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
               "footwear", "trainers", "loafers", "loafer", "heels",
               "sandal", "sandals", "flip flops", "flip-flops", "flipflops",
               "slides", "clogs", "moccasins", "oxfords", "pumps", "cleats")
_GARMENT_WORDS = (
    "top", "croptop", "crop top", "shirt", "t-shirt", "tshirt", "tee",
    "blouse", "tank", "vest", "jacket", "hoodie", "sweater", "cardigan",
    "dress", "gown", "skirt", "shorts", "jeans", "denim", "pants",
    "trousers", "leggings", "swimsuit", "bikini", "jammers", "wetsuit",
    "uniform", "suit", "robe", "overalls", "jumpsuit", "romper", "socks",
    "gloves", "coat", "kimono", "sari", "tunic",
)
_EYE_COLORS = ("green", "blue", "brown", "hazel", "amber", "gray", "grey",
               "dark", "black", "violet")
_DEFAULT_EYE_PROMPT = ("natural dark brown iris, subtle radial fibers, "
                       "off-white sclera, faint veins")


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


def _segments(text: str) -> list[str]:
    """Comma/'and'-separated fragments, lowercased, order preserved."""
    parts = re.split(r",|\band\b", text)
    return [part.strip() for part in parts if part.strip()]


def _clothing_split(text: str) -> tuple[str | None, str | None, str | None]:
    """(identity clause, garment clause, shoe clause) from the description.

    The clothing clause is everything after 'wearing'/'dressed in'/'in a'
    when present, else the fragments containing clothing nouns. Footwear
    fragments split out of the clothing clause into the shoe clause.
    """
    match = re.search(r"\b(?:wearing|dressed in|clad in)\b(.*)$", text)
    if match:
        clothing_text = match.group(1).strip()
        identity_text = text[: match.start()].strip(" ,.")
        clothing_segments = _segments(clothing_text)
    else:
        clothing_segments = [
            segment for segment in _segments(text)
            if any(word in segment for word in _GARMENT_WORDS + _SHOE_WORDS)
        ]
        identity_text = text
        for segment in clothing_segments:
            identity_text = identity_text.replace(segment, " ")
        identity_text = re.sub(r"\s+", " ", identity_text).strip(" ,.")

    shoe_segments = [s for s in clothing_segments
                     if any(word in s for word in _SHOE_WORDS)]
    garment_segments = [s for s in clothing_segments if s not in shoe_segments]

    return (
        identity_text or None,
        ", ".join(garment_segments) or None,
        ", ".join(shoe_segments) or None,
    )


def _eye_clause(text: str) -> tuple[str | None, str]:
    """(eye prompt or None, text with the eye fragment removed).

    Matches a one-or-two-word color phrase directly before "eye(s)" whose
    final word is a known eye color ("green eyes", "deep hazel eyes") —
    never a longer span, so unrelated words cannot ride into the eye slot.
    """
    stopwords = {"and", "with", "a", "the", "has", "her", "his", "their"}
    for match in re.finditer(r"\b([a-z]+(?:[ -][a-z]+)?)[ -]eyes?\b", text):
        words = [w for w in re.split(r"[ -]", match.group(1)) if w not in stopwords]
        phrase = " ".join(words)
        if words and words[-1] in _EYE_COLORS:
            remainder = text[: match.start()] + text[match.end():]
            remainder = re.sub(r"\band\s*(,|$)", r"\1", remainder)
            remainder = re.sub(r"\s+", " ", remainder).strip(" ,.")
            return (
                f"{phrase} iris, off-white sclera, subtle radial fibers",
                remainder,
            )
    return None, text


def rules_interpret(prompt: str) -> Interpretation:
    """Deterministic degraded-mode interpretation.

    A clause decomposer routes clothing words to the garment slot, footwear
    to the shoe slot, eye-color phrases to the eye slot, and the identity
    remainder to the skin slot — so no slot is conditioned on another slot's
    content (a shoe word in the eye prompt paints shoes on the eyeball).
    The model backend replaces all of this with real interpretation.
    """
    text = re.sub(r"\s+", " ", prompt.strip().lower())
    notes = ["rules-fallback interpretation: clause-routed slot prompts, "
             "keyword-mapped hair with conservative defaults"]

    eye_prompt, without_eyes = _eye_clause(text)
    if eye_prompt is None:
        eye_prompt = _DEFAULT_EYE_PROMPT
        notes.append("no eye description found; neutral default used")
    identity, garment, shoe = _clothing_split(without_eyes)

    slots = {
        "skin": identity or text,
        "eye": eye_prompt,
        "garment": garment or "plain fitted t-shirt and plain trousers",
    }
    if garment is None:
        notes.append("no clothing description found; plain default garment used")
    if shoe is not None:
        slots["shoe"] = shoe

    if any(word in text for word in _BALD):
        hair = None
    else:
        family = _first_match(text, _HAIR_FAMILIES) or "crop"
        color = _first_match(text, _HAIR_COLORS) or "dark_brown"
        texture = _first_match(text, _TEXTURES)
        hair = _default_hair(family, color, texture)

    return Interpretation(slot_prompts=slots, hair=hair, notes=notes)
