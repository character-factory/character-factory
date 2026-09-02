"""Multi-call interpretation: one narrow task per model call.

A small local model writes markedly better component prompts when each
call asks one question than when one call asks for the whole document.
Under the single instruction a small model drops slots, echoes the
description, or invents a hair family it was never shown; asked seven
short questions — figure, skin, eye, garment, shoe, hair, proportions —
it answers each one specifically, and the hair call can carry the full
vocabulary inline. The cost is seven generations instead of one (about
fifty seconds per description on a 24 GB-class card with the default
model, roughly what its single prompt costs), which is why the mode is
the default for local models and not for endpoints: a hosted frontier
model gains nothing from the split and loses the descriptive richness
the single instruction elicits (ARCHITECTURE §2.2, "mode").

Every call sees the raw description verbatim and a short task
instruction; the output of every call is grammar- or schema-constrained.
The wording below is deliberately literal — templates with named slots,
no conditional clauses — because a small model treats a hedged
instruction as a request ("add a skin condition only when there is a
reason" produced a sun-damaged astronomer) and treats an abstract one
loosely ("a short phrase with material and color" produced "a singlet of
synthetic fabric and bright orange"). The figure wording is the most
fragile of the set: adding one unrelated sentence to it turned full
physique phrases into word lists on three of seven descriptions. Change
these templates against the bench, never by taste.

The templates are bound to the launch component versions the same way
the registry's per-slot guidance is; a component whose trained format
changes updates its call here.
"""

from __future__ import annotations

from dataclasses import dataclass

from character_factory.interpreter.schema import interpretation_schema

__all__ = ["Call", "build_calls", "hair_vocabulary_lines"]

# Every prompt-writing call returns {"prompt": "<text>"}.
PROMPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt"],
    "properties": {"prompt": {"type": "string"}},
}
_PROMPT_TAIL = 'Reply with JSON only: {"prompt": "<your text>"}'

# Registry shoe-style tokens rendered the way a person names them; a
# token the table does not know is shown with its underscores removed.
_SHOE_STYLE_WORDS = {
    "below_ankle": "flip flops, sandals, loafers, sneakers",
    "high_top": "high tops",
    "ankle_boot": "ankle boots",
    "mid_boot": "mid boots",
    "tall_boot": "tall boots",
}


@dataclass(frozen=True)
class Call:
    name: str            # figure | skin | eye | garment | shoe | hair | proportions
    instruction: str     # the system message; the user message is the description
    schema: dict         # the call's output grammar


def hair_vocabulary_lines() -> str:
    """The hair block's closed vocabulary, one line per group, so a call
    that never sees the JSON grammar still sees every legal value."""
    hair = interpretation_schema()["properties"]["hair"]
    lines = [f"family: {', '.join(hair['properties']['family']['enum'])}"]
    for group in ("part", "hairline", "length", "shape", "drape", "color"):
        fields = hair["properties"][group]["properties"]
        parts = [f"{name} [{', '.join(spec['enum'])}]"
                 for name, spec in fields.items() if "enum" in spec]
        if parts:
            lines.append(f"{group}: " + "; ".join(parts))
    return "\n".join(lines)


def _shoe_styles(styles: list[str] | None) -> str:
    tokens = list(styles) if styles else list(_SHOE_STYLE_WORDS)
    words = [_SHOE_STYLE_WORDS.get(token, token.replace("_", " ")) for token in tokens]
    return ", ".join(words) + " (and other shoes of those heights)"


def build_calls(vocabulary: dict[str, dict] | None = None) -> list[Call]:
    """The call plan, in execution order. `vocabulary` is the installed
    components' declared vocabularies by slot (today: shoe styles)."""
    schema = interpretation_schema()
    vocabulary = vocabulary or {}
    shoe_styles = _shoe_styles(vocabulary.get("shoe", {}).get("styles"))

    figure = [
        "Given this character description, decide what the character's body and face shape should look like. "
        "Derive anything unstated from who they are (age, occupation, background) — specific, never an average.",
        "Write it as one sentence, in this order: sex and age, height, overall build and body fat, musculature, face and head structure (jaw, cheeks, brow). "
        "Physique words only — no name, clothing, occupation, scene, or style words.",
        _PROMPT_TAIL,
    ]
    skin = [
        "Given this character description, decide what the character's skin should look like. "
        "The skin tone must be plausible for the character's stated nationality or ancestry (use the description's own complexion if it gives one). "
        "Invent specific facial features that fit their age and life — no generic placeholders.",
        "Write it like this, comma-separated: <lightness and undertone, e.g. 'deep warm-toned skin'>, <age>, <gender presentation>, <ancestry>, <brow character>, <facial structure>, <lips>, <pores and wrinkles>",
        "Skin only — no clothing, hair, or body-shape words.",
        _PROMPT_TAIL,
    ]
    eye = [
        "Given this character description, decide what the character's eyes should look like (iris color plausible for their ancestry and age unless the description says otherwise).",
        "Each field is a short phrase, not a single word. Sclera is off-white or ivory — never pure white.",
        "Write it like this, comma-separated: <iris color> iris, <radial fiber phrase>, <limbal ring phrase>, <pupil>, <sclera tone>, <sclera veins>",
        _PROMPT_TAIL,
    ]
    garment = [
        "Given this character description, decide what the character wears on their body (clothing only — not footwear).",
        "Pieces the description names are mandatory and stay exactly as described. "
        "If they do not add up to a complete outfit — a top and a bottom, or one full-body garment — add the missing piece(s) this person would wear. "
        "Rule out coverage only when the description does (swimsuit, barefoot, unclothed).",
        "Write it like this, comma-separated: <color> <material> <garment>, <color> <material> <garment>, … — most visible piece first.",
        "Never footwear, never the body.",
        _PROMPT_TAIL,
    ]
    shoe = [
        "Given this character description, decide what footwear the character wears. "
        "If the description names footwear, use exactly that; otherwise choose footwear that suits who they are. "
        "If the character is barefoot or unclothed, reply with an empty prompt.",
        "Choose what this person wears while doing what the description says they do (working, running, at home). Practical over fashionable when the two conflict.",
        f"Styles available: {shoe_styles}.",
        "Write it like this, comma-separated: <style>, <construction details>, <material>, <color palette>, <finish or pattern>",
        _PROMPT_TAIL,
    ]
    hair = [
        "Given this character description, decide the character's hairstyle, length, texture, and color (plausible for their age and background when unstated).",
        "Write it as a JSON hair block using ONLY these values, copied exactly:",
        hair_vocabulary_lines(),
        "Set schema_version to 1 and seed to 0. For a dyed or unusual color use color family \"custom\" with an rgb triple (0–255); otherwise a named family.",
        "Length must fit the family: bun and ponytail need shoulder length or longer; buzz, crop and pixie are cropped or ear length.",
        "Reply with the JSON object only.",
    ]
    proportions = [
        "Given this character description, decide whether it clearly implies an unusual skeletal build (towering, petite, broad-shouldered, long-limbed, and so on).",
        "If it does not, reply {}. If it does, reply with only the affected keys — spine_length, neck_length, shoulder_width, arm_length, hip_width, leg_length — as integers in hundredths from -40 to 40 (25 means +25%; never 0).",
        "Reply with the JSON object only.",
    ]
    return [
        Call("figure", "\n".join(figure), PROMPT_SCHEMA),
        Call("skin", "\n".join(skin), PROMPT_SCHEMA),
        Call("eye", "\n".join(eye), PROMPT_SCHEMA),
        Call("garment", "\n".join(garment), PROMPT_SCHEMA),
        Call("shoe", "\n".join(shoe), PROMPT_SCHEMA),
        Call("hair", "\n".join(hair), schema["properties"]["hair"]),
        Call("proportions", "\n".join(proportions), schema["properties"]["proportions"]),
    ]
