"""HairIntent v1: semantic JSON compiled to deterministic make-wig geometry.

The public vocabulary intentionally describes a haircut rather than exposing
mesh tessellation and solver constants.  An LLM can therefore emit stable,
validated JSON while this compiler remains free to improve how those words
become opaque geometry.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .guides import GuideDrapeSpec
from .presets import PRESETS
from .style import Style
from .texture import TextureSpec


class HairIntentError(ValueError):
    """Raised when HairIntent JSON is structurally or semantically invalid."""


def hair_intent_schema() -> dict[str, Any]:
    """Return the packaged HairIntent v1 JSON Schema."""

    text = resources.files("character_factory.hair.wig.schemas").joinpath("hair_intent_v1.schema.json").read_text()
    return json.loads(text)


@dataclass(frozen=True)
class PartIntent:
    kind: str = "none"
    side: str = "wearer_left"
    position: str = "moderate"
    extent: str = "to_crown"
    width: str = "narrow"


@dataclass(frozen=True)
class HairlineIntent:
    """Wearer-facing controls for the opaque cap's anatomical boundary."""

    height: str = "natural"
    shape: str = "rounded"
    temple_recession: str = "natural"
    sideburns: str = "natural"
    nape: str = "natural"
    irregularity: str = "natural"


@dataclass(frozen=True)
class LengthIntent:
    overall: str = "chin"
    front: str | None = None
    side: str | None = None
    back: str | None = None
    cut_line: str = "soft"


@dataclass(frozen=True)
class ShapeIntent:
    volume: str = "medium"
    density: str = "medium"
    texture: str = "straight"
    wave_size: str = "medium"
    wave_strength: str = "medium"
    root_lift: str = "medium"


@dataclass(frozen=True)
class DrapeIntent:
    gravity: str = "natural"
    stiffness: str = "natural"
    shoulder_routing: str = "split"
    body_clearance: str = "natural"


@dataclass(frozen=True)
class ColorIntent:
    family: str = "dark_brown"
    rgb: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class HairIntent:
    schema_version: int = 1
    seed: int = 0
    family: str = "loose_long"
    part: PartIntent = field(default_factory=PartIntent)
    hairline: HairlineIntent = field(default_factory=HairlineIntent)
    length: LengthIntent = field(default_factory=LengthIntent)
    shape: ShapeIntent = field(default_factory=ShapeIntent)
    drape: DrapeIntent = field(default_factory=DrapeIntent)
    color: ColorIntent = field(default_factory=ColorIntent)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class HairPlan:
    """Resolved deterministic generation plan produced by the compiler."""

    intent: HairIntent
    base_preset: str
    style: Style
    texture: TextureSpec
    compiler_version: str = "hair-intent-v1"

    def manifest(self) -> dict[str, Any]:
        return {
            "compiler_version": self.compiler_version,
            "base_preset": self.base_preset,
            "intent": self.intent.to_dict(),
        }


FAMILIES = {
    "buzz", "crop", "pixie", "side_part", "bob", "loose_long", "coily",
    "ponytail", "bun", "braids", "locs",
}
LENGTHS = {
    "cropped", "ear", "jaw", "chin", "shoulder", "collarbone",
    "below_shoulder", "chest", "mid_back", "waist",
}

_FAMILY_PRESET = {
    "buzz": "buzz",
    "crop": "crop",
    "pixie": "pixie",
    "side_part": "side_part",
    "bob": "bob",
    "loose_long": "long_wavy",
    "coily": "afro",
    "ponytail": "ponytail",
    "bun": "low_bun",
    "braids": "box_braids",
    "locs": "locs",
}
_DEFAULT_LENGTH = {
    "buzz": "cropped", "crop": "cropped", "pixie": "ear",
    "side_part": "ear", "bob": "chin", "loose_long": "mid_back",
    "coily": "ear", "ponytail": "below_shoulder", "bun": "shoulder",
    "braids": "below_shoulder", "locs": "below_shoulder",
}
_LENGTH_SCALE = {
    "cropped": 0.24,
    "ear": 0.72,
    "jaw": 1.02,
    "chin": 1.28,
    "shoulder": 1.82,
    "collarbone": 2.08,
    "below_shoulder": 2.62,
    "chest": 3.02,
    "mid_back": 3.72,
    "waist": 5.05,
}


def _object(value: Any, path: str, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HairIntentError(f"{path} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise HairIntentError(f"{path} has unknown field(s): {', '.join(unknown)}")
    return value


def _enum(value: Any, path: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise HairIntentError(f"{path} must be one of: {', '.join(sorted(choices))}")
    return value


def _optional_enum(value: Any, path: str, choices: set[str]) -> str | None:
    return None if value is None else _enum(value, path, choices)


def parse_hair_intent(payload: Mapping[str, Any]) -> HairIntent:
    """Strictly validate and normalize a HairIntent v1 mapping."""

    root = _object(payload, "$", {
        "schema_version", "seed", "family", "part", "hairline", "length",
        "shape", "drape", "color"
    })
    version = root.get("schema_version", 1)
    if version != 1:
        raise HairIntentError(f"$.schema_version must be 1, got {version!r}")
    seed = root.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise HairIntentError("$.seed must be an integer in [0, 2147483647]")
    family = _enum(root.get("family", "loose_long"), "$.family", FAMILIES)

    default_part = "side" if family == "side_part" else (
        "center" if family in {"bob", "loose_long"} else "none"
    )
    p = _object(root.get("part", {}), "$.part", {"kind", "side", "position", "extent", "width"})
    part = PartIntent(
        kind=_enum(p.get("kind", default_part), "$.part.kind", {"none", "center", "side"}),
        side=_enum(p.get("side", "wearer_left"), "$.part.side", {"wearer_left", "wearer_right"}),
        position=_enum(p.get("position", "moderate"), "$.part.position", {"subtle", "moderate", "deep"}),
        extent=_enum(p.get("extent", "to_crown"), "$.part.extent", {"short", "to_crown", "through_crown"}),
        width=_enum(p.get("width", "narrow"), "$.part.width", {"narrow", "medium", "wide"}),
    )

    hl = _object(root.get("hairline", {}), "$.hairline", {
        "height", "shape", "temple_recession", "sideburns", "nape", "irregularity"
    })
    hairline = HairlineIntent(
        height=_enum(hl.get("height", "natural"), "$.hairline.height", {"low", "natural", "high"}),
        shape=_enum(hl.get("shape", "rounded"), "$.hairline.shape", {"rounded", "straight", "widows_peak"}),
        temple_recession=_enum(hl.get("temple_recession", "natural"), "$.hairline.temple_recession", {
            "none", "natural", "pronounced"
        }),
        sideburns=_enum(hl.get("sideburns", "natural"), "$.hairline.sideburns", {"short", "natural", "long"}),
        nape=_enum(hl.get("nape", "natural"), "$.hairline.nape", {"high", "natural", "low"}),
        irregularity=_enum(hl.get("irregularity", "natural"), "$.hairline.irregularity", {
            "clean", "natural", "textured"
        }),
    )

    length_default = _DEFAULT_LENGTH[family]
    le = _object(root.get("length", {}), "$.length", {"overall", "front", "side", "back", "cut_line"})
    length = LengthIntent(
        overall=_enum(le.get("overall", length_default), "$.length.overall", LENGTHS),
        front=_optional_enum(le.get("front"), "$.length.front", LENGTHS),
        side=_optional_enum(le.get("side"), "$.length.side", LENGTHS),
        back=_optional_enum(le.get("back"), "$.length.back", LENGTHS),
        cut_line=_enum(le.get("cut_line", "soft"), "$.length.cut_line", {"blunt", "soft", "layered"}),
    )

    sh = _object(root.get("shape", {}), "$.shape", {
        "volume", "density", "texture", "wave_size", "wave_strength", "root_lift"
    })
    default_texture = "wavy" if family == "loose_long" else ("coily" if family == "coily" else "straight")
    shape = ShapeIntent(
        volume=_enum(sh.get("volume", "medium"), "$.shape.volume", {"low", "medium", "high"}),
        density=_enum(sh.get("density", "medium"), "$.shape.density", {"light", "medium", "full"}),
        texture=_enum(sh.get("texture", default_texture), "$.shape.texture", {"straight", "wavy", "curly", "coily"}),
        wave_size=_enum(sh.get("wave_size", "medium"), "$.shape.wave_size", {"small", "medium", "large"}),
        wave_strength=_enum(sh.get("wave_strength", "medium"), "$.shape.wave_strength", {"subtle", "medium", "strong"}),
        root_lift=_enum(sh.get("root_lift", "medium"), "$.shape.root_lift", {"low", "medium", "high"}),
    )

    dr = _object(root.get("drape", {}), "$.drape", {
        "gravity", "stiffness", "shoulder_routing", "body_clearance"
    })
    drape = DrapeIntent(
        gravity=_enum(dr.get("gravity", "natural"), "$.drape.gravity", {"light", "natural", "heavy"}),
        stiffness=_enum(dr.get("stiffness", "natural"), "$.drape.stiffness", {"soft", "natural", "firm"}),
        shoulder_routing=_enum(dr.get("shoulder_routing", "split"), "$.drape.shoulder_routing", {
            "natural", "split", "mostly_behind", "all_front", "all_behind"
        }),
        body_clearance=_enum(dr.get("body_clearance", "natural"), "$.drape.body_clearance", {"close", "natural", "loose"}),
    )

    co = _object(root.get("color", {}), "$.color", {"family", "rgb"})
    color_family = _enum(co.get("family", "dark_brown"), "$.color.family", {
        "black", "dark_brown", "brown", "auburn", "copper", "blonde",
        "platinum", "gray", "white", "custom"
    })
    rgb_raw = co.get("rgb")
    rgb = None
    if rgb_raw is not None:
        if not isinstance(rgb_raw, (list, tuple)) or len(rgb_raw) != 3:
            raise HairIntentError("$.color.rgb must be three numbers in [0, 1]")
        try:
            rgb = tuple(float(c) for c in rgb_raw)
        except (TypeError, ValueError) as exc:
            raise HairIntentError("$.color.rgb must be three numbers in [0, 1]") from exc
        if not all(np.isfinite(c) and 0.0 <= c <= 1.0 for c in rgb):
            raise HairIntentError("$.color.rgb must be three numbers in [0, 1]")
    if color_family == "custom" and rgb is None:
        raise HairIntentError("$.color.rgb is required when $.color.family is 'custom'")
    if color_family != "custom" and rgb is not None:
        raise HairIntentError("$.color.rgb is only allowed when $.color.family is 'custom'")
    color = ColorIntent(family=color_family, rgb=rgb)

    return HairIntent(
        schema_version=1, seed=seed, family=family, part=part,
        hairline=hairline, length=length, shape=shape, drape=drape, color=color,
    )


def load_hair_intent(source: str | Path | Mapping[str, Any]) -> HairIntent:
    """Load HairIntent from a mapping, JSON string, or JSON file path."""

    if isinstance(source, Mapping):
        return parse_hair_intent(source)
    if isinstance(source, Path):
        payload = json.loads(source.read_text())
    elif isinstance(source, str) and source.lstrip().startswith("{"):
        payload = json.loads(source)
    else:
        payload = json.loads(Path(source).read_text())
    return parse_hair_intent(payload)


def _clump_specs(style: Style) -> tuple:
    return (() if style.clump_field is None else (style.clump_field,)) + tuple(style.clump_fields)


def _apply_part(style: Style, intent: PartIntent) -> None:
    if intent.kind == "none":
        part_u = None
    elif intent.kind == "center":
        part_u = 0.0
    else:
        offset = {"subtle": 0.17, "moderate": 0.29, "deep": 0.41}[intent.position]
        part_u = offset if intent.side == "wearer_left" else -offset

    widths = {"narrow": (0.034, 0.011), "medium": (0.052, 0.017), "wide": (0.075, 0.025)}
    part_width, open_width = widths[intent.width]
    if style.cap is not None:
        style.cap.part_u = part_u
        style.cap.part_width = part_width
        style.cap.part_open_width = open_width
        style.cap.part_open = part_u is not None
        style.cap.part_depth = 0.94
        style.cap.part_start_v = {"short": 0.52, "to_crown": 0.14, "through_crown": 0.0}[intent.extent]
    for spec in _clump_specs(style):
        # A no-part style still needs a deterministic crown flow origin; it
        # simply omits the visible scalp opening.
        spec.part_u = 0.0 if part_u is None else part_u
        spec.part_open = part_u is not None
        spec.part_gap = open_width * 2.5 if part_u is not None else 0.0


def _apply_hairline(style: Style, intent: HairlineIntent) -> None:
    """Compile semantic boundary choices relative to the archetype's anatomy.

    Hairline chart ``v`` increases down the head.  The deltas intentionally
    preserve each preset's fit while changing recognizable landmarks rather
    than replacing them with one head-specific numeric template.
    """

    if style.cap is None or style.cap.hairline is None:
        return
    line = style.cap.hairline

    frontal_shift = {"low": 0.075, "natural": 0.0, "high": -0.075}[intent.height]
    line.front += frontal_shift
    line.mid += frontal_shift
    line.recess += frontal_shift * 0.75

    if intent.shape == "straight":
        # Reduce the normal centre-to-mid arch without erasing the temples.
        line.front += 0.045
        line.mid -= 0.020
    elif intent.shape == "widows_peak":
        # The centre chart landmark becomes a visibly lower point.
        line.front += 0.110
        line.mid -= 0.015

    if intent.temple_recession == "none":
        line.recess = min(line.recess, line.mid + 0.045)
        line.knee -= 0.025
    elif intent.temple_recession == "pronounced":
        line.recess += 0.100
        line.knee += 0.040

    line.sideburn += {"short": -0.140, "natural": 0.0, "long": 0.140}[intent.sideburns]
    line.nape += {"high": -0.120, "natural": 0.0, "low": 0.120}[intent.nape]
    line.jitter = {"clean": 0.006, "natural": line.jitter, "textured": 0.035}[intent.irregularity]

    # Keep deliberately extreme LLM combinations inside the fitted chart.
    line.front = float(np.clip(line.front, 0.50, 1.00))
    line.mid = float(np.clip(line.mid, 0.58, 1.08))
    line.recess = float(np.clip(line.recess, line.mid + 0.025, 1.22))
    line.knee = float(np.clip(line.knee, 1.04, 1.42))
    line.sideburn = float(np.clip(line.sideburn, 1.12, 1.62))
    line.nape = float(np.clip(line.nape, 1.16, 1.62))


def _apply_length(style: Style, intent: LengthIntent) -> None:
    overall = _LENGTH_SCALE[intent.overall]
    front = _LENGTH_SCALE[intent.front] if intent.front else overall
    side = _LENGTH_SCALE[intent.side] if intent.side else overall
    back = _LENGTH_SCALE[intent.back] if intent.back else overall
    for layer_index, spec in enumerate(_clump_specs(style)):
        if spec.mode != "long":
            continue
        spec.length_front = front
        spec.length_side = side
        spec.length = back
        if intent.cut_line == "blunt":
            spec.lower_width, spec.tip_jitter = 0.96, 0.025
        elif intent.cut_line == "soft":
            spec.lower_width, spec.tip_jitter = 0.78, 0.10
        else:
            # Visible top locks break into varied tips; the broader under-mass
            # remains fuller so layered hair still reads as one hairstyle.
            if layer_index == 0:
                spec.lower_width, spec.tip_jitter = 0.62, 0.18
            else:
                spec.lower_width, spec.tip_jitter = 0.88, 0.08
            if intent.front is None:
                spec.length_front *= 0.74


def _apply_shape(style: Style, texture: TextureSpec, intent: ShapeIntent) -> None:
    volume_factor = {"low": 0.78, "medium": 1.0, "high": 1.28}[intent.volume]
    lift_factor = {"low": 0.62, "medium": 1.0, "high": 1.48}[intent.root_lift]
    density_factor = {"light": 0.76, "medium": 1.0, "full": 1.22}[intent.density]
    if style.cap is not None:
        style.cap.volume *= volume_factor
        style.cap.volume_back *= volume_factor
        style.cap.lift *= lift_factor
    size_frequency = {"small": 2.35, "medium": 1.55, "large": 1.08}[intent.wave_size]
    strength = {"subtle": 0.55, "medium": 1.0, "strong": 1.34}[intent.wave_strength]
    texture_wave = {
        "straight": (0.0, 1.0),
        "wavy": (1.08 * strength, size_frequency),
        "curly": (1.18 * strength, max(2.15, size_frequency * 1.55)),
        "coily": (1.35 * strength, max(3.2, size_frequency * 2.3)),
    }
    texture.wave = {"straight": 0.0, "wavy": 0.55, "curly": 0.32, "coily": 0.18}[intent.texture]
    texture.curl = {"straight": 0.0, "wavy": 0.06, "curly": 0.72, "coily": 1.0}[intent.texture]
    specs = _clump_specs(style)
    layered_long = sum(spec.mode == "long" for spec in specs) > 1
    for layer_index, spec in enumerate(specs):
        spec.volume *= volume_factor
        spec.lift *= lift_factor
        if layered_long and spec.mode == "long":
            # Opaque hair density comes from broad overlapping masses, not
            # strand count.  Dozens of equally narrow guides create a jellyfish
            # fringe even when the coverage sum is mathematically identical.
            counts = {
                "light": (12, 7), "medium": (16, 9), "full": (20, 12),
            }
            widths = {
                "light": (0.27, 0.47), "medium": (0.30, 0.51), "full": (0.32, 0.54),
            }
            slot = min(layer_index, 1)
            spec.count = counts[intent.density][slot]
            spec.width = widths[intent.density][slot]
        else:
            spec.count = max(2, int(round(spec.count * density_factor)))
            # Retain approximately the same aggregate mass as density changes.
            spec.width *= float(np.sqrt(1.0 / density_factor))
        if spec.mode == "long":
            spec.wave, spec.wave_freq = texture_wave[intent.texture]


def _apply_drape(style: Style, intent: DrapeIntent) -> None:
    routing = {
        "natural": "natural", "split": "split", "mostly_behind": "mostly_back",
        "all_front": "front", "all_behind": "back",
    }[intent.shoulder_routing]
    for spec in _clump_specs(style):
        if spec.mode == "long":
            spec.drape = GuideDrapeSpec(
                enabled=True,
                routing=routing,
                gravity={"light": 0.68, "natural": 1.0, "heavy": 1.28}[intent.gravity],
                stiffness={"soft": 0.30, "natural": 0.55, "firm": 0.82}[intent.stiffness],
                collision_margin={"close": 0.065, "natural": 0.10, "loose": 0.16}[intent.body_clearance],
            )


def _apply_color(texture: TextureSpec, intent: ColorIntent) -> None:
    natural = {
        "black": (0.96, 0.16, 0.0),
        "dark_brown": (0.82, 0.24, 0.0),
        "brown": (0.64, 0.34, 0.0),
        "auburn": (0.64, 0.78, 0.0),
        "copper": (0.42, 0.94, 0.0),
        "blonde": (0.16, 0.34, 0.0),
        "platinum": (0.035, 0.12, 0.08),
        "gray": (0.46, 0.20, 0.72),
        "white": (0.02, 0.10, 0.96),
    }
    if intent.family == "custom":
        texture.dye = intent.rgb
        texture.dye_amount = 1.0
        texture.grey = 0.0
    else:
        texture.melanin, texture.redness, texture.grey = natural[intent.family]
        texture.dye = None


def compile_hair_intent(payload: HairIntent | Mapping[str, Any]) -> HairPlan:
    """Compile semantic HairIntent into a deterministic opaque-mesh plan."""

    intent = payload if isinstance(payload, HairIntent) else parse_hair_intent(payload)
    preset = _FAMILY_PRESET[intent.family]
    style, texture = copy.deepcopy(PRESETS[preset])
    style.name = f"intent_{intent.family}"

    _apply_part(style, intent.part)
    _apply_hairline(style, intent.hairline)
    _apply_length(style, intent.length)
    _apply_shape(style, texture, intent.shape)
    _apply_drape(style, intent.drape)
    _apply_color(texture, intent.color)

    if style.cap is not None:
        style.cap.seed = intent.seed
        if style.cap.hairline is not None:
            style.cap.hairline.seed = intent.seed + 101
    for i, spec in enumerate(_clump_specs(style)):
        spec.seed = intent.seed + 1009 * (i + 1)
    texture.seed = intent.seed + 7919
    return HairPlan(intent=intent, base_preset=preset, style=style, texture=texture)
