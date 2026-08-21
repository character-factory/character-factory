"""Footwear: the canvas→atlas foot-chart bake — mirroring, shaft cut,
style-aware occupancy, and style inference from recipe prompts."""

from pathlib import Path

import numpy as np
import pytest

from character_factory.assembly.footwear import FootChart, bake_shoe_overlay

RES = 64          # atlas resolution under test
CANVAS = 64       # canvas resolution under test

# Synthetic layout: the "right foot" quad sits at atlas (0.55..0.90)² and
# maps identically into the chart's upper-foot cell; the "left foot" quad
# sits at atlas (0.05..0.40)² and maps into the same cell mirrored
# (chart x = 0.95 − atlas x), like the real chart's negative-X foot.
RIGHT = [(0.55, 0.55), (0.90, 0.55), (0.90, 0.90), (0.55, 0.90)]
LEFT = [(0.05, 0.55), (0.40, 0.55), (0.40, 0.90), (0.05, 0.90)]
TEXCOORDS = np.array(RIGHT + LEFT, dtype=np.float32)
TEXCOORD_FACES = np.array(
    [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64
)

STYLES = [
    {"name": "low_top", "leg_fraction": 0.0, "treatment": "closed",
     "keywords": []},
    {"name": "flip_flop", "leg_fraction": 0.0, "treatment": "toe_post_straps",
     "keywords": ["flip flop", "flip-flop", "flipflop"]},
    {"name": "open_sandal", "leg_fraction": 0.0, "treatment": "adaptive_straps",
     "keywords": ["sandal"]},
    {"name": "mid_boot", "leg_fraction": 0.2, "treatment": "closed",
     "keywords": ["boot", "boots"]},
]
PARTS = {
    "leg_ankle": {"name": "leg_ankle", "chart_bounds": [0.0, 0.0, 0.45, 1.0]},
    "upper_foot": {"name": "upper_foot", "chart_bounds": [0.5, 0.5, 1.0, 1.0]},
    "sole": {"name": "sole", "chart_bounds": [0.5, 0.0, 1.0, 0.45]},
}


def chart(left_is_leg=False) -> FootChart:
    chart_uv = np.full((8, 2), -1.0, dtype=np.float32)
    part_index = np.full(8, -1, dtype=np.int8)
    for i, (x, y) in enumerate(RIGHT):
        chart_uv[i] = (x, y)
        part_index[i] = 1
    for i, (x, y) in enumerate(LEFT, start=4):
        if left_is_leg:
            chart_uv[i] = (x, y)        # inside the leg cell as-is
            part_index[i] = 0
        else:
            chart_uv[i] = (0.95 - x, y)  # mirrored into the upper cell
            part_index[i] = 1
    return FootChart(
        chart_uv=chart_uv,
        part_index=part_index,
        mask=np.ones((CANVAS, CANVAS), dtype=bool),
        parts=PARTS,
        styles=STYLES,
        texcoord_count=8,
    )


def canvas_halves() -> np.ndarray:
    """Green left of chart x=0.725, blue right of it."""
    image = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    split = int(0.725 * CANVAS)
    image[:, :split] = (20, 200, 20)
    image[:, split:] = (20, 20, 200)
    return image


def at(overlay: np.ndarray, x: float, y: float) -> np.ndarray:
    return overlay[round(y * (RES - 1)), round(x * (RES - 1))]


def bake(c: FootChart, image: np.ndarray, **kwargs) -> np.ndarray:
    return bake_shoe_overlay(
        image, c, TEXCOORDS, TEXCOORD_FACES, resolution=RES, **kwargs
    )


def test_closed_bake_covers_both_feet_and_nothing_else():
    overlay = bake(chart(), canvas_halves(), style="low_top")
    assert at(overlay, 0.7, 0.7)[3] == 255       # right foot covered
    assert at(overlay, 0.2, 0.7)[3] == 255       # left foot covered
    assert at(overlay, 0.5, 0.2)[3] == 0         # off-foot atlas is empty
    assert at(overlay, 0.97, 0.97)[3] == 0


def test_left_foot_mirrors_the_design():
    overlay = bake(chart(), canvas_halves(), style="low_top")
    # Right foot reads the canvas directly: green toward low x, blue high.
    assert at(overlay, 0.60, 0.70)[1] > 150      # green
    assert at(overlay, 0.85, 0.70)[2] > 150      # blue
    # The left foot mirrors: blue toward low atlas x, green toward high.
    assert at(overlay, 0.10, 0.70)[2] > 150      # blue
    assert at(overlay, 0.35, 0.70)[1] > 150      # green


def test_shaftless_style_cuts_the_leg_region():
    overlay = bake(chart(left_is_leg=True), canvas_halves(), style="low_top")
    assert at(overlay, 0.7, 0.7)[3] == 255       # foot island still covered
    assert at(overlay, 0.2, 0.7)[3] == 0         # leg island fully cut


def test_shaft_fraction_keeps_only_the_lower_band():
    # mid_boot keeps the bottom 20% of the leg cell (ankle upward).
    overlay = bake(chart(left_is_leg=True), canvas_halves(), style="mid_boot")
    assert at(overlay, 0.2, 0.6)[3] == 0         # above the shaft line
    assert at(overlay, 0.2, 0.88)[3] == 255      # below it


def test_flip_flop_on_a_strapless_canvas_falls_back_to_schematic_straps():
    uniform = np.full((CANVAS, CANVAS, 3), (240, 235, 228), dtype=np.uint8)
    closed = bake(chart(), uniform, style="low_top")
    open_style = bake(chart(), uniform, style="flip_flop")
    closed_area = int((closed[..., 3] > 0).sum())
    open_area = int((open_style[..., 3] > 0).sum())
    assert 0 < open_area < closed_area           # straps, not a closed shoe


def test_dark_straps_survive_flip_flop_extraction():
    # A larger canvas: the strap-extraction smoothing and morphology are
    # calibrated for generator-sized canvases, not thumbnails. Any square
    # canvas size is legal input (the chart samples it bilinearly).
    size = 256
    image = np.full((size, size, 3), (235, 230, 225), dtype=np.uint8)
    # A saturated dark band from the toe end across the upper cell.
    image[int(0.55 * size):int(0.75 * size), int(0.50 * size):] = (120, 30, 20)
    overlay = bake(chart(), image, style="flip_flop")
    strap_sample = at(overlay, 0.66, 0.66)       # atlas point on the band
    assert strap_sample[3] == 255
    assert strap_sample[0] > 80 and strap_sample[2] < 60


def test_style_inference_from_prompt():
    c = chart()
    assert c.style_for_prompt("red flip flops")["name"] == "flip_flop"
    assert c.style_for_prompt("brown leather sandals")["name"] == "open_sandal"
    assert c.style_for_prompt("white sneakers")["name"] == "low_top"


def test_texcoord_count_mismatch_is_a_hard_error():
    with pytest.raises(ValueError, match="different UV layout"):
        bake_shoe_overlay(
            canvas_halves(), chart(),
            TEXCOORDS[:6], TEXCOORD_FACES[:2], resolution=RES,
        )


def _newest_real_chart():
    root = Path.home() / ".cache/character-factory/components/make-shoe"
    if not root.is_dir():
        return root / "none"
    versions = sorted(d for d in root.iterdir() if (d / "foot_chart.json").is_file())
    return versions[-1] if versions else root / "none"


REAL_CHART = _newest_real_chart()


@pytest.mark.skipif(
    not (REAL_CHART / "foot_chart.json").is_file(),
    reason="make-shoe foot chart not in the local cache",
)
def test_real_chart_loads_and_matches_the_rig_contract():
    real = FootChart.load(REAL_CHART)
    assert real.texcoord_count == 19455          # the rig's unwelded UV count
    assert set(real.parts) == {"leg_ankle", "upper_foot", "sole"}
    mapped = real.part_index >= 0
    assert (real.chart_uv[mapped] >= 0).all() and (real.chart_uv[mapped] <= 1).all()
    assert {s["name"] for s in real.styles} >= {"flip_flop", "open_sandal", "low_top"}
    # Exactly one keywordless default style.
    assert sum(1 for s in real.styles if not s["keywords"]) == 1

def test_shaft_clause_comes_from_registry_data():
    from character_factory.assembly.footwear import shaft_clause

    inference = {"shaft_clause_empty": "no upper strip",
                 "shaft_clause_shafted": "a {percent}% upper strip"}
    assert shaft_clause({"leg_fraction": 0.0}, inference) == "no upper strip"
    assert shaft_clause({"leg_fraction": 0.36}, inference) == "a 36% upper strip"
    with pytest.raises(ValueError, match="declares no"):
        shaft_clause({"leg_fraction": 0.24}, {})


def test_boot_keywords_resolve_with_longest_match():
    c = chart()
    assert c.style_for_prompt("black leather boots")["name"] == "mid_boot"


def test_short_painted_shaft_does_not_composite_background():
    # The generator may paint a shorter shaft than the style's declared
    # height; unpainted background inside the geometric band must not
    # reach the body. Background is estimated from the unmapped region.
    c = chart(left_is_leg=True)
    c.mask = np.ones((CANVAS, CANVAS), dtype=bool)
    c.mask[:8, :8] = False                      # unmapped corner = background
    image = np.full((CANVAS, CANVAS, 3), (28, 28, 32), dtype=np.uint8)
    painted = (slice(int(0.80 * CANVAS), int(0.90 * CANVAS)),
               slice(int(0.05 * CANVAS), int(0.40 * CANVAS)))
    image[painted] = (120, 40, 50)              # the actual shaft band
    overlay = bake(c, image, style="mid_boot")  # keeps y >= 0.8 of leg cell
    assert at(overlay, 0.2, 0.85)[3] == 255     # painted shaft survives
    assert at(overlay, 0.2, 0.62)[3] == 0       # above the band: cut anyway
    # Inside the geometric band but unpainted: background keyed out.
    image2 = image.copy(); image2[painted] = (28, 28, 32)
    bare = bake(c, image2, style="mid_boot")
    assert bare[..., 3].max() >= 0              # runs clean
    assert at(bare, 0.2, 0.85)[3] == 0          # nothing painted -> nothing kept
