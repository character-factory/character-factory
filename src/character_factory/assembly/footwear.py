"""Footwear: mapping the shoe canvas onto the body atlas's foot islands.

`make-shoe` paints one shoe on a fixed single-foot canvas (leg/shaft on the
left, upper foot at upper right, sole at lower right) rather than into the
body atlas directly. The mapping between that canvas and the atlas's six
foot UV islands is the component's **foot chart** — per-texcoord canvas
coordinates shipped with the model (`foot_chart.npz` / `foot_chart.json` /
`foot_chart_mask.png`), because the canvas layout is part of each model
version's output contract. Both feet wear the same design: the
chart maps one foot directly and the other through a horizontal flip, so
callers never mirror the canvas themselves.

The bake recovers a style-aware occupancy for the canvas (open styles keep
only their straps; the shaft region above the ankle is cut down to the
style's shaft height), then rasterizes every foot triangle from canvas
space into atlas space, producing an RGBA overlay for the compositor.

All image coordinates here are image-fraction convention (origin top-left),
matching the rig's texcoords and the chart as shipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FootChart", "bake_shoe_overlay", "shaft_clause"]

# Straps darker/more saturated than the canvas background read as covered;
# these are the fixed thresholds for styles with hard-edged straps.
_STRAP_LUMINANCE_MAX = 110.0
_STRAP_SATURATION_MIN = 0.12
# A strap extraction keeping less than this fraction of the canvas failed;
# a schematic two-band fallback is used instead so open styles never render
# as fully closed shoes.
_STRAP_MINIMUM_FRACTION = 0.003


@dataclass
class FootChart:
    """The canvas↔atlas mapping and style vocabulary of a shoe component."""

    chart_uv: np.ndarray     # (T, 2) float32 canvas coords, -1 where unmapped
    part_index: np.ndarray   # (T,) int8: 0 leg_ankle, 1 upper_foot, 2 sole
    mask: np.ndarray         # (H, W) bool canvas occupancy of the chart
    parts: dict              # name -> {"chart_bounds": [x0, y0, x1, y1], ...}
    styles: list             # [{"name", "leg_fraction", "treatment", "keywords"}]
    texcoord_count: int

    @classmethod
    def load(cls, component_dir: str | Path) -> "FootChart":
        from PIL import Image

        component_dir = Path(component_dir)
        config = json.loads(
            (component_dir / "foot_chart.json").read_text(encoding="utf-8")
        )
        if config.get("format") != "character-factory/foot-chart":
            raise ValueError(f"{component_dir}/foot_chart.json is not a foot chart")
        with np.load(component_dir / "foot_chart.npz") as data:
            chart_uv = data["chart_uv"].astype(np.float32)
            part_index = data["part_index"].astype(np.int8)
        mask = (
            np.asarray(Image.open(component_dir / "foot_chart_mask.png").convert("L"))
            > 127
        )
        return cls(
            chart_uv=chart_uv,
            part_index=part_index,
            mask=mask,
            parts={part["name"]: part for part in config["parts"]},
            styles=config["styles"],
            texcoord_count=config["texcoord_count"],
        )

    def style_for_prompt(self, prompt: str) -> dict:
        """The declared style whose longest keyword appears in the prompt;
        the keywordless declared style is the default. Deterministic, so a
        stored recipe always bakes the same overlay (SPEC.md §9)."""
        text = " ".join(prompt.lower().split())
        best: tuple[int, dict] | None = None
        default: dict | None = None
        for style in self.styles:
            if not style["keywords"]:
                default = style
            for keyword in style["keywords"]:
                if keyword in text and (best is None or len(keyword) > best[0]):
                    best = (len(keyword), style)
        if best is not None:
            return best[1]
        if default is None:
            raise ValueError("foot chart declares no default (keywordless) style")
        return default

    def style(self, name: str) -> dict:
        for style in self.styles:
            if style["name"] == name:
                return style
        raise ValueError(f"style {name!r} is not in the component's vocabulary")


def shaft_clause(style: dict, inference: dict) -> str:
    """The style's shaft clause for a component whose conditioning template
    carries a `{shaft_clause}` hole. The clause wording is the component's
    caption vocabulary and comes from its registry entry
    (`shaft_clause_empty` / `shaft_clause_shafted`, the latter with a
    `{percent}` hole) — data, never code."""
    fraction = float(style["leg_fraction"])
    key = "shaft_clause_empty" if fraction <= 0 else "shaft_clause_shafted"
    template = inference.get(key)
    if template is None:
        raise ValueError(
            f"the component's conditioning template uses {{shaft_clause}} but "
            f"its registry entry declares no {key!r}"
        )
    return template.format(percent=round(fraction * 100))


def _otsu(values: np.ndarray) -> float:
    """Otsu's threshold over 8-bit-range values."""
    histogram, edges = np.histogram(
        np.clip(values, 0.0, 255.0), bins=128, range=(0.0, 255.0)
    )
    weights = histogram.astype(np.float64) / max(histogram.sum(), 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    below = np.cumsum(weights)
    below_mean = np.cumsum(weights * centers)
    spread = below * (1.0 - below)
    score = np.zeros_like(spread)
    valid = spread > 1e-12
    score[valid] = (
        (below_mean[-1] * below[valid] - below_mean[valid]) ** 2 / spread[valid]
    )
    return float(centers[int(np.argmax(score))])


def _local_grid(shape: tuple, bounds: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inside region, local x, local y) for a part's chart bounds."""
    height, width = shape
    x = np.arange(width, dtype=np.float64) / max(width - 1, 1)
    y = np.arange(height, dtype=np.float64) / max(height - 1, 1)
    grid_x, grid_y = np.meshgrid(x, y)
    x0, y0, x1, y1 = bounds
    inside = (grid_x >= x0) & (grid_x <= x1) & (grid_y >= y0) & (grid_y <= y1)
    local_x = np.clip((grid_x - x0) / max(x1 - x0, 1e-9), 0.0, 1.0)
    local_y = np.clip((grid_y - y0) / max(y1 - y0, 1e-9), 0.0, 1.0)
    return inside, local_x, local_y


def _strap_occupancy(
    canvas: np.ndarray, occupied: np.ndarray, chart: FootChart, treatment: str
) -> np.ndarray:
    """Occupancy over the upper-foot region for open styles: keep straps,
    drop the bare-skin openings the model paints as light background."""
    from scipy import ndimage

    height, width = occupied.shape
    upper, local_x, local_y = _local_grid(
        occupied.shape, chart.parts["upper_foot"]["chart_bounds"]
    )

    smoothed = ndimage.gaussian_filter(
        canvas.astype(np.float32), sigma=(4.0, 4.0, 0.0)
    )
    luminance = smoothed.mean(axis=2)
    peak = smoothed.max(axis=2)
    saturation = (peak - smoothed.min(axis=2)) / np.maximum(peak, 1.0)

    if treatment == "adaptive_straps":
        region_luminance = luminance[upper & occupied]
        region_saturation = saturation[upper & occupied]
        dark_cutoff = (
            min(_otsu(region_luminance), float(np.quantile(region_luminance, 0.35)))
            if region_luminance.size
            else 55.0
        )
        saturated_cutoff = (
            max(0.16, float(np.quantile(region_saturation, 0.75)))
            if region_saturation.size
            else 0.18
        )
        straps = upper & occupied & (
            (luminance <= dark_cutoff)
            | ((saturation >= saturated_cutoff) & (luminance < 160.0))
        )
    else:
        straps = (
            upper & occupied
            & (luminance < _STRAP_LUMINANCE_MAX)
            & (saturation > _STRAP_SATURATION_MIN)
        )

    kernel = np.ones((5, 5), dtype=bool)
    straps = ndimage.binary_opening(straps, kernel)
    straps = ndimage.binary_closing(straps, kernel)
    labels, count = ndimage.label(straps)

    if count and treatment == "adaptive_straps":
        sizes = np.bincount(labels.ravel())
        minimum = max(96, int(0.002 * np.count_nonzero(upper)))
        keep = np.flatnonzero(sizes >= minimum)
        straps = np.isin(labels, keep[keep > 0])
        straps = ndimage.binary_closing(straps, np.ones((11, 11), dtype=bool))
        straps = ndimage.binary_fill_holes(straps)
        straps = ndimage.binary_dilation(straps, np.ones((3, 3), dtype=bool))
        straps &= upper & occupied

    if count and treatment == "toe_post_straps":
        # Keep only strap components reaching the toe end (canvas left),
        # clear the heel center, and taper the side anchors so the straps
        # narrow toward their attachment points.
        toe = np.unique(labels[(local_x < 0.35) & (labels > 0)])
        if toe.size:
            straps = np.isin(labels, toe)
        heel_center = (local_x > 0.58) & (local_y > 0.25) & (local_y < 0.75)
        straps &= ~heel_center
        progress = np.clip((local_x - 0.44) / 0.34, 0.0, 1.0)
        eased = progress**2 * (3.0 - 2.0 * progress)
        edge = 0.50 + 0.48 * eased
        straps &= (local_x <= 0.44) | (local_y >= edge) | (local_y <= 1.0 - edge)

    if np.count_nonzero(straps) < _STRAP_MINIMUM_FRACTION * height * width:
        # Schematic V-straps (canvas-left toe to the sides of the ankle).
        straps = (
            (np.abs(local_y - (0.72 - 0.55 * local_x)) < 0.075)
            | (np.abs(local_y - (0.28 + 0.48 * local_x)) < 0.075)
        ) & upper

    result = occupied.copy()
    result &= ~upper | straps
    return result


def _canvas_occupancy(
    canvas: np.ndarray, chart: FootChart, style: dict
) -> np.ndarray:
    """The canvas texels that make it onto the body, for one style."""
    from PIL import Image

    height, width = canvas.shape[:2]
    mask = chart.mask
    if mask.shape != (height, width):
        resized = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        mask = np.asarray(resized) > 127
    occupied = mask.copy()

    # The shaft: the leg region is cut down to the style's shaft height,
    # measured up from the ankle (the bottom edge of the leg cell). A small
    # margin widens the cut so island-edge texels cannot survive it.
    x0, y0, x1, y1 = chart.parts["leg_ankle"]["chart_bounds"]
    margin_x, margin_y = 2.5 / max(width - 1, 1), 2.5 / max(height - 1, 1)
    leg_region, _, _ = _local_grid(
        occupied.shape, [x0 - margin_x, y0 - margin_y, x1 + margin_x, y1 + margin_y]
    )
    fraction = float(style["leg_fraction"])
    if fraction <= 0.0:
        occupied &= ~leg_region
    else:
        grid_y = (
            np.arange(height, dtype=np.float64)[:, None] / max(height - 1, 1)
        ) * np.ones((1, width))
        occupied &= ~leg_region | (grid_y >= y1 - (y1 - y0) * fraction)
        # Within the kept shaft band, drop texels the generator left as
        # canvas background — a shaft painted shorter than the style's
        # declared height must not composite background over the leg. The
        # background reference is the canvas's own unmapped region, so no
        # calibration constant is introduced.
        background_pixels = canvas[~mask]
        if background_pixels.size:
            background = np.median(
                background_pixels.reshape(-1, canvas.shape[2]), axis=0
            )
            deviation = np.abs(
                canvas.astype(np.float64) - background
            ).mean(axis=2)
            occupied &= ~leg_region | (deviation > 12.0)

    if style["treatment"] in ("toe_post_straps", "adaptive_straps"):
        occupied = _strap_occupancy(canvas, occupied, chart, style["treatment"])
    return occupied


def _rasterize(
    chart: FootChart,
    texcoords: np.ndarray,
    texcoord_faces: np.ndarray,
    canvas_rgba: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Every foot triangle, canvas space → atlas space, bilinear-sampled.

    Overlapping texels keep the most-covered sample (highest alpha), so
    seam-adjacent triangles cannot punch holes in each other.
    """
    mapped = (chart.part_index[texcoord_faces] >= 0).all(axis=1)
    overlay = np.zeros((resolution, resolution, 4), dtype=np.float64)
    canvas = canvas_rgba.astype(np.float64)
    canvas_h, canvas_w = canvas.shape[:2]

    for triangle in texcoord_faces[mapped]:
        corners = texcoords[triangle].astype(np.float64) * (resolution - 1)
        low = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(corners.max(axis=0)).astype(int), resolution - 1)
        if np.any(high < low):
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1)
        )
        a, b, c = corners
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        weight_a = (
            (b[1] - c[1]) * (grid_x - c[0]) + (c[0] - b[0]) * (grid_y - c[1])
        ) / denominator
        weight_b = (
            (c[1] - a[1]) * (grid_x - c[0]) + (a[0] - c[0]) * (grid_y - c[1])
        ) / denominator
        weight_c = 1.0 - weight_a - weight_b
        inside = (weight_a >= -1e-5) & (weight_b >= -1e-5) & (weight_c >= -1e-5)
        if not inside.any():
            continue
        source = chart.chart_uv[triangle].astype(np.float64)
        sample_x = np.clip(
            (weight_a * source[0, 0] + weight_b * source[1, 0] + weight_c * source[2, 0])
            * (canvas_w - 1),
            0,
            canvas_w - 1,
        )[inside]
        sample_y = np.clip(
            (weight_a * source[0, 1] + weight_b * source[1, 1] + weight_c * source[2, 1])
            * (canvas_h - 1),
            0,
            canvas_h - 1,
        )[inside]
        x_floor = np.floor(sample_x).astype(np.int64)
        y_floor = np.floor(sample_y).astype(np.int64)
        x_next = np.minimum(x_floor + 1, canvas_w - 1)
        y_next = np.minimum(y_floor + 1, canvas_h - 1)
        fraction_x = (sample_x - x_floor)[:, None]
        fraction_y = (sample_y - y_floor)[:, None]
        top = canvas[y_floor, x_floor] * (1 - fraction_x) + canvas[y_floor, x_next] * fraction_x
        bottom = canvas[y_next, x_floor] * (1 - fraction_x) + canvas[y_next, x_next] * fraction_x
        samples = top * (1 - fraction_y) + bottom * fraction_y

        target_y = grid_y[inside]
        target_x = grid_x[inside]
        keep = samples[:, 3] >= overlay[target_y, target_x, 3]
        overlay[target_y[keep], target_x[keep]] = samples[keep]

    overlay = np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
    overlay[overlay[..., 3] < 8, :3] = 0
    return overlay


def bake_shoe_overlay(
    canvas: np.ndarray,
    chart: FootChart,
    texcoords: np.ndarray,
    texcoord_faces: np.ndarray,
    *,
    style: str | None = None,
    prompt: str = "",
    resolution: int,
) -> np.ndarray:
    """The generated one-foot canvas → an RGBA overlay for the body atlas.

    `canvas` is the make-shoe output, (H, W, 3) uint8, any square size.
    The style comes from `style` when given, else from the prompt via the
    chart's declared keywords. Returns (resolution, resolution, 4) uint8.
    """
    if texcoords.shape[0] != chart.texcoord_count:
        raise ValueError(
            f"foot chart maps {chart.texcoord_count} texcoords but the rig has "
            f"{texcoords.shape[0]} — the chart belongs to a different UV layout"
        )
    chosen = chart.style(style) if style is not None else chart.style_for_prompt(prompt)
    occupancy = _canvas_occupancy(canvas, chart, chosen)
    canvas_rgba = np.zeros((*canvas.shape[:2], 4), dtype=np.uint8)
    canvas_rgba[..., :3] = np.where(occupancy[..., None], canvas, 0)
    canvas_rgba[..., 3] = occupancy.astype(np.uint8) * 255
    return _rasterize(chart, texcoords, texcoord_faces, canvas_rgba, resolution)
