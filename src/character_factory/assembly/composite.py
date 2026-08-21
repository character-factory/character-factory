"""UV compositing: one albedo atlas from the generated texture layers.

The normative order (SPEC.md §9): skin, then garment, then shoe.
The garment layer is painted over an unoccupied (black) background;
coverage is recovered from the image itself by a calibrated luminance
key — the generators keep real garment pixels above a value floor, so a
keying cutoff well below that floor separates cloth from background,
including deliberately dark cloth. The shoe layer arrives as an RGBA
overlay already carrying its coverage: the shoe generator paints a
one-foot canvas, and `assembly.footwear` bakes it through the component's
foot chart into atlas space.

Region masks (where the garment layer may not paint, where the shoe
overlay may) are atlas data from the `assembly-assets` component, never
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["AtlasDefinition", "composite_albedo", "coverage_mask"]

# The keying cutoff, in 8-bit value units. Generators guarantee garment
# pixels keep max(R,G,B) comfortably above this; background is black.
DEFAULT_KEY_CUTOFF = 20
# Feather radius (pixels) applied to recovered coverage so composited edges
# do not alias.
DEFAULT_FEATHER_RADIUS = 1.5


@dataclass
class AtlasDefinition:
    """Atlas metadata: resolution, region masks, and key calibration.

    Loaded from the `assembly-assets` component (`atlas.json` plus mask
    images); tests construct it directly.
    """

    resolution: int
    head_mask: np.ndarray | None = None   # bool (H, W): garment never paints here
    feet_mask: np.ndarray | None = None   # bool (H, W): shoe only paints here
    key_cutoff: int = DEFAULT_KEY_CUTOFF
    feather_radius: float = DEFAULT_FEATHER_RADIUS
    extras: dict = field(default_factory=dict)

    @classmethod
    def load(cls, component_dir: str | Path) -> "AtlasDefinition":
        import json

        from PIL import Image

        component_dir = Path(component_dir)
        config = json.loads(
            (component_dir / "atlas.json").read_text(encoding="utf-8")
        )
        if config.get("format") != "character-factory/atlas-metadata":
            raise ValueError(f"{component_dir}/atlas.json is not atlas metadata")

        def mask(name: str) -> np.ndarray | None:
            path = config.get("masks", {}).get(name)
            if path is None:
                return None
            image = Image.open(component_dir / path).convert("L")
            return np.asarray(image) > 127

        key = config.get("key", {})
        return cls(
            resolution=config["resolution"],
            head_mask=mask("head"),
            feet_mask=mask("feet"),
            key_cutoff=key.get("cutoff", DEFAULT_KEY_CUTOFF),
            feather_radius=key.get("feather_radius", DEFAULT_FEATHER_RADIUS),
            extras=config,
        )

    def at_resolution(self, resolution: int) -> "AtlasDefinition":
        """This atlas with masks rescaled to another working resolution —
        recipes may override the bake resolution (SPEC.md §5.1)."""
        if resolution == self.resolution:
            return self

        def scale(mask: np.ndarray | None) -> np.ndarray | None:
            if mask is None:
                return None
            from PIL import Image

            image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            resized = image.resize((resolution, resolution), Image.NEAREST)
            return np.asarray(resized) > 127

        return AtlasDefinition(
            resolution=resolution,
            head_mask=scale(self.head_mask),
            feet_mask=scale(self.feet_mask),
            key_cutoff=self.key_cutoff,
            feather_radius=self.feather_radius,
            extras=self.extras,
        )


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur. Deliberately dependency-free (numpy only)."""
    if radius <= 0:
        return values
    size = 2 * radius + 1
    kernel = np.ones(size) / size
    padded = np.pad(values, radius, mode="edge")
    horizontal = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), 1, padded
    )[radius:-radius, :]
    return np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"), 0,
        np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge"),
    )


def coverage_mask(
    layer: np.ndarray, cutoff: int, feather_radius: float
) -> np.ndarray:
    """Recover a coverage alpha in [0, 1] from a painted-over-black layer.

    Luminance key, then an inward feather: texels at the covered side of the
    boundary get partial alpha so composited edges do not alias, texels
    outside the keyed region stay exactly 0 (the layer never bleeds onto
    skin), and fully interior texels stay exactly 1.
    """
    occupancy = (layer.max(axis=2) >= cutoff).astype(np.float64)
    if feather_radius <= 0:
        return occupancy
    radius = max(1, round(feather_radius))
    # One box-blur pass: texels more than `radius` inside the boundary stay
    # exactly 1, so the feather is a bounded inward ramp, not a global soften.
    return _box_blur(occupancy, radius) * occupancy


def composite_albedo(
    skin: np.ndarray,
    garment: np.ndarray,
    shoe_overlay: np.ndarray | None,
    atlas: AtlasDefinition,
) -> np.ndarray:
    """The body's final albedo: skin, then garment, then shoe (SPEC.md §9).

    `skin` and `garment` are (H, W, 3) uint8 at the atlas resolution;
    `shoe_overlay` is the (H, W, 4) uint8 result of
    `footwear.bake_shoe_overlay` — its alpha is authoritative coverage, so
    it composites directly (with the same inward edge feather the keyed
    garment layer gets, and confined to the feet mask when one is present).
    """
    expected = (atlas.resolution, atlas.resolution, 3)
    for name, layer in (("skin", skin), ("garment", garment)):
        if layer.shape != expected:
            raise ValueError(f"{name} layer is {layer.shape}, atlas wants {expected}")
    if shoe_overlay is not None and shoe_overlay.shape != (*expected[:2], 4):
        raise ValueError(
            f"shoe overlay is {shoe_overlay.shape}, atlas wants {(*expected[:2], 4)}"
        )

    result = skin.astype(np.float64)

    garment_alpha = coverage_mask(garment, atlas.key_cutoff, atlas.feather_radius)
    if atlas.head_mask is not None:
        garment_alpha = garment_alpha * ~atlas.head_mask
    result = result * (1 - garment_alpha[..., None]) + garment * garment_alpha[..., None]

    if shoe_overlay is not None:
        shoe_alpha = shoe_overlay[..., 3].astype(np.float64) / 255.0
        if atlas.feather_radius > 0:
            radius = max(1, round(atlas.feather_radius))
            shoe_alpha = _box_blur(shoe_alpha, radius) * (shoe_alpha > 0)
        if atlas.feet_mask is not None:
            shoe_alpha = shoe_alpha * atlas.feet_mask
        result = (
            result * (1 - shoe_alpha[..., None])
            + shoe_overlay[..., :3] * shoe_alpha[..., None]
        )

    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
