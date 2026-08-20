"""UV compositing: one albedo atlas from the generated texture layers.

The normative order (SPEC.md §9): skin, then garment, then shoe.
Garment and shoe layers are painted over an unoccupied (black)
background; coverage is recovered from the image itself by a calibrated
luminance key — the generators keep real garment pixels above a value floor,
so a keying cutoff well below that floor separates cloth from background,
including deliberately dark cloth.

Region masks (where the garment layer may not paint, where the shoe layer
may) are atlas data from the `assembly-assets` component, never code.
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
    shoe: np.ndarray | None,
    atlas: AtlasDefinition,
) -> np.ndarray:
    """The body's final albedo: skin, then garment, then shoe (SPEC.md §9).

    All layers are (H, W, 3) uint8 at the atlas resolution.
    """
    expected = (atlas.resolution, atlas.resolution, 3)
    for name, layer in (("skin", skin), ("garment", garment)):
        if layer.shape != expected:
            raise ValueError(f"{name} layer is {layer.shape}, atlas wants {expected}")
    if shoe is not None and shoe.shape != expected:
        raise ValueError(f"shoe layer is {shoe.shape}, atlas wants {expected}")

    result = skin.astype(np.float64)

    garment_alpha = coverage_mask(garment, atlas.key_cutoff, atlas.feather_radius)
    if atlas.head_mask is not None:
        garment_alpha = garment_alpha * ~atlas.head_mask
    result = result * (1 - garment_alpha[..., None]) + garment * garment_alpha[..., None]

    if shoe is not None:
        shoe_alpha = coverage_mask(shoe, atlas.key_cutoff, atlas.feather_radius)
        if atlas.feet_mask is not None:
            shoe_alpha = shoe_alpha * atlas.feet_mask
        result = result * (1 - shoe_alpha[..., None]) + shoe * shoe_alpha[..., None]

    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
