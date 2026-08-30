"""The atlas definition: region masks and keying constants, as data.

The body mesh renders skin only — garments and shoes are geometry
(assembly.garment_shell), never composited into the body albedo. What
remains here is the atlas contract the extractors consume: region masks
(where garment coverage may not key — the head; where shoe coverage is
confined — the feet) from the `assembly-assets` component, never code,
plus the shared keying constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["AtlasDefinition"]

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
