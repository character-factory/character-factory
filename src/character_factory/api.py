"""The library's assembly entry point.

`assemble(character, assets_dir, out_path)` is the deterministic half of the
product promise: a character file plus its baked assets in, a rigged .glb
out (SPEC.md §9). It runs everywhere — CUDA is never required here.

`create` / `bake` / `make` arrive with the interpreter and texture pipelines;
they are deliberately absent rather than stubbed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from character_factory.schema import Character

__all__ = ["AssetError", "assemble"]


class AssetError(ValueError):
    """A required asset is missing or fails its pinned hash."""


def _load_asset(assets_dir: Path, slot: str, character: Character) -> Path:
    path = assets_dir / f"{slot}.png"   # the slot's albedo map (SPEC.md §8)
    if not path.is_file():
        raise AssetError(f"missing asset for slot {slot!r}: {path}")
    pinned = character.asset_maps().get(slot, {}).get("albedo")
    if pinned is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != pinned["sha256"]:
            raise AssetError(
                f"asset {path} does not match the character's pinned hash for "
                f"{slot!r} — refusing to silently substitute (SPEC.md §8)"
            )
    return path


def assemble(
    character: Character | str | Path,
    assets_dir: str | Path,
    out_path: str | Path,
    *,
    registry=None,
    device: str = "cpu",
) -> Path:
    """Build the rigged .glb for a character from its baked assets.

    Albedo asset files are looked up in `assets_dir` by slot name
    (`skin.png`, `eye.png`, `garment.png`, optional `shoe.png`). When the
    character carries an `assets` block, every file is verified against its
    pinned hash before use; a mismatch is a hard error.
    """
    import numpy as np
    from PIL import Image

    from character_factory.assembly import export_character_glb, load_rig
    from character_factory.assembly.composite import AtlasDefinition, composite_albedo
    from character_factory.registry import ComponentNotPublished, Registry

    if not isinstance(character, Character):
        character = Character.load(character)
    assets_dir = Path(assets_dir)
    registry = registry or Registry.default()

    def layer(slot: str) -> np.ndarray:
        path = _load_asset(assets_dir, slot, character)
        return np.asarray(Image.open(path).convert("RGB"))

    skin = layer("skin")
    garment = layer("garment")
    shoe = layer("shoe") if "shoe" in character.textures else None

    try:
        atlas = AtlasDefinition.load(registry.ensure("assembly-assets"))
    except (ComponentNotPublished, FileNotFoundError):
        # Pre-publish fallback: composite without region masks. Correct
        # compositing order and keying still apply; the masks arrive with
        # the assembly-assets component.
        atlas = AtlasDefinition(resolution=skin.shape[0])

    albedo = composite_albedo(skin, garment, shoe, atlas)

    import io

    png = io.BytesIO()
    Image.fromarray(albedo).save(png, format="PNG")

    rig = load_rig(registry.ensure("body-rig"), device=device)
    result = export_character_glb(
        rig,
        character.identity,
        character.resting_expression,
        out_path,
        albedo_png=png.getvalue(),
        name=character.name or character.content_id[:12],
        generator=f"character-factory/{__import__('character_factory').__version__}",
    )
    return result.glb_path
