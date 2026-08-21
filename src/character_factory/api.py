"""The library's generation and assembly entry points.

`assemble(character, assets_dir, out_path)` is the deterministic half of the
product promise: a character file plus its baked assets in, a rigged .glb
out (SPEC.md §9). It runs everywhere — CUDA is never required here.

`create` turns a description into a character file (interpretation +
deterministic identity); `make` chains create → bake → assemble.
Interpretation runs the configured model backend when one is configured
(`interpreter.model` in the cache config) and the documented rules
fallback otherwise; provenance records which.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from character_factory.schema import Character, vocab

__all__ = ["AssetError", "assemble", "create", "make"]


def create(
    prompt: str,
    *,
    seed: int = 0,
    registry=None,
    device: str = "cuda",
    name: str | None = None,
    interpreter: str | None = None,
) -> Character:
    """Description → character file: interpretation fills the symbolic
    recipes; the identity component maps the raw prompt to body parameters
    (deterministic, no seed — the seed governs texture recipes only)."""
    from character_factory.identity import IdentityComponent, IdentityGenerator
    from character_factory.interpreter import INTERPRETER_VERSION, interpret
    from character_factory.registry import Registry

    registry = registry or Registry.default()
    # The interpreter model (if configured) loads, runs, and releases here —
    # before the identity encoder or any diffusion pipeline loads (§2.2).
    # `interpreter` selects a configured backend by alias (per-request; the
    # create UI's model selector); None means the configured default.
    interpretation, _ = interpret(
        prompt, registry=registry, device=device, backend=interpreter
    )

    resolved = registry.resolve_slots(sorted(interpretation.slot_prompts))
    figure_entry = registry.get("make-figure")
    base_ref = figure_entry.document.get("requires", {}).get("base_model")
    generator = IdentityGenerator.with_base_model(
        IdentityComponent.load(registry.ensure("make-figure"), device=device),
        registry.ensure(base_ref),
        device=device,
    )
    identity, resting_expression = generator.generate(prompt)
    del generator  # release the text encoder before any diffusion loads (§2.2)

    textures = {}
    for offset, slot in enumerate(sorted(interpretation.slot_prompts), start=1):
        entry = resolved[slot]
        textures[slot] = {
            "component": entry.name,
            "component_version": str(entry.version),
            "prompt": interpretation.slot_prompts[slot],
            "seed": (seed + offset) % (vocab.SEED_MAX + 1),
        }

    hair = interpretation.hair
    if hair is not None:
        hair = dict(hair, seed=seed % (vocab.SEED_MAX + 1))

    # Provenance records the backend kind, never the model identity — the
    # model is configuration (a local path may even be private).
    if interpretation.backend == "rules-fallback":
        interpreter_version = "0.0.0+rules-fallback"
    else:
        interpreter_version = f"{INTERPRETER_VERSION}+{interpretation.backend}"
    components = {
        "interpreter": {"version": interpreter_version},
        "make-figure": {"version": str(figure_entry.version)},
        **{
            entry.name: {"version": str(entry.version)}
            for entry in resolved.values()
        },
    }
    if hair is not None:
        from character_factory.hair import WigProvider

        components[WigProvider.name] = {"version": WigProvider.version}

    if name is None:
        words = re.findall(r"[a-z0-9]+", prompt.lower())
        name = "-".join(words[:6]) or "character"

    return Character.from_document(
        {
            "format": vocab.FORMAT,
            "schema_version": vocab.SCHEMA_VERSION,
            "name": name,
            "body": {
                "rig": "mhr-lod1@1.0",
                "topology": "closed",
                "identity": identity,
                "resting_expression": resting_expression,
            },
            "textures": textures,
            "hair": hair,
            "provenance": {
                "prompt": prompt,
                "generator": f"character-factory/"
                             f"{__import__('character_factory').__version__}",
                "components": components,
            },
        }
    )


def make(
    prompt: str,
    out_dir: str | Path,
    *,
    seed: int = 0,
    registry=None,
    device: str = "cuda",
) -> Path:
    """create → bake → assemble: description in, rigged .glb out."""
    from character_factory.registry import Registry
    from character_factory.textures import bake

    registry = registry or Registry.default()
    out_dir = Path(out_dir)
    character = create(prompt, seed=seed, registry=registry, device=device)
    baked = bake(character, out_dir / "assets", registry=registry, device=device)
    baked.character.save(out_dir / "character.char.json")
    return assemble(
        baked.character, baked.assets_dir, out_dir / "scene.glb",
        registry=registry,
    )


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

    assets_component: Path | None
    try:
        assets_component = registry.ensure("assembly-assets")
        atlas = AtlasDefinition.load(assets_component)
    except (ComponentNotPublished, FileNotFoundError):
        # Pre-publish fallback: composite without region masks or eyes.
        # Correct compositing order and keying still apply; masks, the
        # eyeball asset, and placement data arrive with assembly-assets.
        assets_component = None
        atlas = AtlasDefinition(resolution=skin.shape[0])

    from character_factory.assembly import (
        Attachment,
        EyeAssets,
        export_character_glb,
        load_rig,
        place_eyes,
    )

    rig = load_rig(registry.ensure("body-rig"), device=device)

    # The shoe generator paints a one-foot canvas; its component's foot
    # chart maps that canvas onto the atlas's foot islands (SPEC.md §9).
    shoe_overlay = None
    if "shoe" in character.textures:
        from character_factory.assembly.footwear import FootChart, bake_shoe_overlay

        recipe = character.texture_maps()["shoe"]["albedo"]
        chart = FootChart.load(
            registry.ensure(recipe["component"], recipe.get("component_version"))
        )
        shoe_overlay = bake_shoe_overlay(
            layer("shoe"),
            chart,
            rig.texcoords,
            rig.texcoord_faces,
            prompt=recipe.get("prompt", ""),
            resolution=skin.shape[0],
        )

    albedo = composite_albedo(
        skin, garment, shoe_overlay, atlas.at_resolution(skin.shape[0])
    )

    import io

    png = io.BytesIO()
    Image.fromarray(albedo).save(png, format="PNG")
    evaluation = rig.evaluate(character.identity, character.resting_expression)

    attachments: list[Attachment] = []
    remove_faces = None

    # Eyes: socket faces removed, eyeballs fitted to the rims, each parented
    # to its eye joint, textured with the slot's albedo.
    if assets_component is not None and (assets_component / "eye_placement.json").is_file():
        eye_assets = EyeAssets.load(assets_component)
        eye_png = _load_asset(assets_dir, "eye", character).read_bytes()
        remove_faces = eye_assets.socket_faces
        for placed in place_eyes(evaluation.vertices, rig.faces, eye_assets):
            attachments.append(
                Attachment(
                    name=f"eye_{placed.side}",
                    vertices=placed.vertices,
                    faces=placed.faces,
                    uv=placed.uv,
                    parent_joint=rig.role_index(f"{placed.side}_eye"),
                    albedo_png=eye_png,
                    roughness=0.15,   # cornea is glossy
                )
            )

    # Hair: synthesized by the provider from the character's semantic block
    # and the evaluated body, parented to the head joint.
    if character.hair is not None:
        from character_factory.hair import HeadGeometry, WigProvider

        eye_joints = [rig.role_index("left_eye"), rig.role_index("right_eye")]
        eye_level = float(evaluation.skeleton[eye_joints, 1].mean())
        result = WigProvider().synthesize(
            character.hair,
            HeadGeometry(
                vertices=evaluation.vertices,
                faces=rig.faces,
                eye_level=eye_level,
            ),
        )
        hair_mesh = result.mesh

        def image_png(image) -> bytes | None:
            if image is None:
                return None
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        material = hair_mesh.visual.material
        attachments.append(
            Attachment(
                name="hair",
                vertices=hair_mesh.vertices,
                faces=hair_mesh.faces,
                uv=hair_mesh.visual.uv,
                parent_joint=rig.role_index("head"),
                albedo_png=image_png(getattr(material, "baseColorTexture", None)),
                normal_png=image_png(getattr(material, "normalTexture", None)),
                double_sided=True,
                roughness=0.62,
            )
        )

    result = export_character_glb(
        rig,
        character.identity,
        character.resting_expression,
        out_path,
        albedo_png=png.getvalue(),
        name=character.name or character.content_id[:12],
        generator=f"character-factory/{__import__('character_factory').__version__}",
        remove_faces=remove_faces,
        attachments=attachments,
        evaluation=evaluation,
    )
    return result.glb_path
