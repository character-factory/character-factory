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
    from character_factory.preflight import require_generation_stack
    from character_factory.registry import Registry

    # Fail in seconds with a named cause (missing dependency, CPU-only
    # torch, dead or too-old driver) instead of minutes into a model load.
    require_generation_stack(device)
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
    generated = generator.generate(prompt)
    identity = generated.identity
    resting_expression = generated.resting_expression
    del generator  # release the text encoder before any diffusion loads (§2.2)

    # Skeletal proportions (§4.3): the identity component owns them (it
    # consumes the raw prompt, like everything identity-class); a writer
    # backend that explicitly emitted proportion fields overrides per key.
    # The rules fallback never emits any. Only deviations are recorded:
    # zero-valued keys are dropped and an all-template result omits the
    # block entirely.
    proportions = dict(generated.proportions)
    if interpretation.proportions:
        proportions.update(interpretation.proportions)
    proportions = {
        key: value for key, value in proportions.items() if value != 0.0
    }

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
                "topology": "mouth-interior",
                "identity": identity,
                **({"proportions": proportions} if proportions else {}),
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
    garment_shells: bool | None = None,
) -> Path:
    """Build the rigged .glb for a character from its baked assets.

    Albedo asset files are looked up in `assets_dir` by slot name
    (`skin.png`, `eye.png`, `garment.png`, optional `shoe.png`). When the
    character carries an `assets` block, every file is verified against its
    pinned hash before use; a mismatch is a hard error.

    `garment_shells` overrides the configured feature gate for this call
    (None = the gate; assembly behavior like turbo, never recorded in the
    character document). With the feature on, a character whose garment
    extraction fails any gate silently keeps the painted composite — the
    manifest's `garments` block records which mode shipped.
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

    # SPEC.md §4.2: facial animation is part of every character, so a rig
    # without the mouth basis is incompatible — never a reduced-quality
    # fallback after texture work has already completed.
    if "mouth" not in rig.metadata:
        raise ValueError(
            "the character contract requires a body-rig component with "
            "mouth data, and the resolved version declares none"
        )

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
    evaluation = rig.evaluate(
        character.identity, character.resting_expression,
        proportions=character.proportions or None,
    )

    attachments: list[Attachment] = []
    remove_faces = None

    # The surface everything downstream is placed against: the rig's own
    # topology, or the coarser tessellation its component declares. Face
    # indices (eye apertures, mouth portals, covered-garment sets) always
    # belong to whichever surface is being built.
    surface_vertices = evaluation.vertices
    surface_faces = rig.faces
    if rig.render is not None:
        surface_vertices = rig.render.vertices_from(evaluation.vertices, rig.faces)
        surface_faces = rig.render.faces

    # Eyes: socket faces removed, eyeballs fitted to the rims, each parented
    # to its eye joint, textured with the slot's albedo.
    if assets_component is not None and (assets_component / "eye_placement.json").is_file():
        import dataclasses

        from character_factory.assembly.eyes import socket_backing

        eye_assets = EyeAssets.load(assets_component)
        eye_png = _load_asset(assets_dir, "eye", character).read_bytes()
        if rig.render is not None:
            # A render LOD carries its own hand-authored aperture: the
            # selection cannot be transferred by mapping the source one.
            eye_assets = dataclasses.replace(
                eye_assets, socket_faces=rig.render.eye_faces)
        remove_faces = eye_assets.socket_faces
        for placed in place_eyes(surface_vertices, surface_faces, eye_assets):
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
            # The dark occluder skirt behind the eyeball: without it the
            # rim-to-eyeball gap (measured ~1.5 mm) reads straight through
            # the head. Skull-parented: eyelid morphs close in front of it.
            backing_v, backing_f = socket_backing(placed.rim, placed.gaze)
            attachments.append(
                Attachment(
                    name=f"eye_{placed.side}_backing",
                    vertices=backing_v,
                    faces=backing_f,
                    uv=None,
                    parent_joint=rig.role_index("head"),
                    base_color=(0.055, 0.032, 0.03, 1.0),
                    double_sided=True,
                    roughness=0.9,
                )
            )

    # Hair: synthesized by the provider from the character's semantic block
    # and the evaluated body, parented to the head joint.
    if character.hair is not None:
        from character_factory.hair import HeadGeometry, WigProvider

        eye_joints = [rig.role_index("left_eye"), rig.role_index("right_eye")]
        eye_level = float(evaluation.skeleton[eye_joints, 1].mean())
        # Density presets are the provider component's data: hair
        # generates at the target density rather than being decimated
        # afterwards. An entry that declares none generates full density.
        wig_entry = registry.get("make-wig")
        provider = WigProvider(
            density_presets=wig_entry.document.get("density_presets"))
        result = provider.synthesize(
            character.hair,
            HeadGeometry(
                vertices=surface_vertices,
                faces=surface_faces,
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

    # Mouth interior (SPEC.md §4.2, §9 step 4): remove the rig version's
    # fixed portal, stitch the socket strip into the skinned body, place
    # the anatomy meshes on the jaw chain, and bake the 72 expression morph
    # targets. This is the one public character assembly path: facial
    # animation is baseline quality, not a topology option.
    mouth_glb, mouth_attachments, mouth_removal = _prepare_mouth(
        rig, assets_component, evaluation, character,
        surface=rig.render, surface_vertices=surface_vertices
    )
    attachments.extend(mouth_attachments)
    remove_faces = (
        mouth_removal if remove_faces is None
        else np.concatenate([np.asarray(remove_faces, dtype=np.int64),
                             mouth_removal])
    )

    # Garment shells (feature-gated assembly behavior, never recipe): the
    # baked garment texture may become a skinned, body-following closed
    # solid over the painted composite. Every failed gate falls back to
    # paint for this character, silently; the manifest's `garments` block
    # records the shipped mode per slot so consumers never sniff.
    from character_factory.assembly import garment_shell as shell_module
    from character_factory.assembly.export import SkinnedAttachment

    skinned_attachments: list[SkinnedAttachment] = []
    garments_manifest: dict = {}
    if "shoe" in character.textures:
        garments_manifest["shoe"] = {"render_mode": "painted"}
    if "garment" in character.textures:
        garments_manifest["garment"] = {"render_mode": "painted"}
        enabled = (shell_module.shells_enabled() if garment_shells is None
                   else garment_shells)
        if enabled:
            shell, rejection = _prepare_garment_shell(
                rig, character, evaluation, garment, atlas,
                surface_vertices, surface_faces)
            if shell is not None:
                # The shell's texture is the baked garment with its
                # boundary colors bled outward (atlas hygiene): boundary
                # faces and rim insets sample cloth, never the keyed-out
                # background. The key itself always comes from the
                # original bytes — dilation cannot grow coverage.
                import io as _io

                from character_factory.assembly.garment_shell import (
                    dilate_garment_colors,
                )

                dilated = dilate_garment_colors(garment, shell.hard_key)
                shell_buffer = _io.BytesIO()
                Image.fromarray(dilated).save(shell_buffer, format="PNG")
                shell_png = shell_buffer.getvalue()
                skinned_attachments.append(SkinnedAttachment(
                    name="garment",
                    vertices=shell.vertices,
                    faces=shell.faces,
                    corner_uv=shell.corner_uv,
                    joints4=shell.joints4,
                    weights4=shell.weights4,
                    albedo_png=shell_png,
                ))
                remove_faces = (
                    shell.covered_body_faces if remove_faces is None
                    else np.concatenate([
                        np.asarray(remove_faces, dtype=np.int64),
                        shell.covered_body_faces]))
                garments_manifest["garment"] = {
                    "render_mode": "shell",
                    "shell": {
                        "constants_version": shell.audit["constants_version"],
                        "components": shell.audit["components"],
                        "solid_vertices": int(len(shell.vertices)),
                        "solid_faces": int(len(shell.faces)),
                        "hidden_body_faces": int(len(shell.covered_body_faces)),
                    },
                }
            else:
                garments_manifest["garment"] = {
                    "render_mode": "painted", "reason": rejection}

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
        skinned_attachments=skinned_attachments,
        evaluation=evaluation,
        mouth=mouth_glb,
        manifest_extra={"garments": garments_manifest} if garments_manifest
        else None,
    )
    return result.glb_path


def _prepare_garment_shell(rig, character, evaluation, garment_rgb, atlas,
                           surface_vertices=None, surface_faces=None):
    """Run the full extraction + certification ladder for one character.
    Returns (shell, None) on success, (None, reason-code) on any gate.

    Extraction runs natively on the surface that ships — a declared
    render LOD is cut, skinned and certified on its own buffers, never
    extracted at a finer topology and reduced afterwards.
    """
    import numpy as np

    from character_factory.assembly import garment_shell as shell_module

    surface = rig.render if rig.render is not None else rig
    constants = shell_module.configured_constants()
    canonical = rig.evaluate(
        [0.0] * len(character.identity),
        [0.0] * len(character.resting_expression))
    canonical_vertices = canonical.vertices
    if rig.render is not None:
        canonical_vertices = rig.render.vertices_from(canonical.vertices, rig.faces)
    if surface_vertices is None:
        surface_vertices = evaluation.vertices
    resolution = garment_rgb.shape[0]
    atlas_valid = shell_module.valid_atlas_mask(surface, resolution)
    # The region contract mirrors the compositor exactly: the head mask
    # (garment never paints there) subtracts from the key. The feet mask
    # is a shoe-side constraint — a broad lower-body region where the
    # shoe may paint — and does NOT remove garment (trouser legs live
    # there in paint and in shell alike).
    excluded = atlas.at_resolution(resolution).head_mask
    try:
        shell = shell_module.prepare_shell(
            surface, garment_rgb, surface_vertices, canonical_vertices,
            atlas_valid, excluded_regions=excluded, constants=constants)
        shell.audit["pose_gate"] = shell_module.pose_gate(
            rig, shell, evaluation, character.identity,
            character.resting_expression,
            proportions=character.proportions or None,
            constants=constants, surface=surface, body_rest=surface_vertices)
    except shell_module.ShellRejected as error:
        return None, error.reason
    return shell, None


def _prepare_mouth(rig, assets_component, evaluation, character,
                   surface=None, surface_vertices=None):
    """Build the MouthGlb bundle, anatomy attachments, and the portal
    removal set for a mouth-interior character. The mouth data belongs to
    the surface being built — a render LOD carries its own portal, lip
    paths and morph basis."""
    from character_factory.assembly import mouth as mouth_assembly
    from character_factory.assembly.export import Attachment, MouthGlb

    if assets_component is None:
        raise ValueError(
            "mouth-interior assembly requires the assembly-assets component"
        )
    surface = surface if surface is not None else rig
    data = mouth_assembly.MouthData.load(
        rig_component_dir(rig), rig.metadata
    )
    rest = surface_vertices if surface_vertices is not None else evaluation.vertices
    strip = mouth_assembly.export_strip(rig, data, evaluation, surface=surface,
                                        rest_vertices=rest)
    body_dense = [data.morph_dense(unit, len(rest))
                  for unit in range(len(data.morph_names))]

    manifest = {
        "expression_morphs": {
            "names": list(data.morph_names),
            "count": len(data.morph_names),
            "encoding": "sparse POSITION+NORMAL deltas; exact (the rig's "
                        "expression is a linear vertex basis)",
            "weights": "unit weights are 0..1 in glTF morph-target "
                       "convention; engines with a rescaled blend-shape "
                       "range (e.g. Unity's 0..100) normalize at import",
            "semantics": data.semantics,
        },
        "jaw": data.jaw,
        "animation_limitations": {
            **data.limitations,
            # Dispatch contract, stated so consumers stop parsing prose:
            # `kind` selects the entry type and `params` is the
            # authoritative machine-readable description of the pose.
            # `case` is a human-readable label — its grammar is not
            # stable and entries exist (neutral-seating) that match no
            # case pattern at all.
            "reading": "dispatch on `kind`; read the pose from `params`. "
                       "`case` is a human-readable label only — do not "
                       "parse it: entry kinds without a morph unit carry "
                       "no unit/weight and match no case grammar.",
        },
    }
    mouth_glb = MouthGlb(
        socket_vertices_cm=strip.vertices,
        socket_faces=strip.faces,
        socket_uv=strip.uv,
        socket_joints=strip.joints,
        socket_weights=strip.weights,
        morph_names=list(data.morph_names),
        body_morph_dense=body_dense,
        socket_morph_dense=strip.morph_deltas,
        manifest=manifest,
        weld_pairs=strip.weld_pairs,
    )
    attachments = [
        Attachment(
            name=piece.name,
            vertices=piece.vertices,
            faces=piece.faces,
            uv=piece.uv,
            parent_joint=rig.joint_index(piece.parent_role),
            base_color=piece.base_color,
            roughness=piece.roughness,
        )
        for piece in mouth_assembly.place_anatomy(
            assets_component, data, rest
        )
    ]
    return mouth_glb, attachments, data.portal_faces


def rig_component_dir(rig) -> Path:
    """The directory the rig component was loaded from (recorded at load)."""
    directory = getattr(rig, "component_dir", None)
    if directory is None:
        raise ValueError(
            "this rig was loaded without a component directory; mouth "
            "assembly needs the component's derived artifacts"
        )
    return Path(directory)
