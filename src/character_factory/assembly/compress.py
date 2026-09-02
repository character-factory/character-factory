"""Opt-in size reduction for an exported .glb.

`scene.glb` is the canonical, uncompressed export: lossless PNG textures,
float32 geometry, byte-deterministic from the character file. That is the
right thing to keep on disk and the wrong thing to ship over the wire —
the textures alone are 4–8 MB of a 7–12 MB file, and every material is
opaque, so a lossy encode at viewer scale is invisible.

`compress_glb` rewrites a finished GLB for one delivery target:

- **`web`** — textures re-encoded as WebP (albedo q85, normal q90) under
  `EXT_texture_webp`. three.js, model-viewer, and Babylon decode it.
- **`unity`** — textures re-encoded as JPEG (same qualities). No
  extension: Unity's image loader (which glTFast goes through) reads
  PNG/JPEG but not WebP, and JPEG is the safe choice for every other
  consumer too.

Both targets first merge byte-identical images (the two eye materials
share one texture — the exporter now writes that directly, but files
from older builds still carry the duplicate). Geometry is left as-is:
quantization and meshopt need either a native decoder on the consumer's
side or an inverse-bind-matrix rewrite on ours, and neither is worth
doing before the texture win, which is most of the bytes. The result is
never the canonical file — `scene.glb` stays lossless; the compressed
sibling is written alongside as `scene.<target>.glb`.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from character_factory.assembly.gltf import GlbWriter, parse_glb

TARGETS = ("web", "unity")

ALBEDO_QUALITY = 85
NORMAL_QUALITY = 90

_FORMATS = {
    "web": ("WEBP", "image/webp"),
    "unity": ("JPEG", "image/jpeg"),
}


def compressed_path(glb_path: str | Path, target: str) -> Path:
    """`scene.glb` → `scene.web.glb`: the sibling a compressed copy goes to."""
    glb_path = Path(glb_path)
    return glb_path.with_name(f"{glb_path.stem}.{target}{glb_path.suffix}")


def compress_glb(
    data: bytes,
    target: str,
    *,
    albedo_quality: int = ALBEDO_QUALITY,
    normal_quality: int = NORMAL_QUALITY,
) -> bytes:
    """Return `data` re-encoded for `target` ("web" or "unity").

    Meshes, skins, morph targets, the animation, and the export manifest
    are carried across untouched; only images (and the bufferViews that
    hold them) change."""
    if target not in TARGETS:
        raise ValueError(
            f"unknown compression target {target!r}; expected one of {TARGETS}"
        )
    from PIL import Image

    gltf, binary = parse_glb(data)
    gltf = dict(gltf)
    images = [dict(image) for image in gltf.get("images", [])]
    textures = [dict(texture) for texture in gltf.get("textures", [])]
    if not images:
        return data

    # Which images are normal maps: a material's normalTexture → texture
    # → image. They get the higher quality; everything else is albedo.
    normal_images: set[int] = set()
    for material in gltf.get("materials", []):
        normal = material.get("normalTexture")
        if normal is not None:
            normal_images.add(textures[normal["index"]]["source"])

    # 1. Dedup: textures whose images are byte-identical point at one
    # image; the orphans are dropped below when the buffer is repacked.
    def image_bytes(image: dict) -> bytes:
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        return binary[start : start + view["byteLength"]]

    canonical: dict[str, int] = {}
    remap: dict[int, int] = {}
    for index, image in enumerate(images):
        if "bufferView" not in image:
            remap[index] = index
            continue
        digest = hashlib.sha256(image_bytes(image)).hexdigest()
        remap[index] = canonical.setdefault(digest, index)
    kept = sorted(set(remap.values()))
    new_index = {old: new for new, old in enumerate(kept)}
    normal_images = {new_index[remap[i]] for i in normal_images}
    images = [images[i] for i in kept]
    for texture in textures:
        texture["source"] = new_index[remap[texture["source"]]]

    # 2. Re-encode each surviving embedded image.
    encoder, mime = _FORMATS[target]
    replacements: dict[int, bytes] = {}
    for index, image in enumerate(images):
        if "bufferView" not in image:
            continue
        quality = normal_quality if index in normal_images else albedo_quality
        picture = Image.open(io.BytesIO(image_bytes(image))).convert("RGB")
        out = io.BytesIO()
        if encoder == "WEBP":
            picture.save(out, format="WEBP", quality=quality, method=6)
        else:
            # 4:4:4 chroma: cheap insurance against colour bleeding on
            # normal maps and hard albedo edges (UV island borders).
            picture.save(out, format="JPEG", quality=quality, subsampling=0,
                         optimize=True)
        replacements[image["bufferView"]] = out.getvalue()
        image["mimeType"] = mime

    # 3. Repack the binary chunk: every bufferView in its original order,
    # image views swapped for the new bytes, orphaned image views dropped
    # and every bufferView reference renumbered.
    dropped = {
        image["bufferView"]
        for index, image in enumerate(gltf["images"])
        if remap[index] != index and "bufferView" in image
    }
    writer = GlbWriter()
    view_index: dict[int, int] = {}
    for old, view in enumerate(gltf["bufferViews"]):
        if old in dropped:
            continue
        start = view.get("byteOffset", 0)
        chunk = replacements.get(old, binary[start : start + view["byteLength"]])
        new = writer.add_view(chunk, view.get("target"))
        for key, value in view.items():
            if key not in ("buffer", "byteOffset", "byteLength", "target"):
                writer.buffer_views[new][key] = value
        view_index[old] = new

    def renumber(obj: dict) -> None:
        if "bufferView" in obj:
            obj["bufferView"] = view_index[obj["bufferView"]]

    accessors = []
    for accessor in gltf["accessors"]:
        accessor = dict(accessor)
        renumber(accessor)
        if "sparse" in accessor:
            sparse = {k: (dict(v) if isinstance(v, dict) else v)
                      for k, v in accessor["sparse"].items()}
            renumber(sparse["indices"])
            renumber(sparse["values"])
            accessor["sparse"] = sparse
        accessors.append(accessor)
    for image in images:
        renumber(image)
    writer.accessors = accessors

    # 4. WebP is an extension; JPEG is core. With no PNG fallback kept,
    # the extension is required — a loader without it must refuse the
    # file rather than render untextured.
    if target == "web":
        for texture in textures:
            texture["extensions"] = {"EXT_texture_webp": {"source": texture.pop("source")}}
        for key in ("extensionsUsed", "extensionsRequired"):
            listed = list(gltf.get(key, []))
            if "EXT_texture_webp" not in listed:
                listed.append("EXT_texture_webp")
            gltf[key] = listed

    gltf["images"] = images
    gltf["textures"] = textures
    return writer.finish(gltf)


def compress_glb_file(glb_path: str | Path, target: str, out_path=None) -> Path:
    """Compress the GLB at `glb_path` for `target`; returns the written path
    (`out_path`, default `compressed_path(glb_path, target)`)."""
    glb_path = Path(glb_path)
    out = Path(out_path) if out_path is not None else compressed_path(glb_path, target)
    out.write_bytes(compress_glb(glb_path.read_bytes(), target))
    return out
