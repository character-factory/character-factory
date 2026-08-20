"""Strand albedo synthesis + material/UV assignment + GLB export.

Texture model: vertical strand streaks (the mesh UV 'v' follows hair flow, so
vertical streaks in texture space = flow-aligned strands on the mesh), warped
for wave/curl, colored by a two-pigment melanin ramp (eumelanin/pheomelanin
absorption per d'Eon/Chiang — the Blender Principled Hair convention).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

EUMELANIN = np.array([0.506, 0.841, 1.653])
PHEOMELANIN = np.array([0.343, 0.733, 1.924])


@dataclass
class TextureSpec:
    melanin: float = 0.7  # 0 = white/grey, 1 = black
    redness: float = 0.25  # pheomelanin ratio (red hair ~0.9 + low melanin)
    wave: float = 0.0  # low-frequency S-waves
    curl: float = 0.0  # high-frequency kinks
    streak_contrast: float = 0.32
    # Broad flow-aligned color groups that remain visible after mipmapping or
    # contact-sheet downsampling.  Fine streaks alone average to a solid fill.
    lock_contrast: float = 0.20
    grey: float = 0.0  # fraction of grey strands
    # --- dyed color (overrides/blends with the melanin base) ---
    dye: tuple | None = None  # RGB in [0,1]; e.g. (0.1, 0.35, 0.8) for blue
    dye_amount: float = 1.0  # 1 = full dye, <1 = melanin shows through
    # --- ombre: blend toward a tip color along the flow direction ---
    ombre: tuple | None = None  # RGB tip color
    ombre_start: float = 0.35  # fraction of tile height where the blend begins
    # --- highlights: per-strand injection of a second color ---
    highlight: tuple | None = None  # RGB; None = off (grey= uses grey strands)
    highlight_frac: float = 0.18  # fraction of strands
    # --- normal map ---
    normal_strength: float = 0.55  # 0 disables the derived normal map
    # Optional image-generated/source plate.  It contributes de-lit color and
    # multi-scale directional detail while procedural pigment, tinting, and
    # normal synthesis remain available on top.
    base_texture: str | None = None
    base_texture_color: float = 0.0
    base_texture_detail: float = 0.0
    size: int = 1024
    seed: int = 0


def _strand_mask(rng, n, frac):
    """Per-column (≈ per-strand) boolean mask covering ~frac of columns."""
    sig = gaussian_filter(rng.standard_normal(n), 2.0, mode="wrap")
    return (sig > np.quantile(sig, 1 - frac))[None, :]


def strand_texture(spec: TextureSpec) -> Image.Image:
    return strand_maps(spec)[0]


def _source_plate(path: str, n: int) -> np.ndarray:
    """Load a source plate and mirror-heal its borders for robust tiling."""

    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    if not p.is_file():
        raise FileNotFoundError(f"hair base texture not found: {p}")
    with Image.open(p) as im:
        src = np.asarray(im.convert("RGB").resize((n, n), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    # Mirrored 2x2 mosaic, cropped through its centre.  Opposing crop borders
    # match exactly, avoiding the seam a generated (nominally tileable) plate
    # can still contain.
    wide = np.concatenate([src, src[:, ::-1]], axis=1)
    mosaic = np.concatenate([wide, wide[::-1]], axis=0)
    lo, hi = n // 2, n // 2 + n
    return mosaic[lo:hi, lo:hi]


def strand_maps(spec: TextureSpec) -> tuple[Image.Image, Image.Image | None]:
    """Returns (albedo, tangent-space normal map or None)."""
    n = spec.size
    rng = np.random.default_rng(spec.seed)
    source = None
    if spec.base_texture is not None:
        source = _source_plate(spec.base_texture, n)

    # vertical streaks: noise smeared along y at several scales
    field = np.zeros((n, n))
    for sigma_y, sigma_x, amp in [(140, 1.0, 1.0), (60, 0.8, 0.65), (18, 0.6, 0.4)]:
        layer = gaussian_filter(rng.standard_normal((n, n)), sigma=(sigma_y, sigma_x), mode="wrap")
        layer /= layer.std() + 1e-9
        field += amp * layer
    field /= field.std() + 1e-9

    # A second, much wider field gives neighboring strands a shared value
    # trend.  This is color variation between locks, not baked illumination.
    lock_field = gaussian_filter(
        rng.standard_normal((n, n)), sigma=(160, 14), mode="wrap"
    )
    lock_field /= lock_field.std() + 1e-9

    if source is not None and spec.base_texture_detail > 0:
        # Pull only de-lit local contrast into the procedural field.  Removing
        # the broad illumination prevents an attractive source image from
        # baking a fake studio highlight into every repeated UV tile.
        lum = source @ np.array([0.2126, 0.7152, 0.0722])
        log_lum = np.log(np.clip(lum, 1e-4, 1.0))
        local = log_lum - gaussian_filter(log_lum, n / 28.0, mode="wrap")
        local /= local.std() + 1e-9
        field = field + spec.base_texture_detail * local
        field /= field.std() + 1e-9

    # wave/curl: shift each row horizontally by a smooth per-column phase
    if spec.wave > 0 or spec.curl > 0:
        y = np.arange(n)[:, None] / n
        phase = gaussian_filter(rng.standard_normal(n), 24, mode="wrap")[None, :] * 2.5
        shift = np.zeros((n, n))
        if spec.wave > 0:
            shift += spec.wave * 0.035 * n * np.sin(2 * np.pi * y * 2.3 + phase)
        if spec.curl > 0:
            phase2 = gaussian_filter(rng.standard_normal(n), 6, mode="wrap")[None, :] * 4.0
            shift += spec.curl * 0.012 * n * np.sin(2 * np.pi * y * 11.0 + phase2)
        cols = (np.arange(n)[None, :] + shift).astype(int) % n
        field = field[np.arange(n)[:, None], cols]

    streak = 1.0 + spec.streak_contrast * np.tanh(field)

    # melanin ramp -> per-pixel albedo
    absorb = EUMELANIN * (1 - spec.redness) + PHEOMELANIN * spec.redness
    # calibrated so melanin 0.6 reads warm brown and 0.95 near-black
    density = 6.55 * spec.melanin**1.75 * np.clip(streak, 0.25, 2.2)
    density *= np.exp(spec.lock_contrast * np.tanh(lock_field))
    rgb = np.exp(-absorb[None, None, :] * density[..., None])

    if source is not None and spec.base_texture_color > 0:
        # Match overall value before blending, retaining source pigment/chroma
        # but not letting the plate unexpectedly brighten a dark hairstyle.
        src_lum = source @ np.array([0.2126, 0.7152, 0.0722])
        dst_lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        value_scale = np.clip(dst_lum.mean() / (src_lum.mean() + 1e-9), 0.45, 1.35)
        source_matched = np.clip(source * value_scale, 0.0, 1.0)
        mix = np.clip(spec.base_texture_color, 0.0, 1.0)
        rgb = rgb * (1.0 - mix) + source_matched * mix

    shade = np.clip(0.55 + 0.45 * np.tanh(field), 0.15, 1.0)[..., None]  # streak value
    if spec.dye is not None:
        dyed = np.asarray(spec.dye)[None, None, :] * shade
        rgb = rgb * (1 - spec.dye_amount) + dyed * spec.dye_amount
    if spec.highlight is not None:
        hmask = _strand_mask(rng, n, spec.highlight_frac)
        hcol = np.asarray(spec.highlight)[None, None, :] * shade
        rgb = np.where(np.broadcast_to(hmask[..., None], rgb.shape), hcol, rgb)
    if spec.grey > 0:
        gmask = _strand_mask(rng, n, spec.grey)
        grey_val = np.clip(0.55 + 0.3 * np.tanh(field), 0, 1)[..., None] * np.ones(3)
        rgb = np.where(np.broadcast_to(gmask[..., None], rgb.shape), grey_val, rgb)

    yy = (np.arange(n) / n)[:, None, None]
    if spec.ombre is not None:
        # tile v follows strand flow -> a v-gradient is a root->tip ombre
        blend = np.clip((yy - spec.ombre_start) / max(1 - spec.ombre_start, 1e-6), 0, 1) ** 1.4
        rgb = rgb * (1 - blend) + (np.asarray(spec.ombre)[None, None, :] * shade) * blend

    # subtle root darkening at the top of the tile
    rgb = rgb * (0.82 + 0.18 * np.clip(yy / 0.15, 0, 1))
    albedo = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")

    normal = None
    if spec.normal_strength > 0:
        # the streak field IS a strand heightfield: derive a tangent-space
        # normal map so shells catch light per-strand instead of reading flat
        h = gaussian_filter(field, 1.0, mode="wrap")
        k = spec.normal_strength * 2.0
        dx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
        dy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
        nvec = np.stack([-k * dx, k * dy, np.ones_like(h)], -1)
        nvec /= np.linalg.norm(nvec, axis=-1, keepdims=True)
        normal = Image.fromarray(((nvec * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")
    return albedo, normal


def apply_material(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    normal: Image.Image | None = None,
    tile=(5.0, 1.6),
) -> trimesh.Trimesh:
    """Attach the strand texture with chart-aligned UVs (u around, v = flow)."""
    chart = mesh.vertex_attributes["chart_uv"]
    uv = np.stack([chart[:, 0] * tile[0], 1.0 - chart[:, 1] * tile[1]], 1)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        normalTexture=normal,
        metallicFactor=0.0,
        roughnessFactor=0.62,
        doubleSided=True,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


def mesh_tangents(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return per-vertex UV tangents for preview normal/anisotropy shading."""

    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    w0, w1, w2 = (uv[faces[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    good = np.abs(det) > 1e-10
    tri = np.zeros_like(e1)
    tri[good] = (
        e1[good] * d2[good, 1, None] - e2[good] * d1[good, 1, None]
    ) / det[good, None]

    tangent = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(tangent, faces[:, corner], tri)
    normal = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tangent -= normal * np.sum(tangent * normal, axis=1, keepdims=True)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    missing = length[:, 0] < 1e-10
    if np.any(missing):
        axis = np.zeros_like(normal[missing])
        axis[:, 1] = 1.0
        parallel = np.abs(np.sum(axis * normal[missing], axis=1)) > 0.9
        axis[parallel] = (1.0, 0.0, 0.0)
        tangent[missing] = np.cross(axis, normal[missing])
        length = np.linalg.norm(tangent, axis=1, keepdims=True)
    still_missing = length[:, 0] < 1e-10
    tangent[still_missing] = (1.0, 0.0, 0.0)
    length[still_missing] = 1.0
    return (tangent / np.maximum(length, 1e-10)).astype(np.float32)


def export_glb(
    mesh: trimesh.Trimesh,
    path: str,
    anisotropy_strength: float = 0.72,
    anisotropy_rotation: float = 0.0,
) -> None:
    """Export smooth-shaded opaque PBR GLB with hair anisotropy.

    Trimesh normally omits ``NORMAL`` unless vertex normals happen to have
    been evaluated and cached earlier in the process.  Request them explicitly
    so exported hair has deterministic, angle-weighted smooth shading instead
    of falling back to flat shading in glTF viewers.
    """

    mesh.export(path, include_normals=True)
    if anisotropy_strength <= 0:
        return
    from pygltflib import GLTF2

    gltf = GLTF2().load_binary(path)
    extension = "KHR_materials_anisotropy"
    used = list(gltf.extensionsUsed or [])
    if extension not in used:
        used.append(extension)
    gltf.extensionsUsed = used
    for material in gltf.materials or []:
        material.extensions = dict(material.extensions or {})
        material.extensions[extension] = {
            "anisotropyStrength": float(np.clip(anisotropy_strength, 0.0, 1.0)),
            "anisotropyRotation": float(anisotropy_rotation),
        }
    gltf.save_binary(path)
