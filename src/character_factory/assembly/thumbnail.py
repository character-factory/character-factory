"""Gallery thumbnails: a software render of the assembled scene.

A small, dependency-free rasterizer (numpy + Pillow) that turns the exported
.glb into a three-quarter full-figure portrait once assembly completes. It
reads only what a consumer would: node transforms, mesh primitives,
materials, embedded textures. Perspective projection with a z-buffer,
perspective-correct UVs, smooth-shaded lambert with a fixed key light,
alpha-masked texels (hair cards), transparent background.

Deliberately not a product renderer: no environment lighting, no
correctives, no shadows — it exists so the gallery shows the character,
not their UV atlas.
"""

from __future__ import annotations

import io

import numpy as np

from character_factory.assembly.gltf import parse_glb, read_accessor
from character_factory.assembly.validate import _global_matrices

__all__ = ["render_thumbnail"]

_YAW_DEGREES = 35.0          # three-quarter view
_FOV_DEGREES = 28.0          # medium-wide framing
_MARGIN = 1.10               # figure height / frame height
_LIGHT = np.array([0.45, 0.35, 0.82])
_AMBIENT, _DIFFUSE = 0.42, 0.58


def _decode_image(gltf: dict, binary: bytes, image_index: int) -> np.ndarray:
    from PIL import Image

    view = gltf["bufferViews"][gltf["images"][image_index]["bufferView"]]
    start = view.get("byteOffset", 0)
    data = binary[start : start + view["byteLength"]]
    image = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.asarray(image, dtype=np.float32) / 255.0


def _draw_list(gltf: dict, binary: bytes):
    """Flatten the scene into world-space primitives with resolved
    materials. Skinned primitives are taken at bind pose (their POSITION
    accessor — the validator proves rest skinning reproduces it)."""
    globals_ = _global_matrices(gltf)
    textures: dict[int, np.ndarray] = {}
    items = []
    for node_index, node in enumerate(gltf["nodes"]):
        if "mesh" not in node:
            continue
        world = np.eye(4) if "skin" in node else globals_[node_index]
        rotation = world[:3, :3]
        for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
            attributes = primitive["attributes"]
            positions = read_accessor(
                gltf, binary, attributes["POSITION"]
            ).astype(np.float64)
            positions = positions @ world[:3, :3].T + world[:3, 3]
            normals = read_accessor(
                gltf, binary, attributes["NORMAL"]
            ).astype(np.float64) @ rotation.T
            uv = (
                read_accessor(gltf, binary, attributes["TEXCOORD_0"])
                .astype(np.float64)
                if "TEXCOORD_0" in attributes else None
            )
            indices = read_accessor(
                gltf, binary, primitive["indices"]
            ).astype(np.int64).reshape(-1, 3)
            material = gltf["materials"][primitive["material"]]
            pbr = material.get("pbrMetallicRoughness", {})
            texture = None
            if "baseColorTexture" in pbr and uv is not None:
                source = gltf["textures"][pbr["baseColorTexture"]["index"]]["source"]
                if source not in textures:
                    textures[source] = _decode_image(gltf, binary, source)
                texture = textures[source]
            color = np.asarray(
                pbr.get("baseColorFactor", [0.75, 0.72, 0.68, 1.0]), dtype=np.float64
            )
            items.append((positions, normals, uv, indices, texture, color,
                          bool(material.get("doubleSided", False))))
    return items


def render_thumbnail(
    glb: bytes, width: int = 480, height: int = 640
) -> bytes:
    """Render one .glb to a PNG (RGBA, transparent background)."""
    from PIL import Image

    gltf, binary = parse_glb(glb)
    items = _draw_list(gltf, binary)

    # Model rotation for the three-quarter view, about the figure's center.
    yaw = np.deg2rad(_YAW_DEGREES)
    c, s = np.cos(yaw), np.sin(yaw)
    turn = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    all_points = np.concatenate([it[0] for it in items])
    center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2.0
    items = [
        ((p - center) @ turn.T, n @ turn.T, uv, idx, tex, col, ds)
        for p, n, uv, idx, tex, col, ds in items
    ]

    spans = np.concatenate([it[0] for it in items])
    figure_height = float(spans[:, 1].max() - spans[:, 1].min())
    fov = np.deg2rad(_FOV_DEGREES)
    distance = (figure_height * _MARGIN / 2.0) / np.tan(fov / 2.0)
    eye = np.array([0.0, figure_height * 0.02, distance])

    focal = (height / 2.0) / np.tan(fov / 2.0)
    light = _LIGHT / np.linalg.norm(_LIGHT)

    color_buffer = np.zeros((height, width, 4), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf)

    for positions, normals, uv, indices, texture, base_color, double_sided in items:
        view = positions - eye                       # camera looks down -Z
        z = -view[:, 2]
        valid_z = z > 1e-6
        sx = width / 2.0 + focal * view[:, 0] / np.maximum(z, 1e-6)
        sy = height / 2.0 - focal * view[:, 1] / np.maximum(z, 1e-6)
        inv_z = 1.0 / np.maximum(z, 1e-6)

        for tri in indices:
            if not valid_z[tri].all():
                continue
            x0, x1, x2 = sx[tri]
            y0, y1, y2 = sy[tri]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if area >= 0 and not double_sided:
                continue                              # backface (CCW front)
            if abs(area) < 1e-12:
                continue
            lo_x = max(int(np.floor(min(x0, x1, x2))), 0)
            hi_x = min(int(np.ceil(max(x0, x1, x2))) + 1, width)
            lo_y = max(int(np.floor(min(y0, y1, y2))), 0)
            hi_y = min(int(np.ceil(max(y0, y1, y2))) + 1, height)
            if lo_x >= hi_x or lo_y >= hi_y:
                continue
            xs, ys = np.meshgrid(
                np.arange(lo_x, hi_x) + 0.5, np.arange(lo_y, hi_y) + 0.5
            )
            w0 = ((x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)) / area
            w1 = ((x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            # perspective-correct interpolation via 1/z. Edge-function
            # weights belong to the OPPOSITE corner: the v0->v1 edge
            # function (w0) weights v2, v1->v2 (w1) weights v0, and the
            # remainder (w2) weights v1.
            iz = w1 * inv_z[tri[0]] + w2 * inv_z[tri[1]] + w0 * inv_z[tri[2]]
            depth = 1.0 / np.maximum(iz, 1e-12)
            rows = ys[inside].astype(np.int64)
            cols = xs[inside].astype(np.int64)
            nearer = depth[inside] < depth_buffer[rows, cols]
            if not nearer.any():
                continue
            rows, cols = rows[nearer], cols[nearer]
            b0, b1, b2 = (w[inside][nearer] for w in (w1, w2, w0))
            d = depth[inside][nearer]

            def lerp(values):
                over_z = (
                    b0[:, None] * values[tri[0]] * inv_z[tri[0]]
                    + b1[:, None] * values[tri[1]] * inv_z[tri[1]]
                    + b2[:, None] * values[tri[2]] * inv_z[tri[2]]
                )
                return over_z * d[:, None]

            if texture is not None and uv is not None:
                st = lerp(uv) % 1.0
                th, tw = texture.shape[:2]
                texel = texture[
                    np.minimum((st[:, 1] * th).astype(np.int64), th - 1),
                    np.minimum((st[:, 0] * tw).astype(np.int64), tw - 1),
                ]
                rgb, alpha = texel[:, :3], texel[:, 3]
            else:
                rgb = np.broadcast_to(base_color[:3], (len(rows), 3))
                alpha = np.full(len(rows), float(base_color[3]))
            opaque = alpha >= 0.5                    # alpha-mask (hair cards)
            if not opaque.any():
                continue
            rows, cols, d = rows[opaque], cols[opaque], d[opaque]
            b0, b1, b2 = b0[opaque], b1[opaque], b2[opaque]
            rgb = rgb[opaque]

            over_z = (
                b0[:, None] * normals[tri[0]] * inv_z[tri[0]]
                + b1[:, None] * normals[tri[1]] * inv_z[tri[1]]
                + b2[:, None] * normals[tri[2]] * inv_z[tri[2]]
            )
            n = over_z * d[:, None]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            if double_sided:
                facing = n[:, 2] < 0
                n[facing] = -n[facing]
            shade = _AMBIENT + _DIFFUSE * np.maximum(n @ light, 0.0)

            depth_buffer[rows, cols] = d
            color_buffer[rows, cols, :3] = rgb * shade[:, None]
            color_buffer[rows, cols, 3] = 1.0

    image = Image.fromarray(
        (np.clip(color_buffer, 0.0, 1.0) * 255).astype(np.uint8), "RGBA"
    )
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
