"""Hair synthesis: the HairProvider boundary and the vendored default engine.

The contract (ARCHITECTURE §5): the character file's hair block in, a
textured triangle mesh in the body's frame out. Providers are
interchangeable behind it; the default is **make-wig**, the procedural
engine vendored at :mod:`character_factory.hair.wig` — pure Python, no
diffusion, deterministic for a fixed (intent, head, engine version) triple.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = ["HairProvider", "HeadGeometry", "HairResult", "WigProvider"]

WIG_PROVIDER_VERSION = "0.1.0"


@dataclass
class HeadGeometry:
    """The assembled head/body geometry a provider fits hair to.

    Rig-native frame: centimeters, Y-up, character facing +Z. A full-body
    mesh gives long styles their shoulder/back drape; head-only works for
    short styles.
    """

    vertices: "np.ndarray"       # (V, 3) cm
    faces: "np.ndarray"          # (F, 3)
    eye_level: float             # world Y of the eye line, cm
    forward: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class HairResult:
    mesh: object                 # trimesh.Trimesh, textured, in the body frame
    manifest: dict = field(default_factory=dict)


class HairProvider(Protocol):
    def synthesize(self, intent: dict, head: HeadGeometry) -> HairResult: ...


class WigProvider:
    """The default provider: the vendored make-wig engine.

    `density_presets` is component data — the make-wig registry entry's
    `density_presets` block, `{preset_name: {control: value}}` plus an
    optional `families` map naming which preset each hair family uses.
    Passing none generates at full density, exactly as before. Presets
    are data because a density is a tuning decision per engine version,
    like a conditioning template.
    """

    name = "make-wig"
    version = WIG_PROVIDER_VERSION

    def __init__(self, density_presets: dict | None = None,
                 preset: str | None = None):
        self.density_presets = dict(density_presets or {})
        self.preset = preset

    def density_for(self, family: str):
        """The density this provider generates `family` at."""
        from character_factory.hair.wig.density import FULL, Density

        table = self.density_presets
        name = self.preset
        if name is None:
            name = (table.get("families") or {}).get(family)
        if name is None:
            return FULL, None
        controls = table.get(name)
        if controls is None:
            raise ValueError(
                f"hair density preset {name!r} is not declared by the "
                f"component (declared: "
                f"{', '.join(sorted(k for k in table if k != 'families'))})")
        return Density.from_mapping(controls), name

    def synthesize(self, intent: dict, head: HeadGeometry) -> HairResult:
        import numpy as np

        from character_factory.hair.wig import Head, compile_hair_intent
        from character_factory.hair.wig import style as wig_style
        from character_factory.hair.wig import texture as wig_texture

        plan = compile_hair_intent(intent)

        # The engine's constructor is file-based; hand it the geometry as a
        # transient OBJ (cheap at body-mesh sizes, and it exercises exactly
        # the loader third parties will use).
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = Path(tmp) / "body.obj"
            with obj_path.open("w", encoding="utf-8") as obj:
                for vertex in np.asarray(head.vertices, dtype=float):
                    obj.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                for face in np.asarray(head.faces, dtype=int) + 1:
                    obj.write(f"f {face[0]} {face[1]} {face[2]}\n")
            fitted = Head.from_file(
                str(obj_path),
                forward=np.asarray(head.forward, dtype=float),
                eye_level=head.eye_level,
            )

        density, density_preset = self.density_for(
            intent.get("family", "loose_long"))
        mesh = wig_style.generate(fitted, plan.style, density=density)
        albedo, normal = wig_texture.strand_maps(plan.texture)
        mesh = wig_texture.apply_material(mesh, albedo, normal)
        return HairResult(
            mesh=mesh,
            manifest={
                "provider": self.name,
                "provider_version": self.version,
                "compiler_version": plan.compiler_version,
                "base_preset": plan.base_preset,
                "density_preset": density_preset,
                "triangles": int(len(mesh.faces)),
            },
        )
