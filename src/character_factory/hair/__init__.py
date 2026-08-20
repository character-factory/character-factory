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
    """The default provider: the vendored make-wig engine."""

    name = "make-wig"
    version = WIG_PROVIDER_VERSION

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

        mesh = wig_style.generate(fitted, plan.style)
        albedo, normal = wig_texture.strand_maps(plan.texture)
        mesh = wig_texture.apply_material(mesh, albedo, normal)
        return HairResult(
            mesh=mesh,
            manifest={
                "provider": self.name,
                "provider_version": self.version,
                "compiler_version": plan.compiler_version,
                "base_preset": plan.base_preset,
            },
        )
