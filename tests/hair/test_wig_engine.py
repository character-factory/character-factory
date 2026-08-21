"""Validation suite for the vendored make-wig engine — fully self-contained (procedural mannequin).

    pytest tests/hair/test_wig_engine.py

Optionally validate against a real head too by setting
    MAKE_WIG_TEST_HEAD=/path/to/head.glb
(e.g. a decompressed VALID avatar; VALID is MIT-licensed).
"""

import os
import json
import sys
from pathlib import Path

import tempfile

import numpy as np
import trimesh

from character_factory.hair.wig import Head
from character_factory.hair.wig.intent import HairIntentError, compile_hair_intent, hair_intent_schema, load_hair_intent
from character_factory.hair.wig.presets import PRESETS, sample_style
from character_factory.hair.wig.style import generate
from character_factory.hair.wig.texture import apply_material, export_glb, mesh_tangents, strand_maps

MANNEQUIN = dict(forward=[0, 0, 1], eye_level=-1.0)
FAIL = []


def check(name, cond, detail=""):
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def penetration_frac(head, hair):
    chart = hair.vertex_attributes["chart_uv"]
    S = (hair.vertices - head.center) / head.radii
    r_now = np.linalg.norm(S, axis=1)
    u = np.clip(chart[:, 0], -1, 1)
    vv = np.clip(chart[:, 1], 0, head.V_MAX - 1e-6)
    r_scalp = head.sample_radius(u, vv)
    depth = (r_scalp - r_now) / np.maximum(r_scalp, 1e-9)
    return float((depth > 0.12).mean())


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="wig-suite-"))
    print("== heads ==")
    mq_path = str(Path(__file__).parent / "assets" / "mannequin.obj")
    head = Head.from_file(mq_path, **MANNEQUIN)
    check("mannequin chart fit", 6 < head.scale < 13, f"scale {head.scale:.1f} cm")

    # a meters-unit copy of the same head proves unit invariance exactly
    m = trimesh.load(mq_path, force="mesh", process=False)
    m.vertices = m.vertices * 0.01
    scaled_path = str(out_dir / "mannequin_m.obj")
    (out_dir).mkdir(exist_ok=True)
    m.export(scaled_path)
    head_m = Head.from_file(scaled_path, forward=[0, 0, 1], eye_level=-0.01)
    check("meters-unit twin fit", abs(head_m.scale * 100 - head.scale) / head.scale < 0.05,
          f"{head_m.scale*100:.2f} vs {head.scale:.2f}")

    print("== determinism ==")
    style, tex = PRESETS["bob_bangs"]
    a = generate(head, style)
    b = generate(head, style)
    check("same style twice -> identical mesh", np.allclose(a.vertices, b.vertices))
    n1, s1, t1 = sample_style(42)
    n2, s2, t2 = sample_style(42)
    check("sampler deterministic", n1 == n2 and t1.seed == t2.seed)
    names = {sample_style(k)[0].rsplit("_s", 1)[0] for k in range(40)}
    check("sampler diversity", len(names) >= 10, f"{len(names)} archetypes in 40 draws")

    print("== mesh sanity + penetration (all presets) ==")
    for pname, (st, tx) in PRESETS.items():
        hair = generate(head, st)
        v, f = hair.vertices, hair.faces
        ok_geom = np.isfinite(v).all() and f.max() < len(v) and len(f) > 50
        frac = penetration_frac(head, hair)
        check(f"{pname}", ok_geom and frac < 0.12, f"deep frac {frac:.3f}, v={len(v)}")

    print("== unit invariance of a style ==")
    st, tx = PRESETS["bob"]
    e1 = np.ptp(generate(head, st).vertices, axis=0) / head.scale
    e2 = np.ptp(generate(head_m, st).vertices, axis=0) / head_m.scale
    rel = np.abs(e1 - e2) / np.maximum(e1, 1e-9)
    check("bob extent scales with head.scale", rel.max() < 0.05, f"rel {np.round(rel,3)}")

    print("== HairIntent v1 + guide drape ==")
    intent_path = Path(__file__).parent / "assets" / "long_wavy_split.hair.json"
    check("packaged schema matches repository schema",
          hair_intent_schema() == json.loads(
              (Path(__file__).parents[2] / "src/character_factory/hair/wig/schemas/hair_intent_v1.schema.json").read_text()))
    intent = load_hair_intent(intent_path)
    plan_a = compile_hair_intent(intent)
    plan_b = compile_hair_intent(json.loads(intent_path.read_text()))
    cf = plan_a.style.clump_field
    check("semantic regional lengths compile",
          cf.length_front < cf.length_side < cf.length,
          f"front/side/back {cf.length_front:.2f}/{cf.length_side:.2f}/{cf.length:.2f}")
    check("semantic part extent compiles",
          plan_a.style.cap.part_u == 0.29 and plan_a.style.cap.part_start_v == 0.14
          and plan_a.style.clump_field.part_open)
    no_part = compile_hair_intent({"family": "loose_long", "part": {"kind": "none"}})
    check("semantic no-part closes cap and clump roots",
          not no_part.style.cap.part_open and not no_part.style.clump_field.part_open
          and no_part.style.clump_field.part_gap == 0.0)
    hairline_plan = compile_hair_intent({
        "family": "loose_long",
        "hairline": {
            "height": "low", "shape": "widows_peak",
            "temple_recession": "pronounced", "sideburns": "long",
            "nape": "low", "irregularity": "textured",
        },
    })
    baseline_line = plan_a.style.cap.hairline
    changed_line = hairline_plan.style.cap.hairline
    check("semantic hairline compiles",
          changed_line.front > baseline_line.front
          and changed_line.recess > baseline_line.recess
          and changed_line.sideburn > baseline_line.sideburn
          and changed_line.nape > baseline_line.nape
          and changed_line.jitter == 0.035)
    check("semantic drape compiles", cf.drape.enabled and cf.drape.routing == "split")
    check("semantic texture compiles to material",
          plan_a.texture.wave == 0.55 and plan_a.texture.curl == 0.06)
    intent_hair_a = generate(head, plan_a.style)
    intent_hair_b = generate(head, plan_b.style)
    check("HairIntent output deterministic",
          np.array_equal(intent_hair_a.vertices, intent_hair_b.vertices))
    check("HairIntent output is finite opaque geometry",
          np.isfinite(intent_hair_a.vertices).all() and len(intent_hair_a.faces) > 1000,
          f"v={len(intent_hair_a.vertices)}, f={len(intent_hair_a.faces)}")
    intent_m = generate(head_m, compile_hair_intent(intent).style)
    ie1 = np.ptp(intent_hair_a.vertices, axis=0) / head.scale
    ie2 = np.ptp(intent_m.vertices, axis=0) / head_m.scale
    irel = np.abs(ie1 - ie2) / np.maximum(ie1, 1e-9)
    check("draped HairIntent is unit invariant", irel.max() < 0.05,
          f"rel {np.round(irel,3)}")
    try:
        compile_hair_intent({"family": "loose_long", "raw_solver_knob": 42})
        strict = False
    except HairIntentError:
        strict = True
    check("unknown LLM fields rejected", strict)

    print("== GLB roundtrip (albedo + normal) ==")
    img, nrm = strand_maps(tx)
    hair = apply_material(generate(head, st), img, normal=nrm)
    tangent = mesh_tangents(hair)
    check("UV tangents are finite unit vectors",
          np.isfinite(tangent).all() and np.max(np.abs(np.linalg.norm(tangent, axis=1) - 1)) < 1e-4)
    out = out_dir / "roundtrip_test.glb"
    export_glb(hair, str(out))
    from pygltflib import GLTF2

    g = GLTF2().load_binary(str(out))
    mat = g.materials[0]
    primitive = g.meshes[0].primitives[0]
    check("GLB has smooth vertex normals",
          primitive.attributes.NORMAL is not None
          and g.accessors[primitive.attributes.NORMAL].count
          == g.accessors[primitive.attributes.POSITION].count)
    check("GLB has baseColor + normal textures",
          mat.pbrMetallicRoughness.baseColorTexture is not None and mat.normalTexture is not None,
          f"{out.stat().st_size//1024} KB, {len(g.images)} images")
    check("GLB material is opaque", mat.alphaMode in (None, "OPAQUE"),
          f"alphaMode={mat.alphaMode!r}")
    check("GLB advertises optional hair anisotropy",
          "KHR_materials_anisotropy" in (g.extensionsUsed or [])
          and mat.extensions["KHR_materials_anisotropy"]["anisotropyStrength"] > 0
          and "KHR_materials_anisotropy" not in (g.extensionsRequired or []))

    real = os.environ.get("MAKE_WIG_TEST_HEAD")
    if real:
        print(f"== optional real head: {real} ==")
        rh = Head.from_file(real)
        hair = generate(rh, PRESETS["bob"][0])
        check("real head bob", penetration_frac(rh, hair) < 0.12)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURES: {FAIL}")
        sys.exit(1)
    print("all checks passed")


def test_wig_engine_suite():
    """The engine's own validation suite, run as one pytest case."""
    FAIL.clear()
    main()


if __name__ == "__main__":
    main()


def test_every_preset_base_texture_plate_is_vendored():
    """The engine raises at synthesis time if a preset's base plate is
    missing — the failure that broke a build must be structurally
    impossible: every referenced plate ships with the package."""
    import dataclasses
    from pathlib import Path

    from character_factory.hair.wig import presets as preset_module

    package_root = Path(preset_module.__file__).resolve().parents[2]
    referenced = set()
    for name in dir(preset_module):
        value = getattr(preset_module, name)
        if isinstance(value, dict):
            for entry in value.values():
                for part in (entry if isinstance(entry, tuple) else (entry,)):
                    if dataclasses.is_dataclass(part):
                        base = getattr(part, "base_texture", None)
                        if base:
                            referenced.add(base)
    assert referenced, "expected at least one preset to reference a plate"
    for relative in referenced:
        assert (package_root / relative).is_file(), f"missing plate: {relative}"
