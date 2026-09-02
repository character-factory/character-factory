# The Character Format

Version 0.1 · schema `character-factory/character`

A character file is a small JSON document (typically 2–6 KB) that records how to build a rigged, textured 3D human: body-model parameters, texture generation recipes, a semantic hair description, and provenance. Meshes, textures, and scene files are build artifacts derived from it.

This document is complete for a third-party reader or writer. Familiarity with the [Momentum Human Rig (MHR)](https://github.com/facebookresearch/MHR) body model helps but is not required.

1. [Design goals](#1-design-goals)
2. [Conventions](#2-conventions)
3. [Document structure](#3-document-structure)
4. [`body`](#4-body)
5. [`textures`](#5-textures)
6. [`hair`](#6-hair)
7. [`provenance`](#7-provenance)
8. [`assets`](#8-assets)
9. [Assembly semantics](#9-assembly-semantics)
10. [Versioning](#10-versioning)
11. [Validation](#11-validation)
12. [What the format is not](#12-what-the-format-is-not)

## 1. Design goals

1. **Symbolic.** Parameters, references, and recipes — diffable, editable by hand, readable by an agent.
2. **Reproducible.** A character file plus pinned generation components rebuilds the same character. Every generative step records its inputs.
3. **Runtime-ready.** Every character builds with the full skeleton, a modeled mouth interior, and 72 expression morph targets.
4. **Small.** Five blocks: `body`, `textures`, `hair`, `provenance`, `assets`.

## 2. Conventions

MUST, MUST NOT, SHOULD, and MAY are as in RFC 2119.

- A character file is one UTF-8 JSON document without a byte order mark. Recommended extension: `.char.json`.
- Model parameters are IEEE 754 float32 values serialized as JSON numbers. Writers MUST emit representations that round-trip to the same float32; the shortest such representation is RECOMMENDED.
- Field names are `snake_case`. Unknown fields are rejected in strict validation and SHOULD produce a warning in default validation (§10).
- Lengths are centimeters; colors are linear RGB in [0, 1]; the coordinate frame is the body model's: Y up, facing +Z, feet at Y = 0.

### 2.1 Canonical form and content ID

The canonical form is the RFC 8785 (JSON Canonicalization Scheme) serialization of the document. The content ID is the lowercase hex SHA-256 of the canonical form. Two files with the same content ID are the same character. The content ID is never stored in the document.

## 3. Document structure

A complete example. The two `"... N values total"` markers and the four `…`-shortened hashes are elisions for the page; with the arrays and hashes written out, this document validates under `character-factory validate --strict`.

```json
{
  "format": "character-factory/character",
  "schema_version": "0.1",
  "name": "marathon-runner",
  "body": {
    "rig": "mhr-lod1@1.0",
    "topology": "mouth-interior",
    "identity": [0.13, 0.9256, 1.0908, "... 45 values total"],
    "proportions": { "leg_length": 0.24, "hip_width": -0.06 },
    "resting_expression": [0.0, 0.0, "... 72 values total"]
  },
  "textures": {
    "skin": {
      "component": "make-skin",
      "component_version": "0.0.4",
      "prompt": "light-medium neutral-toned skin, adult, subtle freckles across the nose",
      "seed": 41002
    },
    "eye": {
      "component": "make-eye",
      "component_version": "0.1.0",
      "prompt": "green iris with amber central ring, fine radial fibers",
      "seed": 41003
    },
    "garment": {
      "component": "make-garment",
      "component_version": "0.1.0",
      "prompt": "teal running vest and black shorts, white piping",
      "seed": 41004
    },
    "shoe": {
      "component": "make-shoe",
      "component_version": "0.0.2",
      "prompt": "low white running shoes with a teal stripe",
      "seed": 41005
    }
  },
  "hair": {
    "schema_version": 1,
    "seed": 7,
    "family": "crop",
    "part": { "kind": "none", "side": "wearer_left", "position": "moderate",
              "extent": "to_crown", "width": "narrow" },
    "hairline": { "height": "natural", "shape": "rounded",
                  "temple_recession": "natural", "sideburns": "natural",
                  "nape": "natural", "irregularity": "natural" },
    "length": { "overall": "cropped", "cut_line": "soft" },
    "shape": { "volume": "low", "density": "medium", "texture": "straight",
               "wave_size": "medium", "wave_strength": "medium",
               "root_lift": "medium" },
    "drape": { "gravity": "natural", "stiffness": "natural",
               "shoulder_routing": "split", "body_clearance": "natural" },
    "color": { "family": "dark_brown" }
  },
  "provenance": {
    "prompt": "a lean marathon runner with cropped dark hair and green eyes, teal running vest, white running shoes",
    "figure_prompt": "adult, lean endurance-runner build, low body fat, long legs, narrow hips, …",
    "seed": 41000,
    "generator": "character-factory/0.1.0",
    "components": {
      "interpreter": { "version": "0.1.0" },
      "make-figure": { "version": "0.1.1" },
      "make-skin": { "version": "0.0.4" },
      "make-eye": { "version": "0.1.0" },
      "make-garment": { "version": "0.1.0" },
      "make-shoe": { "version": "0.0.2" },
      "make-wig": { "version": "0.1.0" }
    },
    "created": "2026-09-01T12:00:00Z"
  },
  "assets": {
    "skin": { "sha256": "9f2c…", "media_type": "image/png", "width": 1024, "height": 1024 },
    "eye": { "sha256": "1b77…", "media_type": "image/png", "width": 1024, "height": 1024 },
    "garment": { "sha256": "c04a…", "media_type": "image/png", "width": 1024, "height": 1024 },
    "shoe": { "sha256": "5e19…", "media_type": "image/png", "width": 1024, "height": 1024 }
  }
}
```

### 3.1 Top-level fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `format` | string | yes | MUST be `"character-factory/character"`. |
| `schema_version` | string | yes | `"0.1"` for documents conforming to this text. |
| `name` | string | no | Display name. Not an identifier. |
| `body` | object | yes | §4. |
| `textures` | object | yes | §5. |
| `hair` | object or null | yes | §6. `null` means no hair. |
| `provenance` | object | yes | §7. |
| `assets` | object | no | §8. |

## 4. `body`

The body is described entirely by parameters of a published parametric body model; no mesh data appears in the file.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `rig` | string | yes | `<model>@<version>`. v0.1 defines exactly `"mhr-lod1@1.0"`. §4.1. |
| `topology` | string | yes | The surface contract. v0.1 defines exactly `"mouth-interior"`. §4.2. |
| `identity` | 45 numbers | yes | MHR identity coefficients, native order. Determines body and face shape. |
| `proportions` | object | no | Skeletal-proportion parameter name → number; `0.0` is the template. Absent or empty means the template skeleton. §4.3. |
| `resting_expression` | 72 numbers | yes | MHR expression coefficients for the *resting* face. Part of identity, not an animation pose. |

### 4.1 The `mhr-lod1@1.0` rig

The Momentum Human Rig, release 1.0 line, LOD1: 18,439 vertices, 36,874 triangles, 127 joints, 45 identity coefficients, a 204-value body pose, and 72 expression coefficients. Published by Meta under Apache-2.0.

Evaluating the rig with identity, a pose, and an expression yields vertex positions and a posed skeleton. Only positions change: vertex and triangle indexing is invariant across characters, and everything index-based in this format (UV layout, attachment regions, the topology patch) is defined against the rig version.

**Render surface.** The rig string names the parameter space and the source topology the rig evaluates. The surface an assembler *exports* is the render topology declared by the pinned `body-rig` component — which MAY be a coarser tessellation of the same rig, carried by a supplied vertex map with skin weights and expression morphs transferred (the launch component renders MHR's LOD3: 4,899 vertices, 9,794 triangles for the full closed surface, before the assembler removes aperture and under-shell faces). The render topology, its vertex map, its checksum, and its aperture patches are component data, pinned in the registry, not part of this specification. A conforming assembler MUST verify the rig and render artifacts it loads against their pinned checksums before trusting index-based data.

**Pose channels.** The rig's 204-value pose holds two kinds of channels: articulation (joint rotations — runtime inputs, never document data) and skeletal proportions (segment lengths — identity-class data, `proportions`, §4.3). A character file never contains articulation, and non-resting expression is a runtime input.

### 4.2 `topology`

`topology` names the surface a document builds. v0.1 defines exactly `"mouth-interior"`: the rig's exterior render surface with a fixed triangle patch removed at the mouth and, behind it, a posterior-lip cuff, an inner-mouth cavity, and teeth, gum, and tongue meshes. The removal set is the same for every character on a rig version; the set, the construction parameters, and the interior meshes are `body-rig` and `assembly-assets` component data. The built artifact also carries the 72 expression coefficients as morph targets `facs_00`–`facs_71` and jaw guidance in its manifest.

Assembling a document requires a `body-rig` version that declares the mouth basis and compatible `assembly-assets`. Missing or incompatible data is an error, never a reduced surface.

Readers encountering an unrecognized `topology` MUST refuse to assemble rather than build a different surface. Writers targeting v0.1 MUST emit `"mouth-interior"`.

### 4.3 `proportions`

For `mhr-lod1@1.0` the vocabulary is six controls: `spine_length`, `neck_length`, `shoulder_width`, `arm_length`, `hip_width`, `leg_length`.

Values are in the rig's proportion parameterization: `0.0` is the template, positive lengthens or widens (roughly 10 cm per unit for length controls), valid range **±0.40** compared at float32. Out-of-range values are validation errors, never clamped. Evaluation is left/right-uniform. The name → rig-parameter mapping is registry metadata on the rig version; the rig's finer per-segment scales are not exposed in v0.1.

An absent block, an empty block, and an absent key all mean the template value. Writers SHOULD omit the block when empty and SHOULD NOT emit `0.0` keys.

A reader that does not implement `proportions`, or meets an unknown key or out-of-range value, MUST refuse to assemble rather than build the template skeleton. Unknown keys are hard errors in every validation mode, with a did-you-mean correction where one is close.

## 5. `textures`

`textures` maps **slot keys** to recipes. Slot keys are singular (`skin`, `eye`, `garment`, `shoe`); a plural spelling is a hard error naming the correction, in every validation mode.

| Slot | Required | Target | Content |
| --- | --- | --- | --- |
| `skin` | yes | The body's UV atlas | Full-body skin albedo. |
| `eye` | yes | The eyeball surface (its own concentric layout) | One eye albedo, applied to both eyes. |
| `garment` | yes | The body's UV atlas | Clothing over a black background; coverage is recovered from the image at assembly. |
| `shoe` | no | The component's single-shoe canvas | One shoe; the component's foot chart maps it onto both feet at assembly (the second foot mirrored by the chart). All footwear derives from the foot regions; wider footwear vocabularies arrive as new versions of the slot's component, never as a sibling slot. A barefoot character has no `shoe` key. |

An unused optional slot MUST be omitted; `null` is invalid for texture slots (unlike `hair`).

Every slot has a first-party default component named `make-<slot>`; other components MAY register against the same slot. Which component generates a map is recorded in the recipe and pinned in provenance.

Slots are additive across minor versions. The anticipated growth path is within slots — named secondary maps and conditioned recipes (§5.2, §5.3).

### 5.1 Recipe fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `component` | string | yes | Registry component name. |
| `component_version` | string | yes | Semantic version used, or to be used, for generation. |
| `prompt` | string | yes | The per-slot prompt that conditions generation — not the original description (`provenance.prompt`). |
| `seed` | integer | yes | 0 ≤ seed ≤ 2³¹ − 1. Each recipe carries its own resolved seed; derivation is not defined by the format. |
| `overrides` | object | no | Sampler overrides: `steps` (integer), `guidance` (number), `resolution` (integer). Defaults come from the component's registry entry. |
| `inputs` | — | reserved | MUST NOT appear in v0.1. §5.3. |

Regeneration from a recipe is reproducible up to GPU kernel nondeterminism; exact bytes are pinned through `assets` (§8).

### 5.2 Named maps and the flat shorthand

A slot holds named maps. v0.1 defines one map name, `albedo`. A slot's value takes either shape:

```json
"skin": { "component": "make-skin", "component_version": "0.0.4", "prompt": "…", "seed": 41002 }
```

```json
"skin": { "albedo": { "component": "make-skin", "component_version": "0.0.4", "prompt": "…", "seed": 41002 } }
```

The flat recipe (recognized by its `component` key) is shorthand for the second and the RECOMMENDED form. Implementations MUST canonicalize a slot whose only map is `albedo` to the flat form before computing the canonical form, so both spellings share a content ID. A slot written as named maps MUST include `albedo`. Unrecognized map names from a newer minor version follow §10: warn and ignore by default, reject in strict mode.

### 5.3 Reserved: `inputs`

A future minor version will let a recipe declare conditioning inputs — other maps' generated output consumed as generation input. The field's semantics are fixed now:

- An input names a (slot, map) pair and the SHA-256 of the exact image consumed. References resolve through asset hashes, never paths.
- Baking is dependency-ordered: inputs are generated or verified first.
- A missing or unpinned input is an error, not unconditioned generation.

In v0.1 documents `inputs` MUST NOT appear; its presence is a hard error in every mode, because a conditioned recipe cannot be honored by ignoring its inputs.

## 6. `hair`

Hair is a small vocabulary of styling decisions, synthesized to geometry at assembly by a hair provider. The file never contains hair geometry. `hair` MAY be `null`.

The block carries its own integer `schema_version`; this section defines version `1`. All fields are required unless marked optional; every enum is closed. Strict validators MUST reject unknown fields in the block.

**Top level**

| Field | Type / values |
| --- | --- |
| `schema_version` | The integer `1`. |
| `seed` | Integer, `0` to `2147483647`. Seeds all stochastic detail. Same block + same head geometry + same provider version ⇒ identical geometry. |
| `family` | `buzz`, `crop`, `pixie`, `side_part`, `bob`, `loose_long`, `coily`, `ponytail`, `bun`, `braids`, `locs` |

**`part`**

| Field | Values |
| --- | --- |
| `kind` | `none`, `center`, `side` |
| `side` | `wearer_left`, `wearer_right` (meaningful when `kind` is `side`) |
| `position` | `subtle`, `moderate`, `deep` |
| `extent` | `short`, `to_crown`, `through_crown` |
| `width` | `narrow`, `medium`, `wide` |

**`hairline`**

| Field | Values |
| --- | --- |
| `height` | `low`, `natural`, `high` |
| `shape` | `rounded`, `straight`, `widows_peak` |
| `temple_recession` | `none`, `natural`, `pronounced` |
| `sideburns` | `short`, `natural`, `long` |
| `nape` | `high`, `natural`, `low` |
| `irregularity` | `clean`, `natural`, `textured` |

**`length`**

The length scale, shared by `overall`, `front`, `side`, and `back`: `cropped`, `ear`, `jaw`, `chin`, `shoulder`, `collarbone`, `below_shoulder`, `chest`, `mid_back`, `waist`.

| Field | Values | Required |
| --- | --- | --- |
| `overall` | length scale | yes |
| `front`, `side`, `back` | length scale | optional — default to `overall` |
| `cut_line` | `blunt`, `soft`, `layered` | yes |

**`shape`**

| Field | Values |
| --- | --- |
| `volume` | `low`, `medium`, `high` |
| `density` | `light`, `medium`, `full` |
| `texture` | `straight`, `wavy`, `curly`, `coily` |
| `wave_size` | `small`, `medium`, `large` |
| `wave_strength` | `subtle`, `medium`, `strong` |
| `root_lift` | `low`, `medium`, `high` |

**`drape`**

| Field | Values |
| --- | --- |
| `gravity` | `light`, `natural`, `heavy` |
| `stiffness` | `soft`, `natural`, `firm` |
| `shoulder_routing` | `natural`, `split`, `mostly_behind`, `all_front`, `all_behind` |
| `body_clearance` | `close`, `natural`, `loose` |

**`color`**

| Field | Type / values |
| --- | --- |
| `family` | `black`, `dark_brown`, `brown`, `auburn`, `copper`, `blonde`, `platinum`, `gray`, `white`, `custom` |
| `rgb` | 3 numbers in [0, 1], linear RGB. REQUIRED when `family` is `custom`; MUST be absent otherwise. |

Writers SHOULD emit fully resolved blocks. The only permitted omissions are `length.front`/`side`/`back` and `color.rgb` (when not `custom`).

The provider consumes this block plus the assembled head and body geometry and returns a textured triangle mesh in the body frame. This section is the complete contract between the format and any provider.

## 7. `provenance`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `prompt` | string or null | yes | The original description. `null` for hand-authored or edited files. |
| `figure_prompt` | string | no | The body-generation prompt the interpreter wrote for the figure component — what the identity model conditioned on. |
| `seed` | integer | no | The creation seed (caller-given, or drawn by the generating pipeline). Slot and hair seeds derive from it. |
| `generator` | string | yes | `<name>/<version>` of the producing software. |
| `components` | object | yes | Component name → `{ "version": string, "sha256": string (optional) }` for every component that produced values in the file: `make-figure` (for `body`), each texture component, `interpreter`, and the hair provider (`make-wig` by default) once geometry has been synthesized. The `hair` block itself carries no component field. |
| `created` | string | no | RFC 3339 timestamp. |
| `notes` | string | no | Free text. |

Provenance is descriptive. Identity generation is a sample from the figure prompt's distribution, seeded by the pipeline; the drawn values in `body` are what the file means, and implementations MUST treat the parameter arrays, not the prompts, as authoritative.

## 8. `assets`

`assets` maps slot keys to content descriptors of generated images, per map — a flat descriptor is shorthand for `albedo`, with the same canonicalization rule as §5.2.

| Field | Type | Description |
| --- | --- | --- |
| `sha256` | string | Lowercase hex SHA-256 of the image file. |
| `media_type` | string | IANA media type; v0.1 generators emit `image/png`. |
| `width`, `height` | integer | Pixel dimensions. |

The format stores hashes, not paths. An assembler given candidate asset files MUST verify hashes before use and MUST NOT substitute a non-matching asset. A file without `assets` is complete; its textures are regenerable from §5.

## 9. Assembly semantics

What a document means in terms of the built artifact, without prescribing an implementation:

1. **Body.** Evaluate the rig with `identity`, `proportions`, `resting_expression`, and a rest pose; transfer to the declared render surface (§4.1).
2. **Surface.** The body's albedo is the `skin` image. Recover garment coverage from the `garment` image (luminance-keyed over black); the covered region becomes a separate body-following garment mesh carrying the garment texture, and body faces under it are omitted, keeping a narrow skin band at the boundary. If `shoe` is present, map its canvas onto both feet through the component's foot chart and build a shoe mesh the same way. Layering: skin, then garment, then shoe. Coverage that cannot produce valid geometry is an assembly error.
3. **Eyes.** Apply the `eye` albedo to both eyeball meshes in the rig's sockets.
4. **Mouth.** Remove the rig version's mouth patch; construct the posterior-lip cuff and cavity from the inner-lip curves; place teeth, gums, and tongue from identity. Upper anatomy is skull-locked; lower anatomy binds to the jaw joint.
5. **Hair.** If `hair` is non-null, synthesize from the block and the assembled head and body; attach rigidly to the head.
6. **Rig.** Export the full joint hierarchy, skinning weights, the 72 expression morph targets, and a manifest carrying jaw guidance, the animation-limitation table, and ground/foot reference data.

The same document, the same pinned assets, and the same assembler version MUST produce an identical scene and SHOULD produce a byte-identical file.

## 10. Versioning

`schema_version` is `"<major>.<minor>"`.

- **Minor versions are additive**: optional top-level and recipe fields, new named maps, new slots, new `rig` and `topology` values. A later minor MUST NOT change the meaning or validity of a document valid under an earlier minor of the same major.
- **Readers** given a newer minor SHOULD process it, ignoring unrecognized optional fields, except for hard errors that change what the document *builds*: an unrecognized `topology` or `rig`, a recipe carrying `inputs`, and a `proportions` block or key the reader does not implement.
- **Major 0.** Breaking changes may occur between minors, each with a documented migration. From 1.0, breaking changes require a major bump.
- **Hair** versions independently; v0.1 accepts hair schema `1`.

## 11. Validation

A conforming validator checks, at minimum:

- required fields and types
- array lengths (45, 72, 3)
- every closed enum, including `topology` and `rig`
- seed ranges
- the `color.rgb` / `custom` co-constraint
- omission of unused optional slots
- singular slot keys (plurals rejected with the correction)
- the `albedo` requirement and shorthand shapes
- the reserved `inputs` field
- the `proportions` vocabulary and range (with did-you-mean for near-miss names, as for `topology`)
- finiteness of all numbers

The reference implementation (`character_factory.schema`, standard library only) publishes a JSON Schema per schema version and exposes validation as a library call, `character-factory validate`, `POST /v0/validate`, and the `validate_character` MCP tool.

## 12. What the format is not

- Not a mesh interchange format — the built artifact is standard glTF.
- Not an animation format — poses, expressions, and clips are runtime data.
- Not a likeness record — a character file stores parameters of a synthetic identity described by text. It records no biometric data, reference images, or real-person identifiers.
