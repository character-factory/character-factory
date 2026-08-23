# The Character Format

**Version 0.1 — draft**

This document specifies the character format used by Character Factory: a
compact, symbolic description of a rigged, textured 3D human. A character file
is a small JSON document — typically 2–6 KB — that records *how to build* a
character, not the built result. Meshes, textures, and rigged scene files are
build artifacts derived from it.

The format is designed to be implemented by third parties. Everything a
conforming reader or writer needs is in this document. Familiarity with the
[Momentum Human Rig (MHR)](https://github.com/facebookresearch/MHR) parametric
body model is helpful but not required.

## 1. Design goals

1. **Symbolic, not binary.** A character is parameters, references, and
   recipes. It can be diffed, versioned in git, sent in a chat message, edited
   by hand, and read by a coding agent.
2. **Reproducible.** A character file plus a pinned set of generation
   components rebuilds the same character. Every generative step records the
   inputs that produced it.
3. **Forward-compatible.** New texture slots, new components, and a future
   mouth-interior body variant are additive changes, not breaking ones.
4. **Small surface.** Five blocks: `body`, `textures`, `hair`, `provenance`,
   `assets`. No block requires another implementation's internals to
   interpret.

## 2. Conventions

The key words MUST, MUST NOT, SHOULD, and MAY are to be interpreted as in RFC
2119.

- A character file is a single JSON document, UTF-8 encoded, with no byte
  order mark. The recommended file extension is `.char.json`.
- All numbers that represent model parameters are IEEE 754 binary32 (float32)
  values serialized as JSON numbers. Writers MUST emit decimal
  representations that round-trip to the same float32 value; the shortest
  round-tripping representation is RECOMMENDED.
- Field names are `snake_case`. Unknown top-level and block-level fields MUST
  be rejected in strict validation and SHOULD produce a warning (not an
  error) in default validation, to allow forward-compatible minor additions
  (see §10).
- Linear lengths are centimeters, colors are linear-light RGB in [0, 1], and
  the coordinate system is the body model's native frame: Y up, character
  facing +Z, feet at Y = 0.

### 2.1 Canonical form and content identity

The **canonical form** of a character document is its JSON Canonicalization
Scheme serialization (RFC 8785). The **content ID** of a character is the
lowercase hex SHA-256 of the canonical form. Two files with the same content
ID describe the same character. Implementations that cache or deduplicate
characters SHOULD key on the content ID. The content ID is never stored
inside the document itself.

## 3. Document structure

A complete v0.1 character file:

```json
{
  "format": "character-factory/character",
  "schema_version": "0.1",
  "name": "marathon-runner",
  "body": {
    "rig": "mhr-lod1@1.0",
    "topology": "closed",
    "identity": [0.1837423, -0.0921118, 1.2210972, "... 45 values total"],
    "resting_expression": [0.0, 0.0, 0.0, "... 72 values total"]
  },
  "textures": {
    "skin": {
      "component": "make-skin",
      "component_version": "0.1.0",
      "prompt": "light-medium skin tone, adult, subtle freckles across the nose",
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
      "component_version": "0.1.0",
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
    "prompt": "a lean marathon runner with cropped dark hair and green eyes, teal running vest",
    "generator": "character-factory/0.1.0",
    "components": {
      "interpreter": { "version": "0.1.0" },
      "make-figure": { "version": "0.1.0" },
      "make-skin": { "version": "0.1.0" },
      "make-eye": { "version": "0.1.0" },
      "make-garment": { "version": "0.1.0" },
      "make-shoe": { "version": "0.1.0" },
      "make-wig": { "version": "0.1.0" }
    },
    "created": "2026-08-18T12:00:00Z"
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
| `format` | string | yes | MUST be `"character-factory/character"`. Identifies the document type independent of file name. |
| `schema_version` | string | yes | The version of this specification the document conforms to. `"0.1"` for documents conforming to this text. |
| `name` | string | no | Human-readable display name. Not an identifier; the content ID (§2.1) identifies the character. |
| `body` | object | yes | Body model parameters. §4. |
| `textures` | object | yes | Texture generation recipes, one per slot. §5. |
| `hair` | object or null | yes | Semantic hair description, or `null` for no hair. §6. |
| `provenance` | object | yes | How this document was produced. §7. |
| `assets` | object | no | Content hashes pinning previously generated texture images. §8. |

## 4. `body` — the parametric body

The body is described entirely by parameters of a published parametric body
model — no mesh data appears in the file.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `rig` | string | yes | The body model and version, as `<model>@<version>`. v0.1 defines exactly one value: `"mhr-lod1@1.0"`. |
| `topology` | string | yes | Surface topology variant. v0.1 defines two values: `"closed"` and `"mouth-interior"`. §4.2. |
| `identity` | array of 45 numbers | yes | MHR identity coefficients, in MHR's native order. Together these determine the body and face shape. |
| `proportions` | object | no | Skeletal-proportion parameters: rig proportion-parameter name → number, `0.0` meaning the rig's template value. Absent (or empty) means the template skeleton. §4.3. |
| `resting_expression` | array of 72 numbers | yes | MHR expression coefficients describing the character's *resting* face (for example, natural eyelid posture). Most entries are typically `0.0`. This is part of identity — it is not an animation pose. |

### 4.1 The `mhr-lod1@1.0` rig

`"mhr-lod1@1.0"` refers to the Momentum Human Rig, release 1.0 line, at level
of detail 1: 18,439 vertices, 36,874 triangles, 127 joints, with 45 identity
coefficients, a 204-value body pose, and 72 expression coefficients. MHR is
published by Meta under the Apache 2.0 license. The exact upstream release
and topology checksum for each rig string are pinned in the Character Factory
component registry; a conforming implementation MUST verify that the rig
asset it evaluates matches the pinned topology (vertex and triangle counts at
minimum, checksum when available) before trusting index-based data derived
from it.

Evaluating the rig with `identity`, a body pose, and an expression yields
posed vertex positions and a posed skeleton. Identity, pose, and expression
change vertex *positions* only — vertex and triangle indexing is invariant.
The character format relies on this: everything index-based (UV layout,
attachment regions, the topology variant below) is defined against the rig
version, not against an individual character.

Animation is out of scope for the format. The rig's 204-value body pose
vector contains two kinds of channels: articulation (joint rotations —
runtime inputs, never properties of a character) and **skeletal
proportions** (segment lengths — identity-class data, carried by
`proportions`, §4.3). A character file never contains articulation;
non-resting expression is likewise a runtime input.

### 4.2 The `topology` variants

`topology` selects the surface variant the document assembles to. v0.1
defines two values.

**`"closed"`** is the full, unmodified MHR surface, with a closed mouth
region. It assembles exactly as the sections above describe: no interior
components, no expression morph targets. This meaning is frozen — a
document that says `"closed"` assembles to the same surface under every
future version of this specification.

**`"mouth-interior"`** is the same exterior surface with a fixed patch of
triangles removed from the mouth region of the rig's triangle buffer, and
interior components assembled behind it: a posterior-lip cuff, an
inner-mouth cavity, and teeth, gums, and tongue meshes. The removal set is
defined purely at the topology level — the same triangle indices for every
character on a given rig version; the exact set, the interior construction
parameters, and the interior mesh data are `body-rig` and `assembly-assets`
component data pinned in the registry, not part of this specification. The
exported artifact for this variant additionally carries the rig's 72
expression coefficients as morph targets (index-stable names `facs_00`
through `facs_71`) plus jaw-animation guidance in its manifest, so the face
is animatable at runtime. This does not change §4.1's semantics: non-resting
expression remains a runtime input, never document data.

Assembling a `"mouth-interior"` document requires a `body-rig` component
version that declares mouth data; assembling it against one that does not is
a defined error — never a silent fall back to a closed surface.

Readers encountering an unrecognized `topology` value MUST treat the
document as requiring a newer schema version rather than silently assembling
a different surface. Writers targeting v0.1 MUST emit one of the two defined
values; both are valid forever, and no migration between them exists or is
implied (they are different characters' surfaces, not versions of one).
Mechanically, the variant ships as a new `body-rig` component version plus
grown `assembly-assets` (the interior meshes are placed by assembly, like
the eyeballs) — no new texture slot and no new component kind.

### 4.3 `proportions` — skeletal proportions

`proportions` maps rig proportion-parameter names to numbers. For
`mhr-lod1@1.0` the vocabulary is exactly six semantic controls:
`spine_length`, `neck_length`, `shoulder_width`, `arm_length`,
`hip_width`, and `leg_length`. Values are in the rig's native proportion
parameterization: `0.0` is the template, positive lengthens or widens the
named dimension (roughly 10 cm per unit for the length controls), and the
valid range is **±0.40**, compared at float32 — the format's canonical
parameter precision (§2.1). Out-of-range values are validation errors,
never clamped. Evaluation is left/right-uniform by construction — the
vocabulary contains no lateralized parameters. The mapping from these
names to rig parameters is registry metadata on the rig version,
alongside the joint-name table; the rig's finer-grained per-segment
scales are not exposed in v0.1 and remain at template values.

An absent block, an empty block, and an absent key all mean the same
thing: the template value. Every document without the block therefore
keeps producing byte-identical output forever.

Readers MUST treat this block like `topology`, not like an optional
annotation: **a reader that does not understand `proportions`, or
encounters an unknown key or out-of-range value in it, MUST refuse to
assemble the document** rather than build the template skeleton — a
proportioned character silently built on the template skeleton is a
different character than the file describes. Unknown keys are hard errors
in every validation mode, with a did-you-mean correction where one is
close. Writers SHOULD omit the block entirely rather than emit an empty
object, and SHOULD NOT emit keys carrying `0.0` — only deviations are
recorded.

Note on the first-party generator: `make-figure` ≥ 0.1.0 authors
`resting_expression` as exact zeros — that field is unchanged in the
format; only the generator's behavior changed.

## 5. `textures` — generation recipes

`textures` maps **slot keys** to slot contents. A slot is a named surface
region that receives generated images; **slot keys are singular nouns,
always** (`skin`, `eye`, `garment`, `shoe`). Validators MUST reject a plural
spelling — the most likely authoring mistake — as a hard error naming the
correction, in every validation mode. v0.1 defines exactly four slots —
three required, one optional:

| Slot | Required | Target surface | Content |
| --- | --- | --- | --- |
| `skin` | yes | The body's canonical UV atlas | Full-body skin albedo: body, face, hands, feet, scalp. |
| `eye` | yes | The eyeball surface (its own concentric UV layout) | One eye albedo (iris, sclera). The single `eye` recipe applies to **both** eyes. |
| `garment` | yes | The body's canonical UV atlas | Clothing painted over an unoccupied (black) background; coverage is recovered from the image itself at assembly time via a calibrated luminance key. |
| `shoe` | no | The generating component's single-shoe canvas (a fixed layout the component declares and ships mapping data for) | One shoe, painted once; at assembly the component's **foot chart** maps the canvas onto the atlas's foot regions of *both* feet (the second foot is mirrored by the chart — the image is never mirrored by hand) and composites above the garment layer (§9). **All footwear composites into the foot regions**; capability growth (say, boots) arrives as a new version of the slot's component widening its declared vocabulary, never as a sibling slot or component. A barefoot character simply has no `shoe` key. |

An optional slot that is not used MUST be omitted entirely — an explicit
`null` is invalid for texture slots (unlike `hair`, which is a required key
with `null` as a meaningful value). This keeps the canonical form (§2.1)
unambiguous: barefoot characters have no `shoe` key at all.

**Every texture slot has a first-party default maker component named
`make-<slot>`** (`make-skin`, `make-eye`, `make-garment`, `make-shoe`); other
components MAY register against the same slot in the component registry. The
pairing is a naming convention, not a monopoly — which component actually
generates a map is resolution machinery recorded in the recipe and pinned in
provenance, never part of the authoring surface.

Future schema minor versions may add further optional slots; slots are
additive and a new slot never changes the meaning of an existing one. The
*anticipated* growth path, however, is within slots — named secondary maps
and conditioned recipes (§5.2, §5.3) — not new slots.

### 5.1 Texture recipe fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `component` | string | yes | Name of the generation component in the component registry (the first-party defaults are `make-skin`, `make-eye`, `make-garment`, `make-shoe`). |
| `component_version` | string | yes | Semantic version of that component used (or to be used) for generation. |
| `prompt` | string | yes | The slot-level text description that conditions generation. This is the *decomposed*, per-slot prompt — not the original character description, which lives in `provenance.prompt`. |
| `seed` | integer | yes | Non-negative seed, ≤ 2³¹ − 1, for the diffusion sampler. Each recipe carries its own explicit resolved seed; the format does not define seed derivation. |
| `overrides` | object | no | Sampler overrides (`steps`: integer, `guidance`: number, `resolution`: integer). Defaults live in component metadata in the registry; a recipe without `overrides` uses them. |
| `inputs` | — | reserved | Reserved for a future minor version; MUST NOT appear in v0.1 documents. See §5.3. |

A recipe is a *claim about how to (re)generate* one image. Regeneration with
the same component version and recipe is reproducible up to GPU kernel
nondeterminism; exact byte-level reproduction is pinned through `assets`
(§8).

### 5.2 Named maps and the flat shorthand

A slot holds **named maps**. v0.1 defines exactly one map name, `albedo`;
each map carries a full recipe (§5.1) and pins its own asset (§8). Because
the overwhelmingly common case is one albedo per slot, a slot's value may
take either of two shapes:

```json
"skin": { "component": "make-skin", "component_version": "0.1.0",
          "prompt": "…", "seed": 41002 }
```

```json
"skin": { "albedo": { "component": "make-skin", "component_version": "0.1.0",
                      "prompt": "…", "seed": 41002 } }
```

The first — a **flat recipe**, recognized by its `component` key — is
shorthand for the second and is the RECOMMENDED authoring form. The two
spellings are one character: implementations MUST canonicalize a slot whose
only map is `albedo` to the flat shorthand before computing the canonical
form (§2.1), so both share one content ID. A slot written as named maps MUST
include `albedo`. Readers encountering an unrecognized map name from a newer
minor version follow the standard unknown-optional rule (§10): warn and
ignore by default, reject in strict mode.

### 5.3 Reserved: conditioning inputs

A future schema minor version will allow a recipe to declare **conditioning
inputs** — references to other maps' generated output consumed as generation
input (the motivating case: a secondary map generated from the same slot's
albedo). v0.1 *reserves* the recipe field `inputs` and pins its semantics
now, so the addition is purely additive later:

- An input reference names a (slot, map) pair and the SHA-256 of the exact
  image consumed: references resolve **through asset hashes,
  content-addressed** — never through file paths or "whatever is currently
  there". A conditioned recipe reproduces its output if and only if its
  declared input hash matches the input actually supplied, which extends the
  determinism story (§9) unchanged to conditioned generation.
- Baking is **dependency-ordered** where inputs exist: a recipe's inputs are
  generated (or supplied and verified) before the recipe runs.
- A reference to a missing or unpinned input is a **defined error**, not a
  fallback to unconditioned generation.

In v0.1 documents `inputs` MUST NOT appear; because a conditioned recipe
cannot be honored by ignoring its inputs, v0.1 validators treat its presence
as a hard error in every mode (the same class as an unrecognized `topology`),
not as an ignorable unknown field.

## 6. `hair` — semantic hair description

Hair is described semantically — a small vocabulary of styling decisions —
and synthesized to geometry at assembly time by a hair provider. The
character file never contains hair geometry. `hair` may be `null`, meaning no
hair (a bald character remains valid and complete).

The hair block is versioned independently of the character schema via its own
integer `schema_version`; this section defines hair schema version `1`. All
fields are required unless marked optional; every enum is closed (unknown
values are invalid). Strict validators MUST reject unknown fields anywhere in
the hair block.

| Field | Type / values |
| --- | --- |
| `schema_version` | The integer `1`. |
| `seed` | Integer, `0` to `2147483647`. Seeds all stochastic detail (strand placement, texture grain). Same hair block + same head geometry + same provider version ⇒ identical geometry. |
| `family` | `buzz`, `crop`, `pixie`, `side_part`, `bob`, `loose_long`, `coily`, `ponytail`, `bun`, `braids`, `locs` — the overall structural archetype. |

**`part`** — where and how the hair parts:

| Field | Values |
| --- | --- |
| `kind` | `none`, `center`, `side` |
| `side` | `wearer_left`, `wearer_right` (meaningful when `kind` is `side`) |
| `position` | `subtle`, `moderate`, `deep` |
| `extent` | `short`, `to_crown`, `through_crown` |
| `width` | `narrow`, `medium`, `wide` |

**`hairline`** — the boundary between hair and skin:

| Field | Values |
| --- | --- |
| `height` | `low`, `natural`, `high` |
| `shape` | `rounded`, `straight`, `widows_peak` |
| `temple_recession` | `none`, `natural`, `pronounced` |
| `sideburns` | `short`, `natural`, `long` |
| `nape` | `high`, `natural`, `low` |
| `irregularity` | `clean`, `natural`, `textured` |

**`length`** — how far the hair reaches. The length scale, common to all four
fields: `cropped`, `ear`, `jaw`, `chin`, `shoulder`, `collarbone`,
`below_shoulder`, `chest`, `mid_back`, `waist`.

| Field | Type | Required |
| --- | --- | --- |
| `overall` | length scale | yes |
| `front`, `side`, `back` | length scale | optional — each falls back to `overall` when omitted |
| `cut_line` | `blunt`, `soft`, `layered` | yes |

**`shape`** — volume and strand character:

| Field | Values |
| --- | --- |
| `volume` | `low`, `medium`, `high` |
| `density` | `light`, `medium`, `full` |
| `texture` | `straight`, `wavy`, `curly`, `coily` |
| `wave_size` | `small`, `medium`, `large` |
| `wave_strength` | `subtle`, `medium`, `strong` |
| `root_lift` | `low`, `medium`, `high` |

**`drape`** — how hair falls against the body:

| Field | Values |
| --- | --- |
| `gravity` | `light`, `natural`, `heavy` |
| `stiffness` | `soft`, `natural`, `firm` |
| `shoulder_routing` | `natural`, `split`, `mostly_behind`, `all_front`, `all_behind` |
| `body_clearance` | `close`, `natural`, `loose` |

**`color`**:

| Field | Type / values |
| --- | --- |
| `family` | `black`, `dark_brown`, `brown`, `auburn`, `copper`, `blonde`, `platinum`, `gray`, `white`, `custom` |
| `rgb` | Array of 3 numbers in [0, 1], linear RGB. REQUIRED when `family` is `custom`; MUST be absent otherwise. |

Writers SHOULD emit fully resolved hair blocks — every non-optional field
explicit — so that a character file reads the same in any implementation.
The only permitted omissions are `length.front`/`side`/`back` (fallback:
`overall`) and `color.rgb` (forbidden unless `custom`).

The hair provider consumes this block plus the character's assembled head and
body geometry (for scalp fitting and long-hair drape) and returns a textured
triangle mesh in the body's frame. §6 is the complete contract between the
format and any provider; providers are interchangeable behind it.

## 7. `provenance` — how the character came to be

Provenance makes a character file self-describing: which text produced it,
by which pipeline, using which component versions. It is required, because
reproducibility is a core promise of the format — but its fields are
descriptive, not instructions to a reader.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `prompt` | string or null | yes | The original free-text character description the file was generated from. `null` for hand-authored or edited files. |
| `generator` | string | yes | The producing software and version, as `<name>/<version>`. |
| `components` | object | yes | Map from **component name** to `{ "version": string, "sha256": string (optional) }` for every generative component that produced values in this file — at minimum `make-figure` (which produced `body.identity` and `body.resting_expression`) and one entry per texture component in use. An `interpreter` entry records the component that mapped the free-text description onto the per-slot prompts and the hair block, like any other generative component. The hair provider's entry (the first-party default is `make-wig`) records its version once geometry has been synthesized — the `hair` block itself carries no component field, because it is semantic vocabulary, not a texture slot. |
| `created` | string | no | RFC 3339 timestamp. |
| `notes` | string | no | Free text. |

Identity generation is deterministic: the identity component maps prompt
text to body parameters as a pure function, with no seed. `provenance.prompt`
plus the pinned `identity` component version therefore reproduces
`body.identity` exactly — and conversely, editing `body.identity` by hand
makes `provenance.prompt` a historical note rather than a regeneration
recipe. Implementations MUST treat the parameter arrays in `body`, not the
prompt, as authoritative.

## 8. `assets` — pinning generated images

`assets` is optional. When present, it maps texture slot keys to content
descriptors of the generated images, **per map**: a slot's entry is either a
flat descriptor — shorthand for its `albedo` map, mirroring §5.2 exactly,
including the canonicalization rule — or an object of map-name →
descriptor. A descriptor:

| Field | Type | Description |
| --- | --- | --- |
| `sha256` | string | Lowercase hex SHA-256 of the image file. |
| `media_type` | string | IANA media type; v0.1 generators emit `image/png`. |
| `width`, `height` | integer | Pixel dimensions. |

The format deliberately stores hashes, not paths or URLs — where assets live
is an implementation concern (a sibling directory, a cache, an object
store). An assembler given both a character file and a set of candidate
asset files MUST verify hashes before use and MUST refuse to silently
substitute a non-matching asset. A character file without `assets` (or with
missing entries) is still complete: its textures are regenerable from the
recipes in §5.

## 9. Assembly semantics

This section defines what a character file *means* in terms of the built
artifact, without prescribing an implementation.

1. **Body.** Evaluate the rig (§4.1) with `body.identity`,
   `body.resting_expression`, and a rest body pose. The result is the body
   mesh and the rest skeleton.
2. **Surface.** Apply the `skin` albedo to the body's canonical UV atlas.
   Recover garment coverage from the `garment` albedo (luminance-keyed
   occupancy over the black background) and composite covered texels over
   the skin image. If a `shoe` slot is present, map its single-shoe canvas
   onto both feet through the generating component's foot-chart data
   (style-aware occupancy: open styles keep only their straps, the shaft
   region is cut to the style's height) and composite the result above the
   garment layer. The compositing order is normative:
   **skin, then garment, then shoe** — shoe occludes garment (socks under
   shoes), garment occludes skin. The final composited image is the body's
   albedo.
3. **Eyes.** Apply the `eye` albedo to both eyeball meshes, placed in the
   rig's eye sockets.
4. **Mouth** (`"mouth-interior"` topology only). Remove the rig version's
   fixed mouth patch from the body's triangle buffer. Construct the
   posterior-lip cuff and inner-mouth cavity from the body's inner-lip
   curves, and place the teeth, gums, and tongue meshes from the body's
   identity (§4.2). Upper anatomy is skull-locked; lower anatomy binds to
   the rig's jaw joint.
5. **Hair.** If `hair` is non-null, synthesize hair geometry from the hair
   block and the assembled head/body geometry; attach it rigidly to the head.
6. **Rig.** The exported artifact carries the rig's full joint hierarchy and
   per-vertex skinning weights, so the result is animatable, not a statue.
   For `"mouth-interior"` documents it also carries the rig's 72 expression
   coefficients as morph targets (§4.2) and, in its manifest, the measured
   animation-limitation table and jaw guidance for consumers.

Assembly is deterministic: the same character file, the same pinned assets
(or byte-identical regenerated ones), and the same assembler version MUST
produce an identical scene, and SHOULD produce a byte-identical scene file.

## 10. Versioning and compatibility

`schema_version` is `"<major>.<minor>"`.

- **Minor versions are additive.** A later minor version may add optional
  top-level fields, optional recipe fields, new named maps, new texture
  slots, new `rig` strings, and new `topology` values. It MUST NOT change
  the meaning or validity of any document that was valid under an earlier
  minor version of the same major. The anticipated additive path is **named
  secondary maps within existing slots plus conditioning inputs (§5.2,
  §5.3), and the widening of the skeletal-proportion vocabulary (§4.3) as
  rig versions expose finer-grained parameters** — not new slots.
- **Readers** encountering a document with a newer minor version than they
  implement SHOULD process it, ignoring unrecognized optional fields, with
  three exceptions that are hard errors: an unrecognized `topology` or
  `rig` value (§4.2), a recipe carrying `inputs` (§5.3) — a conditioning
  input ignored is a different image silently built — and a
  `body.proportions` block or key the reader does not implement (§4.3) — a
  proportioned character silently built on the template skeleton is a
  different character than the file describes. All of these change what
  the document *builds*, not just what it *records*.
- **Major version 0 caveat.** While the major version is 0, breaking changes
  may occur between minors; each will ship with a documented migration. From
  1.0, breaking changes require a major bump.
- The hair block versions independently (§6); a character schema version
  pins the set of hair schema versions it accepts (v0.1 accepts hair schema
  `1` only).

## 11. Validation

A conforming implementation validates, at minimum: presence and types of all
required fields; array lengths (45, 72, 3); enum membership for every closed
vocabulary (including `topology` and `rig`); seed ranges; the
`color.rgb`/`custom` co-constraint; the optional-slot omission rule (§5);
singular slot keys, with plural spellings rejected as hard errors naming the
correction (§5); the albedo requirement and shorthand shapes (§5.2); the
reserved `inputs` field (§5.3, hard error); the `proportions` vocabulary and
range (§4.3 — unknown names and out-of-range values are hard errors in every
mode, with a did-you-mean correction for near-miss names — the same
correction applies to near-miss `topology` values); and finiteness of
all numbers (NaN and infinities are invalid everywhere). The reference implementation publishes a
machine-readable JSON Schema for each schema version and exposes validation
as a library call, a CLI command, and an MCP tool; third parties are
encouraged to validate against the JSON Schema directly.

## 12. What the format is not

- **Not a mesh interchange format.** The built artifact is standard glTF;
  interchange happens there.
- **Not an animation format.** Poses, expressions, and clips are runtime
  data for the rig.
- **Not a likeness record.** A character file stores parameters of a
  synthetic identity described by text. It records no biometric data,
  reference images, or real-person identifiers.
