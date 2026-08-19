# Architecture

**Status: v0 design document.** This describes what Character Factory is,
how the pieces fit together, and what is deliberately not in v0. The
companion document [SPEC.md](SPEC.md) defines the character format; this
document covers everything else.

Character Factory is licensed under the Apache License 2.0 — code, documents,
and published model weights alike. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for third-party attributions.

## 1. What v0 is

Character Factory turns a text description into a rigged, textured, realtime
3D human. One sentence in, one animatable glTF binary out:

```
"a lean marathon runner with cropped dark hair and green eyes, teal running vest"
        │
        ▼
  interpretation                (a small local language model maps the description to
        │                        per-slot texture prompts + the semantic hair block;
        │                        the raw description also passes through untouched)
        ▼
  identity generation           (raw text → 45 body/face shape coefficients + 72 resting-
        │                        expression values on the MHR parametric body — a
        │                        deterministic function of the text, no seed)
        ▼
  texture generation            (skin, eyes, garments: three UV-space images from a
        │                        FLUX.2 Klein 4B base model with per-slot adapters;
        │                        seeded diffusion, ~1024×1024 albedo each)
        ▼
  hair synthesis                (semantic hair JSON → textured hair mesh, procedural,
        │                        deterministic, no diffusion)
        ▼
  assembly                      (evaluate the rig, composite garments over skin in UV
        │                        space, place eyes and lashes, attach hair, emit a
        ▼                        skinned glTF with the full 127-joint skeleton)
  character.char.json  +  scene.glb
```

The intermediate product — and the thing this project treats as the character
itself — is the **character file** (SPEC.md): a few kilobytes of JSON holding
the body parameters, per-slot texture recipes, semantic hair description, and
provenance. The GLB is a build artifact; the character file is the source.

Two properties are load-bearing:

- **Identity is deterministic.** The identity model consumes the raw user
  description — never an interpreter rewrite of it — so the same description
  always produces the same body. Variation between characters comes from the
  description (and from texture seeds), not from identity sampling. This
  makes character files honest recipes rather than lottery tickets.
- **Everything downstream of generation is symbolic.** Textures are recipes
  (prompt + seed + component version), hair is a semantic vocabulary, the
  body is 117 floats. Regeneration and hand-editing are both first-class.

The determinism boundary is worth stating precisely: **prompt → character
file is not promised deterministic** (interpretation is a language-model
step); **character file → GLB is byte-identical** given pinned assets. The
character file is the reproducible artifact, which is why it — not the
prompt — is the thing you commit, share, and edit.

### 1.1 What v0 is not

Stated plainly, because a reader deciding whether to depend on this needs
the boundaries more than the features:

- **No normal maps or other non-albedo maps** for skin and garments. v0
  surfaces are albedo plus fixed material constants. Generated normal/detail
  maps are planned for v0.2; model components ship on a roughly monthly
  cadence after launch (§4), so this does not wait on a code release. (Hair
  is the exception: the hair synthesizer emits its own albedo and normal
  textures.)
- **No mouth interior.** v0 characters have a closed mouth. The character
  format already reserves the topology variant for a mouth-interior body
  (teeth, gums, tongue behind a fixed removed patch of the face surface —
  see SPEC.md §4.2), and the test surface for it is specified (§7.4), but
  v0 ships without it.
- **No open-strap footwear.** Footwear in this pipeline is texture on the
  foot regions of the body atlas — closed shoes can be painted; sandals and
  straps would require garment geometry, which v0 does not build. A footwear
  texture component is expected to be among the first post-launch registry
  additions.
- **No garment geometry at all, in fact.** Garments are generated in UV
  space and composited onto the body surface — visually "worn," structurally
  painted-on. Loose clothing, skirts, and anything that departs from the
  body silhouette are out of v0 scope.
- **No identity randomness.** There is deliberately no "give me a different
  face for the same prompt" sampler in v0. If you want a different
  character, describe a different character.
- **No animation authoring.** The output is a rigged, skinned glTF; playing
  or retargeting animation on it is the consumer's job.

## 2. One package, three doors

Character Factory is a single Python package with three ways in, in strict
layering order:

```
  ┌───────────────────────────────────────────────┐
  │  MCP server        (character_factory.mcp)    │   coding agents
  ├───────────────────────────────────────────────┤
  │  HTTP server       (character_factory.server) │   apps, UIs
  ├───────────────────────────────────────────────┤
  │  Library API       (character_factory)        │   Python users
  ├───────────────────────────────────────────────┤
  │  schema · registry · identity · textures ·    │
  │  hair · assembly   (internal modules)         │
  └───────────────────────────────────────────────┘
```

The library is the foundation; the HTTP server is a thin process wrapper
around it; the MCP server exposes the *same operations* as tools so a coding
agent can create and inspect characters as part of a workflow. Nothing in
the server layers has logic of its own — if a behavior can't be reached from
the library API, it doesn't exist.

One principle governs both server layers: **local and hosted are one product
with two addresses.** The HTTP contract and the MCP tool surface defined here
are the same contract a hosted service exposes; a user or agent who outgrows
local switches by changing an endpoint (and adding a token), never by
learning a new product. Design decisions in §2.3 and §2.4 that look like
over-engineering for a localhost tool exist to keep that true.

### 2.1 The library API

The public surface a user touches is intentionally small — one class, four
functions:

```python
from character_factory import Character, create, bake, assemble, make

character = create("a lean marathon runner …", seed=41000)
    # → Character. Runs interpretation and identity generation and
    #   fills in texture recipes + hair semantics. Needs the identity
    #   component (GPU strongly recommended, small model); does NOT run
    #   texture diffusion.

assets = bake(character, out_dir="runner/")
    # → BakedAssets. Runs texture generation for every slot (GPU required)
    #   and hair synthesis (CPU). Writes images + hair geometry, records
    #   their hashes into character.assets.

path = assemble(character, assets, "runner/scene.glb")
    # → Path to a rigged, skinned glTF binary. Deterministic; CPU-capable.

path = make("a lean marathon runner …", out_dir="runner/")
    # create → bake → assemble, with progress callbacks. The one-liner.

Character.load("runner/character.char.json")   # schema round-trip
character.save("runner/character.char.json")   # (validated on both ends)
```

`Character` is a plain, validated, immutable-by-convention data object with
`load`/`save`/`validate`/`content_id`. The split between `create`, `bake`,
and `assemble` is the product thesis expressed as an API: the symbolic
character is cheap and always available; pixels and triangles are derived,
cacheable, and re-derivable.

Design rules for the internal modules:

- `schema` and `assembly` import neither the diffusion stack nor any network
  code. A machine that can't generate can still validate, edit, and
  assemble.
- `identity`, `textures` depend on torch + CUDA and are imported lazily, so
  `import character_factory` works everywhere.
- `registry` is the only module that touches the network, and only when a
  component is missing from the local cache.

### 2.2 The interpreter

The step that maps a free-text description onto the character file's
structured fields is a language-model task by construction — the hair block
alone is ~30 closed-vocabulary fields that no rule set can fill from prose —
so it is designed as permanent infrastructure, not a convenience:

```python
class Interpreter(Protocol):
    def interpret(
        self,
        text: str,                     # the user's description or edit request
        existing: Character | None,    # the edit path: text amends this character
    ) -> Interpretation:               # per-slot texture prompts + hair block
        ...
```

The raw text is *not* consumed by the interpreter alone — it passes through
to identity generation untouched (§1), and the interpreter's output covers
only the fields that are prose in the character file anyway (texture prompts)
plus the semantic hair block. The edit path is the same protocol: given an
existing character and "give her a ponytail," the interpreter returns an
updated hair block and leaves everything else alone.

- **Default backend: a small local model, shipped as a registry component**
  like every other model — a versioned, hash-pinned artifact containing
  quantized weights, the system prompt, few-shot examples, and a decoding
  grammar derived from the hair block's JSON Schema. Inference is
  grammar-constrained decoding at temperature 0, with schema validation as a
  repair loop. The all-in-one install stays intact: no external LLM server.
- **The model choice is config, not code.** The backend accepts a registry
  component id or a local weights path; nothing in the codebase names a
  model. Swapping candidates must take under a minute, and the step is
  invokable standalone — `character-factory interpret "<text>"` emits the
  decomposition JSON without running any generation — so candidate models
  can be compared side by side on real prompts.
- **Optional backend: any OpenAI-compatible endpoint**, selected by one
  config field, for people running a local inference server or wanting a
  larger model. It must never complicate the default path.
- **Degraded mode: a rules-based fallback** (slot-prompt splitting plus a
  conservative default hair block), used in offline CI and when no
  interpreter backend is available. It is documented as degraded, not
  offered as a quality tier.
- **VRAM discipline:** the interpreter never coexists in VRAM with the base
  model — it runs and releases before the diffusion stack loads, or runs
  CPU-only. The VRAM floor in §6.1 is unchanged by this component.

### 2.3 The HTTP server

`character-factory serve` starts a local FastAPI app: a library front-end
with a job queue, not a platform. Design constraints: single GPU, so a
single-flight generation queue; all state on disk in per-character
directories (the character file is the database); progressive results —
the scene GLB is rebuilt and atomically replaced as each texture lands, so a
polling viewer watches the character get dressed. The bundled browser view is
built as a plain API client of this contract — no server-side rendering, no
local-only endpoints — so the same interface can front a hosted deployment
rather than being a dead end.

Endpoint sketch (v0):

```
POST   /v0/characters                  {prompt, seed?}     → 202 {id}   enqueue full make
GET    /v0/characters                  list (id, name, status, thumbnail)
GET    /v0/characters/{id}             status + character document + revision
GET    /v0/characters/{id}/scene.glb   current build artifact
GET    /v0/characters/{id}/assets/{slot}.png
POST   /v0/characters/{id}/rebuild     {from: "bake"|"assemble", overrides?}
DELETE /v0/characters/{id}
POST   /v0/validate                    character document in body → validation report
GET    /v0/components                  registry view: installed + available components
GET    /v0/health                      GPU present, VRAM, component cache state
```

Uploads of edited character files are just `POST /v0/characters` with a
`character` body instead of a `prompt` — the server builds whatever valid
character it is given.

Per the parity principle (§2), the entire `/v0` surface above is the common
local/hosted contract — no endpoint is a local-only convenience — and it is
designed as if a conformance suite will one day run against both. The auth
story is reserved at the contract level now: clients send
`Authorization: Bearer <token>`, which the v0 local server accepts and
ignores, so no client changes shape when auth becomes real. The one expected
divergence is capacity semantics (`202` queue depth and `/v0/health`
contents), which are declared server-specific, not contractual.

### 2.4 The MCP server

`character-factory mcp` (stdio transport; HTTP optional) exposes:

- tools — `make_character`, `create_character`, `bake_character`,
  `assemble_character`, `get_character`, `list_characters`,
  `validate_character`, `list_components`;
- resources — the character JSON Schema, SPEC.md itself, and each existing
  character's document, so an agent can read the format it is writing
  against without leaving the session.

The intended consumer is a coding agent building something *with* characters
(a game, a scene, a test fixture) that needs "give me a 3D person matching
this description" as a callable primitive. Tool inputs and outputs are
character documents, never opaque handles, so agent workflows compose with
hand editing and version control.

MCP parity mirrors HTTP parity: same tool names, same input/output shapes,
local and hosted; the only difference is transport and endpoint
configuration. An agent workflow built against the local MCP server must
work against a hosted one by editing its MCP config, nothing else.

## 3. Assembly and the rigged export

Assembly (`character_factory.assembly`) is the deterministic half of the
system and deliberately boring:

1. **Rig evaluation.** The MHR TorchScript rig (Apache 2.0, fetched as a
   registry component, ~700 MB, CPU-capable) maps identity + rest pose +
   resting expression to 18,439 vertices and a 127-joint skeleton.
2. **UV compositing.** The garment image's coverage is recovered by a
   calibrated luminance key against its black background (dark garments are
   protected by a value floor in the generator), cleaned against a small
   library of coverage templates, feathered, and composited over the skin
   image; the head region is masked from garment coverage. Result: one
   albedo atlas on one body mesh.
3. **Eyes.** A small patch of faces over each eye socket is removed; a
   stock eyeball mesh (permissively licensed, bundled as a registry asset)
   is placed by a similarity fit of its lid margin to the socket rim, and
   receives the generated eye albedo. Procedural lash cards, caruncle
   patches, and a darkened socket backing complete the region.
4. **Hair.** The hair provider's mesh (§5) is attached rigidly to the head
   joint.
5. **Skinned glTF export.** The exporter emits the full 127-joint node
   hierarchy, inverse bind matrices, and per-vertex skinning weights read
   from the rig (pruned to 4 influences per vertex and renormalized), with
   the body mesh skinned and eyes/lashes/hair bound to their carrier
   joints. Where UV seams force vertex duplication, skinning weights are
   carried through the duplication. Materials are metallic-roughness PBR
   with fixed v0 constants; hair additionally uses the glTF anisotropy
   extension with a standard-PBR fallback.

This is the component that makes the output "a character, not a statue,"
and it runs everywhere — including machines that cannot generate.

## 4. Components and weights

No weights live in this repository or in the Python package. Every model and
static asset is a **component**: a versioned, hash-pinned artifact fetched
from the `character-factory` Hugging Face organization on first use and
cached locally (`~/.cache/character-factory/`, override via
`CHARACTER_FACTORY_HOME`).

### 4.1 v0 components

| Component | Contents | Approx. size |
| --- | --- | --- |
| `interpreter` | Quantized small language model + system prompt + few-shot examples + decoding grammar (§2.2; the specific model is being selected through this pipeline and arrives as registry data) | ~1–3 GB |
| `identity` | Text → body/face parameter heads + normalization stats | ~10 MB |
| `skin` | Texture adapter for the body atlas (skin albedo) | ~90 MB |
| `eyes` | Texture adapter for the eyeball layout | ~90 MB |
| `garments` | Texture adapter for garment-over-black atlas images | ~90 MB |
| `body-rig` | Pinned MHR TorchScript rig + topology metadata (Apache 2.0, mirrored with attribution) | ~700 MB |
| `assembly-assets` | Eyeball/lash meshes and textures, UV occupancy templates, atlas metadata | ~20 MB |
| *base model* | FLUX.2 Klein 4B (transformer + text encoder + VAE), fetched from its upstream repository, shared by `identity` (text encoder) and all texture adapters | ~16 GB |

First-run download for full generation ≈ 18–20 GB (the final figure depends
on the interpreter model choice). Assembly-only use (no generation) needs
only `body-rig` + `assembly-assets` ≈ 720 MB.

The base model is fetched from its upstream repository (it is Apache 2.0),
**pinned to an exact upstream revision hash in the registry entry**, so an
upstream change can never silently alter output; mirroring it into the
organization is the documented contingency if upstream availability ever
becomes a problem, not the default.

### 4.2 The registry

The registry is a signed-by-hash JSON index, itself versioned and fetched
like a component (with a snapshot vendored into each package release as an
offline fallback). Each entry records:

```json
{
  "name": "skin",
  "version": "0.1.0",
  "kind": "texture-adapter",
  "slot": "skin",
  "requires": { "base_model": "flux2-klein-4b", "schema": ">=0.1 <1.0" },
  "artifacts": [ { "path": "adapter.safetensors", "sha256": "…", "bytes": 92000000 } ],
  "inference": { "prompt_template": "…", "steps": 20, "guidance": 4.0, "resolution": 1024 },
  "source": { "hf_repo": "character-factory/skin", "revision": "…" }
}
```

Design consequences, in decreasing order of importance:

- **New capability is data, not code.** A footwear adapter, an updated skin
  component, or the mouth-interior geometry pack arrive as new registry
  entries (a new slot name, a version bump, a new component kind). The
  planned monthly model cadence never requires users to upgrade the package
  unless the schema itself grows.
- **Prompt conditioning is component metadata.** Each adapter's trigger
  phrasing, sampler defaults, and resolution live in its registry entry, so
  an updated component with different conditioning is still just a version
  bump.
- **Character files pin versions; the registry resolves names.** `create`
  resolves each slot to the newest compatible component and records exact
  versions in the character's provenance; `bake` honors pinned versions and
  can be told to upgrade (`rebuild {from: "bake"}` after an update).
- **Integrity is non-optional.** Every artifact is SHA-256 verified after
  fetch and before load. A hash mismatch is a hard error, never a warning.

## 5. The hair boundary

Hair is synthesized by a procedural engine that ships **vendored inside this
package**, under the repository's Apache 2.0 license. Even so, the
architecture is built against an interface, not the engine — extraction to a
standalone dependency remains a post-launch option, and alternative providers
remain possible:

```python
class HairProvider(Protocol):
    def synthesize(
        self,
        intent: dict,          # the character's hair block, SPEC.md §6 — the full contract
        head: HeadGeometry,    # body mesh + forward axis + eye level, body frame, cm
    ) -> HairResult:           # textured triangle mesh + PBR material + provider version
        ...
```

The contract is exactly the character format's hair block on the way in and
a textured mesh in the body's frame on the way out. The engine needs no rig
internals — it fits any roughly-human head mesh — which is what makes the
boundary this narrow. Its output is deterministic for a fixed (intent, head,
engine version) triple, which assembly's determinism guarantee inherits.

Because the boundary is this narrow, the vendoring decision is packaging,
not architecture: the engine could equally ship as a pip dependency or an
entry-point plugin without changing anything above this line. Third-party
providers plug in behind the same protocol; a missing provider degrades to
`hair: null` characters and a clear "hair provider not installed" error.

## 6. Install story

```
pip install character-factory
character-factory make "a lean marathon runner with cropped dark hair" -o runner/
# → runner/character.char.json, runner/scene.glb  (first run downloads components)
```

- **Python:** 3.11 floor, tested against 3.12.
- **Linux + NVIDIA CUDA** is the first-class platform; everything works.
- **Windows** is supported via WSL2 and is the same code path as Linux.
  Native Windows is untested and unclaimed in v0.
- **macOS** installs and runs the CLI, schema tools, server (library mode),
  and assembly; **generation is unsupported** (no CUDA; MPS is not a v0
  target).

### 6.1 VRAM floor

Proposed from actual component sizes, to be confirmed by measurement before
launch: with the base model's transformer and text encoder loaded 8-bit
quantized (the default), texture generation at 1024×1024 fits in
**12 GB VRAM (floor)**; **16 GB is recommended**, and 24 GB removes any need
for quantization or careful load ordering. The pipeline loads the base
model once and swaps ~90 MB adapters between slots, and identity generation
reuses the same text encoder, so the floor is set by the base model, not by
the number of components. The interpreter does not raise the floor: it runs
first and releases its VRAM before the diffusion stack loads (or runs
CPU-only) — the two are never resident together. Indicative timing on an RTX 3090-class card:
well under a second for text → body parameters, tens of seconds per texture
slot, a few minutes end-to-end for a first character (plus one-time
download and model load).

On an unsuitable machine the pipeline degrades honestly: `create`, `bake`,
and `make` probe device and free VRAM up front and exit with a message
stating the floor and pointing at what *does* work (below) — no partial
generation, no silent CPU fallback that would take hours.

### 6.2 What succeeds on a MacBook Air

The final install test target is a MacBook Air, defined as exactly this
passing:

1. `pip install character-factory` succeeds (no CUDA extras resolved).
2. `character-factory validate examples/characters/*.char.json` passes.
3. `character-factory assemble examples/characters/runner.char.json -o out/`
   downloads `body-rig` + `assembly-assets` (~720 MB), assembles against the
   repo's committed example assets, and writes a valid rigged `scene.glb`
   that opens in a standard glTF viewer.
4. `character-factory serve` starts in library mode, lists the example
   characters, and serves their GLBs to a browser.
5. `character-factory make "…"` fails fast with the documented
   generation-unsupported message.

Steps 2–4 are the demonstration that the character format and assembler are
real products independent of the generation stack.

## 7. Test surface

Four families, in decreasing order of how much of the product they protect:

### 7.1 Schema round-trips

- load → save → load is the identity function; canonical form and content ID
  are stable across round-trips and implementations of the serializer.
- Every committed example validates; a corpus of deliberately broken
  documents (wrong array lengths, unknown enum values, NaN, `color.rgb`
  without `custom`, unknown `topology`) fails with the documented error for
  each.
- Forward-compatibility: a synthetic "0.x+1" document with an unknown
  optional field passes default validation with a warning and fails strict
  validation; an unknown `topology` value is a hard error in both modes.

### 7.2 Assembler determinism

- Same character file + same pinned assets ⇒ byte-identical GLB across two
  runs, and across CPU/GPU rig evaluation.
- Rig integrity: 127 joints exported; every vertex's weights sum to 1 within
  tolerance and reference ≤ 4 joints; every UV-seam-duplicated vertex
  carries its original's weights; topology counts and the rig checksum match
  the registry pin.
- Golden-file structural checks on the example characters: mesh/primitive
  inventory, material constants, texture bindings, extension usage — not
  pixel screenshots.

### 7.3 Bake correctness

Diffusion outputs are reproducible only up to GPU kernel nondeterminism, so
bake tests assert structure, not bytes:

- every slot produces an image at the component's declared resolution, and
  recorded asset hashes match the files written;
- garment images key correctly: coverage recovered at the calibrated cutoff
  is stable under the cutoff's documented tolerance band, never bleeds into
  the head region, and an intentionally dark test garment survives the key;
- seeds are respected: same recipe twice on the same machine ⇒ identical
  hashes for the deterministic stages and near-identical (perceptual-diff
  bounded) images for diffusion stages;
- a reference-GPU golden run (hash-exact) runs in CI on self-hosted
  hardware, and is advisory elsewhere.

### 7.4 Mouth-variant contract cases (reserved)

Written now, gated behind the future topology variant, so the variant ships
against a pre-agreed contract:

- the removal patch is validated against the pinned rig topology: exact
  expected face count, closed boundary of the expected edge count, identical
  across identities and expressions;
- the interior geometry's entrance ring coincides with the lip-derived
  boundary within a stated tolerance at every pose — no visible seam;
- identity may *place* interior anatomy (a similarity fit, computed once
  per character, reflection forbidden); expression may move the lower
  anatomy only through the rig's rigid jaw transform; upper anatomy is
  bit-identical under expression-only changes;
- a clearance sweep over the supported pose set proves no interior geometry
  protrudes through the face surface, and the supported pose set is an
  explicit list that tests prevent from widening silently.

## 8. Repository layout

```
character-factory/
├── README.md                  # demo video embed at top; install + first
│                              # command immediately after; then the 60-second
│                              # tour of the character file; links to SPEC.md
├── SPEC.md                    # the character format (this repo's second product)
├── ARCHITECTURE.md            # this document
├── LICENSE                    # Apache 2.0
├── NOTICE                     # third-party attributions
├── pyproject.toml             # one package: character-factory
├── src/character_factory/
│   ├── __init__.py            # Character, create, bake, assemble, make
│   ├── schema/                # format model, validation, canonical form, JSON Schema
│   ├── registry/              # component index, fetch, cache, integrity
│   ├── interpreter/           # Interpreter protocol, local-model + HTTP backends,
│   │                          # rules fallback (local model: lazy import)
│   ├── identity/              # raw text → body parameters       (GPU, lazy import)
│   ├── textures/              # diffusion runner, adapter loading (GPU, lazy import)
│   ├── hair/                  # HairProvider protocol + the vendored procedural engine
│   ├── assembly/              # rig eval, UV compositing, eyes, skinned glTF export
│   ├── server/                # FastAPI app, job queue, per-character state dirs
│   ├── mcp/                   # MCP tools + resources
│   └── cli.py                 # make / create / bake / assemble / interpret /
│                              # validate / serve / mcp / components
├── examples/
│   ├── characters/            # committed .char.json files
│   └── assets/                # committed baked textures for the no-GPU path
├── tests/                     # §7, mirroring the module layout
└── docs/                      # deeper guides as they are earned
```

The README's job is a stranger's first five minutes: see it move (video),
install it, run one command, then discover that the interesting artifact is
the small JSON file next to the GLB.
