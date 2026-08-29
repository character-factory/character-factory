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
        │                        expression values on the MHR parametric body — SAMPLED
        │                        from a generative model; the create seed picks one
        │                        identity from the description's distribution)
        ▼
  texture generation            (skin, eye, garment, optional shoe: UV-space
        │                        images from a FLUX.2 Klein 4B base model with
        │                        per-slot adapters; seeded diffusion, 1024² albedo)
        ▼
  hair synthesis                (semantic hair JSON → textured hair mesh, procedural,
        │                        deterministic, no diffusion)
        ▼
  assembly                      (evaluate the rig, composite garments over skin in UV
        │                        space, place eyes and the mouth interior, attach hair,
        ▼                        emit a skinned glTF with the full 127-joint skeleton)
  character.char.json  +  scene.glb
```

The intermediate product — and the thing this project treats as the character
itself — is the **character file** (SPEC.md): a few kilobytes of JSON holding
the body parameters, per-slot texture recipes, semantic hair description, and
provenance. The GLB is a build artifact; the character file is the source.

Two properties are load-bearing:

- **Identity is generative, and the document records the draw.** The
  identity model consumes the raw user description — never an interpreter
  rewrite of it — and SAMPLES one identity from that description's
  distribution: a single joint model (a semantic-center regressor plus a
  conditional rectified flow over one body+proportions+face state), seeded
  by the create seed, with noise drawn on the CPU so a (description, seed,
  component version) triple reproduces the same body on every device. The
  drawn values land in the character file, so the file remains an honest,
  fully-determined recipe — stochasticity exists at create time only.
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

- **No normal maps or other non-albedo maps** for skin and garments in v0. v0
  surfaces are albedo plus fixed material constants. Generated normal/detail
  maps are planned for v0.2; model components ship on a roughly monthly
  cadence after launch (§4), so this does not wait on a code release. (Hair
  is the exception: the hair synthesizer emits its own albedo and normal
  textures.)
- **Facial animation is baseline data, not an optional tier.** Every
  character carries 72 exact expression morph targets and a jaw that
  animates through `c_jaw` (SPEC.md §4.2, §7.4 here) — but the system
  authors no facial performances: playing expressions, lip-sync, and
  blends is the consumer's job, within the measured limitation table the
  manifest ships. Creation fails if the complete mouth-interior artifact
  cannot be delivered; there is no body-only fallback.
- **Footwear is below-ankle styles only.** Footwear ships at launch as the
  optional `shoe` texture slot on the foot regions of the body atlas —
  closed, below-ankle shoes can be painted; boots above the ankle, sandals,
  and open straps would require geometry the pipeline does not build.
  `make-shoe` *declares* its supported style vocabulary in its registry
  entry, and the interpreter clamps shoe prompts to what the installed
  component declares (§4.2) — so capability growth (boots, say) arrives as a
  `make-shoe` version bump widening its vocabulary, never as a sibling
  component or a code change.
- **Garment geometry is body-following shells, not simulated clothing.**
  Garments are generated in UV space, and each character's baked garment
  becomes its own skinned mesh — lifted off the body, with a cloth
  material distinct from skin (§7). A character whose garment fails the
  extraction certification falls back to the painted composite, recorded
  in the manifest. The shells follow the body silhouette; loose clothing,
  skirts, and anything that departs from the silhouette are out of v0
  scope. Footwear ships the same way: the baked shoe overlay becomes its
  own shell over the feet, cloth-class material included.
- **No identity resampling of an existing character.** Identity is sampled
  at create time (a different seed gives a different take on the same
  description), but there is no operation that redraws the body of a
  character file that already exists — the drawn parameters in the file are
  the character. Want a variant? Create again with another seed.
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

**Skeletal proportions have a fixed precedence stack.** Proportions are
identity-class output: the identity component writes them from the raw
prompt on *every* create, regardless of which interpreter backend ran.
An interpreter backend may additionally emit explicit proportion fields
(only on clear signal in the description); when it does, those values
override the identity component's per key. In short: **head writes →
interpreter overrides per key.**

- **Default backend: a small local model, shipped as a registry component**
  like every other model — a versioned, hash-pinned artifact containing
  quantized weights, the system prompt, few-shot examples, and a decoding
  grammar derived from the hair block's JSON Schema. Inference is
  grammar-constrained decoding at temperature 0, with schema validation as a
  repair loop. The runtime is **in-process and Python-native**: installed by
  pip as part of this package's dependencies, no external daemon to start,
  no runtime with account linkage or telemetry. It rides the same
  torch/transformers stack the generation extra already requires (with a
  pure-Python constrained-decoding layer), so it adds no build step and no
  new platform to the matrix — interpretation runs where generation runs.
  The all-in-one install stays intact: no external LLM server.
- **Load order is a rule, not a habit:** the interpreter loads, runs, and
  **releases its memory before the base image model loads** (or runs
  CPU-only). Keeping both resident is easy to do by accident with an
  in-process runtime and would break the VRAM floor (§6.1) — the pipeline
  enforces the sequencing, and a test asserts the interpreter's weights are
  released before the diffusion stack initializes.
- **`character-factory interpret "<text>"` is also the benchmark harness**
  for choosing the default model: it runs exactly the production path
  (safetensors weights via the in-process runtime with grammar-constrained
  decoding) and reports, alongside the decomposition JSON, per-invocation
  wall time and peak memory — so candidate models are compared on the same
  numbers users will experience.
- **Component vocabularies bound the interpreter's output.** Registry
  components may declare supported-vocabulary constraints (§4.2) — the first
  customer is the launch `make-shoe` component, which supports below-ankle
  styles only. The interpreter reads the installed components' declarations
  and clamps its slot prompts to them, so "knee-high boots" degrades to the
  nearest supported request rather than silently conditioning a component
  outside its competence. The clamp is advisory: it constrains prompt
  authoring, and nothing downstream can verify compliance — the diffusion
  component has no notion of its own vocabulary.
- **The model choice is config, not code.** The backend accepts a registry
  component id or a local weights path; nothing in the codebase names a
  model. Swapping candidates must take under a minute, and the step is
  invokable standalone — `character-factory interpret "<text>"` emits the
  decomposition JSON without running any generation — so candidate models
  can be compared side by side on real prompts.
- **Grammar-derivation note for the implementer:** the constraint layer's
  JSON-Schema support has a known limitation — non-string `const` values
  crash its enum path — so the grammar derivation expresses the hair
  block's `schema_version: 1` as a closed integer range (and should prefer
  ranges over non-string consts generally). The exercised derivation lives
  in the grammar test module.
- **Optional backend: any OpenAI-compatible endpoint**, selected by one
  config field, for people running a local inference server or wanting a
  larger model. Endpoint output uses strict JSON-Schema response formatting
  where the endpoint supports it and still passes through the same validator.
  Empty or length-truncated output receives one bounded retry against the
  same backend; malformed JSON and schema-invalid documents fail immediately.
  It must never complicate the default path.
- **No degraded mode.** There is deliberately no non-model interpretation
  backend: an unconfigured installation or a failed model request is a
  structured, named error, never a silent quality downgrade. Records
  expose the requested and actual backend aliases and warnings.
- **Endpoint diagnostics are private.** If an operator configures
  `CHARACTER_FACTORY_INTERPRETER_AUDIT_LOG`, the server writes a mode-0600
  JSONL stream containing raw prompts, raw responses, HTTP status, response
  size, finish reason, latency, endpoint request id, reported backend version,
  usage, attempt number, and an opaque trace id. Public job failures include
  only that trace id and a safe classification such as `empty_response`,
  `truncated_response`, `invalid_json`, or `schema_invalid`. Invalid endpoint
  output uses `interpreter_invalid_output`; transport and HTTP availability
  failures use `interpreter_unavailable`.
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
POST   /v0/characters                  {prompt, interpreter?, allow_fallback?, turbo?}
                                       → 202 Job; Location + Retry-After
                                       Idempotency-Key makes transport retries
                                       safe; unkeyed requests are new work
                                       interpreter: backend alias from /v0/interpreters —
                                       per-request model selection (hosted tiers use the same field)
GET    /v0/interpreters                selectable interpreter backends: [{alias, kind}]
GET    /v0/jobs                        lightweight job list
GET    /v0/jobs/{id}                   stage, progress, heartbeat, outcome/error
DELETE /v0/jobs/{id}                   cancel queued work or request cancellation
POST   /v0/jobs/{id}/retry             explicit new attempt after failure/cancel
GET    /v0/characters                  completed records array, newest first
GET    /v0/characters/{id}             character + separate artifact/latest-job state
GET    /v0/characters/{id}/scene.glb   current build artifact
GET    /v0/characters/{id}/assets/{slot}.png
GET    /v0/characters/{id}/manifest.json   the scene's embedded export manifest,
                                           served standalone (same bytes as the
                                           GLB's asset extras)
PUT    /v0/characters/{id}/assets/{slot}   image body → replace one baked asset;
                                           the stored hash pin is updated and the
                                           scene rebuilds from the assemble stage
POST   /v0/characters/{id}/rebuild     {from: "bake"|"assemble", overrides?}
                                       → 202 Job; always explicit new work
                                       unless replaying the same Idempotency-Key
DELETE /v0/characters/{id}
POST   /v0/validate                    character document in body → validation report
GET    /v0/components                  registry view: installed + available components
GET    /v0/health                      GPU present, VRAM, component cache state
```

The character list is a bare JSON array of completed records. Browser use is
same-origin only. The server does not provide CORS headers or support
separately hosted browser clients. The bundled docs are self-contained, and
the gallery retains a dependency-free create/list/download view if its 3D
viewer modules are unavailable offline.

Uploads of edited character files are just `POST /v0/characters` with a
`character` body instead of a `prompt` — the server builds whatever valid
character it is given.

Per the parity principle (§2), the entire `/v0` surface above is the common
local/hosted contract — no endpoint is a local-only convenience — and it is
designed as if a conformance suite will one day run against both. The auth
story is reserved at the contract level now: clients send
`Authorization: Bearer <token>`, which the v0 local server accepts and
ignores, so no client changes shape when auth becomes real. The one expected
divergence is capacity (`queue_position` and device fields in `/v0/health`),
which is declared server-specific, not contractual. Job states, terminal
enums, idempotency, cancellation, and retry semantics are common contract.

Creation and rebuild submission use one retry contract. A caller that supplies
`Idempotency-Key` receives the original job when it repeats the same operation,
target, and payload. Reusing that key for different work is a `409` conflict.
Omitting the header always submits a new job; rebuilds are never inferred from
repeating a create body. The key does not alter character identity or artifact
hashing.

Interpreter failure handling is backend-neutral. A client should inspect the
job's structured `error.code`, `error.classification`, `error.retryable`, and
opaque `error.trace_id`; it must not infer backend behavior from the message.
It may retry the original failed job through `POST /v0/jobs/{id}/retry`.
Changing fallback policy is a different request: submit it separately with a
new `Idempotency-Key` and `allow_fallback: true`. Retrying never mutates the
original idempotent request, and fallback is never enabled silently.

The local process binds to loopback by default. Binding it to all interfaces
is intended only for a trusted local network or private overlay network, where
the host firewall and network access rules are the security boundary. Because
the local server does not authenticate requests, it must not be exposed to the
public internet or an untrusted network. CORS would not change that boundary:
it constrains browser scripts, not agents, native clients, or other machines.

### 2.4 The MCP server

`character-factory mcp` (stdio transport; HTTP optional) exposes:

- tools — `create_character`, `assemble_character`, `get_job`, `cancel_job`,
  `retry_job`, `get_character`, `list_characters`, `validate_character`,
  `store_character`, `list_components`;
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
   image; the head region is masked from garment coverage. The `shoe`
   image, when present, is a single-shoe canvas: the component's foot
   chart (per-texcoord canvas coordinates shipped with the model, because
   the canvas layout is part of each version's output contract)
   bakes it onto the foot islands of both feet — the second foot through
   the chart's horizontal mirror — with style-aware occupancy (open styles
   keep only their straps; the shaft is cut to the style's declared
   height), and the resulting RGBA overlay composites above the garment
   layer — the normative order is skin, then garment, then shoe
   (SPEC.md §9). That is the *painted* composite — the fallback form.
   When slots ship as shells (the standard outcome, next step), their
   layers are left out of the body albedo entirely: with garment and
   shoe both shipped as geometry, the body mesh carries pure skin, and
   the narrow skin band retained at each shell's coverage boundary shows
   skin — never painted-on garment — at the rim.

   **Garment shells.** The baked garment texture then becomes geometry
   (standard assembly behavior, never recorded in the character
   document): its own keyed
   coverage — never a canonical or per-style cut — is marched through the
   body triangles, reconstructed on the character's identity, lifted and
   faired under clamps, closed into a watertight solid, and exported as a
   skinned mesh riding the body's own skin. The body faces under the
   coverage are omitted — deleted, not hidden — so the shell is the only
   surface where the shell is: no doubled geometry, no z-fighting, and
   body-through-cloth clipping is impossible where cloth covers, because
   there is no body there. The one intentional overlap is a narrow skin
   band retained at the coverage boundary (the erosion rings), tucked
   under the shell's rim so skin runs continuously under cloth from any
   angle; the band width is the technique's tuning knob. The shoe shell
   extracts identically from the baked overlay's own alpha (authoritative
   occupancy — no luminance keying), confined to the atlas's feet
   region, and carries the same cloth-class material. Extraction is a
   pure function of the published baked asset bytes; certification is a
   fail-closed ladder of structural gates (alpha quality, seam-crack
   detection, topology, closed-solid and weight audits) — a shell that
   builds as a valid solid ships; posing behavior is the consumer's to
   see, not a reason to withhold geometry. A slot that fails a
   structural gate keeps the painted composite,
   and the manifest's `garments` block records the shipped `render_mode`
   per slot (with the rejection reason when an extraction was attempted),
   so consumers never sniff. Validators validate; nothing repairs a
   nonconforming mask. The v0.1 boundary treatment is the feathered
   soft-threshold cut (crossings refined against the texture's own alpha)
   with the garment's boundary colors dilated outward so edge faces
   always sample cloth; a re-triangulated cut-line boundary is a planned
   post-v0.1 refinement, held to the same watertightness and
   weight-transfer invariants.
3. **Eyes.** A small patch of faces over each eye socket is removed; a
   stock eyeball mesh (permissively licensed, bundled as a registry asset)
   is placed by a similarity fit of its lid margin to the socket rim, and
   receives the generated eye albedo. Procedural lash cards, caruncle
   patches, and a darkened socket backing are planned post-v0.1 polish —
   today the eyeball alone fills the socket, and no interior geometry is
   stitched into the body mesh for the eyes.
4. **Mouth** (`"mouth-interior"` topology). The rig version's fixed mouth
   patch is removed; a posterior-lip cuff and cavity strip — built from the
   character's inner-lip curves, skinned by extending the lips' own
   influences, UV-mapped into the removed patch's own atlas region at
   measured-even density — is stitched into the body mesh; GNM-derived
   teeth, gums, and tongue are placed by identity anchors (upper on the
   skull, lower and tongue on the jaw chain). The export gains the rig's
   72 expression coefficients as exact sparse morph targets
   (`facs_00`–`facs_71`) and machine-readable jaw guidance and
   animation-limitation tables in its manifest. Interior geometry obeys
   the interior-UV contract: original vertex UVs bit-exact, no new atlas
   islands, no chart overlap, no UV inversion — all asserted by the
   permanent suite.
5. **Hair.** The hair provider's mesh (§5) is attached rigidly to the head
   joint.
5. **Skinned glTF export.** The exporter emits the full 127-joint node
   hierarchy with human-readable joint names (from the rig component's
   metadata, §4.1), inverse bind matrices, and per-vertex skinning weights
   read from the rig. The rig's native skinning uses at most 4 influences
   per vertex with weights summing to 1, so glTF's JOINTS_0/WEIGHTS_0
   carries the **weights exactly — no pruning loss**. The body mesh is
   skinned; rigid accessories are parented to their carrier joints (eyeballs
   to the eye joints, hair to the head joint). Where UV seams force vertex
   duplication, skinning weights are carried through the duplication.
   Materials are metallic-roughness PBR with fixed v0 constants —
   dielectric throughout, skin at roughness 0.5 and garment and shoe
   shells at
   0.9, so cloth responds to light differently than skin by design
   (roughness contrast is the one portable lever while surfaces are
   albedo-only); hair
   additionally uses the glTF anisotropy extension with a standard-PBR
   fallback.

### 3.1 Exporter conventions

The exporter targets game engines, not just viewers, and follows a fixed set
of conventions chosen so the artifact imports and retargets correctly:

- **One frame, one constant.** The rig's native frame is centimeters, Y-up,
  +Z-forward; glTF is meters, Y-up, +Z-forward. The exporter's entire
  conversion is a uniform scale of 0.01 — no axis flip, no handedness
  change, and mesh and skeleton are always expressed in the same frame.
- **One binding truth.** Inverse bind matrices are the inverse of each
  joint's world bind matrix, written column-major; whenever the rest pose is
  post-processed (below), the IBMs are rebuilt *afterward*, so the bound
  mesh is bit-identical before and after.
- **Rest orientations are re-authored, deliberately.** The rig's native
  per-joint rest rotations carry per-bone roll that is not mirrored between
  left and right limbs — harmless to skinning (the IBM cancels it) but
  hostile to humanoid retargeting systems, which derive hinge axes from rest
  frames. The exporter discards the source rotations and re-authors every
  joint's rest orientation from geometry under one mirror-invariant global
  convention (bone-long axis toward the mean of children; a forward-axis
  reference vector, invariant under the sagittal mirror, orthogonalized
  against it). Joint *positions* are untouched. Bones shorter than two
  millimeters inherit the parent's direction instead of deriving their
  own. This covers the near-zero foot and wrist-twist pairs and the neck's
  procedural twist helper, whose endpoint can cross its parent at valid
  body proportions; deriving an axis from that displacement would force a
  180° local rotation. With the floor, no exported joint approaches that
  half-turn singularity, and the worst left/right frame deviation remains
  ~0.02° across the entire proportion range.
- **A small knee flexion is baked into the exported rest pose** (a
  documented, versioned constant), because a near-straight knee is a
  degenerate hinge that retargeters can resolve backwards. It is applied as
  a plain skinning-space edit about a shared sagittal axis, so left and
  right stay symmetric by construction, and IBMs are rebuilt after.
- Consequence of the two points above, stated honestly: **the exported rest
  pose is deliberately not the rig's verbatim rest** — same class of caveat
  as the correctives note below.
- **Skeleton root.** Joint index 0 is the rig's world-transform node
  (`body_world` — a transform root, not a deformer). It is exported as the
  skeleton's root node and included in the skin's joint list so that glTF
  joint indices equal rig joint indices verbatim; the exporter asserts no
  vertex is weighted to it.
- **File hygiene.** A single self-contained `.glb`: textures embedded as
  PNG, every bufferView 4-byte aligned, POSITION accessors carry min/max,
  vertex attributes and indices use the proper buffer targets, joints as
  unsigned-short VEC4 with float VEC4 weights, no UV V-flip (the bake and
  glTF already agree on a top-left origin), counter-clockwise winding
  verified against outward normals rather than assumed.
- **The GLB is self-describing.** The **bone-role manifest** (JSON: the
  full engine humanoid role → joint name mapping, the explicit
  leave-unmapped set — procedural twist joints, null markers, mouth
  interior, tarsal helpers — with structured flags for the jaw (mappable,
  default-unmapped) and fingers (mapped, verify in-engine), units, axes,
  joint count, and the baked knee constant) embeds in the GLB's
  asset-level `extras` — the
  spec's mechanism for application metadata, ignored by parsers that don't
  read it — so one file is the complete engine deliverable and the
  manifest can never be separated from the mesh it describes. It is a pure
  function of rig version, exporter constants, and the character's
  skeletal proportions (stature is measured from the exported geometry;
  byte-identical across re-exports of the same character); a sidecar
  `manifest.json` exists only on request, as a projection of the same
  bytes (`GET /v0/characters/{id}/manifest.json`). The manifest carries
  its own `schema_version` and `$schema`; the machine-readable schema is
  served at the declared same-origin path and packaged with the library.
  Minor versions are additive, while a field-shape or meaning change bumps
  the major. A consumer that pins a tested contract rejects any other
  version loudly instead of inferring from field shapes. The per-slot
  `garments` block declares `render_mode: "shell" | "painted"` for each
  garment-class slot, with the shell inventory or rejection reason. The jaw
  block states its contract explicitly: rotation sign (positive about the
  local axis opens, in the file's right-handed glTF frame — handedness
  conversion at import flips it) and the two jaw compositions
  (joint-only with `full_open_degrees`, or expression playback pairing
  `facs_24` with `expression_fit_angle_degrees`), which are alternatives
  and never summed. Boundary: the manifest
  is **export metadata** — facts about the GLB as an engine deliverable.
  Character identity, textures, hair, and provenance live in the character
  document exclusively; nothing from it is ever duplicated into GLB
  `extras` — one source of truth per fact. A **baked idle clip** ships
  inside the GLB as a **Generic, native-skeleton clip that is not certified
  for Humanoid retargeting**: a few seconds of subtle breathing and weight sway, with
  the complete local TRS baked for every joint — animation channels
  replace node transforms in a conforming player, so the clip leaves
  nothing to engine defaults. Frame 0 is exactly the rest pose and the
  clip loops seamlessly. The `grounding` block measures the body ground
  plane in scene space, gives the root and left/right foot-joint offsets for
  this character's proportions, declares the idle's tested ground-drift
  tolerance, and states that there are no certified contact frames and
  runtime foot IK is recommended. A consumer never has to guess whether to
  play the embedded clip as Generic or convert it to Humanoid.

One honest documentation line, twice over: the exported rig animates as
clean linear-blend skinning — the generator's own renders additionally apply
learned pose correctives that core glTF cannot express — and the exported
rest pose re-authors joint orientations and knee flexion as described above.
Rest-pose *geometry* is exact in both cases.

This is the component that makes the output "a character, not a statue,"
and it runs everywhere — including machines that cannot generate.

## 4. Components and weights

No weights live in this repository or in the Python package. Every model and
static asset is a **component**: a versioned, hash-pinned artifact fetched
from the `character-factory` Hugging Face organization on first use and
cached locally (`~/.cache/character-factory/`, override via
`CHARACTER_FACTORY_HOME`).

**Naming.** Generator components carry `make-<artifact>` names, where the
noun is the discrete artifact the component produces: `make-figure` makes a
figure (the body parameters), `make-skin` a skin, `make-eye` one eye
texture, `make-garment` a garment layer, `make-shoe` a shoe texture,
`make-wig` a wig mesh. Static data and infrastructure keep plain functional
names (`body-rig`, `assembly-assets`, `interpreter`). Nouns are singular
throughout, matching the texture slot keys they serve.

### 4.1 v0 components

| Component | Contents | Approx. size |
| --- | --- | --- |
| `interpreter` | Quantized small language model + system prompt + few-shot examples + decoding grammar (§2.2; the specific model is being selected through this pipeline and arrives as registry data) | ~1–3 GB |
| `make-figure` | Text → body/face parameter heads + normalization stats | ~10 MB |
| `make-skin` | Texture adapter for the `skin` slot's albedo (body atlas) | ~90 MB |
| `make-eye` | Texture adapter for the `eye` slot's albedo (eyeball layout) | ~90 MB |
| `make-garment` | Texture adapter for the `garment` slot's albedo (garment-over-black atlas images) | ~90 MB |
| `make-shoe` | Texture adapter for the `shoe` slot's albedo (optional slot; declares a below-ankle style vocabulary at launch) | ~90 MB |
| `make-wig` | The default hair provider (vendored procedural engine; registry entry records versions for provenance) | in package |
| `body-rig` | Pinned MHR TorchScript rig + topology metadata + the authoritative 127-joint name table (Apache 2.0, mirrored with attribution) | ~700 MB |
| `assembly-assets` | Eyeball and mouth-interior meshes, UV occupancy templates, atlas metadata | ~20 MB |
| *base model* | FLUX.2 Klein 4B (transformer + text encoder + VAE), fetched from its upstream repository, shared by `identity` (text encoder) and all texture adapters | ~16 GB |

First-run download for full generation ≈ 18–20 GB (the final figure depends
on the interpreter model choice). Assembly-only use (no generation) needs
only `body-rig` + `assembly-assets` ≈ 720 MB.

Everything upstream is **pinned by content hash, never by a floating
"latest"**: the base model is fetched from its upstream repository (it is
Apache 2.0) at an exact revision hash recorded in the registry entry, and
the rig component pins the exact upstream release archive and each consumed
artifact by SHA-256. An upstream change can never silently alter output;
mirroring into the organization is the documented contingency if upstream
availability ever becomes a problem, not the default.

### 4.2 The registry

The registry is a signed-by-hash JSON index, itself versioned and fetched
like a component (with a snapshot vendored into each package release as an
offline fallback). Each entry records:

```json
{
  "name": "make-skin",
  "version": "0.1.0",
  "kind": "texture-adapter",
  "slot": "skin",
  "map": "albedo",
  "requires": { "base_model": "flux2-klein-4b", "schema": ">=0.1 <1.0" },
  "artifacts": [ { "path": "adapter.safetensors", "sha256": "…", "bytes": 92000000 } ],
  "inference": { "prompt_template": "…", "steps": 20, "guidance": 4.0, "resolution": 1024 },
  "constraints": { "vocabulary": { "styles": ["…"] } },
  "source": { "hf_repo": "character-factory/skin", "revision": "…" }
}
```

`constraints` is a general mechanism: a component MAY declare the vocabulary
it actually supports (named enums of styles, categories, or attributes), and
the interpreter clamps its prompts to the declarations of the components
that will run (§2.2). The launch `make-shoe` component is the first user
(below-ankle styles only); any component whose competence is narrower than
its slot's plain-language name should declare, so that capability growth is
a registry edit, not a code release. Texture entries also carry `map`: which
named map of their slot they produce (every current entry: `albedo`), so a
future secondary-map component for an existing slot is pure data.

Design consequences, in decreasing order of importance:

- **New capability is data, not code.** An updated skin component, a
  secondary-map component, or the mouth-interior geometry pack arrive as new
  registry entries (a version bump, a new `map` value, grown assets). The
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
- **Install size:** the base install (schema tools, registry, assembly,
  server — no generation extras) measures **about 1.1 GB**, most of it
  the CPU torch wheel the assembler's rig evaluation needs. Generation
  extras and model components download on top of that.
- **Linux + NVIDIA CUDA** is the first-class platform; everything works.
- **Windows** is supported via WSL2 and is the same code path as Linux.
  Native Windows is untested and unclaimed in v0.
- **macOS** installs and runs the CLI, schema tools, server (library mode),
  and assembly; **generation is unsupported** (no CUDA; MPS is not a v0
  target).

### 6.1 VRAM floor

Measured on a 24 GB card, full four-slot bake at 1024×1024, same
character and seeds per mode:

- **Full precision (the default): 17.4 GB allocated / 20.3 GB reserved**,
  about 132 s of diffusion per character — a **24 GB card** runs it
  without configuration.
- **`nf4` weight quantization** (`textures.quantization` in the local
  config; transformer and text encoder quantize, the VAE stays full
  precision) brings the same bake to **8.9 GB allocated / 9.7 GB
  reserved** with the expandable-segments CUDA allocator, at roughly
  twice the bake time (~265 s). Every GPU stage is strictly serialized
  and releases its memory before the next loads, and no other stage
  peaks higher, so **the complete pipeline — interpretation, identity,
  all texture slots, assembly — runs on a 12 GB card** under `nf4`.
  8 GB is not supported.

The pipeline loads the base model once and swaps ~90 MB adapters between
slots, and identity generation reuses the same text encoder, so the floor
is set by the base model, not by the number of components. On
Ada-generation and newer NVIDIA hardware, fp8 weight formats should bring
the full-precision footprint down further; that is anticipated but
unmeasured, and no number is claimed for it in v0. The
interpreter does not raise the floor: it runs first and releases its VRAM
before the diffusion stack loads (or runs CPU-only) — the two are never
resident together.

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
- **Re-parse acceptance test:** parse the exported `.glb` back from disk,
  walk the node hierarchy to each joint's global matrix, apply the inverse
  bind matrices, skin the rest mesh with the exported weights, and compare
  to the exported positions — the target is exact (≈ 0 mm), plus upright
  orientation and mesh/skeleton co-location. The validator trusts only the
  artifact, never in-memory state.
- Rig integrity: 127 joints exported with the golden name table; parent
  indices are topologically ordered (`parents[i] < i`); joint order matches
  the rig bundle's own buffers; index 0 is the world-transform root and
  carries no skin weights; every vertex's weights sum to 1 and reference
  ≤ 4 joints (exact — the rig's native maximum); every UV-seam-duplicated
  vertex carries its original's weights; topology counts and the rig
  checksum match the registry pin.
- **Mirror-consistency test:** for every left/right joint pair, the left
  rest frame equals the right one reflected across the sagittal plane
  (reflect, then restore handedness); the worst angular deviation is
  reported and bounded.
- Rest-pose conventions: the baked knee-flexion constant matches its
  documented version; IBMs were rebuilt after rest edits (implied and
  verified by the re-parse test above).
- The embedded bone-role manifest is present, complete, consistent with
  the exported joint set, and byte-identical across re-exports; the baked
  idle clip fully drives every joint's TRS, and — substituted for the node
  transforms, exactly as a conforming engine plays it — reproduces the
  rest skin through the hierarchy and IBMs at t=0 within numerical
  tolerance (engine-free, catching bakes that only work when a forgiving
  viewer reconciles them with node state), while over the clip it must
  actually move: some channels vary in time and the peak mesh deviation is
  non-zero yet bounded, so a fully-driven statue fails exactly like an
  explosion does.
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

### 7.4 Mouth-variant contract cases

Written before the variant existed and now implemented as written — the
`mouth-interior` topology ships against this pre-agreed contract.
Mechanically the variant is a `body-rig` version bump plus grown
`assembly-assets` — interior meshes placed by assembly like the eyeballs,
no new texture slot or component kind:

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
  explicit list that tests prevent from widening silently;
- synthetic stress-envelope identities at the mechanism's tested
  mouth-width bounds run the same construction and clearance checks, so
  the guarantee covers the identity space, not the library sample;
- interior geometry stitched into the skinned body carries UVs under the
  interior-UV contract: original vertex UVs bit-exact before and after
  construction, new UVs inside an already-appropriate existing region
  (no new islands), no chart overlap, no UV winding inversion, and
  per-face texel density measured against stated bounds.

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
│   ├── interpreter/           # Interpreter protocol, local-model + HTTP
│   │                          # backends (local model: lazy import)
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
