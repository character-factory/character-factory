# Architecture

How Character Factory is put together: the pipeline, the package and its entry points, assembly and export, components, install, and tests. The character format itself is specified in [SPEC.md](SPEC.md). Licensed under Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

1. [Pipeline](#1-pipeline)
2. [One package, four doors](#2-one-package-four-doors) — library API, interpreter, HTTP server, MCP server
3. [Assembly and export](#3-assembly-and-export) — exporter conventions, manifest and idle clip, compression
4. [Components and the registry](#4-components-and-the-registry)
5. [Hair](#5-hair)
6. [Install and hardware](#6-install-and-hardware)
7. [Tests](#7-tests)
8. [Repository layout](#8-repository-layout)

## 1. Pipeline

```
"a lean marathon runner with cropped dark hair and green eyes, teal running vest"
        │
        ▼
  interpretation      a language model writes every component's prompt in that
        │             component's format: per-slot texture prompts, the figure
        │             prompt, and the semantic hair block
        ▼
  identity            figure prompt → 45 MHR identity coefficients + skeletal
        │             proportions, sampled from a generative model (the create
        │             seed picks the draw)
        ▼
  textures            skin, eye, garment, optional shoe: 1024² UV-space albedo
        │             images from FLUX.2 Klein 4B with a per-slot adapter
        ▼
  hair                semantic hair block → textured hair mesh (procedural, CPU)
        │
        ▼
  assembly            rig evaluation, garment/shoe shells, eyes, mouth interior,
        │             hair, skinned glTF export
        ▼
  character.char.json  +  scene.glb
```

The character file is the product; the GLB is built from it. The file holds body parameters, per-slot texture recipes (prompt + seed + component version), the hair block, and provenance — a few kilobytes of JSON that can be diffed, edited, and regenerated.

### Determinism

**Prompt → character file is not deterministic.** Interpretation is a language-model step, and identity is a sample: the interpreter's figure prompt conditions a rectified-flow model, seeded by the create seed with noise drawn on the CPU, so a (figure prompt, seed, component version) triple reproduces the same body on any device. The drawn values and the figure prompt are both written to the file.

**Character file → GLB is byte-identical** under pinned components. Texture regeneration is reproducible up to GPU kernel nondeterminism; exact bytes are pinned through `assets` hashes.

### 1.1 Scope of v0

- Surfaces are albedo plus fixed material constants. No generated normal or other secondary maps for skin and garments (hair emits its own albedo and normal).
- Every character carries 72 expression morph targets (`facs_00`–`facs_71`) and a jaw joint (`c_jaw`). No facial performances are authored; playback is the consumer's.
- Garments and shoes are single-layer shells extracted from the garment texture; loose or off-silhouette clothing is out of scope.
- No identity resampling of an existing file, and no animation authoring beyond the baked idle clip.

## 2. One package, four doors

```
  ┌──────────────────────────────────────────────┐
  │  MCP server        (character_factory.mcp)   │   coding agents
  ├──────────────────────────────────────────────┤
  │  HTTP server       (character_factory.server)│   apps, UIs, Unity
  ├──────────────────────────────────────────────┤
  │  CLI + library API (character_factory)       │   shell, Python
  ├──────────────────────────────────────────────┤
  │  schema · registry · interpreter · identity  │
  │  textures · hair · assembly                  │
  └──────────────────────────────────────────────┘
```

The servers wrap the library; they add no logic of their own. The `/v0` HTTP contract and the MCP tool surface accept a bearer token and ignore it locally.

### 2.1 Library API

```python
from character_factory import Character, create, assemble, make
from character_factory.textures import bake

character = create("a lean marathon runner …", seed=41000)
    # → Character: interpretation + identity. Fills texture recipes and
    #   the hair block; runs no diffusion. GPU.

result = bake(character, "runner/assets", turbo=False)
    # → BakeResult: one image per slot (GPU), asset hashes recorded on
    #   result.character.

path = assemble(result.character, "runner/assets", "runner/scene.glb",
                compress=None)
    # → rigged, skinned GLB. CPU; hash-verifies every asset first.

path = make("a lean marathon runner …", "runner/", seed=41000,
            turbo=False, compress="web")
    # create → bake → assemble in one call.

Character.load("runner/character.char.json")   # validated on load and save
```

`Character` is a validated data object with `load`, `save`, `validate`, and `content_id`.

Module rules:

- `schema` and `assembly` import no diffusion or network code.
- `identity`, `textures`, and the local interpreter backend import torch lazily, so `import character_factory` is instant everywhere.
- `registry` is the only module that touches the network, and only when a component is missing from the cache.

CLI commands: `make`, `validate`, `assemble`, `compress`, `interpret`, `preflight`, `serve`, `mcp`.

### 2.2 The interpreter

The interpreter turns the description into every component's prompt: the figure prompt for identity, one prompt per texture slot, the hair block, and optional skeletal-proportion overrides. Its output is validated against the interpretation schema before anything runs.

**Default backend: a local model named by the `interpreter` registry component.** Qwen3.5-9B at launch — an exact upstream revision, hash-pinned, no account or token. It runs in-process on the same torch/transformers stack generation uses, in bfloat16, with grammar-constrained decoding. No external daemon.

**Optional backend: any OpenAI-compatible endpoint.** Configured per alias with `CHARACTER_FACTORY_INTERPRETER_ENDPOINT` / `_MODEL` / `_API_KEY` / `_MODE`, `interpreter.backends` in the cache `config.json`, or `PUT /v0/interpreters/{alias}`. Uses strict JSON-Schema response formatting where supported. An empty or truncated response gets one retry with a tripled completion budget; malformed or schema-invalid output fails. This is the recommended configuration when available. On one RTX 3090 with weights on disk, the local default interprets a description in 78 s in multi mode; a hosted frontier model answers the same description in 14 s in single mode, and writes better prompts.

**Two modes.** `single` decodes the whole interpretation in one pass from one instruction that folds in each component's registry guidance. `multi` asks one question per component (figure, skin, eye, garment, shoe, hair, proportions), each with a literal template for that component's format and its own grammar. In our bench, small local models did better in multi mode and hosted models in single mode. `mode: auto` (the default) picks `multi` for local models and `single` for endpoints. The multi-call templates live in `interpreter/multi.py` and are bound to the launch component versions.

**Skeletal proportions.** The identity component writes them on every create; an interpreter backend may emit explicit values on clear signal in the description, and those override per key.

**Vocabulary clamps.** Components may declare `constraints.vocabulary` in the registry (§4.2); the interpreter clamps its slot prompts to the installed components' declarations. The clamp constrains prompt authoring only — nothing downstream can verify it.

**No degraded mode.** There is no non-model backend. A model that cannot be fetched or a failed request is a structured error (`error.code`, `classification`, `retryable`, `trace_id`).

**Readiness.** `GET /v0/interpreters` reports, per configured backend, whether a create would succeed now and if not why (weights not downloaded and how many bytes; declared VRAM exceeds the device; no CUDA device; missing weights path). The checks are file existence and a device property read — no hashing, no model load, no network. A create against a registry model with uncached weights downloads them as a `downloading` job stage with byte progress, after the generation preflight passes; cancelling discards the partial file.

**VRAM sequencing.** The interpreter loads, runs, and releases inside the `interpret` call, before the diffusion stack loads.

**Audit log.** With `CHARACTER_FACTORY_INTERPRETER_AUDIT_LOG` set, the server appends a mode-0600 JSONL record per request (raw prompt and response, status, latency, usage, attempt, trace id). Public job errors carry only the trace id and a classification (`empty_response`, `truncated_response`, `invalid_json`, `schema_invalid`, `transport_error`).

**Comparing models.** `character-factory interpret "<text>" [--backend ALIAS] [--mode single|multi]` runs the production path without generation and prints the interpretation, wall time, peak memory, and per-call seconds.

### 2.3 The HTTP server

`character-factory serve` starts a FastAPI app around the library: a single-flight generation queue (one GPU), per-character directories on disk (the character file is the database), and progressive results — the GLB is rebuilt and atomically replaced as each stage lands. The bundled browser UI is a same-origin client of this contract with no private endpoints.

```
POST   /v0/characters                  {prompt, interpreter?, turbo?, seed?}
                                       | {character}         → 202 Job
                                       Idempotency-Key: same key + same body
                                       returns the original job; different
                                       body → 409; no key → new work
GET    /v0/characters                  completed records, newest first
GET    /v0/characters/{id}             record + artifact and latest-job state
GET    /v0/characters/{id}/character.json
GET    /v0/characters/{id}/scene.glb
GET    /v0/characters/{id}/assets/{slot}.png
PUT    /v0/characters/{id}/assets/{slot}   replace one asset; re-pins the hash
                                           and rebuilds from assemble
GET    /v0/characters/{id}/manifest.json   the GLB's embedded export manifest
GET    /v0/characters/{id}/thumbnail.png
POST   /v0/characters/{id}/rebuild     {from: "bake"|"assemble"} → 202 Job
DELETE /v0/characters/{id}

GET    /v0/jobs                        job list
GET    /v0/jobs/{id}                   stage, stages, stage_progress,
                                       heartbeat, outcome, error
DELETE /v0/jobs/{id}                   cancel
POST   /v0/jobs/{id}/retry             new attempt after failure or cancel

GET    /v0/interpreters                backends: alias, kind, default, label,
                                       ready, reason, download_bytes,
                                       vram_bytes, fits, device_bytes,
                                       endpoint_host, has_key — no model
                                       identities or keys
PUT    /v0/interpreters/{alias}        {endpoint|model, api_key?, mode?,
                                       label?, default?}; key is write-only;
                                       remote deployments may answer 405
DELETE /v0/interpreters/{alias}

POST   /v0/validate                    document → validation report
GET    /v0/components                  registry view
GET    /v0/health                      status, counts, cuda, vram_free_gb,
                                       vram_total_gb
GET    /v0/docs                        rendered OpenAPI
```

Job states, terminal outcomes, idempotency, cancellation, and retry are part of the `/v0` contract; capacity fields (`queue_position`, device fields in `/v0/health`) are server-specific.

**Trust boundary.** The server binds `127.0.0.1` and does not authenticate. On `0.0.0.0` the host firewall and network are the boundary: a trusted LAN or private overlay, not the public internet. There are no CORS headers; CORS constrains browser scripts, not other machines.

### 2.4 The MCP server

`character-factory mcp` (stdio) exposes:

- tools — `create_character`, `assemble_character`, `get_job`, `cancel_job`, `retry_job`, `get_character`, `list_characters`, `validate_character`, `store_character`, `list_components`
- resources — `character-factory://schema/character`, `character-factory://characters/{id}`

Tool inputs and outputs are character documents, not handles, so agent workflows compose with hand editing and version control. The tools mirror the `/v0` contract's names and shapes.

## 3. Assembly and export

`character_factory.assembly` is deterministic and CPU-capable. It does not import the diffusion stack. The stages, in order:

**1. Rig evaluation.** The MHR TorchScript rig (`body-rig` component) maps identity, proportions, rest pose, and resting expression to the LOD1 surface (18,439 vertices) and a 127-joint skeleton.

**2. Render topology.** The `body-rig` component may declare a coarser render surface; the launch component (1.2.1) declares MHR's own LOD3 (4,899 vertices, 9,794 triangles — the full closed body surface before the eye-socket, mouth, and under-shell faces below are removed), carried by the supplied barycentric map with skin weights and expression morphs transferred. Everything below operates on the render surface. Absent a declaration, the source topology is the render surface.

**3. Skin.** The body albedo is the skin texture, unmodified.

**4. Garment and shoe shells.** Garment coverage is recovered from the garment image by a calibrated luminance key against its black background (head region masked). The `shoe` image is a single-shoe canvas: the component's foot chart bakes it onto both feet's atlas islands (the second foot through the chart's mirror), with style-aware occupancy, and its alpha is the coverage. Each coverage is marched through the body triangles, lifted, faired, closed into a watertight solid, and exported as its own skinned mesh riding the body's weights. Body faces under a shell are deleted; a narrow skin band (`band_cm`: 3.0 for garments, 0.4 for shoes) is kept under the rim. A shell that fails a structural gate (alpha quality, seam cracks, topology, closed solid, weight audit) is an assembly error naming the gate. Layering is shoe over garment over skin.

**5. Eyes.** A patch over each socket is removed; a stock eyeball mesh is placed by a similarity fit of its lid margin to the socket rim and takes the eye albedo. Lashes, caruncles, and socket backing are not in v0.

**6. Mouth.** The rig version's fixed mouth patch is removed; a posterior-lip cuff and cavity strip built from the inner-lip curves, skinned by extending the lips' influences and UV-mapped into the removed patch's atlas region, are stitched into the body. Teeth, gums, and tongue (GNM-derived, `assembly-assets`) are placed by identity anchors — upper on the skull, lower and tongue on the jaw chain. Interior geometry keeps original vertex UVs bit-exact and adds no atlas islands.

**7. Hair.** The hair provider's mesh (§5) is parented rigidly to the head joint.

**8. Export.** Skinned glTF binary with the full 127-joint hierarchy, joint names from the rig component's metadata, inverse bind matrices, and the rig's own weights (≤ 4 influences, summing to 1 — carried exactly). Eyeballs are parented to the eye joints, hair to the head. Materials are metallic-roughness with constants: dielectric, skin roughness 0.5, garment and shoe shells 0.9; hair carries an albedo and a normal map. Every material is opaque. The 72 expression coefficients export as sparse morph targets `facs_00`–`facs_71`.

### 3.1 Exporter conventions

**Frame.** Rig: centimeters, Y-up, +Z-forward. glTF: meters, Y-up, +Z-forward. The conversion is a uniform 0.01 scale; no axis flip.

**Rest orientations are re-authored.** MHR's native rest rotations carry per-bone roll that is not mirrored between limbs, which breaks humanoid retargeters that derive hinge axes from rest frames. The exporter re-authors every joint's rest orientation from geometry under one mirror-invariant convention (long axis toward the mean of children; a sagittal-invariant forward reference orthogonalized against it). Bones shorter than 2 mm inherit the parent's direction. Joint positions are untouched; worst left/right deviation is ≈0.02°.

**Knee flexion.** 5° (`KNEE_FLEXION_DEGREES`, versioned) is baked into the rest pose about a shared sagittal axis.

**Inverse bind matrices** are rebuilt after both edits; the bound mesh is bit-identical to the rig's rest geometry.

**Correctives.** The rig animates as linear-blend skinning; MHR's learned pose correctives are not exported.

**Root.** Joint 0 is `body_world`, the rig's transform root, included in the skin's joint list so glTF joint indices equal rig indices; no vertex is weighted to it.

**File hygiene.** One self-contained `.glb`: PNG textures embedded once each (both eyes share one image), 4-byte-aligned buffer views, POSITION min/max, proper buffer targets, unsigned-short JOINTS_0 / float WEIGHTS_0, no UV V-flip, counter-clockwise winding verified against outward normals.

### 3.2 Manifest and idle clip

The **export manifest** (schema `0.6`, served at `/v0/schemas/export-manifest-0.6.json` and packaged with the library) is embedded in the GLB's asset-level `extras` and served standalone at `/v0/characters/{id}/manifest.json`. It carries:

- the engine humanoid role → joint-name map (54 bones for Unity Humanoid) and the leave-unmapped set
- units and axes, joint count, the knee constant
- per-slot `garments` shell inventory (`render_mode: "shell"`)
- a `jaw` block: rotation sign, `full_open_degrees`, and the `facs_24` + `expression_fit_angle_degrees` composition (alternatives, not summed)
- the measured animation-limitation table
- a `grounding` block: ground plane, root and foot offsets, idle drift tolerance, foot-IK recommendation
- the render LOD

It is a function of rig version, exporter constants, and the character's proportions — byte-identical across re-exports. Character identity, textures, hair, and provenance stay in the character document and are not duplicated into `extras`.

A **baked idle clip** ships in the GLB: a few seconds of breathing and weight sway with full local TRS for every joint, frame 0 exactly the rest pose, seamless loop. It is a Generic, native-skeleton clip, not validated for Humanoid retargeting.

### 3.3 Delivery compression

`scene.glb` is lossless and is what the determinism promise covers.

| Output | Command | Textures |
| --- | --- | --- |
| `scene.web.glb` | `--compress web` (on `make` and `assemble`) or `character-factory compress` | albedo as WebP q85, hair normal as WebP q90, under `EXT_texture_webp` (in `extensionsRequired` — no PNG fallback) |
| `scene.unity.glb` | `--compress unity` | JPEG at the same qualities, no extension — for glTFast and other loaders without WebP |

Meshes, skins, morph targets, the idle clip, and the manifest are unchanged. Geometry compression is not done: on a skinned mesh dequantization must be folded into the inverse bind matrices.

## 4. Components and the registry

No weights live in the repository or the package. Every model and static asset is a **component**: versioned, hash-pinned, fetched on first use from the `character-factory` Hugging Face organization (or upstream), cached under `~/.cache/character-factory/` (`CHARACTER_FACTORY_HOME`).

Generators are named `make-<artifact>` after the singular artifact they produce; data and infrastructure keep plain names.

| Component | Version | Contents | Size |
| --- | --- | --- | --- |
| `interpreter` | 0.1.0 | Qwen3.5-9B, exact upstream revision (Apache-2.0) | 19.33 GB (19,329,302,129 B) |
| `make-figure` | 0.1.1 | Text → identity + proportions generative model and normalization stats | 34.0 MB (33,960,218 B) |
| `make-skin` | 0.0.4 | Adapter for the `skin` albedo (body atlas) | 92.4 MB (92,426,392 B) |
| `make-eye` | 0.1.0 | Adapter for the `eye` albedo (eyeball layout) | 92.4 MB (92,426,386 B) |
| `make-garment` | 0.1.0 | Adapter for the `garment` albedo (garment over black) | 92.4 MB (92,426,390 B) |
| `make-shoe` | 0.0.2 | Adapter for the `shoe` albedo (single-shoe canvas) plus the foot chart (47,186 B of the total) | 92.5 MB (92,473,322 B) |
| `make-wig` | 0.1.0 | Procedural hair engine, vendored; registry entry records the version and density presets | in package |
| `body-rig` | 1.2.1 | Pinned MHR v1.0.1 release: TorchScript rig, LOD1 + LOD3 topology, expression morphs, 127-joint name table, humanoid map, mouth data | 709 MB (709,024,724 B) |
| `assembly-assets` | 0.3.0 | Eyeballs, dental arches, tongue, placement data, atlas metadata | 46,598 B |
| `flux2-klein-base-4b` | 1.0.0 | FLUX.2 Klein 4B base (transformer, text encoder, VAE, tokenizer, scheduler — 18 files), fetched from `black-forest-labs/FLUX.2-klein-base-4B` at an exact revision; the text encoder also embeds the figure prompt | 15.98 GB (15,980,131,711 B) |
| `flux2-klein-4b` | 1.0.0 | FLUX.2 Klein 4B distilled variant, same 18-file layout, fetched from `black-forest-labs/FLUX.2-klein-4B` at an exact revision; used only by `--turbo` | 15.98 GB (15,980,131,745 B) |

Sizes are the sum of each entry's `artifacts[].bytes` in the vendored snapshot. Download totals:

| Use | Components | Total |
| --- | --- | --- |
| First run, default configuration | `interpreter` + `flux2-klein-base-4b` + `make-figure` + the four adapters + `body-rig` + `assembly-assets` | 36.42 GB (36,422,217,870 B) |
| First run, endpoint interpreter | as above without `interpreter` | 17.09 GB (17,092,915,741 B) |
| `--turbo` | adds `flux2-klein-4b` | +15.98 GB |
| Assembly only (`assemble` on an existing character file) | `body-rig` + `assembly-assets` | 709 MB (709,071,322 B) |
| Assembly only, character wears shoes | as above plus the whole `make-shoe` component | 802 MB (801,544,644 B) |

Everything is pinned by content hash: upstream models by revision hash, the MHR release archive and each consumed artifact by SHA-256. Every artifact is verified after fetch and before load; a mismatch is an error.

### 4.1 The registry

The registry is a JSON index fetched like a component, with a snapshot vendored into each release as the offline default. `CHARACTER_FACTORY_REGISTRY_URL` and `CHARACTER_FACTORY_AUTH_TOKEN` (or `registry_url` / `auth_token` in `config.json`) point at an alternate index. An entry:

```json
{
  "name": "make-skin",
  "version": "0.0.4",
  "kind": "texture-adapter",
  "slot": "skin",
  "map": "albedo",
  "requires": { "base_model": "flux2-klein-base-4b", "schema": ">=0.1 <1.0" },
  "artifacts": [
    { "path": "config.json", "sha256": "c281165d41b53c33f0a1af8c3021be079db6d6c1e245b6aedc93d6a20bb9815c", "bytes": 256 },
    { "path": "weights.safetensors", "sha256": "7b09947747dba9a86d798fba7c076b6b075ca39eaee927dadef12cd43a0d177f", "bytes": 92426136 }
  ],
  "inference": { "prompt_template": "MHR face albedo texture map, neutral delit lighting, {prompt}", "steps": 20, "guidance": 4.0, "resolution": 1024 },
  "interpretation": { "fields": "…" },
  "source": { "hf_repo": "character-factory/make-skin", "revision": "main" },
  "description": "Full-body skin albedo on the body's canonical UV atlas."
}
```

- `inference` holds each adapter's conditioning template and sampler defaults; `interpretation.fields` is the per-component prompt guidance the interpreter folds into its instructions. An updated component with different conditioning is a version bump.
- `constraints.vocabulary` (optional) declares what a component supports; the interpreter clamps to it (§2.2). `make-shoe` uses it.
- `map` names which of a slot's maps a texture component produces (all current: `albedo`).
- `create` resolves each slot to the newest compatible component and records exact versions in provenance; `bake` honors the pins; `rebuild {from: "bake"}` re-resolves after an update.

## 5. Hair

Hair is synthesized by the vendored `make-wig` engine behind one protocol:

```python
class HairProvider(Protocol):
    def synthesize(self, intent: dict, head: HeadGeometry) -> HairResult: ...
    # intent: the character's hair block (SPEC.md §6)
    # head:   body mesh, forward axis, eye level — body frame, cm
    # result: textured triangle mesh + material + provider version
```

The engine fits any roughly human head mesh and needs no rig internals. Its output is deterministic for a fixed (intent, head, engine version). Wigs are opaque sculpted shells bound rigidly to the head bone, with albedo and normal maps; the convention follows the VALID avatar library's hair (NOTICE).

## 6. Install and hardware

```sh
pip install "character-factory[generation]"
character-factory make "a lean marathon runner with cropped dark hair" -o runner/
```

- Python ≥ 3.11. Base install 5.6 GB (torch with CUDA); `[generation]`, `[server]`, `[mcp]` extras on top.
- Linux + NVIDIA CUDA is first-class. Native Windows runs the full pipeline; `pip install` on Windows installs CPU torch; install torch from the PyTorch CUDA index (`https://download.pytorch.org/whl/<cuXXX>`) first. WSL2 runs the Linux path. macOS runs the CLI, schema tools, server, and assembly.
- `character-factory preflight` checks the generation import set, the torch CUDA build, and the driver with a real CUDA call. `make` and every server generation job run it before loading weights.

### 6.1 Measured VRAM

Measured on one RTX 3090 (24 GiB) with the launch components and weights on disk; peak torch allocation, including pipeline load.

| Stage | Precision | Allocated | Reserved | Time |
| --- | --- | --- | --- | --- |
| Four-slot bake at 1024² | bf16 (default) | 17.4 GiB | 20.3 GiB | 137 s |
| Local interpreter (Qwen3.5-9B, multi mode) | bf16 | 16.9 GiB | — | 78 s; 68 s with the model files in the OS page cache |

GPU stages are serialized and release before the next loads, so the bake sets the floor.

## 7. Tests

`tests/` mirrors the module layout. What the suite protects:

**Schema.** Load → save → load is the identity; canonical form and content ID are stable; every example validates; a corpus of broken documents fails with the documented error; unknown optional fields warn by default and fail strict; unknown `topology`, `rig`, `proportions` keys, and `inputs` are hard errors.

**Export.** Byte-determinism across runs; a re-parse acceptance test (read the GLB back, skin the rest mesh through node matrices and inverse bind matrices, compare to exported positions, ≈0 mm); 127 joints with the golden name table, topological parent order, no weights on joint 0, weights sum to 1 with ≤ 4 influences, UV-seam duplicates carry their weights; left/right rest frames mirror within a bound; knee constant; manifest present, complete, byte-stable; the idle clip reproduces the rest skin at t=0 through the hierarchy and moves within bounds over its length; buffer alignment, scale, winding, no V-flip.

**Shells, mouth, footwear.** Extraction gates, watertightness, weight transfer, interior-UV invariants (original UVs bit-exact, no new islands, no overlap, no inversion), clearance sweeps over the supported pose set, and the same battery re-run natively on a declared render LOD (skipped when no such component is cached).

**Bake.** One image per slot at the declared resolution with hashes recorded; seeds and templates honored; recipe overrides beat defaults; the base model loads once; assembly refuses non-matching assets.

**Interpreter, server, registry, CLI.** Grammar derivation, backend readiness and config writes, job lifecycle (idempotency, cancel, retry, download stage), route contracts, integrity checks, command behavior.

Runs that need cached components skip when they are absent.

## 8. Repository layout

```
├── README.md · SPEC.md · ARCHITECTURE.md · LICENSE · NOTICE
├── pyproject.toml
├── src/character_factory/
│   ├── __init__.py       Character, create, assemble, make
│   ├── api.py            create / make / assemble
│   ├── cli.py
│   ├── preflight.py
│   ├── schema/           format model, validation, canonical form, JSON Schema
│   ├── registry/         index, fetch, cache, integrity; vendored snapshot
│   ├── interpreter/      backends, multi-call templates, grammar, config
│   ├── identity/         figure prompt → body parameters (lazy torch)
│   ├── textures/         diffusion runner, adapters (lazy torch)
│   ├── hair/             HairProvider + vendored make-wig engine
│   ├── assembly/         rig, shells, eyes, mouth, rest pose, export, manifest, compress
│   ├── server/           FastAPI app, service, jobs, static UI
│   └── mcp/
├── examples/characters/  committed .char.json files
└── tests/
```
