# Character Factory

A free, open-source, locally-run text-to-3D character pipeline.

## Quickstart

```
pip install "character-factory[generation]"
character-factory make "A retired astronomy professor, tweed waistcoat, round spectacles, white beard" -o professor
```

```
$ character-factory make "A retired astronomy professor, tweed waistcoat, round spectacles, white beard" -o professor
create      77.6 s
bake       134.3 s
assemble     5.2 s
professor/character.char.json
professor/scene.glb
```

(One RTX 3090, weights on disk, default local interpreter.) The first run
downloads 36.4 GB of model weights, or 17.1 GB if you point the
interpreter at an endpoint (see [The interpreter](#the-interpreter)).
After that, generation is local and offline.

## The character file

`character.char.json` is the character. Trimmed from the [SPEC.md](SPEC.md)
§3 example:

```json
{
  "format": "character-factory/character",
  "schema_version": "0.1",
  "body": {
    "rig": "mhr-lod1@1.0",
    "identity": ["…"],
    "proportions": { "leg_length": 0.24, "hip_width": -0.06 },
    "resting_expression": ["…"]
  },
  "textures": {
    "garment": {
      "component": "make-garment",
      "component_version": "0.1.0",
      "prompt": "teal running vest and black shorts, white piping",
      "seed": 41004
    },
    "…": "…"
  },
  "hair": { "family": "crop", "color": { "family": "dark_brown" }, "…": "…" },
  "provenance": {
    "components": { "make-figure": { "version": "0.1.1" }, "make-garment": { "version": "0.1.0" }, "…": "…" },
    "…": "…"
  }
}
```

Change `textures.garment.prompt` and hand the file back to the pipeline
(`POST /v0/characters` with `{"character": …}` on a running server, or
`bake` then `assemble` in Python): same character, different clothes.
Plain JSON. Assembly is deterministic: the same file produces the same
GLB. The format is specified in [SPEC.md](SPEC.md).

## What you get

- **`character.char.json`** — body parameters, texture recipes, a semantic
  hair description, and provenance; 5,973 bytes for the quickstart
  character.
- **`scene.glb`** — a rigged, skinned glTF built from the character file:
  a skin-only body, garment and shoe shells with their own materials,
  hair geometry with albedo and normal maps, and a modeled mouth (teeth,
  gums, tongue, inner cavity).

Every character:

| | |
|---|---|
| Triangles | 17,850 in the quickstart character (body 8,337, garment 4,528, shoe 1,432, hair 1,059, mouth 1,798, eyes 696); hair is 1.1–2.3k for most families, 31k for locs, 45k for braids |
| Textures | 1024² per surface — skin, eyes, garment, shoes; hair albedo + normal |
| Rig | 127 joints, exact linear-blend skinning, bone-role manifest embedded; 54-bone Unity Humanoid map in the manifest |
| Facial animation | 72 expression morph targets (`facs_00`–`facs_71`); the jaw animates through the `c_jaw` joint |
| Idle | a baked breathing/weight-sway clip, playable as Generic or Humanoid |
| Materials / draw calls | 11 primitives with garment and shoes |
| Alpha passes | 0 — fully opaque |
| GLB size | 8.8 MB lossless (PNG) for the quickstart character; 3.9 MB with `--compress web`, 4.6 MB with `--compress unity` |
| Generation time | 3 min 38 s wall for the quickstart character |

`--compress web` writes `scene.web.glb` with WebP textures
(`EXT_texture_webp`); `--compress unity` writes `scene.unity.glb` with
JPEG textures and no extension, for glTFast and other loaders without
WebP. Meshes, morph targets, the idle clip, and the manifest are the same
in every variant.

## Known limitations (v0.1)

- Garment coverage is recovered from the garment texture by a luminance
  key against its black background; the shell edge is where that key
  crosses its threshold. Shells follow the body surface: loose clothing,
  skirts, and anything leaving the body silhouette is out of scope.
- The garment model has no sari or wrapped-garment coverage.
- Footwear styles run from below-ankle shoes to tall boots, as declared by
  `make-shoe`. Both feet wear the same design (the canvas is mirrored).
  Open styles (sandals, flip-flops) keep only the straps the bake recovers;
  when that recovery keeps too little of the canvas, a schematic two-band
  strap layout is substituted.
- Hair is a closed set of eleven procedural families (buzz, crop, pixie,
  side_part, bob, loose_long, coily, ponytail, bun, braids, locs), one
  color per character; "dyed tips" cannot be represented. An improved hair
  path is in progress for v0.2.
- The local interpreter misses on some prompts (bottom garments, one-piece
  garments); an endpoint does better on the same prompts.
- The render surface is LOD3 only; no generated normal or secondary maps
  for skin and garments.
- Skeletal proportions are six controls (`spine_length`, `neck_length`,
  `shoulder_width`, `arm_length`, `hip_width`, `leg_length`); the rig's
  finer per-segment scales stay at template values.
- Keep compound facial expressions moderate while the mouth is nearly
  closed; the manifest's limitation table lists the combinations that clip.
- Ground contact in-engine needs foot IK.
- A garment or shoe shell that fails a structural gate is an assembly error
  naming the gate; the body is never painted instead.
- 12 GB cards run the bake under `nf4` with an endpoint interpreter; 8 GB
  is not supported.
- A failed interpretation returns a structured error; there is no fallback
  backend.

## Interfaces

- **CLI** — `character-factory make` (above; `--seed` pins the identity
  and texture draw, `--backend` picks a configured interpreter by alias,
  `--turbo` trades texture quality for bake time, `--compress web|unity`
  also writes a delivery-sized GLB), `validate`, `assemble` (character
  file → GLB, no GPU), `compress`, `interpret` (prints what a description
  decomposes into, with timings), `preflight`.
- **Server + browser UI** — `pip install "character-factory[server]"`,
  then `character-factory serve`: the `/v0` HTTP API and a gallery UI on
  `127.0.0.1:8400` (`--host 0.0.0.0` for other machines on a trusted
  network); `/v0/docs` documents the API. A bearer token is accepted and
  ignored.
- **MCP** — `pip install "character-factory[mcp]"`, then add
  `character-factory mcp` to your agent's MCP config: `create_character`,
  `get_job`, `get_character`, `list_components`, and friends, on stdio.
- **Unity (6000.0+)** — run `character-factory serve`, add
  `"com.character-factory.unity": "https://github.com/character-factory/character-factory-unity.git"`
  to `Packages/manifest.json`, then `unity cmd cf-create --prompt "<description>" --walking true --json`
  and `unity cmd cf-verify --target "<scene-object-name>" --json`. The
  [Unity package README](https://github.com/character-factory/character-factory-unity)
  covers the editor window, the Humanoid avatar, and the verify checks.

## Hardware and install

Measured on a single RTX 3090 (details in
[ARCHITECTURE.md §6](ARCHITECTURE.md)):

| | |
|---|---|
| Install | 1.1 GB |
| Weights | 36.4 GB on first use: 19.3 GB interpreter + 16.0 GB base image model + 1.1 GB components; 17.1 GB with an endpoint interpreter |
| Generation, 24 GB card | the default bf16 pipeline: bake 17.4 GiB allocated, 137 s; whole character 3 min 38 s with the local interpreter |
| Generation, 12 GB card | `nf4` quantization (`textures.quantization` in the cache config): bake 8.9 GiB allocated, 267 s (slower: dequantization), with an endpoint interpreter. 8 GB is not supported |
| Assembly and consumption | no GPU — character file → GLB runs on CPU, including macOS |

`character-factory preflight` checks the generation stack in seconds —
the `[generation]` import set, the torch CUDA build, and the driver (via
a real CUDA call). `make` and every server generation job run the same
check before touching any weights.

## The interpreter

The interpreter is the language model that turns your description into
each component's prompt. It runs locally by default: the registry's
`interpreter` component names Qwen3.5-9B (Apache-2.0, ungated — no
token, no account), 19.3 GB to download, 16.9 GiB of VRAM on its own,
78 s per description on the 3090 (25 s to load, 52 s to answer one
question per component). It never shares VRAM with the image model.

**For speed and quality, point it at an OpenAI-compatible endpoint
instead.** A current hosted frontier model (an OpenAI GPT-5.6-class model
in our bench) takes 14 s and wrote better garment prompts in our bench.
Configure it with `CHARACTER_FACTORY_INTERPRETER_ENDPOINT`, `_MODEL`, and
`_API_KEY`, with `interpreter.backends` in the cache `config.json`, or
through `PUT /v0/interpreters/{alias}` on a running server. The browser
UI offers the same two choices in its interpreter setup panel.

## Built on

MHR (Meta, Apache-2.0), FLUX.2 Klein 4B (Black Forest Labs, Apache-2.0),
Qwen3.5-9B (Apache-2.0), UnityEyes2 (MIT), GNM (Google, Apache-2.0) —
see [NOTICE](NOTICE).

## Status

v0.1 is the architecture launch, built in public; v0.2 targets hair and
seeded variation. File issues at
[github.com/character-factory/character-factory/issues](https://github.com/character-factory/character-factory/issues).

## Going deeper

- [SPEC.md](SPEC.md) — the character format. `character_factory.schema`
  is its reference implementation (standard library only) with the
  published JSON Schema, example characters, and test corpus.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the system: interpreter, bake,
  assembly and export conventions, components and registry, install
  story, test surface.
- `/v0/docs` on a running server — the HTTP contract, generated from the
  OpenAPI document.

```python
from character_factory import Character

c = Character.load("examples/characters/freediver.char.json")
print(c.content_id, c.rig, sorted(c.textures))
```

## Trust boundary

The local server binds to `127.0.0.1` by default and does not
authenticate requests. Bound to `0.0.0.0`, your firewall and network
rules are the security boundary — a trusted LAN or private overlay, never
the public internet. Details in [ARCHITECTURE.md §2.3](ARCHITECTURE.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
