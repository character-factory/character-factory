# Character Factory

A free, open-source, locally-run text-to-3D character pipeline. Prompt
in, rigged character out.

## Quickstart

```
pip install "character-factory[generation]"
character-factory make "A retired astronomy professor, tweed waistcoat, round spectacles, white beard" -o professor
```

The first run downloads ≈36 GB of model weights (≈17 GB if you point the
interpreter at an endpoint — see [The interpreter](#the-interpreter)).
After that, generation is local and offline. Stage timings print to
stderr; stdout is exactly the two output paths:

```
$ character-factory make "A retired astronomy professor, tweed waistcoat, round spectacles, white beard" -o professor
create      80.9 s
bake       131.2 s
assemble     4.6 s
professor/character.char.json
professor/scene.glb
```

(24 GB card, weights already on disk, default local interpreter.)
`--seed` pins the identity and texture draw, `--backend` picks a
configured interpreter by alias, `--turbo` trades texture quality for
bake time, `--compress web|unity` also writes a delivery-sized GLB.

## What you get

- **`character.char.json`** — the character itself: a few kilobytes of
  JSON holding body parameters, texture recipes, a semantic hair
  description, and provenance. Diff it, commit it, edit it by hand, hand
  it to an agent. It regenerates the character under pinned component
  versions; character file → GLB is byte-identical.
- **`scene.glb`** — a rigged, skinned glTF built deterministically from
  the character file. Garments and shoes are separate body-following
  shells with their own cloth materials over a skin-only body; hair is
  its own geometry with albedo and normal maps; the mouth is modeled
  (teeth, gums, tongue, inner cavity).

Every character:

| | |
|---|---|
| Triangles | ~15–50k, hair-dependent (body ≈16k; locs and braids are the high end) |
| Textures | 1024² per surface — skin, eyes, garment, shoes; hair albedo + normal |
| Rig | 127 joints, exact linear-blend skinning, bone-role manifest embedded; 54-bone Unity Humanoid map in the manifest |
| Facial animation | 72 expression morph targets (`facs_00`–`facs_71`); the jaw animates through the `c_jaw` joint |
| Idle | a baked breathing/weight-sway clip, playable as Generic or Humanoid |
| Materials / draw calls | 10–11 primitives |
| Alpha passes | 0 — fully opaque by design |
| GLB size | 7–12 MB lossless (PNG); `--compress web` ≈ a third of that |
| Source document | 2–6 KB JSON |
| Generation time | ~1–5 min depending on hardware and interpreter |

`scene.glb` is lossless. For delivery, `--compress web` writes
`scene.web.glb` with WebP textures (`EXT_texture_webp`), and
`--compress unity` writes `scene.unity.glb` with JPEG textures and no
extension, for glTFast and other loaders without WebP.
`character-factory compress` does the same to an existing file. Meshes,
morph targets, the idle clip, and the manifest are untouched either way.

Scope notes for v0.1: identity drives the face, build, and surface form,
and skeletal proportions vary within six semantic controls (spine, neck,
shoulders, arms, hips, legs — `body.proportions`); the rig's finer
per-segment scales stay at template values. Ground contact in-engine
needs foot IK either way. Keep compound facial expressions moderate while
the mouth is nearly closed — the manifest's limitation table lists the
exact combinations that clip.

## Doors

Four ways onto the same pipeline. Everything below the CLI is a client
of the `/v0` HTTP contract; nothing is a local-only convenience.

- **CLI** — `make` (above), `validate`, `assemble` (character file →
  GLB, no GPU), `compress`, `interpret` (see what a description
  decomposes into, for benchmarking interpreter models side by side),
  `preflight`.
- **Server + browser UI** — `pip install "character-factory[server]"`,
  then `character-factory serve`: the local `/v0` HTTP API and a gallery
  UI on `127.0.0.1:8400` (`--host 0.0.0.0` for agents and other machines
  on a trusted network). `/v0/docs` on a running server documents the
  API. Local and hosted are one product with two addresses: same
  contract, and a bearer token is accepted (and ignored) locally so no
  client changes shape when auth becomes real.
- **MCP** — `pip install "character-factory[mcp]"`, then add
  `character-factory mcp` to your agent's MCP config: `create_character`,
  `get_job`, `get_character`, `list_components`, and friends, on stdio.
  No API key or account involved.
- **Unity (6000.0+)** — run `character-factory serve`, add
  `"com.character-factory.unity": "https://github.com/character-factory/character-factory-unity.git"`
  to `Packages/manifest.json`, then `unity cmd cf-create --prompt "<description>" --walking true --json`
  and `unity cmd cf-verify --target "<scene-object-name>" --json`. The
  [Unity package README](https://github.com/character-factory/character-factory-unity)
  covers the editor window, the Humanoid avatar, and the verify checks.

If you are handing this to a coding agent, paste in the agent block from
the website — it is this section in five lines, and points back here.

## Hardware and install

Measured numbers, not aspirations (details in
[ARCHITECTURE.md §6](ARCHITECTURE.md)):

| | |
|---|---|
| Install | ≈1.1 GB (most of it the CPU torch wheel the rig evaluation needs) |
| Weights | ≈36 GB on first use: ≈19 GB interpreter + ≈16 GB base image model + ≈1 GB components; ≈17 GB with an endpoint interpreter |
| Generation, 24 GB card | the default bf16 pipeline: bake measured at 17.4 GB allocated (≈132 s), whole character ≈3.5 min with the local interpreter, ≈2.5 min with an endpoint |
| Generation, 12 GB card | `nf4` quantization (`textures.quantization` in the cache config): bake measured at 8.9 GB (≈265 s), with an endpoint interpreter. 8 GB is not supported |
| Assembly and consumption | no GPU — character file → GLB runs on CPU, including macOS |

`character-factory preflight` checks the generation stack in seconds —
the `[generation]` import set, the torch CUDA build, and the driver (via
a real CUDA call) — and names what is broken instead of letting it
surface minutes into the first model load. `make` and every server
generation job run the same check before touching any weights.

## The interpreter

The interpreter is the language model that turns your description into
each component's prompt. It runs locally by default: the registry's
`interpreter` component names Qwen3.5-9B (Apache-2.0, ungated — no
token, no account), ≈19 GB to download, ≈18 GB of VRAM on its own, about
a minute per description asked one component at a time. It never shares
VRAM with the image model.

**For speed and quality, point it at an OpenAI-compatible endpoint
instead.** A current hosted frontier model (an OpenAI GPT-5.6-class model
in our bench) takes 10–15 s and writes noticeably richer clothing.
Configure it with `CHARACTER_FACTORY_INTERPRETER_ENDPOINT`, `_MODEL`, and
`_API_KEY`, with `interpreter.backends` in the cache `config.json`, or
through `PUT /v0/interpreters/{alias}` on a running server. The browser
UI offers the same two choices in its interpreter setup panel and says
up front what pressing Create will do — create, download the local model
first (a visible job step with byte progress), or nothing until an
interpreter is set up. There is no degraded mode: the server never
silently swaps the requested backend, and a failed interpretation is a
structured, retryable error, not a worse character.

## Going deeper

- [SPEC.md](SPEC.md) — the character format, if you are judging the
  format or implementing against it. `character_factory.schema` is its
  reference implementation (standard library only) with the published
  JSON Schema, example characters, and test corpus.
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

**Trust boundary.** The local server binds to `127.0.0.1` by default and
does not authenticate requests. Bound to `0.0.0.0`, your firewall and
network rules are the security boundary — a trusted LAN or private
overlay, never the public internet. The bundled UI is same-origin; there
are no CORS headers, and CORS would not protect the port from other
machines anyway. Details in [ARCHITECTURE.md §2.3](ARCHITECTURE.md).

Licensed under Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
