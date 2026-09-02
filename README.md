# Character Factory

Turn a text description into a rigged, textured, realtime 3D human.

<!-- demo video embed lands here at launch -->

> **Pre-release.** This repository is being built in the open ahead of its
> first release. The character format ([SPEC.md](SPEC.md)) and the system
> design ([ARCHITECTURE.md](ARCHITECTURE.md)) are stable enough to read and
> implement against; the package below is under construction and not yet on
> PyPI.

```
pip install "character-factory[generation]"
character-factory make "a lean marathon runner with cropped dark hair" -o runner/
```

Stage timings print to stderr as they finish; stdout is the two output
paths (`runner/character.char.json`, `runner/scene.glb`). `--seed` pins
the identity and texture draw, `--backend` picks a configured interpreter
by alias, `--turbo` trades texture quality for bake time. The first run
downloads the model components (sizes below).

`scene.glb` is lossless (7–12 MB, most of it PNG textures). For
delivery, `--compress web` also writes `scene.web.glb` with WebP
textures (`EXT_texture_webp`; about a third of the size), and
`--compress unity` writes `scene.unity.glb` with JPEG textures and no
extension, for glTFast and other loaders without WebP.
`character-factory compress` does the same to an existing file. Meshes,
morph targets, the idle clip, and the manifest are untouched either way.

Two more doors onto the same pipeline: `character-factory serve` (extra
`[server]`) runs the local `/v0` HTTP API with a browser UI, and
`character-factory mcp` (extra `[mcp]`) runs an MCP server on stdio for
coding agents — `create_character`, `get_job`, `get_character`,
`list_components`, and friends, no API key or account involved.

One sentence in; two artifacts out:

- **`character.char.json`** — the character itself: a few kilobytes of JSON
  holding body parameters, texture recipes, a semantic hair description, and
  provenance. Diff it, commit it, edit it by hand, hand it to an agent.
- **`scene.glb`** — a rigged, skinned glTF built deterministically from the
  character file: 127-joint skeleton, embedded bone-role manifest, baked
  idle clip, modeled mouth (teeth, gums, tongue, inner cavity), and 72 exact
  facial-expression morph targets
  (`facs_00`–`facs_71`); the jaw animates through the `c_jaw` joint, with
  the certified rotation and the measured animation limitations stated
  machine-readably in the embedded manifest. Keep compound facial
  expressions moderate while the mouth is nearly closed — the manifest's
  limitation table lists the exact combinations that clip, and demo
  footage should avoid them.

The repository contains the format's reference implementation and test
corpus (`character_factory.schema` — standard library only), the published
JSON Schema, example characters, the component registry, the interpreter,
the texture bake pipeline, the assembler/exporter, and the local server —
one `/v0` HTTP contract shared between local and hosted, with a bundled
browser UI that is a plain client of it.

```python
from character_factory import Character

c = Character.load("examples/characters/freediver.char.json")
print(c.content_id, c.rig, sorted(c.textures))
```

## Install and hardware

Measured numbers, not aspirations (details in
[ARCHITECTURE.md §6](ARCHITECTURE.md)):

- **Base install ~1.1 GB** (schema tools, registry, assembly, server —
  most of it the CPU torch wheel the rig evaluation needs). Model
  components download on first use.
- **Generation: 24 GB of VRAM runs the default full-precision pipeline**
  (~132 s of diffusion per character). **12 GB works with `nf4`
  quantization** (measured under 10 GB reserved, roughly twice the bake
  time) and an endpoint interpreter. 8 GB is not supported.
- **Interpretation — the language model that turns your description into
  each component's prompt — runs locally by default:** the registry's
  `interpreter` component names Qwen3.5-9B (Apache-2.0, no account or
  token needed), ≈19 GB to download, ≈18 GB of VRAM, about a minute per
  description asked one component at a time. **For speed and quality,
  point it at an OpenAI-compatible endpoint instead** — a current hosted
  frontier model (an OpenAI GPT-5.6-class model in our bench) takes
  10–15 s and writes noticeably richer clothing — with
  `CHARACTER_FACTORY_INTERPRETER_ENDPOINT`, `_MODEL`, and `_API_KEY`, or
  `interpreter.backends` in the cache `config.json`
  ([ARCHITECTURE.md §2.2](ARCHITECTURE.md)).
- **Assembly without generation runs anywhere** — including CPU-only
  machines and macOS: validate, assemble, and serve existing character
  files with no GPU at all.
- **`character-factory preflight` checks the generation stack in
  seconds** — the `[generation]` import set, the torch CUDA build, and
  the driver (via a real CUDA call) — and names what is broken, instead
  of letting it surface minutes into the first model load. `create` and
  `bake` run the same check before touching any weights.

One honest scope note: in v0.1, identity drives the face, build, and
surface form, and **skeletal proportions vary within six semantic
controls** (spine, neck, shoulders, arms, hips, legs — `body.proportions`
in the character file); the rig's finer per-segment scales stay at
template values. Ground contact in-engine needs foot IK either way.

Start with [SPEC.md](SPEC.md) if you are judging the format,
[ARCHITECTURE.md](ARCHITECTURE.md) if you are judging the system, and
`/v0/docs` on a running server if you are integrating against the API.

## Server trust boundary

The local server binds to `127.0.0.1` by default. An operator may bind it to
`0.0.0.0` for agents and native clients on a trusted local network or private
overlay network. In that mode, the operator's firewall and network access
rules are the security boundary: the v0 local server does not authenticate
requests, and a bearer token is accepted but ignored. Do not expose it to the
public internet or an untrusted network.

The bundled browser UI is served by the same process and uses same-origin
requests. Cross-origin browser clients are not supported, and the server does
not provide CORS headers. CORS is a browser policy and would not protect this
service from other machines or native clients that can reach its port.

`GET /v0/characters` returns a bare newest-first array of completed character
records. Character creation and explicit rebuilds are new work by default. If
an HTTP response is lost, repeat the request with the same `Idempotency-Key` to
recover the original job; using that key for a different request returns
`409`. Export guarantees such as mouth topology, the Humanoid mapping, facial
morph inventory, jaw behavior, and grounding live in the versioned export
manifest rather than being duplicated on every record and job.

Interpreter failures are structured and backend-neutral. Clients can use the
job error's `code`, safe `classification`, `retryable` flag, and opaque
`trace_id`, then retry the original job through `/v0/jobs/{id}/retry`. The
server never silently changes the requested interpreter or mutates the
original request — there is no degraded interpretation mode to fall back to.

Licensed under Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
