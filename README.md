# Character Factory

A free, open-source, locally-run text-to-3D character pipeline. Prompt in, rigged character out.

[Website](https://characterfactory.ai) · [Hugging Face](https://huggingface.co/character-factory) · [PyPI](https://pypi.org/project/character-factory/) · [Unity package](https://github.com/character-factory/character-factory-unity)

- [Quickstart](#quickstart)
- [Agent quickstart](#agent-quickstart)
- [What you get](#what-you-get)
- [The character file](#the-character-file)
- [Interfaces](#interfaces)
- [Hardware and install](#hardware-and-install)
- [The interpreter](#the-interpreter)
- [Known limitations](#known-limitations-v01)
- [Going deeper](#going-deeper)

## Quickstart

Requires Python 3.11+.

```sh
pip install "character-factory[generation]"
```

```console
$ character-factory make "A retired astronomy professor, tweed waistcoat, round spectacles, white beard" -o professor
create      77.6 s
bake       134.3 s
assemble     5.2 s
professor/character.char.json
professor/scene.glb
```

Timings: one RTX 3090, all weights on disk, default local interpreter. On the first run, `create` includes the base-model download.

The first run downloads 36.4 GB of model weights (17.1 GB with an endpoint interpreter). After that, generation is local and offline.

## Agent quickstart

Paste into Claude Code, Codex, or any coding agent.

```
Set up Character Factory (NVIDIA GPU, 24 GB):
Requires Python 3.11+.
On Windows, install torch from https://download.pytorch.org/whl/<cuXXX> before step 1.
1. pip install "character-factory[generation,server]"
2. character-factory make "<description>" -o <dir>
   Writes <dir>/character.char.json and <dir>/scene.glb; stdout is exactly those two paths.
   Interpreter: Qwen3.5-9B, local by default (Apache-2.0, ungated — no token, no account).
   First run downloads 36.4 GB of model weights.
Optional:
- MCP: pip install "character-factory[mcp]"; add `character-factory mcp` to the agent's MCP config.
- Unity 6000.0+: character-factory serve; add
  "com.character-factory.unity": "https://github.com/character-factory/character-factory-unity.git"
  to Packages/manifest.json; then unity cmd cf-create --prompt "<description>" --walking true --json
  and unity cmd cf-verify --target "<name>" --json.
- Endpoint interpreter (faster): set CHARACTER_FACTORY_INTERPRETER_ENDPOINT, _MODEL, _API_KEY.
```

Or:

```
Build a character-driven game using https://characterfactory.ai
```

No additional API key or account required.

## What you get

Two files per character:

- **`character.char.json`** — body parameters, texture recipes, a hair description, and provenance. A few KB.
- **`scene.glb`** — a rigged, skinned glTF: a skin-only body, garment and shoe shells with their own materials, hair with albedo and normal maps, and a modeled mouth (teeth, gums, tongue, inner cavity).

| Property | Every character |
| --- | --- |
| Triangles | ~15–50k, hair-dependent |
| Textures | 1024² per surface — skin, eyes, garment, shoes; hair albedo + normal |
| Rig | 127 joints, linear-blend skinning; 54-bone Unity Humanoid map in the manifest |
| Facial animation | 72 morph targets (`facs_00`–`facs_71`) + jaw joint |
| Idle | a baked breathing clip, Generic or Humanoid |
| Materials / draw calls | 10–11 primitives |
| Alpha passes | 0 — fully opaque |
| GLB size | ~9 MB; ~4 MB with `--compress` |

Compression: `--compress web` writes `scene.web.glb` with WebP textures; `--compress unity` writes `scene.unity.glb` with JPEG textures for glTFast and other loaders without WebP.

## The character file

`character.char.json` is the character; the GLB is built from it. Trimmed from the [SPEC.md](https://github.com/character-factory/character-factory/blob/main/SPEC.md) §3 example:

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

Edit the file and resubmit it — `POST /v0/characters` with `{"character": …}`, or `bake` then `assemble` in Python. Assembly is deterministic. The format is specified in [SPEC.md](https://github.com/character-factory/character-factory/blob/main/SPEC.md).

## Interfaces

**CLI.** `make` (`--seed`, `--backend`, `--turbo`, `--compress web|unity`), `validate`, `assemble` (character file → GLB, no GPU), `compress`, `interpret`, `preflight`.

**Server + browser UI.** `pip install "character-factory[server]"`, then `character-factory serve`: the `/v0` HTTP API and a gallery UI on `127.0.0.1:8400` (`--host 0.0.0.0` for other machines). `/v0/docs` documents the API.

**MCP.** `pip install "character-factory[mcp]"`, then add `character-factory mcp` to your agent's MCP config. Tools on stdio: `create_character`, `get_job`, `get_character`, `list_components`.

**Unity (6000.0+).** Run `character-factory serve`, add

```json
"com.character-factory.unity": "https://github.com/character-factory/character-factory-unity.git"
```

to `Packages/manifest.json`, then:

```sh
unity cmd cf-create --prompt "<description>" --walking true --json
unity cmd cf-verify --target "<scene-object-name>" --json
```

See the [package README](https://github.com/character-factory/character-factory-unity).

**Python.**

```python
from character_factory import Character
c = Character.load("examples/characters/freediver.char.json")
print(c.content_id, c.rig, sorted(c.textures))
```

## Hardware and install

Measured on one RTX 3090; details in [ARCHITECTURE.md §6](https://github.com/character-factory/character-factory/blob/main/ARCHITECTURE.md#6-install-and-hardware).

| | Size / time |
| --- | --- |
| Install | 5.6 GB (torch with CUDA) |
| Windows | `pip install` installs CPU torch; install torch from `https://download.pytorch.org/whl/<cuXXX>` first |
| Weights, first use | 36.4 GB: 19.3 GB interpreter + 16.0 GB base image model + 1.1 GB components. 17.1 GB with an endpoint interpreter |
| Generation, 24 GB card | bf16: bake 17.4 GiB, 137 s. Whole character 3 min 38 s with the local interpreter |
| Generation, 8 GB card | not supported |
| Assembly and consumption | no GPU — character file → GLB runs on CPU, including macOS |

`character-factory preflight` checks the install, CUDA build, and driver. `make` runs it first.

## The interpreter

The interpreter is the language model that turns your description into each component's prompt.

**Local (default).** The registry's `interpreter` component names Qwen3.5-9B (Apache-2.0, ungated — no token, no account). 19.3 GB to download, 16.9 GiB of VRAM, 78 s per description on the 3090.

**Endpoint (faster, better prompts).** Point it at any OpenAI-compatible endpoint. A hosted frontier model (an OpenAI GPT-5.6-class model in our bench) takes 14 s and writes better prompts. Configure with one of:

- environment: `CHARACTER_FACTORY_INTERPRETER_ENDPOINT`, `_MODEL`, `_API_KEY`
- `interpreter.backends` in the cache `config.json`
- `PUT /v0/interpreters/{alias}` on a running server

## Known limitations (v0.1)

- Generated textures are albedo only, no normal or material maps.
- The initial hair provider is a finite set of procedural components.
- Garment textures may have warped details and edge artifacts.
- Garment and shoe geometry are a single layer shell separated from the body.
- No facial hair.
- Clothing is one layer; outer garments are not represented.
- Spec and architecture are an initial draft and will change rapidly.

## Trust boundary

The server binds to `127.0.0.1` and does not authenticate. Do not expose it to the public internet.

## Going deeper

- [SPEC.md](https://github.com/character-factory/character-factory/blob/main/SPEC.md) — the character format.
- [ARCHITECTURE.md](https://github.com/character-factory/character-factory/blob/main/ARCHITECTURE.md) — the system.
- `/v0/docs` on a running server — the HTTP API.

## Built on

MHR (Meta, Apache-2.0), FLUX.2 Klein 4B (Black Forest Labs, Apache-2.0), Qwen3.5-9B (Apache-2.0), UnityEyes2 (MIT), GNM (Google, Apache-2.0). See [NOTICE](https://github.com/character-factory/character-factory/blob/main/NOTICE).

## Status and license

v0.1. File issues at [github.com/character-factory/character-factory/issues](https://github.com/character-factory/character-factory/issues).

Apache-2.0 ([LICENSE](https://github.com/character-factory/character-factory/blob/main/LICENSE), [NOTICE](https://github.com/character-factory/character-factory/blob/main/NOTICE)).
