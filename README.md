# Character Factory

Turn a text description into a rigged, textured, realtime 3D human.

<!-- demo video embed lands here at launch -->

> **Pre-release.** This repository is being built in the open ahead of its
> first release. The character format ([SPEC.md](SPEC.md)) and the system
> design ([ARCHITECTURE.md](ARCHITECTURE.md)) are stable enough to read and
> implement against; the package below is under construction and not yet on
> PyPI.

```
pip install character-factory
character-factory make "a lean marathon runner with cropped dark hair" -o runner/
```

One sentence in; two artifacts out:

- **`character.char.json`** — the character itself: a few kilobytes of JSON
  holding body parameters, texture recipes, a semantic hair description, and
  provenance. Diff it, commit it, edit it by hand, hand it to an agent.
- **`scene.glb`** — a rigged, skinned glTF built deterministically from the
  character file: 127-joint skeleton, embedded bone-role manifest, baked
  idle clip. Drop it into any engine or viewer.

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
  time). 8 GB is not supported.
- **Assembly without generation runs anywhere** — including CPU-only
  machines and macOS: validate, assemble, and serve existing character
  files with no GPU at all.

One honest scope note: in v0.1, identity drives the face, build, and
surface form; **skeletal proportions are uniform** — every character
shares the template skeleton. Ground contact in-engine needs foot IK
either way.

Start with [SPEC.md](SPEC.md) if you are judging the format,
[ARCHITECTURE.md](ARCHITECTURE.md) if you are judging the system, and
`/v0/docs` on a running server if you are integrating against the API.

Licensed under Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
