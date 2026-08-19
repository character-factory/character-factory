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
  character file. Drop it into any engine or viewer.

Today the repository contains the character format's reference
implementation (`character_factory.schema` — standard library only), the
published JSON Schema, example characters, and the format's test corpus:

```python
from character_factory import Character

c = Character.load("examples/characters/freediver.char.json")
print(c.content_id, c.rig, sorted(c.textures))
```

Start with [SPEC.md](SPEC.md) if you are judging the format, and
[ARCHITECTURE.md](ARCHITECTURE.md) if you are judging the system.

Licensed under Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
