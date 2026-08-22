"""The character service: every operation the HTTP and MCP layers expose.

Parity by construction (ARCHITECTURE §2): HTTP routes and MCP tools are thin
delegates over this one class, so the two surfaces cannot drift. All state
lives on disk in per-character directories — the character file is the
database; a sidecar ``state.json`` records job status; the scene file is
atomically replaced so a polling viewer never reads a torn artifact.

Generation operations (`create`/`bake`/`make`) surface a clear
"not available yet" error while their components are unpublished — absent
capability is reported, never stubbed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from character_factory.registry import Registry
from character_factory.schema import Character, CharacterError

__all__ = ["CharacterService", "ServiceError", "NotAvailable"]

_ASSET_SLOTS_FILE_RE = None


class ServiceError(ValueError):
    """A client-caused failure (bad input, unknown id, conflicting state)."""


class NotAvailable(ServiceError):
    """The operation needs components that are not published yet."""


@dataclass
class CharacterRecord:
    id: str
    name: str | None
    status: str
    detail: str | None
    revision: int
    has_scene: bool
    created_at: str | None = None    # ISO 8601 UTC; sidecar metadata,
    updated_at: str | None = None    # never the character document


class CharacterService:
    def __init__(self, library_dir: str | Path, registry: Registry | None = None,
                 device: str = "cuda"):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._registry_override = registry
        self._registry_cache: Registry | None = None
        self._registry_mtime: float | None = None
        self.device = device
        # One GPU, one worker: assembly/bake jobs run single-flight.
        self._job_lock = threading.Lock()

    @property
    def registry(self) -> Registry:
        """The live registry: a restaged local index is picked up on its
        next use, without a server restart — the components view and new
        creates always reflect current state (per-character provenance
        keeps whatever was resolved at its create)."""
        if self._registry_override is not None:
            return self._registry_override
        from character_factory.registry import cache_dir

        path = cache_dir() / "registry.json"
        mtime = path.stat().st_mtime if path.is_file() else None
        if self._registry_cache is None or mtime != self._registry_mtime:
            self._registry_cache = Registry.default()
            self._registry_mtime = mtime
        return self._registry_cache

    # -- paths and state -----------------------------------------------------

    def _dir(self, character_id: str) -> Path:
        if not character_id.replace("-", "").isalnum():
            raise ServiceError(f"malformed character id {character_id!r}")
        path = self.library_dir / character_id
        if not path.is_dir():
            raise ServiceError(f"unknown character {character_id!r}")
        return path

    @staticmethod
    def _now() -> str:
        import datetime

        return datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )

    def _state(self, directory: Path) -> dict:
        path = directory / "state.json"
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
        else:
            state = {"status": "stored", "detail": None, "revision": 0}
        if "created_at" not in state:
            # Records from before timestamps existed: the character file's
            # mtime is the best available creation time.
            import datetime

            source = directory / "character.char.json"
            if source.is_file():
                state["created_at"] = datetime.datetime.fromtimestamp(
                    source.stat().st_mtime, datetime.timezone.utc
                ).isoformat(timespec="seconds")
            else:
                state["created_at"] = self._now()
        return state

    def _write_state(self, directory: Path, state: dict) -> None:
        state["updated_at"] = self._now()
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, suffix=".tmp", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(state, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, directory / "state.json")

    # -- operations -----------------------------------------------------------

    def store_character(self, document: dict) -> CharacterRecord:
        """Validate and store a character document; the id is derived from
        the content ID, so storing the same character twice is idempotent."""
        try:
            character = Character.from_document(document)
        except CharacterError as error:
            raise ServiceError(str(error)) from error
        character_id = character.content_id[:16]
        directory = self.library_dir / character_id
        directory.mkdir(parents=True, exist_ok=True)
        character.save(directory / "character.char.json")
        state = self._state(directory)
        self._write_state(directory, state)
        return self.get(character_id)

    def create_from_prompt(
        self, prompt: str, interpreter: str | None = None
    ) -> CharacterRecord:
        """Full make: create the character file now (interpretation +
        deterministic identity), then bake + assemble in a background job.
        `interpreter` selects a configured backend by alias (the create
        UI's model selector); None uses the configured default. The create
        stage runs GPU models too (the interpreter, the identity encoder),
        so it takes the same job lock as bake/assemble: one GPU stage at a
        time, ever — concurrent requests queue."""
        from character_factory.api import create
        from character_factory.registry import ComponentNotPublished

        if interpreter is not None:
            from character_factory.interpreter.config import (
                load_interpreter_config,
            )

            try:
                load_interpreter_config(alias=interpreter)
            except ValueError as error:
                raise ServiceError(str(error)) from error
        try:
            with self._job_lock:
                character = create(
                    prompt, registry=self.registry, device=self.device,
                    interpreter=interpreter,
                )
        except (ComponentNotPublished, FileNotFoundError) as error:
            raise NotAvailable(
                f"text-to-character needs generation components that are not "
                f"available here yet: {error}"
            ) from error
        record = self.store_character(character.to_document())
        directory = self.library_dir / record.id
        state = self._state(directory)
        state.update(status="queued", detail="waiting for the generation worker")
        self._write_state(directory, state)
        threading.Thread(
            target=self._run_generation, args=(record.id,), daemon=True
        ).start()
        return self.get(record.id)

    def regenerate(self, character_id: str) -> CharacterRecord:
        """Re-run bake + assemble for a stored character (its recipes are
        unchanged; component updates and template changes take effect)."""
        directory = self._dir(character_id)
        state = self._state(directory)
        state.update(status="queued", detail="waiting for the generation worker")
        self._write_state(directory, state)
        threading.Thread(
            target=self._run_generation, args=(character_id,), daemon=True
        ).start()
        return self.get(character_id)

    def _run_generation(self, character_id: str) -> None:
        from character_factory.textures import bake

        directory = self.library_dir / character_id
        with self._job_lock:
            state = self._state(directory)
            try:
                state.update(status="baking", detail=None)
                self._write_state(directory, state)
                character = Character.load(directory / "character.char.json")
                baked = bake(
                    character, directory / "assets",
                    registry=self.registry, device=self.device,
                )
                baked.character.save(directory / "character.char.json")
            except Exception as error:  # noqa: BLE001 - job state must record it
                state.update(status="error", detail=f"bake failed: {error}")
                self._write_state(directory, state)
                return
        try:
            self.assemble(character_id)
        except ServiceError:
            pass  # assemble already recorded the error state

    def list(self) -> list[CharacterRecord]:
        """Every stored character, newest first."""
        records = []
        for path in sorted(self.library_dir.iterdir()):
            if path.is_dir() and (path / "character.char.json").is_file():
                records.append(self.get(path.name))
        records.sort(key=lambda r: r.created_at or "", reverse=True)
        return records

    def get(self, character_id: str) -> CharacterRecord:
        directory = self._dir(character_id)
        character = Character.load(directory / "character.char.json")
        state = self._state(directory)
        return CharacterRecord(
            id=character_id,
            name=character.name,
            status=state["status"],
            detail=state.get("detail"),
            revision=state.get("revision", 0),
            has_scene=(directory / "scene.glb").is_file(),
            created_at=state.get("created_at"),
            updated_at=state.get("updated_at"),
        )

    def document(self, character_id: str) -> dict:
        return Character.load(
            self._dir(character_id) / "character.char.json"
        ).to_document()

    def delete(self, character_id: str) -> None:
        shutil.rmtree(self._dir(character_id))

    def validate(self, document: dict, strict: bool = False) -> dict:
        from character_factory.schema import validate_document

        report = validate_document(document, strict=strict)
        return {
            "ok": report.ok,
            "errors": [str(issue) for issue in report.errors],
            "warnings": [str(issue) for issue in report.warnings],
        }

    def manifest(self, character_id: str) -> dict:
        """The scene's embedded export manifest — a projection of the same
        bytes carried in the GLB's asset extras (one authored source, two
        deliveries)."""
        from character_factory.assembly.gltf import parse_glb

        scene = self._dir(character_id) / "scene.glb"
        if not scene.is_file():
            raise ServiceError(
                f"character {character_id!r} has no built scene yet"
            )
        gltf, _ = parse_glb(scene.read_bytes())
        manifest = gltf.get("asset", {}).get("extras")
        if not manifest:
            raise ServiceError(
                "this scene predates the embedded manifest; rebuild it "
                '(rebuild {"from": "assemble"}) to get one'
            )
        return manifest

    def interpreters(self) -> list[dict]:
        """Selectable interpreter backends: aliases and kinds only —
        model identity is local configuration and never leaves it."""
        from character_factory.interpreter.config import available_backends

        return available_backends()

    def components(self) -> list[dict]:
        # `active` marks the version that unpinned resolution picks today —
        # the one a new create uses. Older versions remain listed because
        # stored recipes may still pin them.
        index = self.registry.index
        active: set[str] = set()
        for name in {entry.name for entry in index.entries}:
            try:
                active.add(index.get(name).ref)
            except Exception:  # noqa: BLE001 - no compatible version: none active
                pass
        rows = []
        for entry in index.entries:
            rows.append(
                {
                    "name": entry.name,
                    "version": str(entry.version),
                    "kind": entry.kind,
                    "slot": entry.slot,
                    "published": entry.source is not None,
                    "active": entry.ref in active,
                    "vocabulary": entry.vocabulary,
                }
            )
        return rows

    # -- assets and building ---------------------------------------------------

    def put_asset(self, character_id: str, slot: str, data: bytes) -> dict:
        from character_factory.schema import vocab

        directory = self._dir(character_id)
        if slot not in vocab.ALL_SLOTS:
            hint = vocab.SLOT_MISTAKES.get(slot)
            raise ServiceError(
                f"unknown slot {slot!r}"
                + (f" — texture slots are singular; did you mean {hint!r}?" if hint else "")
            )
        character = Character.load(directory / "character.char.json")
        pinned = character.asset_maps().get(slot, {}).get("albedo")
        digest = hashlib.sha256(data).hexdigest()
        if pinned is not None and digest != pinned["sha256"]:
            raise ServiceError(
                f"uploaded {slot} asset does not match the character's pinned "
                f"hash — refusing to silently substitute (SPEC.md §8)"
            )
        assets_dir = directory / "assets"
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / f"{slot}.png").write_bytes(data)
        return {"slot": slot, "sha256": digest, "bytes": len(data)}

    def asset_path(self, character_id: str, slot: str) -> Path:
        path = self._dir(character_id) / "assets" / f"{slot}.png"
        if not path.is_file():
            raise ServiceError(f"no {slot} asset for character {character_id!r}")
        return path

    def scene_path(self, character_id: str) -> Path:
        path = self._dir(character_id) / "scene.glb"
        if not path.is_file():
            raise ServiceError(
                f"character {character_id!r} has no built scene yet — "
                f"run assemble first"
            )
        return path

    def assemble(self, character_id: str) -> CharacterRecord:
        """Build the rigged scene from stored assets. Single-flight; the scene
        file is replaced atomically and the revision bumped on success."""
        from character_factory.api import AssetError, assemble

        directory = self._dir(character_id)
        with self._job_lock:
            state = self._state(directory)
            state.update(status="assembling", detail=None)
            self._write_state(directory, state)
            try:
                with tempfile.TemporaryDirectory(dir=directory) as tmp:
                    built = assemble(
                        directory / "character.char.json",
                        directory / "assets",
                        Path(tmp) / "scene.glb",
                        registry=self.registry,
                    )
                    os.replace(built, directory / "scene.glb")
            except (AssetError, FileNotFoundError, ValueError) as error:
                state.update(status="error", detail=str(error))
                self._write_state(directory, state)
                raise ServiceError(str(error)) from error
            state.update(
                status="built", detail=None, revision=state.get("revision", 0) + 1
            )
            self._write_state(directory, state)
        return self.get(character_id)

    def health(self) -> dict:
        report: dict = {"library": str(self.library_dir), "characters": len(self.list())}
        try:
            import torch

            report["cuda"] = torch.cuda.is_available()
            if report["cuda"]:
                free, total = torch.cuda.mem_get_info()
                report["vram_free_gb"] = round(free / 2**30, 1)
                report["vram_total_gb"] = round(total / 2**30, 1)
        except Exception:  # noqa: BLE001 - health must never fail
            report["cuda"] = False
        return report
