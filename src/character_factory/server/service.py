"""The character service: every operation the HTTP and MCP layers expose.

Parity by construction (ARCHITECTURE §2): HTTP routes and MCP tools are thin
delegates over this one class, so the two surfaces cannot drift. Character
data lives in per-character directories; persisted jobs are independent,
lightweight resources. The scene file is atomically replaced so a polling
viewer never reads a torn artifact, and records expose artifact state apart
from the latest job.

Unavailable generation components and failed backends become structured,
retryable job errors — absent capability is reported, never stubbed.
"""

from __future__ import annotations

import hashlib
import base64
import binascii
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from character_factory.registry import Registry
from character_factory.schema import Character, CharacterError
from character_factory.server.jobs import JobConflict, JobNotFound, JobStore

__all__ = [
    "CharacterService", "IncompatibleCharacter", "ResourceNotFound",
    "ServiceError"
]

_ASSET_SLOTS_FILE_RE = None


def _delivery_warnings(manifest: dict, request: dict) -> list[dict]:
    warnings = []
    if request.get("garment_shells") is True:
        for slot, delivered in manifest.get("garments", {}).items():
            if slot != "garment":
                continue
            if delivered.get("render_mode") != "shell":
                warnings.append({
                    "code": "requested_geometry_not_delivered",
                    "message": (
                        f"{slot} shell was requested but the artifact uses "
                        "painted rendering"
                    ),
                    "details": {
                        "slot": slot,
                        "requested": "shell",
                        "actual": delivered.get("render_mode"),
                        "reason": delivered.get("reason"),
                    },
                })
    return warnings


class ServiceError(ValueError):
    """A client-caused failure (bad input, unknown id, conflicting state)."""


class ResourceNotFound(ServiceError):
    """A requested character, job, or artifact does not exist."""


class IncompatibleCharacter(ServiceError):
    """A stored character does not satisfy the server's current format."""


@dataclass
class CharacterRecord:
    id: str
    name: str | None
    artifact: dict
    latest_job: dict | None
    capabilities: dict
    creation: dict
    created_at: str | None = None    # ISO 8601 UTC; sidecar metadata,
    updated_at: str | None = None    # never the character document


class CharacterService:
    def __init__(self, library_dir: str | Path, registry: Registry | None = None,
                 device: str = "cuda", *, start_worker: bool = True,
                 stage_timeout_seconds: float = 3600.0):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._registry_override = registry
        self._registry_cache: Registry | None = None
        self._registry_mtime: float | None = None
        self.device = device
        # One GPU, one worker: assembly/bake jobs run single-flight.
        self._job_lock = threading.Lock()
        self.jobs = JobStore(
            self.library_dir / ".jobs",
            self._execute_job,
            start_worker=start_worker,
            stage_timeout_seconds=stage_timeout_seconds,
        )

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
            raise ResourceNotFound(f"unknown character {character_id!r}")
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

    @staticmethod
    def _load_character(directory: Path) -> Character:
        try:
            return Character.load(directory / "character.char.json")
        except (CharacterError, json.JSONDecodeError) as error:
            raise IncompatibleCharacter(
                f"stored character {directory.name!r} is incompatible with "
                f"the current character format: {error}"
            ) from error

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
        self, prompt: str, interpreter: str | None = None,
        turbo: bool = False, allow_fallback: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        """Submit a full create without running model work on the request.

        The canonical request is its default idempotency key, so replaying a
        timed-out POST returns the same job. A caller key may be supplied to
        make that intent explicit across equivalent client representations.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ServiceError("prompt must be a non-empty string")
        if interpreter is not None:
            from character_factory.interpreter.config import (
                load_interpreter_config,
            )

            try:
                load_interpreter_config(alias=interpreter)
            except ValueError as error:
                raise ServiceError(str(error)) from error
        try:
            return self.jobs.submit(
                "create",
                {
                    "prompt": prompt,
                    "interpreter": interpreter or "default",
                    "turbo": bool(turbo),
                    "allow_fallback": bool(allow_fallback),
                },
                idempotency_key=idempotency_key,
            )
        except JobConflict as error:
            raise ServiceError(str(error)) from error

    def regenerate(self, character_id: str,
                   turbo: bool = False) -> dict:
        """Explicitly submit a fresh bake + assembly job."""
        self._dir(character_id)
        return self._submit_character_job(
            "bake", character_id, {"turbo": bool(turbo)}
        )

    def rebuild(
        self, character_id: str, *, stage: str = "assemble",
        turbo: bool = False, garment_shells: bool | None = None,
    ) -> dict:
        if stage not in {"assemble", "bake"}:
            raise ServiceError(f"unknown rebuild stage {stage!r}")
        request = {"turbo": bool(turbo)} if stage == "bake" else {
            "garment_shells": garment_shells
        }
        return self._submit_character_job(stage, character_id, request)

    def _submit_character_job(
        self, operation: str, character_id: str, request: dict
    ) -> dict:
        directory = self._dir(character_id)
        job = self.jobs.submit(
            operation,
            {"character_id": character_id, **request},
            force_new=True,
        )
        state = self._state(directory)
        state.update(
            status="queued", detail="waiting for the worker",
            active_job_id=job["id"], last_job_id=job["id"],
        )
        self._write_state(directory, state)
        return job

    def _bake(self, character_id: str, turbo: bool = False) -> None:
        from character_factory.textures import bake

        directory = self.library_dir / character_id
        with self._job_lock:
            character = Character.load(directory / "character.char.json")
            baked = bake(
                character, directory / "assets",
                registry=self.registry, device=self.device, turbo=turbo,
            )
            baked.character.save(directory / "character.char.json")

    def _execute_job(self, job_id: str, job: dict) -> None:
        from character_factory.api import create
        from character_factory.interpreter.backend import InterpreterError
        from character_factory.preflight import PreflightError
        from character_factory.registry import ComponentNotPublished

        request = job["request"]
        operation = job["operation"]
        character_id = request.get("character_id")
        try:
            if operation == "create":
                if not self.jobs.stage(
                    job_id, "creating", 0.05,
                    "interpreting the prompt and creating the character recipe",
                ):
                    return
                requested = request.get("interpreter", "default")
                with self._job_lock:
                    result = create(
                        request["prompt"], registry=self.registry,
                        device=self.device,
                        interpreter=None if requested == "default" else requested,
                        allow_fallback=request.get("allow_fallback", False),
                        _with_report=True,
                    )
                if not self.jobs.active(job_id):
                    return
                record = self.store_character(result.character.to_document())
                character_id = record.id
                self.jobs.update(job_id, **result.interpretation)
                directory = self.library_dir / character_id
                state = self._state(directory)
                state.update(
                    status="baking", detail=None, active_job_id=job_id,
                    last_job_id=job_id,
                    **result.interpretation,
                )
                self._write_state(directory, state)
            if operation in {"create", "bake"}:
                if not self.jobs.stage(
                    job_id, "baking", 0.35, "generating texture assets"
                ):
                    return
                self._bake(character_id, turbo=bool(request.get("turbo", False)))
            if not self.jobs.active(job_id):
                return
            if not self.jobs.stage(
                job_id, "assembling", 0.8, "building the runtime GLB"
            ):
                return
            record = self.assemble(
                character_id,
                garment_shells=request.get("garment_shells"),
            )
            manifest = self.manifest(character_id)
            job_state = self.jobs.internal(job_id)
            warnings = list(job_state.get("warnings", []))
            warnings.extend(_delivery_warnings(manifest, request))
            self.jobs.update(job_id, warnings=warnings)
            state = self._state(self.library_dir / character_id)
            state.update(active_job_id=None, last_job_id=job_id)
            self._write_state(self.library_dir / character_id, state)
            self.jobs.succeed(
                job_id,
                {
                    "character_id": character_id,
                    "revision": record.artifact["revision"],
                    "actual_capabilities": {
                        "topology": manifest["topology"],
                        "humanoid": bool(manifest.get("humanoid_map", {}).get("map")),
                        "facial_animation": {
                            "morph_count": manifest["expression_morphs"]["count"],
                            "morph_names": manifest["expression_morphs"]["names"],
                        },
                    },
                },
            )
        except InterpreterError as error:
            self.jobs.fail(
                job_id, "interpreter_unavailable", str(error), retryable=True
            )
        except (ComponentNotPublished, FileNotFoundError, PreflightError) as error:
            self.jobs.fail(
                job_id, "generation_unavailable", str(error), retryable=True
            )
        except Exception as error:  # every submitted job becomes terminal
            self.jobs.fail(job_id, f"{operation}_failed", str(error), retryable=True)
        finally:
            if character_id:
                directory = self.library_dir / character_id
                if directory.is_dir():
                    state = self._state(directory)
                    current = self.jobs.get(job_id)
                    if current["status"] in {"failed", "cancelling", "cancelled"}:
                        artifact_ready = (directory / "scene.glb").is_file()
                        state.update(
                            status=("error" if current["status"] == "failed"
                                    else "built" if artifact_ready else "stored"),
                            detail=(current["error"]["message"]
                                    if current["status"] == "failed"
                                    else "job cancelled"),
                            active_job_id=None,
                            last_job_id=job_id,
                        )
                        self._write_state(directory, state)

    def get_job(self, job_id: str) -> dict:
        try:
            return self.jobs.get(job_id)
        except JobNotFound as error:
            raise ResourceNotFound(f"unknown job {job_id!r}") from error

    def list_jobs(self) -> list[dict]:
        return self.jobs.list()

    def cancel_job(self, job_id: str) -> dict:
        try:
            return self.jobs.cancel(job_id)
        except JobNotFound as error:
            raise ResourceNotFound(f"unknown job {job_id!r}") from error

    def retry_job(self, job_id: str) -> dict:
        try:
            return self.jobs.retry(job_id)
        except (JobNotFound, JobConflict) as error:
            if isinstance(error, JobNotFound):
                raise ResourceNotFound(f"unknown job {job_id!r}") from error
            raise ServiceError(str(error)) from error

    def _scan_library(self) -> tuple[list[CharacterRecord], list[str]]:
        """Return current records and quarantine incompatible directories.

        A stale or damaged on-disk document must not make health, listing, or
        unrelated characters unavailable. The bytes are left untouched for
        an operator to inspect or remove deliberately.
        """
        records = []
        incompatible = []
        for path in sorted(self.library_dir.iterdir()):
            if path.is_dir() and (path / "character.char.json").is_file():
                try:
                    records.append(self.get(path.name))
                except IncompatibleCharacter:
                    incompatible.append(path.name)
        records.sort(key=lambda r: r.created_at or "", reverse=True)
        return records, incompatible

    def list(self) -> list[CharacterRecord]:
        """Every compatible stored character, newest first."""
        records, _ = self._scan_library()
        return records

    def list_page(self, *, limit: int = 50, cursor: str | None = None) -> dict:
        if not 1 <= limit <= 100:
            raise ServiceError("limit must be in [1, 100]")
        records = sorted(
            self.list(), key=lambda record: (record.created_at or "", record.id),
            reverse=True,
        )
        if cursor:
            try:
                padding = "=" * (-len(cursor) % 4)
                decoded = json.loads(
                    base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
                )
                boundary = (decoded["created_at"], decoded["id"])
            except (ValueError, KeyError, UnicodeDecodeError, binascii.Error) as error:
                raise ServiceError("invalid character-list cursor") from error
            records = [
                record for record in records
                if (record.created_at or "", record.id) < boundary
            ]
        page = records[:limit]
        next_cursor = None
        if len(records) > limit:
            tail = page[-1]
            payload = json.dumps(
                {"created_at": tail.created_at or "", "id": tail.id},
                separators=(",", ":"),
            ).encode("utf-8")
            next_cursor = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        return {"items": page, "next_cursor": next_cursor}

    def get(self, character_id: str) -> CharacterRecord:
        directory = self._dir(character_id)
        character = self._load_character(directory)
        state = self._state(directory)
        scene = directory / "scene.glb"
        artifact = dict(state.get("artifact") or {})
        artifact.setdefault("available", scene.is_file())
        artifact.setdefault("revision", state.get("revision", 0))
        if scene.is_file():
            if artifact.get("bytes") is None or artifact.get("sha256") is None:
                data = scene.read_bytes()
                artifact.update(
                    bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
                )
            if artifact.get("built_at") is None:
                import datetime

                artifact["built_at"] = datetime.datetime.fromtimestamp(
                    scene.stat().st_mtime, datetime.timezone.utc
                ).isoformat(timespec="seconds")
        else:
            artifact.update(bytes=None, sha256=None, built_at=None)
        latest_job = None
        latest_job_id = state.get("active_job_id") or state.get("last_job_id")
        if latest_job_id:
            try:
                latest_job = self.jobs.get(latest_job_id)
            except JobNotFound:
                latest_job = None
        return CharacterRecord(
            id=character_id,
            name=character.name,
            artifact=artifact,
            latest_job=latest_job,
            capabilities={
                "topology": "mouth-interior",
                "humanoid": True,
                "facial_animation": {
                    "morph_count": 72,
                    "morph_names": [f"facs_{index:02d}" for index in range(72)],
                },
            },
            creation={
                "requested_interpreter": state.get("requested_interpreter"),
                "actual_interpreter": state.get("actual_interpreter"),
                "fallback_reason": state.get("fallback_reason"),
                "warnings": state.get("warnings", []),
            },
            created_at=state.get("created_at"),
            updated_at=state.get("updated_at"),
        )

    def document(self, character_id: str) -> dict:
        return self._load_character(self._dir(character_id)).to_document()

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
            raise ResourceNotFound(
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
        if len(data) > 16 * 1024 * 1024:
            raise ServiceError("PNG asset exceeds the 16 MiB upload limit")
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ServiceError("asset body must be a PNG image")
        character = self._load_character(directory)
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
            raise ResourceNotFound(
                f"no {slot} asset for character {character_id!r}"
            )
        return path

    def scene_path(self, character_id: str) -> Path:
        path = self._dir(character_id) / "scene.glb"
        if not path.is_file():
            raise ResourceNotFound(
                f"character {character_id!r} has no built scene yet — "
                f"run assemble first"
            )
        return path

    def assemble(self, character_id: str,
                 garment_shells: bool | None = None) -> CharacterRecord:
        """Build the rigged scene from stored assets. Single-flight; the scene
        file is replaced atomically and the revision bumped on success.
        `garment_shells` overrides the configured feature gate for this
        build only (the review app's shell-vs-painted comparison)."""
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
                        garment_shells=garment_shells,
                    )
                    os.replace(built, directory / "scene.glb")
            except (AssetError, FileNotFoundError, ValueError) as error:
                state.update(status="error", detail=str(error))
                self._write_state(directory, state)
                raise ServiceError(str(error)) from error
            self._write_thumbnail(directory)
            revision = state.get("revision", 0) + 1
            scene_data = (directory / "scene.glb").read_bytes()
            state.update(
                status="built", detail=None, revision=revision,
                artifact={
                    "available": True,
                    "revision": revision,
                    "bytes": len(scene_data),
                    "sha256": hashlib.sha256(scene_data).hexdigest(),
                    "built_at": self._now(),
                },
            )
            self._write_state(directory, state)
        return self.get(character_id)

    def _write_thumbnail(self, directory: Path) -> None:
        """Render the gallery portrait from the just-built scene. Best-effort:
        a thumbnail failure must never fail an assembly."""
        try:
            from character_factory.assembly.thumbnail import render_thumbnail

            png = render_thumbnail((directory / "scene.glb").read_bytes())
            (directory / "thumb.png").write_bytes(png)
        except Exception:  # noqa: BLE001 - cosmetic artifact, never fatal
            pass

    def thumbnail_path(self, character_id: str) -> Path:
        path = self._dir(character_id) / "thumb.png"
        if not path.is_file():
            raise ResourceNotFound(
                f"character {character_id} has no thumbnail yet"
            )
        return path

    def health(self) -> dict:
        records, incompatible = self._scan_library()
        report: dict = {
            "status": "ok",
            "characters": len(records),
            "incompatible_characters": len(incompatible),
            "jobs": len(self.list_jobs()),
        }
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
