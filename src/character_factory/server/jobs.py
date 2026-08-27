"""Persistent single-worker jobs for the HTTP and MCP surfaces.

Submission is cheap. Callers opt into retry-safe submission with an explicit
idempotency key; unkeyed submissions always create new work. Job state is a
small JSON resource, while character documents and artifacts remain separate
resources. A process restart recovers queued work and converts interrupted
stages to a retryable terminal error instead of leaving them apparently active
forever.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

__all__ = ["JobConflict", "JobNotFound", "JobStore"]

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
ACTIVE = frozenset({"queued", "running", "cancelling"})


class JobNotFound(KeyError):
    pass


class JobConflict(ValueError):
    pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def _parse_time(value: str) -> float:
    return datetime.datetime.fromisoformat(value).timestamp()


class JobStore:
    def __init__(
        self,
        root: Path,
        handler,
        *,
        stage_timeout_seconds: float = 3600.0,
        heartbeat_seconds: float = 5.0,
        start_worker: bool = True,
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._handler = handler
        self._timeout = stage_timeout_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._recover()
        if start_worker:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _fingerprint(operation: str, request: dict) -> str:
        body = json.dumps(
            {"operation": operation, "request": request},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    def _path(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise JobNotFound(job_id)
        path = self.root / f"{job_id}.json"
        if not path.is_file():
            raise JobNotFound(job_id)
        return path

    def _read(self, job_id: str) -> dict:
        return json.loads(self._path(job_id).read_text(encoding="utf-8"))

    def _write(self, job: dict) -> None:
        job["updated_at"] = _now()
        with tempfile.NamedTemporaryFile(
            "w", dir=self.root, suffix=".tmp", delete=False, encoding="utf-8"
        ) as output:
            json.dump(job, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(output.name, self.root / f"{job['id']}.json")

    def _recover(self) -> None:
        for path in sorted(self.root.glob("*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("status") == "queued":
                self._queue.put(job["id"])
            elif job.get("status") in {"running", "cancelling"}:
                job.update(
                    status="failed",
                    stage="failed",
                    progress=job.get("progress", 0.0),
                    error={
                        "code": "worker_restarted",
                        "message": "the worker stopped before this stage completed",
                        "retryable": True,
                    },
                    finished_at=_now(),
                )
                self._write(job)

    def submit(
        self,
        operation: str,
        request: dict,
        *,
        idempotency_key: str | None = None,
        force_new: bool = False,
    ) -> dict:
        fingerprint = self._fingerprint(operation, request)
        if force_new:
            job_id = uuid.uuid4().hex[:24]
        elif idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                raise ValueError("Idempotency-Key must be a non-empty string")
            if len(idempotency_key) > 255:
                raise ValueError("Idempotency-Key must not exceed 255 characters")
            job_id = hashlib.sha256(
                f"idempotency:{idempotency_key}".encode("utf-8")
            ).hexdigest()[:24]
        else:
            job_id = uuid.uuid4().hex[:24]
        with self._lock:
            path = self.root / f"{job_id}.json"
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("request_fingerprint") != fingerprint:
                    raise JobConflict(
                        "the Idempotency-Key was already used for a different request"
                    )
                return self.public(existing)
            now = _now()
            job = {
                "id": job_id,
                "operation": operation,
                "request": request,
                "request_fingerprint": fingerprint,
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "detail": "waiting for the worker",
                "error": None,
                "result": None,
                "requested_interpreter": request.get("interpreter", "default"),
                "actual_interpreter": None,
                "fallback_reason": None,
                "warnings": [],
                "cancel_requested": False,
                "created_at": now,
                "updated_at": now,
                "stage_started_at": now,
                "last_heartbeat": None,
                "finished_at": None,
            }
            self._write(job)
            self._queue.put(job_id)
            return self.public(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            return self.public(self._read(job_id))

    def internal(self, job_id: str) -> dict:
        with self._lock:
            return self._read(job_id)

    def public(self, job: dict) -> dict:
        result = {
            key: job.get(key)
            for key in (
                "id", "operation", "status", "stage", "progress", "detail",
                "error", "result", "requested_interpreter",
                "actual_interpreter", "fallback_reason", "warnings",
                "created_at", "updated_at", "stage_started_at",
                "last_heartbeat", "finished_at",
            )
        }
        if isinstance(result.get("result"), dict):
            # The manifest is the sole authority for mandatory export
            # properties. Keep old on-disk job payloads private if a
            # pre-release server wrote duplicated capability summaries.
            result["result"] = dict(result["result"])
            for key in (
                "capabilities", "requested_capabilities", "actual_capabilities"
            ):
                result["result"].pop(key, None)
        if job.get("status") == "queued":
            queued = sorted(
                (
                    (candidate.get("created_at", ""), candidate["id"])
                    for candidate in self._all_internal()
                    if candidate.get("status") == "queued"
                )
            )
            result["queue_position"] = next(
                (index + 1 for index, (_, value) in enumerate(queued)
                 if value == job["id"]),
                None,
            )
        else:
            result["queue_position"] = None
        return result

    def _all_internal(self) -> list[dict]:
        jobs = []
        for path in self.root.glob("*.json"):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return jobs

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(
                self._all_internal(), key=lambda item: item.get("created_at", ""),
                reverse=True,
            )
            return [self.public(job) for job in jobs]

    def update(self, job_id: str, **values) -> dict:
        with self._lock:
            job = self._read(job_id)
            job.update(values)
            self._write(job)
            return self.public(job)

    def stage(
        self, job_id: str, stage: str, progress: float, detail: str | None = None
    ) -> bool:
        with self._lock:
            job = self._read(job_id)
            if job.get("status") in TERMINAL or job.get("cancel_requested"):
                return False
            now = _now()
            job.update(
                status="running",
                stage=stage,
                progress=float(progress),
                detail=detail,
                stage_started_at=now,
                last_heartbeat=now,
            )
            self._write(job)
            return True

    def active(self, job_id: str) -> bool:
        with self._lock:
            job = self._read(job_id)
            return job.get("status") not in TERMINAL and not job.get(
                "cancel_requested"
            )

    def succeed(self, job_id: str, result: dict) -> dict:
        return self.update(
            job_id,
            status="succeeded",
            stage="complete",
            progress=1.0,
            detail=None,
            error=None,
            result=result,
            finished_at=_now(),
        )

    def fail(
        self, job_id: str, code: str, message: str, *, retryable: bool
    ) -> dict:
        return self.update(
            job_id,
            status="failed",
            stage="failed",
            detail=message,
            error={"code": code, "message": message, "retryable": retryable},
            finished_at=_now(),
        )

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._read(job_id)
            if job.get("status") in TERMINAL:
                return self.public(job)
            if job.get("status") == "queued":
                job.update(
                    status="cancelled", stage="cancelled", detail=None,
                    cancel_requested=True, finished_at=_now(),
                )
            else:
                job.update(
                    status="cancelling", detail="cancellation requested",
                    cancel_requested=True,
                )
            self._write(job)
            return self.public(job)

    def retry(self, job_id: str) -> dict:
        with self._lock:
            job = self._read(job_id)
            if job.get("status") not in {"failed", "cancelled"}:
                raise JobConflict("only failed or cancelled jobs can be retried")
            return self.submit(
                job["operation"], job["request"], force_new=True
            )

    def _heartbeat(self, job_id: str, stopped: threading.Event) -> None:
        while not stopped.wait(self._heartbeat_seconds):
            with self._lock:
                try:
                    job = self._read(job_id)
                except JobNotFound:
                    return
                if job.get("status") not in {"running", "cancelling"}:
                    return
                started = job.get("stage_started_at")
                if started and time.time() - _parse_time(started) > self._timeout:
                    self.fail(
                        job_id,
                        "stage_timeout",
                        f"stage {job.get('stage')!r} exceeded its time limit",
                        retryable=True,
                    )
                    return
                job["last_heartbeat"] = _now()
                self._write(job)

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    job = self._read(job_id)
                    if job.get("status") != "queued":
                        continue
                stopped = threading.Event()
                heartbeat = threading.Thread(
                    target=self._heartbeat, args=(job_id, stopped), daemon=True
                )
                heartbeat.start()
                try:
                    self._handler(job_id, self.internal(job_id))
                except Exception as error:  # the resource must become terminal
                    if self.active(job_id):
                        self.fail(
                            job_id, "job_failed", str(error), retryable=True
                        )
                finally:
                    stopped.set()
                    with self._lock:
                        current = self._read(job_id)
                        if current.get("status") == "cancelling":
                            current.update(
                                status="cancelled", stage="cancelled",
                                detail=None, finished_at=_now(),
                            )
                            self._write(current)
            finally:
                self._queue.task_done()
