"""Persistent job lifecycle: idempotency, recovery, cancellation, timeout."""

import threading
import time

import pytest

from character_factory.server.jobs import JobConflict, JobStore


def test_identical_submission_returns_the_same_job(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    first = store.submit("create", {"prompt": "one person"})
    replay = store.submit("create", {"prompt": "one person"})
    assert replay["id"] == first["id"]
    assert len(store.list()) == 1


def test_idempotency_key_cannot_name_two_requests(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    store.submit("create", {"prompt": "one"}, idempotency_key="request-7")
    with pytest.raises(JobConflict, match="different request"):
        store.submit("create", {"prompt": "two"}, idempotency_key="request-7")


def test_cancelled_job_can_only_restart_through_retry(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    submitted = store.submit("assemble", {"character_id": "abc"})
    cancelled = store.cancel(submitted["id"])
    assert cancelled["status"] == "cancelled"
    retried = store.retry(submitted["id"])
    assert retried["id"] != submitted["id"]
    assert retried["status"] == "queued"


def test_process_restart_turns_interrupted_work_into_retryable_error(tmp_path):
    root = tmp_path / "jobs"
    store = JobStore(root, lambda *_: None, start_worker=False)
    submitted = store.submit("create", {"prompt": "one"})
    store.stage(submitted["id"], "baking", 0.4)

    recovered = JobStore(root, lambda *_: None, start_worker=False)
    job = recovered.get(submitted["id"])
    assert job["status"] == "failed"
    assert job["error"] == {
        "code": "worker_restarted",
        "message": "the worker stopped before this stage completed",
        "retryable": True,
    }


def test_wedged_stage_reaches_a_documented_timeout(tmp_path):
    release = threading.Event()
    holder = {}

    def wedge(job_id, _job):
        holder["store"].stage(job_id, "baking", 0.4)
        release.wait(0.5)

    store = JobStore(
        tmp_path / "jobs", wedge,
        stage_timeout_seconds=0.05,
        heartbeat_seconds=0.01,
    )
    holder["store"] = store
    submitted = store.submit("create", {"prompt": "one"})
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        job = store.get(submitted["id"])
        if job["status"] == "failed":
            break
        time.sleep(0.01)
    release.set()
    assert job["error"]["code"] == "stage_timeout"
    assert job["error"]["retryable"] is True
    assert job["last_heartbeat"] is not None
