"""Persistent job lifecycle: idempotency, recovery, cancellation, timeout."""

import threading
import time

import pytest

from character_factory.server.jobs import JobConflict, JobStore


def test_unkeyed_identical_submissions_create_distinct_jobs(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    first = store.submit("create", {"prompt": "one person"})
    second = store.submit("create", {"prompt": "one person"})
    assert second["id"] != first["id"]
    assert len(store.list()) == 2


def test_same_idempotency_key_and_request_returns_the_original_job(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    request = {"prompt": "one person"}
    first = store.submit("create", request, idempotency_key="request-7")
    replay = store.submit("create", request, idempotency_key="request-7")
    assert replay["id"] == first["id"]
    assert len(store.list()) == 1


def test_idempotency_key_cannot_name_two_requests(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    store.submit("create", {"prompt": "one"}, idempotency_key="request-7")
    with pytest.raises(JobConflict, match="different request"):
        store.submit("create", {"prompt": "two"}, idempotency_key="request-7")


def test_idempotency_key_is_scoped_to_exact_operation_and_target(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    store.submit(
        "assemble", {"character_id": "first"}, idempotency_key="request-7"
    )
    with pytest.raises(JobConflict, match="different request"):
        store.submit(
            "assemble", {"character_id": "second"}, idempotency_key="request-7"
        )
    with pytest.raises(JobConflict, match="different request"):
        store.submit("create", {"prompt": "one"}, idempotency_key="request-7")


def test_idempotency_key_must_be_nonempty(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    with pytest.raises(ValueError, match="non-empty"):
        store.submit("create", {"prompt": "one"}, idempotency_key="")


def test_public_job_projection_hides_precontract_capability_summaries(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    submitted = store.submit("create", {"prompt": "one"})
    store.succeed(submitted["id"], {
        "character_id": "abc",
        "actual_capabilities": {"topology": "mouth-interior"},
    })
    result = store.get(submitted["id"])["result"]
    assert result == {"character_id": "abc"}


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


def test_failure_can_expose_safe_interpreter_diagnostics(tmp_path):
    store = JobStore(tmp_path / "jobs", lambda *_: None, start_worker=False)
    submitted = store.submit("create", {"prompt": "one"})
    store.fail(
        submitted["id"], "interpreter_invalid_output",
        "interpreter failure: truncated_response (trace opaque123)",
        retryable=True, classification="truncated_response",
        trace_id="opaque123",
    )
    assert store.get(submitted["id"])["error"] == {
        "code": "interpreter_invalid_output",
        "message": "interpreter failure: truncated_response (trace opaque123)",
        "retryable": True,
        "classification": "truncated_response",
        "trace_id": "opaque123",
    }
