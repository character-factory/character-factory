"""Registry: index validation, resolution, constraints, and integrity."""

import copy
import hashlib
import json
import re

import pytest

from character_factory.registry import (
    ComponentNotPublished,
    IntegrityError,
    Registry,
    RegistryError,
    RegistryIndex,
    cache_dir,
    parse_ref,
)
from character_factory.registry.store import ensure_component
from character_factory.registry.versions import Version, schema_range_allows


def make_index(components) -> dict:
    return {
        "format": "character-factory/registry",
        "registry_version": "0.1",
        "components": components,
    }


def adapter(name, version, slot, *, schema=">=0.1 <1.0", constraints=None, source=None,
            artifacts=None):
    entry = {
        "name": name,
        "version": version,
        "kind": "texture-adapter",
        "slot": slot,
        "requires": {"base_model": "flux2-klein-4b", "schema": schema},
        "artifacts": artifacts or [],
        "source": source,
    }
    if constraints:
        entry["constraints"] = constraints
    return entry


# --- versions ---------------------------------------------------------------


def test_version_ordering():
    assert Version("0.2.0") > Version("0.1.9")
    assert Version("1.0.0") > Version("0.99.99")
    with pytest.raises(ValueError):
        Version("1.0")


@pytest.mark.parametrize(
    ("range_text", "schema", "expected"),
    [
        (">=0.1 <1.0", "0.1", True),
        (">=0.1 <1.0", "0.2", True),
        (">=0.2 <1.0", "0.1", False),
        (">=0.1 <1.0", "1.0", False),
        ("==0.1", "0.1", True),
        ("==0.1", "0.2", False),
    ],
)
def test_schema_ranges(range_text, schema, expected):
    assert schema_range_allows(range_text, schema) is expected


def test_bad_schema_range_rejected():
    with pytest.raises(ValueError):
        schema_range_allows("~0.1", "0.1")


# --- snapshot ----------------------------------------------------------------


def test_packaged_snapshot_is_valid_and_complete():
    registry = Registry.default()
    names = {entry.name for entry in registry.index.entries}
    assert {"interpreter", "make-figure", "make-skin", "make-eye",
            "make-garment", "make-shoe", "make-wig",
            "body-rig", "flux2-klein-4b", "assembly-assets"} <= names


def test_snapshot_shoe_maker_declares_vocabulary():
    # The packaged snapshot specifically — a refreshed local index may
    # carry newer staged declarations.
    from character_factory.registry import RegistryIndex, _snapshot_document

    registry = Registry(RegistryIndex(_snapshot_document()))
    assert registry.vocabulary_for("make-shoe") == {
        "styles": ["below_ankle", "high_top", "ankle_boot", "mid_boot",
                   "tall_boot"]
    }
    assert registry.vocabulary_for("make-skin") == {}
    assert registry.get("make-shoe").map == "albedo"


def test_snapshot_body_rig_is_hash_pinned():
    # The packaged snapshot specifically (a refreshed local index may
    # differ): the upstream body model file is pinned by content hash.
    from character_factory.registry import RegistryIndex, _snapshot_document

    entry = Registry(RegistryIndex(_snapshot_document())).get("body-rig")
    paths = {a["path"]: a for a in entry.artifacts}
    assert paths["mhr_model.pt"]["bytes"] == 696110248
    assert len(paths["mhr_model.pt"]["sha256"]) == 64


def test_snapshot_base_models_are_pinned_upstream():
    # Both image base models are fetched from their upstream repositories
    # at an exact revision, every file the pipeline opens hash-pinned, and
    # the identity component's declared base is one of them.
    from character_factory.registry import RegistryIndex, _snapshot_document

    registry = Registry(RegistryIndex(_snapshot_document()))
    for name in ("flux2-klein-base-4b", "flux2-klein-4b"):
        entry = registry.get(name)
        assert len(entry.document["source"]["revision"]) == 40
        paths = {a["path"] for a in entry.artifacts}
        assert {"model_index.json", "transformer/config.json",
                "text_encoder/config.json", "tokenizer/tokenizer.json",
                "vae/config.json"} <= paths
        assert all(len(a["sha256"]) == 64 and a["bytes"] > 0
                   for a in entry.artifacts)
        assert entry.document["upstream"]["gated"] is False
    base = registry.get("make-figure").document["requires"]["base_model"]
    assert registry.get(base).kind == "base-model"


def test_unpublished_component_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path))
    # The hair engine's weights are not yet published: declared with no
    # artifacts, so the failure names itself.
    with pytest.raises(ComponentNotPublished, match="declares no artifacts"):
        Registry.default().ensure("make-wig")


def test_snapshot_interpreter_default_model_is_pinned_upstream():
    # The default interpreter model is registry data: an exact upstream
    # revision with every artifact hash-pinned, fetched anonymously — the
    # install-and-run promise depends on the upstream not being gated, and
    # the entry records that where a reader looks first.
    entry = Registry.default().get("interpreter")
    assert entry.document["source"]["hf_repo"]
    assert len(entry.document["source"]["revision"]) == 40
    paths = {a["path"] for a in entry.artifacts}
    assert {"config.json", "tokenizer.json", "model.safetensors.index.json"} <= paths
    assert any(p.endswith(".safetensors") for p in paths)
    assert all(len(a["sha256"]) == 64 and a["bytes"] > 0 for a in entry.artifacts)
    assert entry.document["upstream"]["gated"] is False
    assert entry.document["upstream"]["license"] == "Apache-2.0"
    assert "token" in entry.document["description"]   # says none is needed


# --- resolution -----------------------------------------------------------------


def test_resolution_prefers_newest_compatible():
    index = RegistryIndex(
        make_index(
            [
                adapter("skin", "0.1.0", "skin"),
                adapter("skin", "0.2.0", "skin"),
                adapter("skin", "0.3.0", "skin", schema=">=0.9 <1.0"),  # too new
            ]
        )
    )
    registry = Registry(index)
    assert str(registry.get("skin").version) == "0.2.0"
    assert str(registry.get("skin@0.1.0").version) == "0.1.0"
    assert str(registry.get("skin", "0.1.0").version) == "0.1.0"


def test_resolution_errors():
    registry = Registry(RegistryIndex(make_index([adapter("skin", "0.1.0", "skin")])))
    # An unknown name lists what the index does know.
    with pytest.raises(RegistryError, match="known components: skin"):
        registry.get("nonexistent")
    # An unknown version of a known name lists the available versions —
    # distinct from the not-published-yet error, and it names the fix for
    # a stale pin (e.g. from before a version renumbering).
    with pytest.raises(
        RegistryError, match=r"skin@9\.9\.9 not found; available: 0\.1\.0"
    ):
        registry.get("skin@9.9.9")


def test_resolve_slots_and_vocabulary_by_slot():
    index = RegistryIndex(
        make_index(
            [
                adapter("make-skin", "0.1.0", "skin"),
                adapter("make-garment", "0.1.0", "garment"),
                adapter(
                    "make-shoe", "0.1.0", "shoe",
                    constraints={"vocabulary": {"styles": ["below_ankle"]}},
                ),
            ]
        )
    )
    registry = Registry(index)
    by_slot = registry.vocabulary_by_slot(["skin", "garment", "shoe"])
    assert by_slot == {
        "skin": {},
        "garment": {},
        "shoe": {"styles": ["below_ankle"]},
    }
    with pytest.raises(RegistryError):
        registry.resolve_slots(["eye"])  # no component serves it in this index


def test_parse_ref():
    assert parse_ref("skin@0.1.0") == ("skin", "0.1.0")
    assert parse_ref("skin") == ("skin", None)


# --- index validation --------------------------------------------------------------


def test_duplicate_entries_rejected():
    entry = adapter("skin", "0.1.0", "skin")
    with pytest.raises(RegistryError):
        RegistryIndex(make_index([entry, copy.deepcopy(entry)]))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.__setitem__("kind", "mystery"),
        lambda e: e.__setitem__("slot", "cape"),
        lambda e: e.__setitem__("version", "1.0"),
        lambda e: e.__setitem__("name", "Bad Name"),
        lambda e: e.__setitem__("artifacts", [{"path": "../escape", "sha256": "ab" * 32, "bytes": 1}]),
        lambda e: e.__setitem__("artifacts", [{"path": "w", "sha256": "nope", "bytes": 1}]),
        lambda e: e.__setitem__("source", {"hf_repo": "x"}),
        lambda e: e.__setitem__("constraints", {"vocabulary": {"styles": []}}),
    ],
    ids=["kind", "slot", "version", "name", "traversal-path", "bad-sha", "bad-source",
         "empty-vocab"],
)
def test_malformed_entries_rejected(mutate):
    entry = adapter("skin", "0.1.0", "skin")
    mutate(entry)
    with pytest.raises(RegistryError):
        RegistryIndex(make_index([entry]))


# --- store: fetch + integrity ------------------------------------------------------


def _fetching_registry(payload: bytes, tmp_path, monkeypatch, *, lie_sha=False):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path))
    sha = hashlib.sha256(payload).hexdigest()
    if lie_sha:
        sha = "0" * 64
    index = RegistryIndex(
        make_index(
            [
                adapter(
                    "skin", "0.1.0", "skin",
                    source={"hf_repo": "character-factory/skin", "revision": "abc123"},
                    artifacts=[{"path": "adapter.safetensors", "sha256": sha,
                                "bytes": len(payload)}],
                )
            ]
        )
    )

    calls = []

    def fake_fetch(url, target, expected_bytes):
        calls.append(url)
        target.write_bytes(payload)

    return Registry(index), fake_fetch, calls


def test_fetch_verifies_and_caches(tmp_path, monkeypatch):
    payload = b"pretend weights"
    registry, fake_fetch, calls = _fetching_registry(payload, tmp_path, monkeypatch)
    directory = registry.ensure("skin", fetch=fake_fetch)
    assert (directory / "adapter.safetensors").read_bytes() == payload
    assert directory == cache_dir() / "components" / "skin" / "0.1.0"
    assert calls == [
        "https://huggingface.co/character-factory/skin/resolve/abc123/adapter.safetensors"
    ]
    # Second call: cached, verified, no re-download.
    registry.ensure("skin", fetch=fake_fetch)
    assert len(calls) == 1


def test_hash_mismatch_is_a_hard_error(tmp_path, monkeypatch):
    registry, fake_fetch, _ = _fetching_registry(
        b"pretend weights", tmp_path, monkeypatch, lie_sha=True
    )
    with pytest.raises(IntegrityError):
        registry.ensure("skin", fetch=fake_fetch)
    # Nothing partially fetched may remain in place.
    assert not list((cache_dir() / "components").rglob("adapter.safetensors"))


def test_corrupted_cache_detected(tmp_path, monkeypatch):
    payload = b"pretend weights"
    registry, fake_fetch, _ = _fetching_registry(payload, tmp_path, monkeypatch)
    directory = registry.ensure("skin", fetch=fake_fetch)
    (directory / "adapter.safetensors").write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        registry.ensure("skin", fetch=fake_fetch)


def test_ensure_component_size_check(tmp_path, monkeypatch):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path))
    payload = b"payload"
    sha = hashlib.sha256(payload).hexdigest()
    from character_factory.registry.model import ComponentEntry

    entry = ComponentEntry(
        adapter(
            "skin", "0.1.0", "skin",
            source={"hf_repo": "r", "revision": "v"},
            artifacts=[{"path": "w", "sha256": sha, "bytes": len(payload)}],
        )
    )
    directory = ensure_component(entry, fetch=lambda u, t, b: t.write_bytes(payload))
    assert (directory / "w").exists()


def test_ensure_component_logs_fetch_start_and_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path))
    payload = b"payload"
    sha = hashlib.sha256(payload).hexdigest()
    from character_factory.registry.model import ComponentEntry

    entry = ComponentEntry(
        adapter(
            "skin", "0.1.0", "skin",
            source={"hf_repo": "r", "revision": "v"},
            artifacts=[{"path": "w", "sha256": sha, "bytes": len(payload)}],
        )
    )
    ensure_component(entry, fetch=lambda u, t, b: t.write_bytes(payload))
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0] == f"fetching skin 0.1.0 ({len(payload)} B)"
    assert re.fullmatch(r"fetched skin 0\.1\.0 \d+\.\d s", lines[1])
    assert len(lines) == 2

    # Already on disk: nothing to fetch, nothing said.
    ensure_component(entry, fetch=lambda u, t, b: t.write_bytes(payload))
    assert capsys.readouterr().err == ""


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path / "override"))
    assert cache_dir() == tmp_path / "override"
    monkeypatch.delenv("CHARACTER_FACTORY_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "character-factory"


def _streaming_response(payload: bytes, chunk: int):
    """A urlopen() stand-in that hands `payload` out `chunk` bytes at a time."""
    import io

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request):
        assert request.get_header("User-agent") == "character-factory"
        return Response(payload)

    return fake_urlopen


def test_missing_bytes_counts_only_absent_artifacts(tmp_path, monkeypatch):
    from character_factory.registry.store import missing_bytes

    payload = b"pretend weights"
    registry, fake_fetch, _ = _fetching_registry(payload, tmp_path, monkeypatch)
    entry = registry.get("skin")
    assert missing_bytes(entry) == len(payload)
    registry.ensure("skin", fetch=fake_fetch)
    assert missing_bytes(entry) == 0


def test_default_fetch_streams_byte_progress(tmp_path, monkeypatch):
    from character_factory.registry import store

    payload = b"0123456789" * 5
    registry, _, _ = _fetching_registry(payload, tmp_path, monkeypatch)
    monkeypatch.setattr(store, "_CHUNK", 16)
    monkeypatch.setattr(
        store.urllib.request, "urlopen", _streaming_response(payload, 16)
    )
    seen = []
    directory = registry.ensure(
        "skin", progress=lambda received, total: seen.append((received, total))
    )
    assert (directory / "adapter.safetensors").read_bytes() == payload
    # Cumulative bytes against the total still to fetch, ending exactly there.
    assert seen == [(16, 50), (32, 50), (48, 50), (50, 50)]


def test_progress_callback_can_abandon_a_download(tmp_path, monkeypatch):
    from character_factory.registry import store

    payload = b"0123456789" * 5
    registry, _, _ = _fetching_registry(payload, tmp_path, monkeypatch)
    monkeypatch.setattr(store, "_CHUNK", 16)
    monkeypatch.setattr(
        store.urllib.request, "urlopen", _streaming_response(payload, 16)
    )

    class Stop(Exception):
        pass

    def progress(received, total):
        if received >= 32:
            raise Stop()

    with pytest.raises(Stop):
        registry.ensure("skin", progress=progress)
    # No partial file survives; the next attempt starts over.
    assert not list((cache_dir() / "components").rglob("adapter.safetensors*"))
