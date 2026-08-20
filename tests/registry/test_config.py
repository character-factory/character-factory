"""Alternate registry index + authenticated fetches."""

import json

import pytest

from character_factory.registry import Registry, RegistryError, cache_dir
from character_factory.registry.config import (
    ENV_AUTH_TOKEN,
    ENV_REGISTRY_URL,
    load_config,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path))
    monkeypatch.delenv(ENV_REGISTRY_URL, raising=False)
    monkeypatch.delenv(ENV_AUTH_TOKEN, raising=False)
    return tmp_path


def test_defaults_are_anonymous_snapshot(home):
    config = load_config()
    assert config.registry_url is None
    assert config.headers() == {}


def test_config_file_values(home):
    (home / "config.json").write_text(
        json.dumps({"registry_url": "https://example.test/index.json",
                    "auth_token": "sekrit"})
    )
    config = load_config()
    assert config.registry_url == "https://example.test/index.json"
    assert config.headers() == {"Authorization": "Bearer sekrit"}


def test_env_overrides_config_file(home, monkeypatch):
    (home / "config.json").write_text(json.dumps({"registry_url": "https://file.test"}))
    monkeypatch.setenv(ENV_REGISTRY_URL, "https://env.test/index.json")
    monkeypatch.setenv(ENV_AUTH_TOKEN, "env-token")
    config = load_config()
    assert config.registry_url == "https://env.test/index.json"
    assert config.headers()["Authorization"] == "Bearer env-token"


def test_malformed_config_file_is_a_clear_error(home):
    (home / "config.json").write_text("not json")
    with pytest.raises(ValueError):
        load_config()


def test_refresh_requires_configured_url(home):
    with pytest.raises(RegistryError):
        Registry.refresh()


def test_refresh_fetches_validates_and_caches(home, monkeypatch):
    index = {
        "format": "character-factory/registry",
        "registry_version": "0.1",
        "components": [],
    }
    seen = {}

    def fake_fetch_json(url, headers=None):
        seen["url"], seen["headers"] = url, headers
        return index

    monkeypatch.setenv(ENV_REGISTRY_URL, "https://staging.test/index.json")
    monkeypatch.setenv(ENV_AUTH_TOKEN, "staging-token")
    monkeypatch.setattr("character_factory.registry.store.fetch_json", fake_fetch_json)
    registry = Registry.refresh()
    assert seen["url"] == "https://staging.test/index.json"
    assert seen["headers"] == {"Authorization": "Bearer staging-token"}
    assert registry.index.entries == []
    # The refreshed index is what default() now serves.
    assert json.loads((cache_dir() / "registry.json").read_text())["components"] == []
    assert Registry.default().index.entries == []


def test_refresh_rejects_invalid_remote_index(home, monkeypatch):
    monkeypatch.setenv(ENV_REGISTRY_URL, "https://staging.test/index.json")
    monkeypatch.setattr(
        "character_factory.registry.store.fetch_json",
        lambda url, headers=None: {"format": "wrong"},
    )
    with pytest.raises(RegistryError):
        Registry.refresh()
    assert not (cache_dir() / "registry.json").exists()


def test_auth_header_reaches_artifact_downloads(home, monkeypatch):
    """ensure() threads the configured bearer token into the fetch path."""
    import hashlib

    from character_factory.registry import RegistryIndex
    from character_factory.registry import store

    monkeypatch.setenv(ENV_AUTH_TOKEN, "artifact-token")
    payload = b"weights"
    index = RegistryIndex(
        {
            "format": "character-factory/registry",
            "registry_version": "0.1",
            "components": [
                {
                    "name": "make-skin", "version": "0.1.0",
                    "kind": "texture-adapter", "slot": "skin",
                    "requires": {"schema": ">=0.1 <1.0"},
                    "artifacts": [{"path": "w", "bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest()}],
                    "source": {"hf_repo": "org/private", "revision": "r1"},
                }
            ],
        }
    )
    captured = {}

    def fake_download(url, target, expected, headers=None):
        captured["headers"] = headers
        target.write_bytes(payload)

    monkeypatch.setattr(store, "_download", fake_download)
    Registry(index).ensure("make-skin")
    assert captured["headers"] == {"Authorization": "Bearer artifact-token"}
