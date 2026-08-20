"""CharacterService semantics plus the /v0 HTTP surface over it.

The HTTP tests run through FastAPI's TestClient; MCP tools delegate to the
identical service methods (parity by construction), so service coverage is
tool coverage.
"""

import json
from pathlib import Path

import pytest

from character_factory.server.service import (
    CharacterService,
    NotAvailable,
    ServiceError,
)

EXAMPLES = Path(__file__).parents[2] / "examples" / "characters"


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("CHARACTER_FACTORY_HOME", str(tmp_path / "cache-isolated"))
    return CharacterService(tmp_path / "library")


@pytest.fixture
def stored(service):
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    return service.store_character(document)


# --- service ---------------------------------------------------------------


def test_store_is_idempotent_by_content(service, stored):
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    again = service.store_character(document)
    assert again.id == stored.id
    assert len(service.list()) == 1


def test_store_rejects_invalid_documents(service):
    with pytest.raises(ServiceError):
        service.store_character({"format": "nope"})


def test_create_from_prompt_reports_unavailable(service):
    with pytest.raises(NotAvailable) as excinfo:
        service.create_from_prompt("a tall person")
    assert "not been published" in str(excinfo.value)


def test_get_unknown_character(service):
    with pytest.raises(ServiceError):
        service.get("doesnotexist000")


def test_asset_upload_rejects_plural_slot(service, stored):
    with pytest.raises(ServiceError) as excinfo:
        service.put_asset(stored.id, "eyes", b"data")
    assert "singular" in str(excinfo.value)


def test_asset_upload_rejects_pinned_mismatch(service):
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    document["assets"] = {
        "skin": {"sha256": "ab" * 32, "media_type": "image/png",
                 "width": 4, "height": 4}
    }
    record = service.store_character(document)
    with pytest.raises(ServiceError):
        service.put_asset(record.id, "skin", b"not the pinned bytes")


def test_delete(service, stored):
    service.delete(stored.id)
    assert service.list() == []


def test_components_view(service):
    rows = {row["name"]: row for row in service.components()}
    assert rows["make-shoe"]["vocabulary"] == {"styles": ["below_ankle"]}
    assert rows["body-rig"]["published"] is False


def test_validate_passthrough(service):
    report = service.validate({"format": "wrong"})
    assert not report["ok"] and report["errors"]


def test_scene_before_build_is_a_clear_error(service, stored):
    with pytest.raises(ServiceError) as excinfo:
        service.scene_path(stored.id)
    assert "assemble" in str(excinfo.value)


# --- HTTP -------------------------------------------------------------------


@pytest.fixture
def client(service):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from character_factory.server.app import create_app

    return TestClient(create_app(service))


def test_http_round_trip(client):
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    created = client.post("/v0/characters", json={"character": document})
    assert created.status_code == 201
    character_id = created.json()["id"]

    listed = client.get("/v0/characters").json()
    assert [row["id"] for row in listed] == [character_id]

    fetched = client.get(f"/v0/characters/{character_id}")
    assert fetched.json()["character"]["name"] == "freediver"

    raw = client.get(f"/v0/characters/{character_id}/character.json")
    assert raw.json()["format"] == "character-factory/character"

    assert client.delete(f"/v0/characters/{character_id}").status_code == 204
    assert client.get("/v0/characters").json() == []


def test_http_prompt_returns_501_not_available(client):
    response = client.post("/v0/characters", json={"prompt": "a tall person"})
    assert response.status_code == 501
    assert "not been published" in response.json()["error"]


def test_http_validate_endpoint(client):
    document = json.loads((EXAMPLES / "marathon-runner.char.json").read_text())
    assert client.post("/v0/validate", json=document).json()["ok"]
    document["textures"]["eyes"] = document["textures"].pop("eye")
    report = client.post("/v0/validate", json=document).json()
    assert not report["ok"]
    assert any("singular" in error for error in report["errors"])


def test_http_bearer_token_accepted_and_ignored(client):
    response = client.get(
        "/v0/health", headers={"Authorization": "Bearer anything"}
    )
    assert response.status_code == 200
    assert "characters" in response.json()


def test_http_components_and_health(client):
    components = client.get("/v0/components").json()
    assert any(row["name"] == "make-wig" for row in components)
    assert client.get("/v0/health").status_code == 200


def test_http_bad_input_is_400(client):
    assert client.post("/v0/characters", json={}).status_code == 400
    assert client.get("/v0/characters/unknown0000/scene.glb").status_code == 400


def test_http_rebuild_bake_is_501(client):
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    character_id = client.post(
        "/v0/characters", json={"character": document}
    ).json()["id"]
    response = client.post(
        f"/v0/characters/{character_id}/rebuild", json={"from": "bake"}
    )
    assert response.status_code == 501


# --- MCP parity (tools delegate to the same service) --------------------------


def test_mcp_tools_delegate_to_service(service, stored):
    import asyncio

    mcp_module = pytest.importorskip("mcp")  # noqa: F841
    from character_factory.mcp import build_mcp

    mcp = build_mcp(service)
    tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert {"validate_character", "store_character", "create_character",
            "get_character", "list_characters", "assemble_character",
            "list_components"} <= tool_names
