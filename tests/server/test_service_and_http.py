"""CharacterService semantics plus the /v0 HTTP surface over it.

The HTTP tests run through FastAPI's TestClient; MCP tools delegate to the
identical service methods (parity by construction), so service coverage is
tool coverage.
"""

import json
import time
from pathlib import Path

import pytest

from character_factory.server.service import (
    CharacterService,
    ServiceError,
    _delivery_warnings,
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


def test_incompatible_disk_record_is_quarantined(service, stored):
    incompatible = service.library_dir / "precontract"
    incompatible.mkdir()
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    document["body"]["topology"] = "closed"
    (incompatible / "character.char.json").write_text(json.dumps(document))

    assert [record.id for record in service.list()] == [stored.id]
    health = service.health()
    assert health["characters"] == 1
    assert health["incompatible_characters"] == 1


def _wait_for_job(service, job_id, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not become terminal")


def test_create_from_prompt_reports_unavailable_as_a_job(service):
    # Submission stays prompt even when the isolated cache has no generation
    # components; capability failure belongs to the accepted job resource.
    submitted = service.create_from_prompt("a tall person")
    failed = _wait_for_job(service, submitted["id"])
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "generation_unavailable"
    assert failed["error"]["retryable"] is True


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
    assert rows["make-shoe"]["vocabulary"] == {
        "styles": ["below_ankle", "high_top", "ankle_boot", "mid_boot",
                   "tall_boot"]
    }
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
    assert [row["id"] for row in listed["items"]] == [character_id]
    assert listed["next_cursor"] is None

    fetched = client.get(f"/v0/characters/{character_id}")
    assert fetched.json()["character"]["name"] == "freediver"
    record = fetched.json()
    assert "status" not in record and "has_scene" not in record
    assert record["artifact"] == {
        "available": False, "revision": 0, "bytes": None,
        "sha256": None, "built_at": None,
    }
    assert record["latest_job"] is None
    assert record["capabilities"]["topology"] == "mouth-interior"
    assert record["capabilities"]["facial_animation"]["morph_count"] == 72
    assert len(record["capabilities"]["facial_animation"]["morph_names"]) == 72

    raw = client.get(f"/v0/characters/{character_id}/character.json")
    assert raw.json()["format"] == "character-factory/character"

    assert client.delete(f"/v0/characters/{character_id}").status_code == 204
    assert client.get("/v0/characters").json() == {
        "items": [], "next_cursor": None
    }


def test_http_prompt_returns_an_async_job(client):
    response = client.post("/v0/characters", json={"prompt": "a tall person"})
    assert response.status_code == 202
    assert response.headers["location"].startswith("/v0/jobs/")
    assert response.headers["retry-after"] == "2"
    job = client.get(response.headers["location"]).json()
    assert job["operation"] == "create"
    assert job["status"] in {"queued", "running", "failed"}


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
    health = client.get("/v0/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "library" not in health.json()  # never expose a server path


def test_http_bad_input_is_400(client):
    assert client.post("/v0/characters", json={}).status_code == 400
    assert client.get("/v0/characters/unknown0000/scene.glb").status_code == 404


def test_http_incompatible_disk_record_is_409_without_breaking_list(client, service):
    incompatible = service.library_dir / "precontract"
    incompatible.mkdir()
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    document["body"]["topology"] = "closed"
    (incompatible / "character.char.json").write_text(json.dumps(document))

    assert client.get("/v0/characters").json()["items"] == []
    response = client.get("/v0/characters/precontract")
    assert response.status_code == 409
    assert response.json()["code"] == "incompatible_character"


def test_http_rebuild_bake_queues_regeneration(client):
    document = json.loads((EXAMPLES / "freediver.char.json").read_text())
    character_id = client.post(
        "/v0/characters", json={"character": document}
    ).json()["id"]
    response = client.post(
        f"/v0/characters/{character_id}/rebuild", json={"from": "bake"}
    )
    assert response.status_code == 202
    assert response.json()["operation"] == "bake"
    assert response.json()["status"] in ("queued", "running", "failed")


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


def test_http_interpreters_lists_selectable_backends(client, monkeypatch, tmp_path):
    import json as jsonlib

    (tmp_path / "config.json").write_text(jsonlib.dumps({"interpreter": {
        "backends": {"local-a": {"model": "x"},
                     "cloud": {"endpoint": "http://h/v1", "model": "m"}},
    }}))
    monkeypatch.setattr(
        "character_factory.interpreter.config.cache_dir", lambda: tmp_path
    )
    rows = client.get("/v0/interpreters").json()
    assert [r["alias"] for r in rows] == ["cloud", "local-a", "rules"]
    # No model identity leaves the config: aliases and kinds only.
    assert all(set(r) == {"alias", "kind"} for r in rows)


def test_http_create_rejects_unknown_interpreter_alias(client, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        "character_factory.interpreter.config.cache_dir", lambda: tmp_path
    )
    response = client.post(
        "/v0/characters", json={"prompt": "someone", "interpreter": "nope"}
    )
    assert response.status_code == 400
    assert "unknown interpreter backend" in response.json()["error"]


def test_records_carry_timestamps_and_list_is_newest_first(service, stored):
    import time as timelib

    first = service.get(stored.id)
    assert first.created_at and first.updated_at

    # A second character stored later must list first.
    timelib.sleep(1.1)   # timestamps have second resolution
    other = json.loads((service.library_dir / stored.id / "character.char.json")
                       .read_text())
    other["name"] = "later-one"
    other["body"]["identity"][0] += 0.25   # different content id
    later = service.store_character(other)
    rows = service.list()
    assert [r.id for r in rows][0] == later.id
    assert rows[0].created_at >= rows[-1].created_at


def test_character_list_uses_bounded_opaque_cursor_pages(service, stored):
    source = service.document(stored.id)
    for index in (1, 2):
        document = json.loads(json.dumps(source))
        document["name"] = f"page-{index}"
        document["body"]["identity"][0] += index * 0.1
        service.store_character(document)

    first = service.list_page(limit=2)
    assert len(first["items"]) == 2
    assert first["next_cursor"]
    second = service.list_page(limit=2, cursor=first["next_cursor"])
    assert len(second["items"]) == 1
    assert second["next_cursor"] is None
    assert {record.id for record in first["items"]}.isdisjoint(
        {record.id for record in second["items"]}
    )
    with pytest.raises(ServiceError, match="cursor"):
        service.list_page(cursor="not-a-cursor")


def test_manifest_route_serves_the_embedded_extras(client, service, stored):
    # No scene yet: a resource-appropriate absence, not a 500.
    response = client.get(f"/v0/characters/{stored.id}/manifest.json")
    assert response.status_code == 404
    assert "no built scene" in response.json()["error"]


def test_components_carry_an_active_marker(client):
    rows = client.get("/v0/components").json()
    assert all("active" in row for row in rows)
    # Exactly one active version per component name that has any.
    by_name: dict = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    for name, versions in by_name.items():
        active = [row for row in versions if row["active"]]
        assert len(active) == 1, f"{name}: {len(active)} active versions"
        # The active one is the newest listed (unpinned resolution).
        newest = max(versions,
                     key=lambda r: tuple(int(p) for p in r["version"].split(".")))
        assert active[0]["version"] == newest["version"]


def test_openapi_schemas_are_real(client):
    spec = client.get("/v0/openapi.json").json()
    schemas = spec["components"]["schemas"]
    # The character document schema is the published JSON Schema, injected
    # as-is — one source of truth.
    assert schemas["CharacterDocument"]["properties"]["format"]["const"] == (
        "character-factory/character"
    )
    for name in ("CharacterRecord", "CharacterPage", "ValidationReport",
                 "Component", "Interpreter", "Health", "Job", "AssetReceipt",
                 "Error"):
        assert name in schemas
    # Key routes reference real response schemas — no bare `{}` bodies.
    listing = spec["paths"]["/v0/characters"]["get"]["responses"]["200"]
    assert listing["content"]["application/json"]["schema"]["$ref"] \
        == "#/components/schemas/CharacterPage"
    create = spec["paths"]["/v0/characters"]["post"]
    body = create["requestBody"]["content"]["application/json"]["schema"]
    assert body["properties"]["character"]["$ref"] \
        == "#/components/schemas/CharacterDocument"
    assert len(body["oneOf"]) == 2
    assert body["properties"]["allow_fallback"]["default"] is False
    validate = spec["paths"]["/v0/validate"]["post"]["responses"]["200"]
    assert validate["content"]["application/json"]["schema"]["$ref"] \
        == "#/components/schemas/ValidationReport"


def test_api_index_links_to_docs(client):
    index = client.get("/v0").json()
    assert index["docs"] == "/v0/docs"
    assert index["openapi"] == "/v0/openapi.json"
    # The dependency-free docs page serves without public-CDN references.
    docs = client.get("/v0/docs")
    assert docs.status_code == 200
    assert "/v0/openapi.json" in docs.text
    assert "https://unpkg.com" not in docs.text


def test_gallery_has_a_dependency_free_offline_fallback(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "basic offline view" in page.text
    assert "window.characterFactoryUiReady" in page.text


def test_thumbnail_route_missing_is_400_then_serves(client, service, stored, tmp_path):
    # No thumbnail yet: a resource-appropriate 404, not a broken image.
    assert client.get(f"/v0/characters/{stored.id}/thumbnail.png").status_code == 404
    # Once one exists (assembly writes it best-effort), the route serves it.
    directory = service.library_dir / stored.id
    (directory / "thumb.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    response = client.get(f"/v0/characters/{stored.id}/thumbnail.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert f"{stored.id}-thumbnail.png" in response.headers["content-disposition"]
    cached = client.get(
        f"/v0/characters/{stored.id}/thumbnail.png",
        headers={"If-None-Match": response.headers["etag"]},
    )
    assert cached.status_code == 304
    assert cached.content == b""


def test_prompt_create_is_idempotent_by_default_and_by_header(client):
    body = {"prompt": "the same person", "interpreter": "rules"}
    first = client.post("/v0/characters", json=body)
    replay = client.post("/v0/characters", json=body)
    assert replay.json()["id"] == first.json()["id"]

    keyed = client.post(
        "/v0/characters", json={**body, "prompt": "another person"},
        headers={"Idempotency-Key": "client-request-1"},
    )
    conflict = client.post(
        "/v0/characters", json={**body, "prompt": "different person"},
        headers={"Idempotency-Key": "client-request-1"},
    )
    assert keyed.status_code == 202
    assert conflict.status_code == 400
    assert "different request" in conflict.json()["error"]


def test_cors_preflight_uses_an_explicit_allow_list(service):
    from fastapi.testclient import TestClient

    from character_factory.server.app import create_app

    allowed = "https://viewer.example"
    cors_client = TestClient(create_app(service, cors_origins=[allowed]))
    response = cors_client.options(
        "/v0/characters",
        headers={
            "Origin": allowed,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,idempotency-key",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == allowed


def test_openapi_and_runtime_describe_binary_resources(client, service, stored):
    spec = client.get("/v0/openapi.json").json()
    scene = spec["paths"]["/v0/characters/{character_id}/scene.glb"]["get"]
    assert "model/gltf-binary" in scene["responses"]["200"]["content"]
    upload = spec["paths"]["/v0/characters/{character_id}/assets/{slot}"]["put"]
    assert "image/png" in upload["requestBody"]["content"]
    manifest_schema = client.get(
        "/v0/schemas/export-manifest-0.6.json"
    ).json()
    assert manifest_schema["properties"]["schema_version"]["const"] == "0.6"
    assert spec["components"]["schemas"]["ExportManifest"]["$id"] == (
        "/v0/schemas/export-manifest-0.6.json"
    )

    response = client.put(
        f"/v0/characters/{stored.id}/assets/skin",
        content=b"not a png", headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert "Content-Type" in response.json()["error"]


def test_declined_geometry_override_is_a_structured_job_warning():
    warnings = _delivery_warnings(
        {"garments": {
            "garment": {"render_mode": "painted", "reason": "pose-gate"},
            "shoe": {"render_mode": "painted"},
        }},
        {"garment_shells": True},
    )
    assert warnings == [
        {
            "code": "requested_geometry_not_delivered",
            "message": "garment shell was requested but the artifact uses painted rendering",
            "details": {
                "slot": "garment", "requested": "shell", "actual": "painted",
                "reason": "pose-gate",
            },
        },
    ]
