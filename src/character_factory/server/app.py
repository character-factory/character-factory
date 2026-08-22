"""The /v0 HTTP surface: a thin wiring of CharacterService into FastAPI.

Every route delegates to the service; no logic lives here (ARCHITECTURE §2).
The whole surface is the common local/hosted contract — no local-only
endpoints. Auth is reserved at the contract level: clients send
``Authorization: Bearer <token>``, which this local server accepts and
ignores, so no client changes shape when auth becomes real.
"""

# No `from __future__ import annotations` here: FastAPI resolves route
# annotations at definition time, and the framework imports are local to
# create_app so the base install never needs them.
from pathlib import Path

from character_factory.server.service import (
    CharacterService,
    NotAvailable,
    ServiceError,
)

__all__ = ["create_app"]


def create_app(service: CharacterService):
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(
        title="character-factory",
        version=__import__("character_factory").__version__,
        docs_url="/v0/docs",
        openapi_url="/v0/openapi.json",
    )

    def record_json(record):
        return {
            "id": record.id,
            "name": record.name,
            "status": record.status,
            "detail": record.detail,
            "revision": record.revision,
            "has_scene": record.has_scene,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    # -- OpenAPI response/request schemas --------------------------------------
    # One source of truth: the character document schema is the published
    # JSON Schema (character_factory.schema.character_json_schema); the
    # record schema below mirrors record_json above and nothing else.
    def _ref(name: str) -> dict:
        return {"$ref": f"#/components/schemas/{name}"}

    _RECORD_SCHEMA = {
        "type": "object",
        "description": "A library record: build state and timestamps. The "
                       "character document itself is a separate resource.",
        "properties": {
            "id": {"type": "string", "description": "16-hex library id"},
            "name": {"type": ["string", "null"]},
            "status": {"type": "string",
                       "enum": ["queued", "interpreting", "creating", "baking",
                                "assembling", "built", "error"]},
            "detail": {"type": ["string", "null"]},
            "revision": {"type": "integer"},
            "has_scene": {"type": "boolean"},
            "created_at": {"type": ["string", "null"],
                           "description": "ISO 8601, set once at creation"},
            "updated_at": {"type": ["string", "null"],
                           "description": "ISO 8601, refreshed on every write"},
            "character": {**_ref("CharacterDocument"),
                          "description": "Present on single-record reads only."},
        },
        "required": ["id", "status", "revision", "has_scene"],
    }
    _SCHEMAS = {
        "CharacterRecord": _RECORD_SCHEMA,
        "Error": {
            "type": "object",
            "properties": {"error": {"type": "string"}},
            "required": ["error"],
        },
        "ValidationReport": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ok", "errors", "warnings"],
        },
        "Component": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "kind": {"type": "string"},
                "slot": {"type": ["string", "null"]},
                "published": {"type": "boolean"},
                "active": {"type": "boolean",
                           "description": "True on the version unpinned "
                                          "resolution picks for new creates; "
                                          "older versions stay listed because "
                                          "stored recipes may pin them."},
                "vocabulary": {"type": ["object", "null"]},
            },
            "required": ["name", "version", "kind", "active"],
        },
        "Interpreter": {
            "type": "object",
            "properties": {
                "alias": {"type": "string"},
                "kind": {"type": "string",
                         "enum": ["local", "endpoint", "rules"]},
            },
            "required": ["alias", "kind"],
        },
        "ExportManifest": {
            "type": "object",
            "description": "The export manifest embedded in the GLB's asset "
                           "extras; this route serves the same bytes as a "
                           "convenience projection.",
            "additionalProperties": True,
        },
        "Health": {
            "type": "object",
            "properties": {
                "library": {"type": "string"},
                "characters": {"type": "integer"},
                "cuda": {"type": "boolean"},
                "vram_free_gb": {"type": "number"},
                "vram_total_gb": {"type": "number"},
            },
            "required": ["library", "characters", "cuda"],
        },
    }

    def _json_response(schema: dict, description: str) -> dict:
        return {"description": description,
                "content": {"application/json": {"schema": schema}}}

    _ERROR_400 = {400: _json_response(_ref("Error"), "Invalid request")}
    _RECORD_OK = _json_response(_ref("CharacterRecord"), "The library record")

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        from character_factory.schema import character_json_schema

        spec = get_openapi(title=app.title, version=app.version,
                           routes=app.routes)
        document_schema = dict(character_json_schema())
        document_schema.pop("$schema", None)
        spec.setdefault("components", {}).setdefault("schemas", {}).update(
            {"CharacterDocument": document_schema, **_SCHEMAS}
        )
        app.openapi_schema = spec
        return spec

    app.openapi = custom_openapi

    @app.exception_handler(NotAvailable)
    async def not_available(request: Request, error: NotAvailable):
        return JSONResponse(status_code=501, content={"error": str(error)})

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(status_code=400, content={"error": str(error)})

    @app.get("/v0", include_in_schema=False)
    async def api_index():
        return {"service": app.title, "version": app.version,
                "docs": "/v0/docs", "openapi": "/v0/openapi.json"}

    @app.post(
        "/v0/characters", status_code=201,
        responses={201: _RECORD_OK, **_ERROR_400},
        openapi_extra={"requestBody": {"required": True, "content": {
            "application/json": {"schema": {
                "type": "object",
                "description": 'Either a full character document ("character") '
                               'or a text description ("prompt") to interpret.',
                "properties": {
                    "character": _ref("CharacterDocument"),
                    "prompt": {"type": "string"},
                    "interpreter": {
                        "type": "string",
                        "description": "Optional interpreter alias from "
                                       "GET /v0/interpreters; omit for the "
                                       "configured default.",
                    },
                },
            }}}}},
    )
    async def create_character(payload: dict = Body(...)):
        if "character" in payload:
            return record_json(service.store_character(payload["character"]))
        if "prompt" in payload:
            return record_json(service.create_from_prompt(
                payload["prompt"], interpreter=payload.get("interpreter")
            ))
        raise ServiceError('the body must contain "character" or "prompt"')

    @app.get("/v0/characters", responses={
        200: _json_response({"type": "array", "items": _ref("CharacterRecord")},
                            "All library records, newest first")})
    async def list_characters():
        return [record_json(record) for record in service.list()]

    @app.get("/v0/characters/{character_id}",
             responses={200: _json_response(_ref("CharacterRecord"),
                        "The record with its character document"),
                        **_ERROR_400})
    async def get_character(character_id: str):
        record = record_json(service.get(character_id))
        record["character"] = service.document(character_id)
        return record

    @app.get("/v0/characters/{character_id}/character.json",
             responses={200: _json_response(_ref("CharacterDocument"),
                        "The character document alone"), **_ERROR_400})
    async def get_character_document(character_id: str):
        return service.document(character_id)

    @app.get("/v0/characters/{character_id}/scene.glb")
    async def get_scene(character_id: str):
        return FileResponse(service.scene_path(character_id),
                            media_type="model/gltf-binary")

    @app.put("/v0/characters/{character_id}/assets/{slot}")
    async def put_asset(character_id: str, slot: str, request: Request):
        return service.put_asset(character_id, slot, await request.body())

    @app.get("/v0/characters/{character_id}/assets/{slot}.png")
    async def get_asset(character_id: str, slot: str):
        return FileResponse(service.asset_path(character_id, slot),
                            media_type="image/png")

    @app.get("/v0/characters/{character_id}/thumbnail.png",
             responses=_ERROR_400)
    async def get_thumbnail(character_id: str):
        return FileResponse(service.thumbnail_path(character_id),
                            media_type="image/png")

    @app.post(
        "/v0/characters/{character_id}/rebuild",
        responses={200: _RECORD_OK, **_ERROR_400},
        openapi_extra={"requestBody": {"required": False, "content": {
            "application/json": {"schema": {
                "type": "object",
                "properties": {"from": {
                    "type": "string", "enum": ["assemble", "bake"],
                    "default": "assemble",
                    "description": '"assemble" rebuilds the scene from stored '
                                   'assets; "bake" re-runs the stored texture '
                                   "recipes first.",
                }},
            }}}}},
    )
    async def rebuild(character_id: str, payload: dict = Body(default={})):
        stage = payload.get("from", "assemble")
        if stage == "assemble":
            import anyio

            record = await anyio.to_thread.run_sync(
                service.assemble, character_id
            )
            return record_json(record)
        if stage == "bake":
            return record_json(service.regenerate(character_id))
        raise ServiceError(f'unknown rebuild stage {stage!r}')

    @app.delete("/v0/characters/{character_id}", status_code=204)
    async def delete_character(character_id: str):
        service.delete(character_id)

    @app.post(
        "/v0/validate",
        responses={200: _json_response(_ref("ValidationReport"),
                                       "Validation outcome")},
        openapi_extra={"requestBody": {"required": True, "content": {
            "application/json": {"schema": _ref("CharacterDocument")}}}},
    )
    async def validate(payload: dict = Body(...), strict: bool = False):
        return service.validate(payload, strict=strict)

    @app.get("/v0/components", responses={
        200: _json_response({"type": "array", "items": _ref("Component")},
                            "Every registry component version")})
    async def components():
        return service.components()

    @app.get("/v0/characters/{character_id}/manifest.json",
             responses={200: _json_response(_ref("ExportManifest"),
                        "The embedded export manifest"), **_ERROR_400})
    async def manifest(character_id: str):
        return service.manifest(character_id)

    @app.get("/v0/interpreters", responses={
        200: _json_response({"type": "array", "items": _ref("Interpreter")},
                            "Selectable interpreter backends")})
    async def interpreters():
        return service.interpreters()

    @app.get("/v0/health", responses={
        200: _json_response(_ref("Health"), "Service health and device state")})
    async def health():
        return service.health()

    # The bundled browser view: a plain client of the /v0 contract above —
    # no server-side rendering, no local-only endpoints (ARCHITECTURE §2.3).
    @app.get("/", include_in_schema=False)
    async def index():
        from fastapi.responses import HTMLResponse
        from importlib import resources

        page = resources.files("character_factory.server").joinpath(
            "static/index.html"
        )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    return app


def serve(library_dir: str | Path, host: str = "127.0.0.1", port: int = 8400):
    import uvicorn

    app = create_app(CharacterService(library_dir))
    uvicorn.run(app, host=host, port=port)
