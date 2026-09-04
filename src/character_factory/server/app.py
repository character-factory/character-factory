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
    ResourceNotFound,
    ServiceConflict,
    ServiceError,
)

__all__ = ["create_app"]


def create_app(service: CharacterService):
    from character_factory.assembly.manifest import (
        MANIFEST_SCHEMA_PATH,
        export_manifest_schema,
    )
    from fastapi import Body, FastAPI, HTTPException, Request, Response
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(
        title="character-factory",
        version=__import__("character_factory").__version__,
        docs_url=None,
        openapi_url="/v0/openapi.json",
    )
    def record_json(record):
        return {
            "id": record.id,
            "name": record.name,
            "artifact": record.artifact,
            "latest_job": record.latest_job,
            "creation": record.creation,
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
        "description": "A library record with artifact state separate from "
                       "the latest job. The character document is a separate "
                       "resource.",
        "properties": {
            "id": {"type": "string", "description": "16-hex library id"},
            "name": {"type": ["string", "null"]},
            "artifact": {
                "type": "object",
                "properties": {
                    "available": {"type": "boolean"},
                    "revision": {"type": "integer"},
                    "bytes": {"type": ["integer", "null"]},
                    "sha256": {"type": ["string", "null"]},
                    "built_at": {"type": ["string", "null"]},
                },
                "required": ["available", "revision", "bytes", "sha256", "built_at"],
            },
            "latest_job": {"oneOf": [_ref("Job"), {"type": "null"}]},
            "creation": {
                "type": "object",
                "properties": {
                    "requested_interpreter": {"type": ["string", "null"]},
                    "actual_interpreter": {"type": ["string", "null"]},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["requested_interpreter", "actual_interpreter",
                             "warnings"],
            },
            "created_at": {"type": ["string", "null"],
                           "description": "ISO 8601, set once at creation"},
            "updated_at": {"type": ["string", "null"],
                           "description": "ISO 8601, refreshed on every write"},
            "character": {**_ref("CharacterDocument"),
                          "description": "Present on single-record reads only."},
        },
        "required": ["id", "artifact", "latest_job", "creation"],
    }
    _SCHEMAS = {
        "CharacterRecord": _RECORD_SCHEMA,
        "Error": {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "code": {"type": "string"},
                "retryable": {"type": "boolean"},
            },
            "required": ["error"],
        },
        "AssetReceipt": {
            "type": "object",
            "properties": {
                "slot": {"type": "string",
                         "enum": ["skin", "eye", "garment", "shoe"]},
                "sha256": {"type": "string"},
                "bytes": {"type": "integer"},
            },
            "required": ["slot", "sha256", "bytes"],
        },
        "Warning": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "object"},
            },
            "required": ["code", "message"],
        },
        "ValidationReport": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": _ref("Warning")},
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
            "description": "A selectable interpreter backend and what a "
                           "create against it would do right now. Aliases "
                           "and kinds only — no model identity.",
            "properties": {
                "alias": {"type": "string"},
                "kind": {"type": "string",
                         "enum": ["local-model", "endpoint"]},
                "default": {"type": "boolean",
                            "description": "True on the row an unaliased "
                                           "request resolves to."},
                "label": {"type": ["string", "null"],
                          "description": "Operator-given display name."},
                "ready": {"type": "boolean",
                          "description": "False when a create against this "
                                         "backend would fail today; see "
                                         "reason."},
                "reason": {"type": ["string", "null"]},
                "download_bytes": {
                    "type": ["integer", "null"],
                    "description": "local-model: weight bytes a create would "
                                   "fetch first (0 = cached; null = unknown).",
                },
                "vram_bytes": {"type": ["integer", "null"],
                               "description": "local-model: declared peak "
                                              "VRAM, when the registry "
                                              "states it."},
                "fits": {"type": ["boolean", "null"],
                         "description": "local-model: whether the device "
                                        "has that much (null = unknown)."},
                "device_bytes": {"type": ["integer", "null"],
                                 "description": "local-model: total memory "
                                                "of the generation device "
                                                "(null = none detected)."},
                "description": {"type": ["string", "null"],
                                "description": "local-model backed by a "
                                               "registry component: that "
                                               "component's public "
                                               "description."},
                "endpoint_host": {"type": ["string", "null"]},
                "has_key": {"type": "boolean",
                            "description": "endpoint: an API key is stored. "
                                           "The key itself is never "
                                           "returned."},
            },
            "required": ["alias", "kind", "default", "ready"],
        },
        "InterpreterConfig": {
            "type": "object",
            "description": "A backend to configure. An endpoint entry needs "
                           "endpoint (+ model, the served model name); a "
                           "local entry needs model (registry component id "
                           "or weights path). Omitting api_key keeps a "
                           "stored key; an empty string removes it.",
            "additionalProperties": False,
            "properties": {
                "endpoint": {"type": "string", "format": "uri"},
                "model": {"type": "string"},
                "api_key": {"type": "string", "writeOnly": True},
                "repetition_penalty": {"type": "number", "exclusiveMinimum": 0},
                "instruction": {"type": "string"},
                "label": {"type": "string"},
                "default": {"type": "boolean",
                            "description": "True makes this alias the "
                                           "default backend; false clears "
                                           "that if it is."},
            },
        },
        "ExportManifest": {
            **export_manifest_schema(),
            "description": "The versioned export manifest embedded in the "
                           "GLB's asset extras; the character route is a "
                           "convenience projection of those same bytes.",
        },
        "Health": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "const": "ok"},
                "characters": {"type": "integer"},
                "jobs": {"type": "integer"},
                "cuda": {"type": "boolean"},
                "vram_free_gb": {"type": "number"},
                "vram_total_gb": {"type": "number"},
            },
            "required": ["status", "characters", "jobs", "cuda"],
        },
        "Job": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "operation": {"type": "string",
                              "enum": ["create", "bake", "assemble"]},
                "status": {"type": "string", "enum": [
                    "queued", "running", "cancelling", "succeeded",
                    "failed", "cancelled",
                ]},
                "stage": {"type": "string"},
                "stages": {
                    "type": ["array", "null"], "items": {"type": "string"},
                    "description": "The stages this job walks, in order, "
                                   "set when it starts (a create may begin "
                                   "with \"downloading\" when its "
                                   "interpreter weights are not cached).",
                },
                "progress": {"type": "number", "minimum": 0, "maximum": 1},
                "stage_progress": {
                    "type": ["number", "null"], "minimum": 0, "maximum": 1,
                    "description": "Progress within the current stage when "
                                   "the stage reports it (downloads do); "
                                   "null otherwise.",
                },
                "detail": {"type": ["string", "null"]},
                "queue_position": {"type": ["integer", "null"]},
                "request": {
                    "type": "object",
                    "description": "The submitted request, verbatim (prompt, "
                                   "interpreter, turbo, seed for create; "
                                   "character_id for bake/assemble).",
                },
                "requested_interpreter": {"type": ["string", "null"]},
                "actual_interpreter": {"type": ["string", "null"]},
                "warnings": {"type": "array", "items": _ref("Warning")},
                "result": {"type": ["object", "null"]},
                "error": {"type": ["object", "null"], "properties": {
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean"},
                    "classification": {
                        "type": "string",
                        "description": "Safe failure class; raw endpoint "
                                       "content remains in protected logs.",
                    },
                    "trace_id": {
                        "type": "string",
                        "description": "Opaque correlation identifier for "
                                       "operator diagnostics.",
                    },
                }},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
                "stage_started_at": {"type": "string"},
                "last_heartbeat": {"type": ["string", "null"]},
                "finished_at": {"type": ["string", "null"]},
            },
            "required": [
                "id", "operation", "status", "stage", "progress",
                "warnings", "created_at", "updated_at",
            ],
        },
    }

    def _json_response(schema: dict, description: str) -> dict:
        return {"description": description,
                "content": {"application/json": {"schema": schema}}}

    _ERROR_400 = {400: _json_response(_ref("Error"), "Invalid request")}
    _ERROR_409 = {409: _json_response(
        _ref("Error"), "Idempotency key conflicts with an earlier request"
    )}
    _ERROR_404 = {404: _json_response(_ref("Error"), "Resource not found")}
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

    @app.exception_handler(ResourceNotFound)
    async def resource_not_found(request: Request, error: ResourceNotFound):
        return JSONResponse(
            status_code=404,
            content={"error": str(error), "code": "not_found", "retryable": False},
        )

    @app.exception_handler(ServiceConflict)
    async def service_conflict(request: Request, error: ServiceConflict):
        return JSONResponse(
            status_code=409,
            content={
                "error": str(error),
                "code": "idempotency_conflict",
                "retryable": False,
            },
        )

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(status_code=400, content={"error": str(error)})

    @app.get("/v0", include_in_schema=False)
    async def api_index():
        return {"service": app.title, "version": app.version,
                "docs": "/v0/docs", "openapi": "/v0/openapi.json"}

    @app.get(MANIFEST_SCHEMA_PATH, include_in_schema=False)
    async def manifest_schema():
        return export_manifest_schema()

    @app.get("/v0/docs", include_in_schema=False)
    async def docs():
        from fastapi.responses import HTMLResponse
        from importlib import resources

        page = resources.files("character_factory.server").joinpath(
            "static/docs.html"
        )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    idempotency_parameter = {
        "name": "Idempotency-Key",
        "in": "header",
        "required": False,
        "schema": {"type": "string", "minLength": 1, "maxLength": 255},
        "description": "Retry key for asynchronous creation. Reusing the "
                       "same key with the same request returns the original "
                       "job; reusing it for different work returns 409. "
                       "Omit it to create a new job.",
    }

    @app.post(
        "/v0/characters", status_code=202,
        responses={
            201: _RECORD_OK,
            202: _json_response(_ref("Job"), "Accepted create job"),
            **_ERROR_400, **_ERROR_409,
        },
        openapi_extra={"parameters": [idempotency_parameter],
                       "requestBody": {"required": True, "content": {
            "application/json": {"schema": {
                "type": "object",
                "description": 'Either a full character document ("character") '
                               'or a text description ("prompt") to interpret.',
                "oneOf": [
                    {"required": ["character"], "not": {"required": ["prompt"]}},
                    {"required": ["prompt"], "not": {"required": ["character"]}},
                ],
                "additionalProperties": False,
                "properties": {
                    "character": _ref("CharacterDocument"),
                    "prompt": {"type": "string"},
                    "interpreter": {
                        "type": "string",
                        "description": "Optional interpreter alias from "
                                       "GET /v0/interpreters; omit for the "
                                       "configured default.",
                    },
                    "turbo": {
                        "type": "boolean",
                        "default": False,
                        "description": "Bake textures on the fast distilled "
                                       "base variant (speed over fidelity). "
                                       "A bake-time option; not recorded in "
                                       "the character document.",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Generation seed. Honored when given; "
                                       "otherwise the server draws one at "
                                       "random and records it in the "
                                       "document's provenance.seed.",
                    },
                },
            }}}}},
    )
    async def create_character(
        response: Response, request: Request, payload: dict = Body(...)
    ):
        if "character" in payload:
            response.status_code = 201
            return record_json(service.store_character(payload["character"]))
        if "prompt" in payload:
            job = service.create_from_prompt(
                payload["prompt"], interpreter=payload.get("interpreter"),
                turbo=bool(payload.get("turbo", False)),
                seed=payload.get("seed"),
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
            response.headers["Location"] = f"/v0/jobs/{job['id']}"
            response.headers["Retry-After"] = "2"
            return job
        raise ServiceError('the body must contain "character" or "prompt"')

    @app.get("/v0/characters", responses={
        200: _json_response(
            {"type": "array", "items": _ref("CharacterRecord")},
            "Completed library records, newest first",
        )})
    async def list_characters():
        return [record_json(record) for record in service.list()]

    @app.get("/v0/characters/{character_id}",
             responses={200: _json_response(_ref("CharacterRecord"),
                        "The record with its character document"),
                        **_ERROR_400, **_ERROR_404})
    async def get_character(character_id: str):
        record = record_json(service.get(character_id))
        record["character"] = service.document(character_id)
        return record

    @app.get("/v0/characters/{character_id}/character.json",
             responses={200: _json_response(_ref("CharacterDocument"),
                        "The character document alone"),
                        **_ERROR_400, **_ERROR_404})
    async def get_character_document(character_id: str):
        return service.document(character_id)

    def _download(
        path: Path, request: Request, *, media_type: str, filename: str
    ):
        import hashlib

        etag = f'"{hashlib.sha256(path.read_bytes()).hexdigest()}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return FileResponse(
            path,
            media_type=media_type,
            filename=filename,
            headers={"ETag": etag},
        )

    _BINARY_ERRORS = {
        400: _json_response(_ref("Error"), "Invalid request"),
        404: _json_response(_ref("Error"), "Resource not found"),
    }

    @app.get("/v0/characters/{character_id}/scene.glb", responses={
        200: {"description": "Character GLB", "content": {
            "model/gltf-binary": {"schema": {"type": "string", "format": "binary"}}
        }},
        **_BINARY_ERRORS,
    })
    async def get_scene(character_id: str, request: Request):
        return _download(
            service.scene_path(character_id), request,
            media_type="model/gltf-binary",
            filename=f"character-{character_id}.glb",
        )

    @app.put(
        "/v0/characters/{character_id}/assets/{slot}",
        responses={200: _json_response(_ref("AssetReceipt"), "Stored asset"),
                   **_BINARY_ERRORS},
        openapi_extra={"requestBody": {"required": True, "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}}
        }}},
    )
    async def put_asset(character_id: str, slot: str, request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0] != "image/png":
            raise ServiceError("asset Content-Type must be image/png")
        return service.put_asset(character_id, slot, await request.body())

    @app.get("/v0/characters/{character_id}/assets/{slot}.png", responses={
        200: {"description": "PNG texture", "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}}
        }},
        **_BINARY_ERRORS,
    })
    async def get_asset(character_id: str, slot: str, request: Request):
        return _download(
            service.asset_path(character_id, slot), request,
            media_type="image/png", filename=f"{character_id}-{slot}.png",
        )

    @app.get("/v0/characters/{character_id}/thumbnail.png", responses={
        200: {"description": "PNG thumbnail", "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}}
        }},
        **_BINARY_ERRORS,
    })
    async def get_thumbnail(character_id: str, request: Request):
        return _download(
            service.thumbnail_path(character_id), request,
            media_type="image/png", filename=f"{character_id}-thumbnail.png",
        )

    @app.post(
        "/v0/characters/{character_id}/rebuild",
        status_code=202,
        responses={202: _json_response(_ref("Job"), "Accepted rebuild job"),
                   **_ERROR_400, **_ERROR_404, **_ERROR_409},
        openapi_extra={"parameters": [{
            **idempotency_parameter,
            "description": "Retry key for this rebuild target and payload. "
                           "The same key and request returns the original job; "
                           "different work returns 409. Omit it for a new revision.",
        }], "requestBody": {"required": False, "content": {
            "application/json": {"schema": {
                "type": "object",
                "properties": {"from": {
                    "type": "string", "enum": ["assemble", "bake"],
                    "default": "assemble",
                    "description": '"assemble" rebuilds the scene from stored '
                                   'assets; "bake" re-runs the stored texture '
                                   "recipes first.",
                }, "turbo": {
                    "type": "boolean", "default": False,
                    "description": "For \"bake\": use the fast distilled "
                                   "base variant.",
                }},
            }}}}},
    )
    async def rebuild(
        character_id: str, response: Response, request: Request,
        payload: dict = Body(default={}),
    ):
        stage = payload.get("from", "assemble")
        job = service.rebuild(
            character_id,
            stage=stage,
            turbo=bool(payload.get("turbo", False)),
            idempotency_key=request.headers.get("Idempotency-Key"),
        )
        response.headers["Location"] = f"/v0/jobs/{job['id']}"
        response.headers["Retry-After"] = "2"
        return job

    @app.get("/v0/jobs", responses={
        200: _json_response(
            {"type": "array", "items": _ref("Job")}, "Jobs, newest first"
        )})
    async def list_jobs():
        return service.list_jobs()

    @app.get("/v0/jobs/{job_id}", responses={
        200: _json_response(_ref("Job"), "The lightweight job status"),
        **_ERROR_400, **_ERROR_404,
    })
    async def get_job(job_id: str):
        return service.get_job(job_id)

    @app.delete("/v0/jobs/{job_id}", responses={
        200: _json_response(_ref("Job"), "Cancellation state"),
        **_ERROR_400, **_ERROR_404,
    })
    async def cancel_job(job_id: str):
        return service.cancel_job(job_id)

    @app.post("/v0/jobs/{job_id}/retry", status_code=202, responses={
        202: _json_response(_ref("Job"), "Accepted retry job"),
        **_ERROR_400, **_ERROR_404,
    })
    async def retry_job(job_id: str, response: Response):
        job = service.retry_job(job_id)
        response.headers["Location"] = f"/v0/jobs/{job['id']}"
        response.headers["Retry-After"] = "2"
        return job

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
                        "The embedded export manifest"),
                        **_ERROR_400, **_ERROR_404})
    async def manifest(character_id: str):
        return service.manifest(character_id)

    @app.get("/v0/interpreters", responses={
        200: _json_response({"type": "array", "items": _ref("Interpreter")},
                            "Selectable interpreter backends with readiness, "
                            "the default first")})
    async def interpreters():
        return service.interpreters()

    # Backends are the operator's: these two write the local config file.
    # A hosted deployment, whose backends are its tiers, answers 405.
    @app.put(
        "/v0/interpreters/{alias}",
        responses={200: _json_response(_ref("Interpreter"),
                                       "The configured backend's listing row"),
                   **_ERROR_400},
        openapi_extra={"requestBody": {"required": True, "content": {
            "application/json": {"schema": _ref("InterpreterConfig")}}}},
    )
    async def configure_interpreter(alias: str, payload: dict = Body(...)):
        return service.configure_interpreter(alias, payload)

    @app.delete("/v0/interpreters/{alias}", status_code=204,
                responses={**_ERROR_404})
    async def remove_interpreter(alias: str):
        service.remove_interpreter(alias)

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


def serve(
    library_dir: str | Path, host: str = "127.0.0.1", port: int = 8400,
):
    import uvicorn

    app = create_app(CharacterService(library_dir))
    uvicorn.run(app, host=host, port=port)
