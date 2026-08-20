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
        }

    @app.exception_handler(NotAvailable)
    async def not_available(request: Request, error: NotAvailable):
        return JSONResponse(status_code=501, content={"error": str(error)})

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse(status_code=400, content={"error": str(error)})

    @app.post("/v0/characters", status_code=201)
    async def create_character(payload: dict = Body(...)):
        if "character" in payload:
            return record_json(service.store_character(payload["character"]))
        if "prompt" in payload:
            return record_json(service.create_from_prompt(payload["prompt"]))
        raise ServiceError('the body must contain "character" or "prompt"')

    @app.get("/v0/characters")
    async def list_characters():
        return [record_json(record) for record in service.list()]

    @app.get("/v0/characters/{character_id}")
    async def get_character(character_id: str):
        record = record_json(service.get(character_id))
        record["character"] = service.document(character_id)
        return record

    @app.get("/v0/characters/{character_id}/character.json")
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

    @app.post("/v0/characters/{character_id}/rebuild")
    async def rebuild(character_id: str, payload: dict = Body(default={})):
        stage = payload.get("from", "assemble")
        if stage == "assemble":
            import anyio

            record = await anyio.to_thread.run_sync(
                service.assemble, character_id
            )
            return record_json(record)
        if stage == "bake":
            raise NotAvailable(
                "bake is not available yet: texture components are unpublished"
            )
        raise ServiceError(f'unknown rebuild stage {stage!r}')

    @app.delete("/v0/characters/{character_id}", status_code=204)
    async def delete_character(character_id: str):
        service.delete(character_id)

    @app.post("/v0/validate")
    async def validate(payload: dict = Body(...), strict: bool = False):
        return service.validate(payload, strict=strict)

    @app.get("/v0/components")
    async def components():
        return service.components()

    @app.get("/v0/health")
    async def health():
        return service.health()

    return app


def serve(library_dir: str | Path, host: str = "127.0.0.1", port: int = 8400):
    import uvicorn

    app = create_app(CharacterService(library_dir))
    uvicorn.run(app, host=host, port=port)
