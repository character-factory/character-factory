"""The local HTTP server: the /v0 contract over CharacterService.

Install extra ``[server]`` (FastAPI + uvicorn). The MCP layer exposes the
same service as tools — see :mod:`character_factory.mcp`.
"""

from character_factory.server.service import (
    CharacterService,
    ServiceError,
)

__all__ = ["CharacterService", "ServiceError", "create_app", "serve"]


def __getattr__(name: str):
    if name in ("create_app", "serve"):
        from character_factory.server import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
