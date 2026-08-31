"""The MCP server: the same operations as the HTTP surface, as tools.

Parity rule (ARCHITECTURE §2.4): same tool names and input/output shapes
local and hosted; both layers delegate to the one CharacterService, so they
cannot drift. Resources expose the character JSON Schema and each stored
character's document — an agent can read the format it is writing against
without leaving the session.

Install extra ``[mcp]``; run with ``character-factory mcp``.
"""

from __future__ import annotations

import json
from pathlib import Path

from character_factory.server.service import CharacterService

__all__ = ["build_mcp"]


def build_mcp(service: CharacterService):
    try:  # MCP SDK 2.x
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:  # MCP SDK 1.x
        from mcp.server.fastmcp import FastMCP as _Server

    mcp = _Server("character-factory")

    @mcp.tool()
    def validate_character(character: dict, strict: bool = False) -> dict:
        """Validate a character document against the format spec."""
        return service.validate(character, strict=strict)

    @mcp.tool()
    def store_character(character: dict) -> dict:
        """Validate and store a character document; returns its record."""
        record = service.store_character(character)
        return record.__dict__

    @mcp.tool()
    def create_character(
        prompt: str, interpreter: str | None = None, turbo: bool = False,
        seed: int | None = None, idempotency_key: str | None = None,
    ) -> dict:
        """Submit a character-creation job. Supply idempotency_key only when
        retrying an ambiguous submission; omit it to create new work. Omit
        seed for a fresh random one (recorded in provenance.seed); pass one
        to reproduce. Poll get_job until it reaches succeeded, failed, or
        cancelled."""
        return service.create_from_prompt(
            prompt, interpreter=interpreter, turbo=turbo, seed=seed,
            idempotency_key=idempotency_key,
        )

    @mcp.tool()
    def get_character(character_id: str) -> dict:
        """A stored character's record and full document."""
        record = service.get(character_id).__dict__
        record["character"] = service.document(character_id)
        return record

    @mcp.tool()
    def list_characters() -> list[dict]:
        """All completed stored characters, newest first."""
        return [record.__dict__ for record in service.list()]

    @mcp.tool()
    def assemble_character(
        character_id: str, idempotency_key: str | None = None
    ) -> dict:
        """Submit a job to build the rigged GLB from stored assets. Supply
        idempotency_key only when retrying an ambiguous submission."""
        return service.rebuild(
            character_id, stage="assemble", idempotency_key=idempotency_key
        )

    @mcp.tool()
    def get_job(job_id: str) -> dict:
        """Get lightweight status, progress, outcome, and errors for a job."""
        return service.get_job(job_id)

    @mcp.tool()
    def cancel_job(job_id: str) -> dict:
        """Cancel queued work or request cancellation at the next stage boundary."""
        return service.cancel_job(job_id)

    @mcp.tool()
    def retry_job(job_id: str) -> dict:
        """Explicitly retry a failed or cancelled job as new work."""
        return service.retry_job(job_id)

    @mcp.tool()
    def list_components() -> list[dict]:
        """Registry view: every known component and its publication state."""
        return service.components()

    @mcp.resource("character-factory://schema/character")
    def character_schema() -> str:
        """The published JSON Schema for the character format."""
        from character_factory.schema import character_json_schema

        return json.dumps(character_json_schema(), indent=2)

    @mcp.resource("character-factory://characters/{character_id}")
    def character_document(character_id: str) -> str:
        """One stored character's document."""
        return json.dumps(service.document(character_id), indent=2)

    return mcp


def run(library_dir: str | Path) -> None:
    build_mcp(CharacterService(library_dir)).run()
