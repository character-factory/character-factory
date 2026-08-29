"""Interpreter configuration: which model writes the slot prompts.

Model choice is configuration, never code. The default backend is a local
model run in-process; `model` names either a registry component id or a
filesystem path to weights. Several backends can be configured at once
under short **aliases** and selected per request (the create UI's model
selector; the SaaS offers tiers the same way) — the alias→model mapping
is local configuration, so no model identity ever appears in code or in
a character's provenance.

The ``interpreter`` object in the cache root's ``config.json``:

    {
      "interpreter": {
        "default": "local-a",
        "instruction": "…optional system-prompt override…",
        "backends": {
          "local-a": {"model": "<registry id or weights path>"},
          "cloud":   {"endpoint": "https://…/v1", "model": "…",
                      "api_key": "…"},
          }
      }
    }

Environment overrides
(``CHARACTER_FACTORY_INTERPRETER_MODEL`` / ``_ENDPOINT`` / ``_API_KEY``)
take precedence over the file and describe the default backend. An
installation with no backend configured cannot interpret: there is no
non-model mode.

Endpoint operators may set ``CHARACTER_FACTORY_INTERPRETER_AUDIT_LOG`` to a
protected JSONL path. That diagnostic log contains raw prompts and endpoint
responses and must be treated as sensitive; it is never exposed over HTTP.

``endpoint`` selects the OpenAI-compatible chat-completions backend
(``model`` is then the served model name). ``instruction`` (global or
per-backend) replaces the built-in task header — conditioning-grade
prompt engineering is data, like the registry's per-slot guidance, and
never lives in the repo.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from character_factory.registry.store import cache_dir

__all__ = [
    "InterpreterConfig",
    "available_backends",
    "load_interpreter_config",
    "resolve_interpreter_config",
]

ENV_MODEL = "CHARACTER_FACTORY_INTERPRETER_MODEL"
ENV_ENDPOINT = "CHARACTER_FACTORY_INTERPRETER_ENDPOINT"
ENV_API_KEY = "CHARACTER_FACTORY_INTERPRETER_API_KEY"
ENV_AUDIT_LOG = "CHARACTER_FACTORY_INTERPRETER_AUDIT_LOG"


@dataclass(frozen=True)
class InterpreterConfig:
    model: str | None = None      # registry component id OR local weights path
    endpoint: str | None = None   # OpenAI-compatible chat-completions base URL
    api_key: str | None = None
    max_new_tokens: int = 768
    instruction: str | None = None   # system-prompt override (data, not code)
    repetition_penalty: float = 1.0  # >1 damps greedy repetition loops
    audit_log: str | None = None     # protected JSONL; never served over HTTP

    @property
    def configured(self) -> bool:
        return self.model is not None or self.endpoint is not None


def _file_section() -> dict:
    path = cache_dir() / "config.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable config {path}: {error}") from error
    if not isinstance(document, dict):
        return {}
    section = document.get("interpreter", {})
    if not isinstance(section, dict):
        raise ValueError(f"{path}: 'interpreter' must be a JSON object")
    return section


def _config_from(values: dict, instruction: str | None) -> InterpreterConfig:
    return InterpreterConfig(
        model=values.get("model"),
        endpoint=values.get("endpoint"),
        api_key=values.get("api_key"),
        max_new_tokens=int(values.get("max_new_tokens", 768)),
        instruction=values.get("instruction") or instruction,
        repetition_penalty=float(values.get("repetition_penalty", 1.0)),
        audit_log=os.environ.get(ENV_AUDIT_LOG) or values.get("audit_log"),
    )


def resolve_interpreter_config(
    alias: str | None = None,
) -> tuple[str, InterpreterConfig]:
    """Resolve a request to its public alias and configuration.

    With no alias (or the alias ``default``): the default backend —
    environment overrides first, then the entry the file's `default` names
    in the `backends` table. With any other alias: that entry of the
    `backends` table. An installation with nothing configured resolves to
    an unconfigured config; `interpret` raises on it.
    """
    section = _file_section()
    instruction = section.get("instruction")
    backends = section.get("backends", {})
    if not isinstance(backends, dict):
        raise ValueError("interpreter 'backends' must be a JSON object")

    if alias is not None and alias != "default" and alias not in backends:
        raise ValueError(
            f"unknown interpreter backend {alias!r}; configured: "
            f"{', '.join(sorted(backends)) or '(none)'}"
        )
    if alias is not None and alias in backends:
        return alias, _config_from(backends[alias], instruction)

    env = {
        "model": os.environ.get(ENV_MODEL),
        "endpoint": os.environ.get(ENV_ENDPOINT),
        "api_key": os.environ.get(ENV_API_KEY),
    }
    if env["model"] or env["endpoint"]:
        return "default", _config_from(
            {**{k: v for k, v in env.items() if v}}, instruction
        )
    default = section.get("default")
    if default and default in backends:
        return default, _config_from(backends[default], instruction)
    return "default", InterpreterConfig(instruction=instruction)


def load_interpreter_config(alias: str | None = None) -> InterpreterConfig:
    """The configuration for one interpreter backend."""
    return resolve_interpreter_config(alias)[1]


def available_backends() -> list[dict]:
    """The selectable backends: [{alias, kind}] — aliases and kinds only,
    never model identities."""
    section = _file_section()
    backends = section.get("backends", {})
    rows = []
    if isinstance(backends, dict):
        for alias in sorted(backends):
            entry = backends[alias] or {}
            kind = "endpoint" if entry.get("endpoint") else "local-model"
            rows.append({"alias": alias, "kind": kind})
    return rows
