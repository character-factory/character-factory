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

``rules`` is always available as an alias for the deterministic fallback
and needs no entry. The flat legacy keys (``model``, ``endpoint``,
``api_key``, ``max_new_tokens``) still work and act as the default
backend when no ``backends`` table is present. Environment overrides
(``CHARACTER_FACTORY_INTERPRETER_MODEL`` / ``_ENDPOINT`` / ``_API_KEY``)
take precedence over the file and describe the default backend.

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
    "RULES_ALIAS",
    "available_backends",
    "load_interpreter_config",
    "resolve_interpreter_config",
]

ENV_MODEL = "CHARACTER_FACTORY_INTERPRETER_MODEL"
ENV_ENDPOINT = "CHARACTER_FACTORY_INTERPRETER_ENDPOINT"
ENV_API_KEY = "CHARACTER_FACTORY_INTERPRETER_API_KEY"

RULES_ALIAS = "rules"


@dataclass(frozen=True)
class InterpreterConfig:
    model: str | None = None      # registry component id OR local weights path
    endpoint: str | None = None   # OpenAI-compatible chat-completions base URL
    api_key: str | None = None
    max_new_tokens: int = 768
    instruction: str | None = None   # system-prompt override (data, not code)
    repetition_penalty: float = 1.0  # >1 damps greedy repetition loops

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
    )


def resolve_interpreter_config(
    alias: str | None = None,
) -> tuple[str, InterpreterConfig]:
    """Resolve a request to its public alias and configuration.

    With no alias: the default backend — environment overrides first, then
    the file's `default` alias (or its flat legacy keys). With an alias:
    that entry of the `backends` table; the reserved alias `rules` returns
    an unconfigured config, which callers treat as the rules fallback.
    """
    section = _file_section()
    instruction = section.get("instruction")
    backends = section.get("backends", {})
    if not isinstance(backends, dict):
        raise ValueError("interpreter 'backends' must be a JSON object")

    if alias is not None:
        if alias == RULES_ALIAS:
            return RULES_ALIAS, InterpreterConfig(instruction=instruction)
        if alias == "default" and alias not in backends:
            config = _config_from(section, instruction)
            return ("default" if config.configured else RULES_ALIAS), config
        if alias not in backends:
            raise ValueError(
                f"unknown interpreter backend {alias!r}; configured: "
                f"{', '.join(sorted(backends)) or '(none)'} plus '{RULES_ALIAS}'"
            )
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
    if default == RULES_ALIAS:
        return RULES_ALIAS, InterpreterConfig(instruction=instruction)
    config = _config_from(section, instruction)
    return ("default" if config.configured else RULES_ALIAS), config


def load_interpreter_config(alias: str | None = None) -> InterpreterConfig:
    """The configuration for one interpreter backend."""
    return resolve_interpreter_config(alias)[1]


def available_backends() -> list[dict]:
    """The selectable backends: [{alias, kind}] — aliases and kinds only,
    never model identities. `rules` is always present, last."""
    section = _file_section()
    backends = section.get("backends", {})
    rows = []
    if isinstance(backends, dict):
        for alias in sorted(backends):
            entry = backends[alias] or {}
            kind = "endpoint" if entry.get("endpoint") else "local-model"
            rows.append({"alias": alias, "kind": kind})
    if not rows and (section.get("model") or section.get("endpoint")):
        rows.append({
            "alias": "default",
            "kind": "endpoint" if section.get("endpoint") else "local-model",
        })
    rows.append({"alias": RULES_ALIAS, "kind": "rules"})
    return rows
