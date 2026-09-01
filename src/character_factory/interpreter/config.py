"""Interpreter configuration: which model writes the slot prompts.

Model choice is configuration, never code. The default backend is a local
model run in-process; `model` names either a registry component id or a
filesystem path to weights. Several backends can be configured at once
under short **aliases** and selected per request (the create UI's model
selector; the SaaS offers tiers the same way) — the alias→model mapping
is local configuration, so no model identity ever appears in code or in
a character's provenance.

With nothing configured, the default is the registry's ``interpreter``
component: the blessed local model, fetched and hash-verified like any
other component. The model behind that id is registry data.

``mode`` chooses how the model is asked: ``single`` (one instruction, one
JSON document) or ``multi`` (one narrow question per component — see
``interpreter/multi.py``). The default ``auto`` is ``multi`` for local
models and ``single`` for endpoints, which is where each does its best
work.

The ``interpreter`` object in the cache root's ``config.json``:

    {
      "interpreter": {
        "default": "local-a",
        "mode": "auto",
        "instruction": "…optional system-prompt override (single mode)…",
        "backends": {
          "local-a": {"model": "<registry id or weights path>"},
          "cloud":   {"endpoint": "https://…/v1", "model": "…",
                      "api_key": "…", "mode": "single"},
          }
      }
    }

Environment overrides
(``CHARACTER_FACTORY_INTERPRETER_MODEL`` / ``_ENDPOINT`` / ``_API_KEY`` /
``_MODE``) take precedence over the file and describe the default
backend. There is no non-model mode: when the default component cannot
be fetched and nothing else is configured, interpretation is a hard,
named error.

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
    "DEFAULT_MODEL_COMPONENT",
    "MODES",
    "InterpreterConfig",
    "available_backends",
    "load_interpreter_config",
    "resolve_interpreter_config",
]

ENV_MODEL = "CHARACTER_FACTORY_INTERPRETER_MODEL"
ENV_ENDPOINT = "CHARACTER_FACTORY_INTERPRETER_ENDPOINT"
ENV_API_KEY = "CHARACTER_FACTORY_INTERPRETER_API_KEY"
ENV_MODE = "CHARACTER_FACTORY_INTERPRETER_MODE"
ENV_AUDIT_LOG = "CHARACTER_FACTORY_INTERPRETER_AUDIT_LOG"

# The registry component that names the blessed local model. The id is
# code; the model behind it is data.
DEFAULT_MODEL_COMPONENT = "interpreter"
MODES = ("auto", "single", "multi")


@dataclass(frozen=True)
class InterpreterConfig:
    model: str | None = None      # registry component id OR local weights path
    endpoint: str | None = None   # OpenAI-compatible chat-completions base URL
    api_key: str | None = None
    max_new_tokens: int = 768
    instruction: str | None = None   # system-prompt override (data, not code)
    repetition_penalty: float = 1.0  # >1 damps greedy repetition loops
    audit_log: str | None = None     # protected JSONL; never served over HTTP
    mode: str = "auto"               # auto | single | multi

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(
                f"interpreter mode must be one of {', '.join(MODES)}; got {self.mode!r}"
            )

    @property
    def configured(self) -> bool:
        return self.model is not None or self.endpoint is not None

    @property
    def effective_mode(self) -> str:
        """`auto` resolved for this backend: multi-call for a local model,
        the single instruction for an endpoint."""
        if self.mode != "auto":
            return self.mode
        return "single" if self.endpoint is not None else "multi"


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


def _config_from(values: dict, instruction: str | None,
                 mode: str | None = None) -> InterpreterConfig:
    return InterpreterConfig(
        model=values.get("model"),
        endpoint=values.get("endpoint"),
        api_key=values.get("api_key"),
        max_new_tokens=int(values.get("max_new_tokens", 768)),
        instruction=values.get("instruction") or instruction,
        repetition_penalty=float(values.get("repetition_penalty", 1.0)),
        audit_log=os.environ.get(ENV_AUDIT_LOG) or values.get("audit_log"),
        mode=os.environ.get(ENV_MODE) or values.get("mode") or mode or "auto",
    )


def resolve_interpreter_config(
    alias: str | None = None,
) -> tuple[str, InterpreterConfig]:
    """Resolve a request to its public alias and configuration.

    With no alias (or the alias ``default``): the default backend —
    environment overrides first, then the entry the file's `default` names
    in the `backends` table, then the registry's ``interpreter`` component.
    With any other alias: that entry of the `backends` table.
    """
    section = _file_section()
    instruction = section.get("instruction")
    mode = section.get("mode")
    backends = section.get("backends", {})
    if not isinstance(backends, dict):
        raise ValueError("interpreter 'backends' must be a JSON object")

    if alias is not None and alias != "default" and alias not in backends:
        raise ValueError(
            f"unknown interpreter backend {alias!r}; configured: "
            f"{', '.join(sorted(backends)) or '(none)'}"
        )
    if alias is not None and alias in backends:
        return alias, _config_from(backends[alias], instruction, mode)

    env = {
        "model": os.environ.get(ENV_MODEL),
        "endpoint": os.environ.get(ENV_ENDPOINT),
        "api_key": os.environ.get(ENV_API_KEY),
    }
    if env["model"] or env["endpoint"]:
        return "default", _config_from(
            {**{k: v for k, v in env.items() if v}}, instruction, mode
        )
    default = section.get("default")
    if default and default in backends:
        return default, _config_from(backends[default], instruction, mode)
    return "default", _config_from(
        {"model": DEFAULT_MODEL_COMPONENT}, instruction, mode
    )


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
