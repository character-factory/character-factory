"""Interpreter configuration: which model writes the slot prompts.

Model choice is configuration, never code. The default backend is a local
model run in-process; `model` names either a registry component id or a
filesystem path to weights, so swapping candidates is a one-line config
change — that is what makes `character-factory interpret` usable as a
side-by-side model bench.

Precedence, highest first:

1. Environment: ``CHARACTER_FACTORY_INTERPRETER_MODEL``,
   ``CHARACTER_FACTORY_INTERPRETER_ENDPOINT``,
   ``CHARACTER_FACTORY_INTERPRETER_API_KEY``.
2. The ``interpreter`` object in the cache root's ``config.json``
   (same file the registry config reads), keys ``model``, ``endpoint``,
   ``api_key``, ``max_new_tokens``.
3. Nothing configured: interpretation falls back to the documented rules
   mode.

``endpoint`` selects the alternative backend: any OpenAI-compatible
chat-completions server (``model`` is then the served model name).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from character_factory.registry.store import cache_dir

__all__ = ["InterpreterConfig", "load_interpreter_config"]

ENV_MODEL = "CHARACTER_FACTORY_INTERPRETER_MODEL"
ENV_ENDPOINT = "CHARACTER_FACTORY_INTERPRETER_ENDPOINT"
ENV_API_KEY = "CHARACTER_FACTORY_INTERPRETER_API_KEY"


@dataclass(frozen=True)
class InterpreterConfig:
    model: str | None = None      # registry component id OR local weights path
    endpoint: str | None = None   # OpenAI-compatible chat-completions base URL
    api_key: str | None = None
    max_new_tokens: int = 768

    @property
    def configured(self) -> bool:
        return self.model is not None or self.endpoint is not None


def load_interpreter_config() -> InterpreterConfig:
    file_values: dict = {}
    path = cache_dir() / "config.json"
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"unreadable config {path}: {error}") from error
        if isinstance(document, dict):
            section = document.get("interpreter", {})
            if not isinstance(section, dict):
                raise ValueError(f"{path}: 'interpreter' must be a JSON object")
            file_values = section
    return InterpreterConfig(
        model=os.environ.get(ENV_MODEL) or file_values.get("model"),
        endpoint=os.environ.get(ENV_ENDPOINT) or file_values.get("endpoint"),
        api_key=os.environ.get(ENV_API_KEY) or file_values.get("api_key"),
        max_new_tokens=int(file_values.get("max_new_tokens", 768)),
    )
