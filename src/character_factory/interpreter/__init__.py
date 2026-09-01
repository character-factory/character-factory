"""Interpretation: free text → per-slot prompts + the semantic hair block.

The backend is a language model with grammar-constrained decoding — the
default runs locally, in-process (ARCHITECTURE §2.2). Which model is pure
configuration (`interpreter.backends` in the cache config, or the
environment) — a registry component id or a local weights path — so
candidates swap in seconds and `character-factory interpret` doubles as
the side-by-side bench. With nothing configured the registry's
``interpreter`` component is the model. There is no non-model
interpretation mode: a failing backend is a hard, named error, never a
silent quality downgrade.

How the model is asked is the `mode`: one instruction for the whole
document (``single``, what a hosted frontier model does best) or one
narrow call per component (``multi``, what a small local model does
best — `interpreter/multi.py`). ``auto``, the default, picks by backend.

The interpreter writes every component's prompt, in that component's
trained format: per-slot texture prompts, the semantic hair block, and
the figure prompt that conditions identity generation. The raw
description is what the user said; the component prompts are what each
model was trained to read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from character_factory.schema import vocab

__all__ = [
    "INTERPRETER_VERSION",
    "Interpretation",
    "interpret",
]

INTERPRETER_VERSION = "0.1.0"


@dataclass
class Interpretation:
    slot_prompts: dict[str, str]
    hair: dict | None
    # The body-generation prompt, written by the backend in the figure
    # component's trained format — identity conditions on this, never on
    # the raw description.
    figure: str | None = None
    backend: str = "model"
    notes: list[str] = field(default_factory=list)
    # Explicit skeletal-proportion overrides (§4.3), name → value in the
    # document's units, emitted only when the writer chose to steer them.
    proportions: dict | None = None


def interpret(
    prompt: str,
    *,
    registry=None,
    device: str = "cuda",
    config=None,
    backend: str | None = None,
) -> tuple[Interpretation, dict]:
    """Description → (Interpretation, metrics).

    `backend` selects a configured backend by alias (the create UI's
    model selector); without it the configured default applies. The model
    is loaded, run, and released inside this call: no interpreter VRAM
    survives into the diffusion stages (ARCHITECTURE §2.2). A model that
    cannot be fetched or a failed model request raises InterpreterError —
    interpretation quality is the model's job, and there is nothing
    acceptable to degrade to.
    """
    from character_factory.interpreter.backend import (
        InterpreterError,
        ModelInterpreter,
    )
    from character_factory.interpreter.config import resolve_interpreter_config

    if config is None:
        actual_alias, config = resolve_interpreter_config(alias=backend)
    else:
        actual_alias = backend or "default"
    requested_alias = backend or "default"
    if not config.configured:
        raise InterpreterError(
            "no interpreter model is configured — declare one under "
            "interpreter.backends in the cache config.json, or set "
            "CHARACTER_FACTORY_INTERPRETER_MODEL",
            code="interpreter_unconfigured",
            retryable=False,
        )
    start = time.monotonic()
    runner = ModelInterpreter(config, registry=registry, device=device)
    try:
        interpretation = runner.interpret(
            prompt, _slot_guidance(registry), _vocabulary(registry))
        metrics = runner.metrics.as_dict()
    finally:
        runner.close()   # release before any diffusion loads (§2.2)
    metrics.setdefault("requested_interpreter", requested_alias)
    metrics.setdefault("actual_interpreter", actual_alias)
    metrics["wall_seconds"] = time.monotonic() - start
    return interpretation, metrics


def _vocabulary(registry=None) -> dict[str, dict]:
    """Installed components' declared vocabularies by slot (registry
    `constraints.vocabulary`), for the multi-call plan — the shoe call
    lists the styles the installed footwear component supports."""
    from character_factory.registry import Registry, RegistryError

    try:
        registry = registry or Registry.default()
    except Exception:
        return {}
    vocabulary: dict[str, dict] = {}
    for slot in vocab.ALL_SLOTS:
        try:
            resolved = registry.resolve_slots([slot])
        except RegistryError:
            continue
        for name, entry in resolved.items():
            declared = entry.document.get("constraints", {}).get("vocabulary", {})
            if isinstance(declared, dict) and declared:
                vocabulary[name] = declared
    return vocabulary


def _slot_guidance(registry=None) -> dict[str, str]:
    """Per-slot field guidance from the installed components' registry
    entries (`interpretation.fields`) — version-bound data, because each
    component version declares what it wants to be told — plus declared
    vocabulary constraints (the interpreter clamps prompts to what the
    installed component supports)."""
    from character_factory.registry import Registry, RegistryError

    try:
        registry = registry or Registry.default()
    except Exception:
        return {}
    resolved = {}
    for slot in vocab.ALL_SLOTS:
        try:
            resolved.update(registry.resolve_slots([slot]))
        except RegistryError:
            continue   # a slot nothing serves simply gets no guidance
    guidance: dict[str, str] = {}
    try:
        figure_entry = registry.get("make-figure")
        fields = figure_entry.document.get("interpretation", {}).get("fields")
        if isinstance(fields, str):
            guidance["figure"] = fields
    except RegistryError:
        pass   # no installed figure component: generic guidance only
    for slot, entry in resolved.items():
        document = entry.document
        fields = document.get("interpretation", {}).get("fields")
        parts = [fields] if isinstance(fields, str) else []
        vocabulary = document.get("constraints", {}).get("vocabulary", {})
        for kind, values in vocabulary.items():
            if isinstance(values, list) and values:
                parts.append(
                    f"supported {kind}: {', '.join(str(v) for v in values)} — "
                    f"stay within this vocabulary"
                )
        if parts:
            guidance[slot] = "; ".join(parts)
    return guidance
