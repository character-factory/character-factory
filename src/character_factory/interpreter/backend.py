"""The interpreter's model backend: a local language model, in-process.

Runs the configured model (a registry component or a local weights path)
with grammar-constrained decoding against the interpretation schema, so
the output is structurally valid by construction. No daemon, no account,
no telemetry — `transformers` in this process, like every other generation
stage (ARCHITECTURE §2.2). An OpenAI-compatible endpoint can be selected
instead through the `endpoint` config field.

The backend releases its weights (and CUDA cache) in `close()`; callers
must close before any diffusion pipeline loads — interpretation and
texture generation never hold VRAM at the same time.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from character_factory.interpreter.config import InterpreterConfig
from character_factory.interpreter.schema import interpretation_schema

__all__ = ["InterpreterError", "InterpreterMetrics", "ModelInterpreter"]


class InterpreterError(RuntimeError):
    """The model backend could not produce a usable interpretation."""


@dataclass
class InterpreterMetrics:
    backend: str
    load_seconds: float = 0.0
    generate_seconds: float = 0.0
    peak_gpu_bytes: int | None = None
    peak_rss_bytes: int | None = None

    def as_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items()
                if value is not None}


@dataclass
class ModelInterpreter:
    """One loaded interpreter model, reusable across prompts until closed.

    `generate` is injectable for tests: (instruction, description, schema)
    → the model's raw text. When None, the real backend is chosen from the
    config: the OpenAI-compatible endpoint if configured, else the local
    transformers model.
    """

    config: InterpreterConfig
    registry: object = None
    device: str = "cuda"
    generate: object = None
    metrics: InterpreterMetrics = field(init=False)
    _model: object = field(default=None, init=False)
    _tokenizer: object = field(default=None, init=False)

    def __post_init__(self):
        if self.generate is not None:
            backend = "injected"
        elif self.config.endpoint is not None:
            backend = "endpoint"
        elif self.config.model is not None:
            backend = "local-model"
        else:
            raise InterpreterError("no interpreter model configured")
        self.metrics = InterpreterMetrics(backend=backend)

    # -- model resolution ---------------------------------------------------

    def _weights_dir(self) -> str:
        """The configured model: a filesystem path as-is, anything else a
        registry component id (fetched and verified by the registry)."""
        from pathlib import Path

        source = self.config.model
        if Path(source).expanduser().exists():
            return str(Path(source).expanduser())
        from character_factory.registry import Registry

        registry = self.registry or Registry.default()
        return str(registry.ensure(source))

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        start = time.monotonic()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        weights = self._weights_dir()
        self._tokenizer = AutoTokenizer.from_pretrained(weights)
        self._model = AutoModelForCausalLM.from_pretrained(
            weights,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self._model.eval()
        self.metrics.load_seconds = time.monotonic() - start

    def close(self) -> None:
        """Release the model — required before any diffusion load."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- generation ---------------------------------------------------------

    def _generate_local(self, instruction: str, description: str, schema: dict) -> str:
        import torch
        from lmformatenforcer import JsonSchemaParser

        # transformers 5.x moved PreTrainedTokenizerBase out of
        # tokenization_utils; lm-format-enforcer (≤0.11) still imports the
        # old location and mis-reports it as transformers being absent.
        import transformers
        import transformers.tokenization_utils as _tokenization_utils

        if not hasattr(_tokenization_utils, "PreTrainedTokenizerBase"):
            _tokenization_utils.PreTrainedTokenizerBase = (
                transformers.PreTrainedTokenizerBase
            )
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        self._ensure_loaded()
        tokenizer = self._tokenizer
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": description},
        ]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"{instruction}\n\nDescription: {description}\n\nJSON: "
        inputs = tokenizer(text, return_tensors="pt").to(self._model.device)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            tokenizer, JsonSchemaParser(schema)
        )
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _generate_endpoint(self, instruction: str, description: str) -> str:
        import urllib.request

        body = {
            "model": self.config.model or "default",
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": description},
            ],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            self.config.endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise InterpreterError(
                f"endpoint returned an unexpected payload: {error}"
            ) from error

    # -- interpretation -----------------------------------------------------

    def interpret(self, prompt: str, slot_guidance: dict[str, str] | None = None):
        """Description → Interpretation. Raises InterpreterError on any
        failure; the caller decides whether to fall back to rules mode."""
        from character_factory.interpreter import Interpretation

        schema = interpretation_schema()
        instruction = build_instruction(slot_guidance or {})
        start = time.monotonic()
        try:
            import torch

            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        try:
            if self.generate is not None:
                raw = self.generate(instruction, prompt, schema)
            elif self.config.endpoint is not None:
                raw = self._generate_endpoint(instruction, prompt)
            else:
                raw = self._generate_local(instruction, prompt, schema)
        except InterpreterError:
            raise
        except Exception as error:
            raise InterpreterError(f"interpreter model failed: {error}") from error
        self.metrics.generate_seconds = time.monotonic() - start
        self._record_memory()

        document = _parse_json(raw)
        slots, hair, notes = _validate(document, prompt)
        return Interpretation(
            slot_prompts=slots,
            hair=hair,
            backend=self.metrics.backend,
            notes=notes,
        )

    def _record_memory(self) -> None:
        import resource

        self.metrics.peak_rss_bytes = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        try:
            import torch

            if self.device != "cpu" and torch.cuda.is_available():
                self.metrics.peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass


def build_instruction(slot_guidance: dict[str, str]) -> str:
    """The system instruction: the task, the slot surface, and each
    component's declared field guidance (version-bound registry data —
    what the installed component versions want to be told)."""
    from character_factory.schema import vocab

    lines = [
        "You turn one character description into texture-generation prompts "
        "for a rigged 3D human, one prompt per texture slot, plus a hair "
        "block. Respond with a single JSON object and nothing else.",
        "",
        f"Texture slots (keys are singular, exactly these): "
        f"required {', '.join(vocab.REQUIRED_SLOTS)}; "
        f"optional {', '.join(vocab.OPTIONAL_SLOTS)}. "
        "Omit an optional slot entirely when the description gives it "
        "nothing (a barefoot character has no shoe key).",
        "Each slot prompt describes only that surface — never mention "
        "another slot's content: no clothing words in the skin prompt, no "
        "footwear in the eye prompt, and footwear appears ONLY in the shoe "
        "prompt, never in the garment prompt.",
    ]
    for slot, guidance in slot_guidance.items():
        lines.append(f"- {slot}: {guidance}")
    lines += [
        "",
        "The hair block uses closed vocabularies (the JSON grammar enforces "
        "them); pick conservative, natural values for anything the "
        "description leaves unsaid. Set seed to 0.",
    ]
    return "\n".join(lines)


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    try:
        document = json.loads(text)
    except ValueError as error:
        raise InterpreterError(f"model output is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise InterpreterError("model output is not a JSON object")
    return document


_BALD_WORDS = ("bald", "shaved head", "hairless")


def _validate(document: dict, prompt: str) -> tuple[dict, dict | None, list]:
    """Shape-check the decoded interpretation and decide baldness.

    The grammar already constrains structure when decoding is constrained;
    this re-checks it anyway (the endpoint backend is unconstrained) and
    applies the decisions the grammar cannot make: dropping empty optional
    slots and mapping an explicitly bald description to hair = null."""
    from character_factory.schema import vocab

    textures = document.get("textures")
    if not isinstance(textures, dict):
        raise InterpreterError("interpretation has no textures object")
    slots: dict[str, str] = {}
    for slot, value in textures.items():
        if slot not in vocab.ALL_SLOTS:
            raise InterpreterError(f"unknown texture slot {slot!r}")
        text = value.get("prompt") if isinstance(value, dict) else None
        if not isinstance(text, str) or not text.strip():
            if slot in vocab.OPTIONAL_SLOTS:
                continue
            raise InterpreterError(f"slot {slot!r} has no prompt")
        slots[slot] = " ".join(text.split())
    missing = [slot for slot in vocab.REQUIRED_SLOTS if slot not in slots]
    if missing:
        raise InterpreterError(f"interpretation is missing slots: {missing}")

    notes = []
    hair = document.get("hair")
    if hair is not None and not isinstance(hair, dict):
        raise InterpreterError("hair must be an object")
    if any(word in prompt.lower() for word in _BALD_WORDS):
        hair = None
        notes.append("description reads as bald; hair set to null")
    if hair is not None:
        hair = _repair_hair(hair, notes)
        from character_factory.schema.validation import hair_block_errors

        problems = hair_block_errors(hair)
        if problems:
            raise InterpreterError(
                "hair block invalid after repair: " + "; ".join(problems)
            )
    return slots, hair, notes


def _repair_hair(hair: dict, notes: list) -> dict:
    """The repair loop: fix what the decoding grammar deliberately leaves
    to validation (ARCHITECTURE §2.2). Today that is the color rgb/custom
    co-constraint — an rgb triple is only meaningful with the "custom"
    family, and models like writing 0–255 channels — plus the constants a
    less constrained backend (the endpoint) can get wrong."""
    from character_factory.schema import vocab

    hair = dict(hair)
    hair["schema_version"] = vocab.HAIR_SCHEMA_VERSION
    hair.setdefault("seed", 0)
    color = hair.get("color")
    if isinstance(color, dict) and "rgb" in color:
        color = dict(color)
        if color.get("family") not in (None, "custom"):
            del color["rgb"]   # the named family wins; rgb was decoration
            notes.append("hair color rgb dropped: a named family was given")
        else:
            rgb = color.get("rgb")
            if (isinstance(rgb, list) and len(rgb) == 3
                    and all(isinstance(v, (int, float)) for v in rgb)):
                if any(v > 1 for v in rgb):
                    rgb = [v / 255 for v in rgb]
                    notes.append("hair color rgb rescaled from 0-255 to 0-1")
                color["rgb"] = [min(max(float(v), 0.0), 1.0) for v in rgb]
                color["family"] = "custom"
        hair["color"] = color
    return hair
