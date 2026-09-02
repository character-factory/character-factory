"""The interpreter's model backend: a local language model, in-process.

Runs the configured model (a registry component or a local weights path)
with grammar-constrained decoding against the interpretation schema, so
the output is structurally valid by construction. No daemon, no account,
no telemetry — `transformers` in this process, like every other generation
stage (ARCHITECTURE §2.2). An OpenAI-compatible endpoint can be selected
instead through the `endpoint` config field.

Two modes ask the same model differently: ``single`` sends one instruction
and decodes the whole interpretation document; ``multi`` sends one narrow
call per component (`interpreter/multi.py`) and assembles the document
from the answers. The config's `effective_mode` picks (multi for local
models, single for endpoints, unless configured otherwise); validation
is the same function either way.

The backend releases its weights (and CUDA cache) in `close()`; callers
must close before any diffusion pipeline loads — interpretation and
texture generation never hold VRAM at the same time.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from character_factory.interpreter.config import (
    DEFAULT_MODEL_COMPONENT,
    InterpreterConfig,
)
from character_factory.interpreter.multi import build_calls
from character_factory.interpreter.schema import (
    endpoint_schema,
    interpretation_schema,
)

__all__ = ["InterpreterError", "InterpreterMetrics", "ModelInterpreter"]


class InterpreterError(RuntimeError):
    """The model backend could not produce a usable interpretation.

    Endpoint failures carry only safe public metadata. Raw prompts and
    responses, when audit logging is configured, stay in the protected log.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "interpreter_unavailable",
        classification: str | None = None,
        retryable: bool = True,
        trace_id: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.classification = classification
        self.retryable = retryable
        self.trace_id = trace_id

    @property
    def public_message(self) -> str:
        if self.classification is None:
            return str(self)
        message = f"interpreter failure: {self.classification}"
        if self.trace_id:
            message += f" (trace {self.trace_id})"
        return message


@dataclass
class EndpointResult:
    content: str | None
    finish_reason: str | None
    refusal: str | None
    audit: dict


@dataclass
class InterpreterMetrics:
    backend: str
    mode: str = "single"
    load_seconds: float = 0.0
    generate_seconds: float = 0.0
    calls: dict[str, float] | None = None   # multi mode: seconds per call
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
        try:
            return str(registry.ensure(source))
        except Exception as error:
            hint = ""
            if source == DEFAULT_MODEL_COMPONENT:
                hint = (
                    "; the default model is fetched from its upstream "
                    "repository on first use — check network access and "
                    "disk space, or configure another backend "
                    "(CHARACTER_FACTORY_INTERPRETER_MODEL / _ENDPOINT)"
                )
            raise InterpreterError(
                f"interpreter model {source!r} is not available: {error}{hint}",
                retryable=False,
            ) from error

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        start = time.monotonic()
        weights = self._weights_dir()   # a missing model fails before any import
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Hand any reserved-but-free blocks from earlier GPU stages back to
        # the driver before claiming room for this model; an over-subscribed
        # card degrades to shared-memory paging instead of failing loudly.
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._tokenizer = AutoTokenizer.from_pretrained(weights)
        self._model = AutoModelForCausalLM.from_pretrained(
            weights,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self._model.eval()
        self.metrics.load_seconds = time.monotonic() - start

    def close(self) -> None:
        """Release the model — required before any diffusion load."""
        import gc

        self._model = None
        self._tokenizer = None
        # Reference cycles keep the weights alive past the del; collect
        # before returning the cache or the VRAM stays resident.
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- generation ---------------------------------------------------------

    def _generate_local(self, instruction: str, description: str, schema: dict) -> str:
        self._ensure_loaded()
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

        tokenizer = self._tokenizer
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": description},
        ]
        if getattr(tokenizer, "chat_template", None):
            # Grammar-constrained output has no room for a hidden reasoning
            # block, so tell templates that offer one to close it in the
            # prefix. Left open, a reasoning model writes the JSON "inside"
            # the block and then can only pad: the close tag is illegal
            # under the grammar and EOS never comes. Templates without the
            # switch ignore the argument.
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = f"{instruction}\n\nDescription: {description}\n\nJSON: "
        inputs = tokenizer(text, return_tensors="pt").to(self._model.device)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            tokenizer, JsonSchemaParser(schema)
        )
        with torch.no_grad():
            generate_kwargs = {}
            if self.config.repetition_penalty != 1.0:
                # Greedy decoding can loop inside a grammar-legal string;
                # a mild penalty breaks the loop without fighting the
                # grammar the way hard n-gram bans would.
                generate_kwargs["repetition_penalty"] = self.config.repetition_penalty
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=self._stop_ids(),
                **generate_kwargs,
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _stop_ids(self) -> list[int]:
        """Every token generation should stop on: the model's configured
        EOS plus the tokenizer's end-of-turn token. Some model releases ship
        without a generation config, leaving the model stopping only on
        end-of-text while the chat template ends turns with a different
        token; the grammar allows that end-of-turn token after the closing
        brace and nothing but whitespace afterwards, so the model emits it,
        generation carries on, and the call pads to the token budget."""
        configured = self._model.generation_config.eos_token_id
        if configured is None:
            configured = []
        elif isinstance(configured, int):
            configured = [configured]
        ids = list(configured)
        end_of_turn = self._tokenizer.eos_token_id
        if end_of_turn is not None and end_of_turn not in ids:
            ids.append(end_of_turn)
        return ids

    def _generate_endpoint(
        self,
        instruction: str,
        description: str,
        schema: dict,
        *,
        trace_id: str,
        attempt: int,
    ) -> EndpointResult:
        import urllib.error
        import urllib.request

        # Reasoning-capable endpoints account for hidden reasoning and
        # visible JSON inside the same limit. The old 900-token floor could
        # be exhausted entirely before content was emitted; a retry after a
        # truncated or empty answer triples the budget.
        budget = max(self.config.max_new_tokens, 1800)
        if attempt > 1:
            budget *= 3
        body = {
            "model": self.config.model or "default",
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": description},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "character_interpretation",
                    "strict": True,
                    "schema": endpoint_schema(schema),
                },
            },
            "max_completion_tokens": budget,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            self.config.endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        started = time.monotonic()
        status = None
        response_headers = {}
        raw_body = b""
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = response.status
                response_headers = dict(response.headers.items())
                raw_body = response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            response_headers = dict(error.headers.items()) if error.headers else {}
            raw_body = error.read()
            audit = self._endpoint_audit(
                trace_id=trace_id, attempt=attempt, instruction=instruction,
                description=description, status=status, raw_body=raw_body,
                response_headers=response_headers,
                latency_seconds=time.monotonic() - started,
                classification="http_error",
            )
            self._write_audit(audit)
            raise InterpreterError(
                "interpreter endpoint request failed",
                classification="http_error",
                retryable=status in {408, 409, 425, 429} or status >= 500,
                trace_id=trace_id,
            ) from error
        except Exception as error:
            audit = self._endpoint_audit(
                trace_id=trace_id, attempt=attempt, instruction=instruction,
                description=description, status=status, raw_body=raw_body,
                response_headers=response_headers,
                latency_seconds=time.monotonic() - started,
                classification="transport_error",
                transport_error=type(error).__name__,
            )
            self._write_audit(audit)
            raise InterpreterError(
                "interpreter endpoint request failed",
                classification="transport_error", retryable=True,
                trace_id=trace_id,
            ) from error
        latency_seconds = time.monotonic() - started
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            audit = self._endpoint_audit(
                trace_id=trace_id, attempt=attempt, instruction=instruction,
                description=description, status=status, raw_body=raw_body,
                response_headers=response_headers,
                latency_seconds=latency_seconds,
                classification="invalid_response",
            )
            self._write_audit(audit)
            raise InterpreterError(
                "interpreter endpoint returned an invalid response",
                code="interpreter_invalid_output",
                classification="invalid_response", retryable=True,
                trace_id=trace_id,
            ) from error
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as error:
            audit = self._endpoint_audit(
                trace_id=trace_id, attempt=attempt, instruction=instruction,
                description=description, status=status, raw_body=raw_body,
                response_headers=response_headers,
                latency_seconds=latency_seconds,
                classification="invalid_response", payload=payload,
            )
            self._write_audit(audit)
            raise InterpreterError(
                "interpreter endpoint returned an unexpected payload",
                code="interpreter_invalid_output",
                classification="invalid_response", retryable=True,
                trace_id=trace_id,
            ) from error
        audit = self._endpoint_audit(
            trace_id=trace_id, attempt=attempt, instruction=instruction,
            description=description, status=status, raw_body=raw_body,
            response_headers=response_headers,
            latency_seconds=latency_seconds, payload=payload,
            content=content, finish_reason=finish_reason,
        )
        return EndpointResult(
            content=content,
            finish_reason=finish_reason,
            refusal=message.get("refusal"),
            audit=audit,
        )

    def _endpoint_audit(self, **values) -> dict:
        payload = values.pop("payload", None)
        response_headers = values.pop("response_headers", {})
        raw_body = values.pop("raw_body", b"")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "configured_model": self.config.model,
            "endpoint": self.config.endpoint,
            "http_status": values.pop("status", None),
            "response_bytes": len(raw_body),
            "request_id": response_headers.get("x-request-id"),
            "response_model": payload.get("model") if isinstance(payload, dict) else None,
            "system_fingerprint": payload.get("system_fingerprint")
            if isinstance(payload, dict) else None,
            "usage": payload.get("usage") if isinstance(payload, dict) else None,
            "raw_response": raw_body.decode("utf-8", errors="replace"),
            **values,
        }

    def _write_audit(self, record: dict) -> None:
        if not self.config.audit_log:
            return
        try:
            path = Path(self.config.audit_log).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:
            raise InterpreterError(
                "protected interpreter audit logging is unavailable",
                classification="audit_log_error", retryable=False,
                trace_id=record.get("trace_id"),
            ) from error

    # -- interpretation -----------------------------------------------------

    def interpret(
        self,
        prompt: str,
        slot_guidance: dict[str, str] | None = None,
        vocabulary: dict[str, dict] | None = None,
    ):
        """Description → Interpretation. Raises InterpreterError on any
        failure; the caller decides whether to fall back to rules mode.

        `slot_guidance` (registry per-slot guidance) feeds the single
        instruction; `vocabulary` (installed components' declared
        vocabularies by slot) feeds the multi-call plan."""
        from character_factory.interpreter import Interpretation

        mode = self.config.effective_mode
        self.metrics.mode = mode
        try:
            import torch

            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        trace_id = secrets.token_hex(12) if self.config.endpoint is not None else None
        try:
            # Load (and on first use, fetch) the local model before the
            # generation clock starts, so a cold start is reported once as
            # load time rather than folded into the first call's latency.
            if self.config.endpoint is None and self.generate is None:
                self._ensure_loaded()
            start = time.monotonic()
            if mode == "multi":
                figure, slots, hair, notes, proportions = self._interpret_multi(
                    prompt, vocabulary, trace_id)
            else:
                figure, slots, hair, notes, proportions = self._interpret_single(
                    prompt, slot_guidance, trace_id)
        except InterpreterError:
            raise
        except Exception as error:
            raise InterpreterError(f"interpreter model failed: {error}") from error
        self.metrics.generate_seconds = time.monotonic() - start
        self._record_memory()

        return Interpretation(
            figure=figure,
            slot_prompts=slots,
            hair=hair,
            backend=self.metrics.backend,
            notes=notes,
            proportions=proportions,
        )

    def _interpret_single(self, prompt, slot_guidance, trace_id):
        schema = interpretation_schema()
        instruction = build_instruction(
            slot_guidance or {},
            header=self.config.instruction,
            # Endpoint output is constrained directly by its strict response
            # schema; repeating that schema in the prompt wastes context.
            schema=None,
        )
        if self.config.endpoint is not None and self.generate is None:
            return self._endpoint_document(
                instruction, prompt, schema, trace_id,
                validate=lambda document: _validate(document, prompt),
            )
        if self.generate is not None:
            raw = self.generate(instruction, prompt, schema)
        else:
            raw = self._generate_local(instruction, prompt, schema)
        return _validate(_parse_json(raw), prompt)

    def _interpret_multi(self, prompt, vocabulary, trace_id):
        """Run the call plan and assemble one interpretation document from
        the answers; the shared validator then judges it whole. A bald
        description skips the hair call — the validator would discard the
        answer anyway."""
        bald = any(word in prompt.lower() for word in _BALD_WORDS)
        document: dict = {"textures": {}}
        self.metrics.calls = {}
        for call in build_calls(vocabulary):
            if call.name == "hair" and bald:
                continue
            started = time.monotonic()
            if self.config.endpoint is not None and self.generate is None:
                value = self._endpoint_document(
                    call.instruction, prompt, call.schema, trace_id,
                    validate=lambda document: document, name=call.name,
                )
            else:
                if self.generate is not None:
                    raw = self.generate(call.instruction, prompt, call.schema)
                else:
                    raw = self._generate_local(call.instruction, prompt, call.schema)
                value = _drop_nulls(_parse_json(raw))
            self.metrics.calls[call.name] = round(time.monotonic() - started, 2)
            if call.name == "figure":
                document["figure"] = value
            elif call.name == "hair":
                document["hair"] = value
            elif call.name == "proportions":
                document["proportions"] = value or None
            else:
                document["textures"][call.name] = value
        return _validate(document, prompt)

    def _endpoint_document(
        self, instruction: str, prompt: str, schema: dict, trace_id: str,
        *, validate, name: str | None = None,
    ):
        """One endpoint question, asked up to twice: the first attempt's
        answer is retried only when it was empty or cut off; a refusal, a
        content filter, or an invalid document is final. `validate` turns
        the decoded (null-stripped) document into the value returned."""
        for attempt in (1, 2):
            result = self._generate_endpoint(
                instruction, prompt, schema,
                trace_id=trace_id, attempt=attempt,
            )
            if name is not None:
                result.audit["call"] = name
            classification = None
            retryable = True
            if result.refusal:
                classification = "refusal"
                retryable = False
            elif result.finish_reason == "content_filter":
                classification = "content_filtered"
                retryable = False
            elif result.finish_reason == "length":
                classification = "truncated_response"
            elif not isinstance(result.content, str) or not result.content.strip():
                classification = "empty_response"
            if classification is None:
                try:
                    document = _drop_nulls(_parse_json(result.content))
                except InterpreterError:
                    classification = "invalid_json"
                else:
                    try:
                        validated = validate(document)
                    except InterpreterError:
                        classification = "schema_invalid"
                    else:
                        result.audit["classification"] = "valid"
                        self._write_audit(result.audit)
                        return validated
            result.audit["classification"] = classification
            self._write_audit(result.audit)
            if classification in {"empty_response", "truncated_response"} \
                    and attempt == 1:
                continue
            raise InterpreterError(
                "interpreter endpoint produced no usable document",
                code="interpreter_invalid_output",
                classification=classification,
                retryable=retryable,
                trace_id=trace_id,
            )
        raise AssertionError("endpoint retry loop exhausted")

    def _record_memory(self) -> None:
        self.metrics.peak_rss_bytes = peak_rss_bytes()
        try:
            import torch

            if self.device != "cpu" and torch.cuda.is_available():
                self.metrics.peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass


def peak_rss_bytes() -> int:
    """Peak resident memory of this process, in bytes. `resource` is
    POSIX-only; on Windows, psutil's peak working set stands in when psutil
    is installed. 0 when neither is available — this is a metric, never a
    dependency of the pipeline."""
    try:
        import resource
    except ImportError:
        resource = None
    if resource is not None:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    try:
        import psutil
    except ImportError:
        return 0
    memory = psutil.Process().memory_info()
    return int(getattr(memory, "peak_wset", 0) or 0)


def build_instruction(slot_guidance: dict[str, str],
                      header: str | None = None,
                      schema: dict | None = None) -> str:
    """The system instruction: the task header, the slot surface, and each
    component's declared field guidance (version-bound registry data —
    what the installed component versions want to be told). A configured
    `instruction` replaces the built-in header — prompt engineering is
    conditioning-grade data and lives in configuration, not code."""
    from character_factory.schema import vocab

    if header is None:
        header = (
            "You turn one character description into generation prompts "
            "for a rigged 3D human: one body-generation prompt (the "
            "\"figure\"), one prompt per texture slot, and a hair block. "
            "Respond with a single JSON object and nothing else.\n"
            "\n"
            "The downstream component models are NOT intelligent. They "
            "cannot reason about the character, infer what was meant, or "
            "fill gaps sensibly — each one renders exactly and only what "
            "its prompt literally states, and anything you leave unsaid "
            "is left to chance. YOU are the only reasoning step in the "
            "pipeline: decide everything about this character, then spell "
            "every decision out explicitly in the component's own terms."
        )
    lines = [
        header,
        "",
        "The figure prompt generates the body and face shape: one dense "
        "physique-first sentence — sex/age, height, build, body fat and "
        "musculature, face structure. No name, no clothing, no backstory, "
        "no scene or style words. When the description leaves physique or "
        "face unstated, DERIVE them from who the character is — "
        "occupation, discipline, age, demographics: an Olympic sprinter "
        "gets an elite sprinter's build, an elderly scholar an elderly "
        "scholar's. Specific and plausible, never a generic average.",
        "",
        f"Texture slots (keys are singular, exactly these): "
        f"required {', '.join(vocab.REQUIRED_SLOTS)}; "
        f"optional {', '.join(vocab.OPTIONAL_SLOTS)}. ",
        "Characters default to a complete, appropriate outfit: when the "
        "description does not specify clothing or footwear, invent a top "
        "garment, a bottom garment, and shoes that suit who the "
        "character is. When the description DOES specify the outfit, "
        "follow it exactly — a swimsuit stays a swimsuit, a barefoot or "
        "unclothed character gets no invented coverage (omit the shoe "
        "slot for bare feet; the skin texture renders no graphic "
        "nudity). Never add pieces the description rules out.",
        "Each slot prompt describes only that surface — never mention "
        "another slot's content: no clothing words in the skin prompt, no "
        "footwear in the eye prompt, and footwear appears ONLY in the shoe "
        "prompt, never in the garment prompt.",
    ]
    for slot, guidance in slot_guidance.items():
        lines.append(f"- {slot}: {guidance}")
    lines += [
        "",
        "The hair block uses closed vocabularies; copy enum values exactly "
        "— never paraphrase — and set seed to 0. Pick natural values for "
        "anything the description leaves unsaid.",
        "",
        "Skeletal proportions: include the optional \"proportions\" object "
        "ONLY when the description clearly implies unusual build "
        "(towering, petite, broad-shouldered, long-limbed…). Values are "
        "INTEGERS in hundredths, -40..40 (25 means +0.25; ~10 cm per 100). "
        "Keys: spine_length, neck_length, shoulder_width, arm_length, "
        "hip_width, leg_length. Omit the object — and any key — you have "
        "no clear signal for; never write 0.",
        "",
        'Output shape: {"figure": {"prompt": "…"}, '
        '"textures": {"<slot>": {"prompt": "…"}, …}, '
        '"hair": {…}} — slot keys at the textures level, each holding one '
        '"prompt" string.',
    ]
    if schema is not None:
        # Backends without a decoding grammar (the endpoint) get the full
        # schema in-prompt — the proven way to hold an unconstrained model
        # to closed vocabularies.
        lines += [
            "",
            "Your reply must validate against exactly this JSON Schema:",
            json.dumps(schema),
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


def _drop_nulls(value):
    """Normalize strict-endpoint nullable optionals back to omissions."""
    if isinstance(value, dict):
        return {
            key: _drop_nulls(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, list):
        return [_drop_nulls(child) for child in value]
    return value


_BALD_WORDS = ("bald", "shaved head", "hairless")


def _validate(document: dict, prompt: str):
    """Shape-check the decoded interpretation and decide baldness.

    The grammar already constrains structure when decoding is constrained;
    this re-checks it anyway (the endpoint backend is unconstrained) and
    applies the decisions the grammar cannot make: dropping empty optional
    slots and mapping an explicitly bald description to hair = null."""
    from character_factory.schema import vocab

    figure_entry = document.get("figure")
    figure = figure_entry.get("prompt") if isinstance(figure_entry, dict) else None
    if not isinstance(figure, str) or not figure.strip():
        raise InterpreterError("interpretation has no figure prompt")
    figure = " ".join(figure.split())
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

    # Skeletal-proportion overrides: the schema carries integers in
    # hundredths (closed integer ranges are what the decoding grammar can
    # actually enforce); the document unit is the float. Zero entries mean
    # "no deviation" and are dropped; an empty or absent object means the
    # writer chose not to steer proportions at all.
    proportions = None
    raw_proportions = document.get("proportions")
    if isinstance(raw_proportions, dict) and raw_proportions:
        proportions = {}
        for name, value in raw_proportions.items():
            if name not in vocab.PROPORTION_NAMES:
                raise InterpreterError(f"unknown proportion field {name!r}")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InterpreterError(f"proportion {name!r} is not a number")
            hundredths = max(-40.0, min(40.0, float(value)))
            if hundredths != float(value):
                notes.append(f"proportion {name} clamped to the ±0.40 range")
            if hundredths != 0.0:
                proportions[name] = hundredths / 100.0
        proportions = proportions or None
    return figure, slots, hair, notes, proportions


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
