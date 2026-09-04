"""Interpreter configuration: which model writes the slot prompts.

Model choice is configuration, never code. The default backend is a local
model run in-process; `model` names either a registry component id or a
filesystem path to weights. Several backends can be configured at once
under short **aliases** and selected per request (the create UI's model
selector; the SaaS offers tiers the same way) — the alias→model mapping
is local configuration, so no model identity ever appears in code or in
a character's provenance.

With nothing configured, the default is the registry's ``interpreter``
component: the default local model, fetched and hash-verified like any
other component. The model behind that id is registry data.

``mode`` chooses how the model is asked: ``single`` (one instruction, one
JSON document) or ``multi`` (one narrow question per component — see
``interpreter/multi.py``). The default ``auto`` is ``multi`` for local
models and ``single`` for endpoints, which is where each does its best
work.

``quantization`` chooses the weight format a local model is loaded in:
``nf4`` (4-bit, the default), ``int8`` or ``bf16``. The download is the
same either way — the model is quantized as it loads — so this is purely
a VRAM-versus-speed choice; the default keeps the whole generation path
inside about 10 GB so a card also driving a desktop and other programs
still fits. A CPU device ignores it and runs at full precision.

The ``interpreter`` object in the cache root's ``config.json``:

    {
      "interpreter": {
        "default": "local-a",
        "mode": "auto",
        "quantization": "nf4",
        "instruction": "…optional system-prompt override (single mode)…",
        "backends": {
          "local-a": {"model": "<registry id or weights path>"},
          "local-b": {"model": "<weights path>", "quantization": "bf16"},
          "cloud":   {"endpoint": "https://…/v1", "model": "…",
                      "api_key": "…", "mode": "single"},
          }
      }
    }

Environment overrides
(``CHARACTER_FACTORY_INTERPRETER_MODEL`` / ``_ENDPOINT`` / ``_API_KEY`` /
``_MODE`` / ``_QUANTIZATION``) take precedence over the file and describe
the default backend. There is no non-model mode: when the default
component cannot be fetched and nothing else is configured,
interpretation is a hard, named error.

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
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from character_factory.registry.store import cache_dir

__all__ = [
    "BACKEND_FIELDS",
    "DEFAULT_MODEL_COMPONENT",
    "DEFAULT_QUANTIZATION",
    "MODES",
    "QUANTIZATIONS",
    "InterpreterConfig",
    "available_backends",
    "delete_backend",
    "load_interpreter_config",
    "resolve_interpreter_config",
    "save_backend",
    "validate_backend",
]

ENV_MODEL = "CHARACTER_FACTORY_INTERPRETER_MODEL"
ENV_ENDPOINT = "CHARACTER_FACTORY_INTERPRETER_ENDPOINT"
ENV_API_KEY = "CHARACTER_FACTORY_INTERPRETER_API_KEY"
ENV_MODE = "CHARACTER_FACTORY_INTERPRETER_MODE"
ENV_QUANTIZATION = "CHARACTER_FACTORY_INTERPRETER_QUANTIZATION"
ENV_AUDIT_LOG = "CHARACTER_FACTORY_INTERPRETER_AUDIT_LOG"

# The registry component that names the default local model. The id is
# code; the model behind it is data.
DEFAULT_MODEL_COMPONENT = "interpreter"
MODES = ("auto", "single", "multi")
# Weight formats a local model can be loaded in. The default is the
# smallest: the generation path is sized to fit beside other GPU users.
QUANTIZATIONS = ("nf4", "int8", "bf16")
DEFAULT_QUANTIZATION = "nf4"


@dataclass(frozen=True)
class InterpreterConfig:
    model: str | None = None      # registry component id OR local weights path
    endpoint: str | None = None   # OpenAI-compatible chat-completions base URL
    api_key: str | None = None
    instruction: str | None = None   # system-prompt override (data, not code)
    repetition_penalty: float = 1.0  # >1 damps greedy repetition loops
    audit_log: str | None = None     # protected JSONL; never served over HTTP
    mode: str = "auto"               # auto | single | multi
    quantization: str = DEFAULT_QUANTIZATION   # nf4 | int8 | bf16 (local model)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(
                f"interpreter mode must be one of {', '.join(MODES)}; got {self.mode!r}"
            )
        if self.quantization not in QUANTIZATIONS:
            raise ValueError(
                "interpreter quantization must be one of "
                f"{', '.join(QUANTIZATIONS)}; got {self.quantization!r}"
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


def _section_defaults(section: dict) -> dict:
    """The section-level settings every backend entry inherits unless it
    sets its own."""
    return {key: section.get(key) for key in ("instruction", "mode", "quantization")}


def _config_from(values: dict, defaults: dict) -> InterpreterConfig:
    return InterpreterConfig(
        model=values.get("model"),
        endpoint=values.get("endpoint"),
        api_key=values.get("api_key"),
        instruction=values.get("instruction") or defaults.get("instruction"),
        repetition_penalty=float(values.get("repetition_penalty", 1.0)),
        audit_log=os.environ.get(ENV_AUDIT_LOG) or values.get("audit_log"),
        mode=os.environ.get(ENV_MODE) or values.get("mode")
        or defaults.get("mode") or "auto",
        quantization=os.environ.get(ENV_QUANTIZATION) or values.get("quantization")
        or defaults.get("quantization") or DEFAULT_QUANTIZATION,
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
    defaults = _section_defaults(section)
    backends = section.get("backends", {})
    if not isinstance(backends, dict):
        raise ValueError("interpreter 'backends' must be a JSON object")

    if alias is not None and alias != "default" and alias not in backends:
        raise ValueError(
            f"unknown interpreter backend {alias!r}; configured: "
            f"{', '.join(sorted(backends)) or '(none)'}"
        )
    if alias is not None and alias in backends:
        return alias, _config_from(backends[alias], defaults)

    env = {
        "model": os.environ.get(ENV_MODEL),
        "endpoint": os.environ.get(ENV_ENDPOINT),
        "api_key": os.environ.get(ENV_API_KEY),
    }
    if env["model"] or env["endpoint"]:
        return "default", _config_from(
            {**{k: v for k, v in env.items() if v}}, defaults
        )
    default = section.get("default")
    if default and default in backends:
        return default, _config_from(backends[default], defaults)
    return "default", _config_from({"model": DEFAULT_MODEL_COMPONENT}, defaults)


def load_interpreter_config(alias: str | None = None) -> InterpreterConfig:
    """The configuration for one interpreter backend."""
    return resolve_interpreter_config(alias)[1]


def _entry_kind(entry: dict) -> str:
    return "endpoint" if entry.get("endpoint") else "local-model"


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def peak_vram_bytes(inference: dict, quantization: str) -> int | None:
    """A component's declared peak VRAM for one weight format.

    ``inference.peak_vram_bytes`` is either one number (the same whatever
    the format) or a table keyed by quantization; a format the table does
    not state is unknown, not zero.
    """
    declared = inference.get("peak_vram_bytes")
    if isinstance(declared, dict):
        declared = declared.get(quantization)
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return declared
    return None


def _local_readiness(config: InterpreterConfig, registry, device: str) -> dict:
    """Readiness of a local model: are its weights on disk, and does it fit
    the device — cheap checks only (existence and sizes; no hashing, no
    model load, no network)."""
    from pathlib import Path

    from character_factory.preflight import device_memory
    from character_factory.registry.store import component_dir, missing_bytes

    row: dict = {"quantization": config.quantization, "download_bytes": None,
                 "vram_bytes": None, "fits": None, "device_bytes": None,
                 "description": None}
    reasons: list[str] = []
    source = config.model or ""
    path = Path(source).expanduser()
    if path.exists():
        if not (path / "config.json").is_file():
            reasons.append("weights path holds no model config")
        row["download_bytes"] = 0
    else:
        if registry is None:
            from character_factory.registry import Registry

            registry = Registry.default()
        try:
            entry = registry.get(source)
        except Exception:  # noqa: BLE001 — not a path, not a component
            reasons.append("neither a registry component nor a weights path")
        else:
            if entry.artifacts:
                missing = missing_bytes(entry)
                row["download_bytes"] = missing
                if missing:
                    reasons.append(f"weights not downloaded ({_gb(missing)})")
            elif not component_dir(entry).is_dir():
                reasons.append("component not published yet")
            else:
                row["download_bytes"] = 0
            needed = peak_vram_bytes(entry.inference, config.quantization)
            if needed is not None:
                row["vram_bytes"] = needed
            # A published component is public knowledge (its index entry
            # says what it is); a private weights path stays a path.
            description = entry.document.get("description")
            row["description"] = description if isinstance(description, str) else None
    if device.partition(":")[0] != "cpu":
        available = device_memory(device)
        row["device_bytes"] = available
        if available is None:
            row["fits"] = False
            reasons.append("no CUDA device detected")
        elif row["vram_bytes"] is not None:
            row["fits"] = available >= row["vram_bytes"]
            if not row["fits"]:
                reasons.append(
                    f"needs {_gb(row['vram_bytes'])} of VRAM; "
                    f"{_gb(available)} detected"
                )
    row["ready"] = not reasons
    row["reason"] = "; ".join(reasons) or None
    return row


def _describe(alias: str, entry: dict, config: InterpreterConfig, *,
              default: bool, registry, device: str) -> dict:
    row = {
        "alias": alias,
        "kind": _entry_kind(entry),
        "default": default,
        "label": entry.get("label") if isinstance(entry.get("label"), str) else None,
        "mode": config.effective_mode,
    }
    if row["kind"] == "endpoint":
        from urllib.parse import urlsplit

        row.update(
            ready=True, reason=None,
            endpoint_host=urlsplit(config.endpoint or "").hostname,
            has_key=bool(config.api_key),
        )
    else:
        row.update(_local_readiness(config, registry, device))
    return row


def available_backends(*, registry=None, device: str = "cuda") -> list[dict]:
    """The selectable backends with their readiness, default first.

    Aliases and kinds only, never model identities: an endpoint row carries
    its host and whether a key is configured, a local-model row the bytes
    still to download and whether the model fits `device`. ``ready`` is
    false with a human ``reason`` when a create against that backend would
    fail today; ``download_bytes`` > 0 with ``fits`` not false means a
    create will fetch the weights first. The ``default`` row is what an
    unaliased request resolves to right now — the file's `default` entry
    (marked on its own row) or, with nothing configured, the registry's
    ``interpreter`` component under the alias ``default``.
    """
    section = _file_section()
    defaults = _section_defaults(section)
    backends = section.get("backends", {})
    if not isinstance(backends, dict):
        backends = {}
    default_alias, default_config = resolve_interpreter_config(None)
    rows = []
    if default_alias not in backends:
        entry = {"endpoint": default_config.endpoint,
                 "model": default_config.model}
        rows.append(_describe(default_alias, entry, default_config,
                              default=True, registry=registry, device=device))
    for alias in sorted(backends):
        entry = backends[alias] or {}
        config = _config_from(entry, defaults)
        rows.append(_describe(alias, entry, config,
                              default=alias == default_alias,
                              registry=registry, device=device))
    rows.sort(key=lambda row: not row["default"])
    return rows


# -- writing the backends table ---------------------------------------------
# The server's setup flow (and anything else that wants to configure a
# backend without hand-editing the file) goes through these. The file may
# hold API keys: writes are atomic and leave it owner-readable only.

_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")
BACKEND_FIELDS = frozenset({
    "endpoint", "model", "api_key", "mode", "quantization",
    "repetition_penalty", "instruction", "label",
})


def _read_document() -> tuple[Path, dict]:
    path = cache_dir() / "config.json"
    document: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"unreadable config {path}: {error}") from error
        if isinstance(loaded, dict):
            document = loaded
    section = document.setdefault("interpreter", {})
    if not isinstance(section, dict):
        raise ValueError(f"{path}: 'interpreter' must be a JSON object")
    if not isinstance(section.get("backends"), dict):
        section["backends"] = {}
    return path, document


def _write_document(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
        delete=False, encoding="utf-8",
    ) as output:
        json.dump(document, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.chmod(output.name, 0o600)
    os.replace(output.name, path)


def validate_backend(alias: str, values: dict) -> dict:
    """Check one backends-table entry as a client would submit it and
    return it normalized. Raises ValueError with the reason."""
    if not isinstance(alias, str) or not _ALIAS_RE.match(alias):
        raise ValueError(
            "backend alias must be 1–32 characters of a–z, 0–9, '.', '_' or "
            "'-', starting with a letter or digit"
        )
    if alias == "default":
        raise ValueError("'default' is reserved for the configured default")
    if not isinstance(values, dict):
        raise ValueError("backend must be a JSON object")
    unknown = set(values) - BACKEND_FIELDS
    if unknown:
        raise ValueError(
            f"unknown backend field(s): {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(sorted(BACKEND_FIELDS))}"
        )
    clean: dict = {}
    for key in ("endpoint", "model", "api_key", "instruction", "label"):
        if key in values and values[key] is not None:
            if not isinstance(values[key], str):
                raise ValueError(f"{key} must be a string")
            value = values[key].strip()
            if value or key == "api_key":
                clean[key] = value
    if "endpoint" in clean:
        from urllib.parse import urlsplit

        parts = urlsplit(clean["endpoint"])
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("endpoint must be an http(s) URL")
    if not clean.get("endpoint") and not clean.get("model"):
        raise ValueError(
            "a backend needs an endpoint URL or a model (registry component "
            "id or local weights path)"
        )
    if "mode" in values and values["mode"] is not None:
        if values["mode"] not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        clean["mode"] = values["mode"]
    if "quantization" in values and values["quantization"] is not None:
        if values["quantization"] not in QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {', '.join(QUANTIZATIONS)}"
            )
        clean["quantization"] = values["quantization"]
    if "repetition_penalty" in values and values["repetition_penalty"] is not None:
        penalty = values["repetition_penalty"]
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) \
                or penalty <= 0:
            raise ValueError("repetition_penalty must be a positive number")
        clean["repetition_penalty"] = float(penalty)
    return clean


def save_backend(alias: str, values: dict, *,
                 default: bool | None = None) -> None:
    """Create or replace the backends-table entry `alias`.

    A submission without ``api_key`` keeps the stored key (so a URL or
    model can be edited without re-entering it); an empty ``api_key``
    removes it. `default` True makes the alias the default backend, False
    clears that if it currently is; None leaves the default alone.
    """
    clean = validate_backend(alias, values)
    path, document = _read_document()
    section = document["interpreter"]
    existing = section["backends"].get(alias) or {}
    if "api_key" not in clean and existing.get("api_key"):
        clean["api_key"] = existing["api_key"]
    elif clean.get("api_key") == "":
        del clean["api_key"]
    section["backends"][alias] = clean
    if default is True:
        section["default"] = alias
    elif default is False and section.get("default") == alias:
        del section["default"]
    _write_document(path, document)


def delete_backend(alias: str) -> None:
    """Remove the backends-table entry `alias` (and the default pointer if
    it named it). Raises KeyError for an unknown alias."""
    path, document = _read_document()
    section = document["interpreter"]
    if alias not in section["backends"]:
        raise KeyError(alias)
    del section["backends"][alias]
    if section.get("default") == alias:
        del section["default"]
    _write_document(path, document)
