"""Generation-stack preflight: fail in seconds, with a named cause.

The generation path (identity, textures, interpreter) sits on a tall
stack — the ``[generation]`` import set, a torch build with CUDA support,
a live and sufficiently new driver. Any of these can be broken in a way
that otherwise surfaces minutes later, out of the middle of a
multi-gigabyte model load, as an error naming none of them. The preflight
runs the cheap checks up front — including the first real CUDA call — and
fails before any weights are touched.

``character-factory preflight`` runs it standalone; ``create`` and
``bake`` run it implicitly before loading anything.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

__all__ = [
    "GENERATION_IMPORTS",
    "PreflightCheck",
    "PreflightError",
    "check_generation_stack",
    "device_memory",
    "require_generation_stack",
]

# (import name, pip requirement it comes from): the true import set of the
# generation path — the core torch dependency plus the [generation] extra.
# Kept in sync with pyproject.toml.
GENERATION_IMPORTS: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("diffusers", "diffusers"),
    ("safetensors", "safetensors"),
    ("lmformatenforcer", "lm-format-enforcer"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
)

_REMEDY = "install the generation extra: pip install 'character-factory[generation]'"


@dataclass(frozen=True)
class PreflightCheck:
    """One check result. `name` is the named cause when it failed."""

    name: str
    ok: bool
    detail: str


class PreflightError(RuntimeError):
    """The generation stack cannot run; `failures` name every cause."""

    def __init__(self, failures: list[PreflightCheck]):
        self.failures = list(failures)
        lines = [f"[{check.name}] {check.detail}" for check in self.failures]
        super().__init__(
            "generation preflight failed:\n  " + "\n  ".join(lines)
        )


def check_generation_stack(
    device: str = "cuda",
    imports: tuple[tuple[str, str], ...] | None = None,
) -> list[PreflightCheck]:
    """Run every check and return all results, passed and failed alike.

    Import checks always run. The CUDA checks run when `device` is a cuda
    device and torch imported: build support first, then a one-element
    tensor allocation — the same first CUDA call that would otherwise
    happen mid model-load, so a missing or too-old driver fails here, in
    seconds, under its own name.
    """
    if imports is None:
        imports = GENERATION_IMPORTS
    checks: list[PreflightCheck] = []
    imported: set[str] = set()
    for module, requirement in imports:
        try:
            importlib.import_module(module)
        except ImportError as error:
            checks.append(PreflightCheck(
                "missing-dependency", False,
                f"cannot import {module!r} (from {requirement!r}): {error} — "
                f"{_REMEDY}",
            ))
        else:
            imported.add(module)
            checks.append(PreflightCheck(
                f"import:{module}", True, f"{module} imports"))

    if device.partition(":")[0] == "cuda" and "torch" in imported:
        import torch

        if torch.version.cuda is None:
            checks.append(PreflightCheck(
                "torch-cpu-build", False,
                f"this torch build ({torch.__version__}) has no CUDA "
                f"support; install a CUDA-enabled torch wheel from the "
                f"PyTorch CUDA index for your CUDA version "
                f"(https://download.pytorch.org/whl/<cuXXX>): "
                f"pip install --index-url "
                f"https://download.pytorch.org/whl/<cuXXX> torch",
            ))
        else:
            try:
                torch.zeros(1, device=device)
            except Exception as error:  # noqa: BLE001 — classified below
                message = str(error).strip()
                lowered = message.lower()
                if "driver" in lowered:
                    too_old = ("too old" in lowered or "update" in lowered
                               or "insufficient" in lowered)
                    name = "driver-too-old" if too_old else "driver-unavailable"
                else:
                    name = "cuda-init-failed"
                checks.append(PreflightCheck(
                    name, False,
                    f"CUDA initialization failed on {device!r}: {message}",
                ))
            else:
                properties = torch.cuda.get_device_properties(device)
                checks.append(PreflightCheck(
                    "cuda", True,
                    f"{properties.name}, "
                    f"{properties.total_memory / 2**30:.1f} GiB, "
                    f"toolkit {torch.version.cuda}",
                ))
    return checks


def device_memory(device: str = "cuda") -> int | None:
    """Total memory of `device` in bytes, or None when it is not a usable
    CUDA device (no torch, CPU-only build, no card, or a bad index). A
    property read, not an allocation — cheap enough for a request path."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:  # noqa: BLE001 — "not usable" is the answer, not an error
        return None


# Devices that already passed in this process: repeated creates/bakes on a
# long-lived server pay for the preflight once.
_passed: set[str] = set()


def require_generation_stack(device: str = "cuda") -> None:
    """Raise `PreflightError` (naming every cause) unless the generation
    stack can run on `device`. A pass is remembered per device for the
    life of the process."""
    key = str(device)
    if key in _passed:
        return
    failures = [c for c in check_generation_stack(device) if not c.ok]
    if failures:
        raise PreflightError(failures)
    _passed.add(key)
