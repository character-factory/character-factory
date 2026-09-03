"""The generation-stack preflight: fast failure, named causes.

Everything here runs against injected import lists and stub torch modules
— the suite must not depend on which of the [generation] packages the test
environment happens to have, or on a GPU.
"""

import sys
import types

import pytest

import character_factory.preflight as preflight_module
from character_factory.preflight import (
    PreflightError,
    check_generation_stack,
    require_generation_stack,
)


def _fake_torch(cuda_toolkit, zeros=None):
    module = types.ModuleType("torch")
    module.__version__ = "0.0-preflight-test"
    module.version = types.SimpleNamespace(cuda=cuda_toolkit)
    if zeros is not None:
        module.zeros = zeros
    return module


def test_available_imports_pass_on_cpu():
    checks = check_generation_stack(device="cpu", imports=(("json", "json"),))
    assert checks and all(check.ok for check in checks)


def test_missing_dependency_is_named_with_remedy():
    checks = check_generation_stack(
        device="cpu",
        imports=(("cf_preflight_definitely_absent", "some-dist"),),
    )
    [failure] = [check for check in checks if not check.ok]
    assert failure.name == "missing-dependency"
    assert "some-dist" in failure.detail
    assert "character-factory[generation]" in failure.detail


def test_require_raises_naming_every_cause(monkeypatch):
    monkeypatch.setattr(
        preflight_module, "GENERATION_IMPORTS",
        (("cf_absent_one", "dist-one"), ("cf_absent_two", "dist-two")),
    )
    monkeypatch.setattr(preflight_module, "_passed", set())
    with pytest.raises(PreflightError) as excinfo:
        require_generation_stack("cpu")
    message = str(excinfo.value)
    assert "[missing-dependency]" in message
    assert "dist-one" in message and "dist-two" in message


def test_pass_is_remembered_per_device(monkeypatch):
    monkeypatch.setattr(
        preflight_module, "GENERATION_IMPORTS", (("json", "json"),))
    monkeypatch.setattr(preflight_module, "_passed", set())
    require_generation_stack("cpu")
    # Break the stack afterwards: the remembered pass short-circuits, so a
    # long-lived server pays for the preflight once per device.
    monkeypatch.setattr(
        preflight_module, "GENERATION_IMPORTS", (("cf_absent", "dist"),))
    require_generation_stack("cpu")


def test_cpu_only_torch_build_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(None))
    checks = check_generation_stack(device="cuda", imports=(("torch", "torch"),))
    [failure] = [check for check in checks if not check.ok]
    assert failure.name == "torch-cpu-build"
    assert "https://download.pytorch.org/whl/<cuXXX>" in failure.detail
    assert "pip install --index-url" in failure.detail


def test_missing_driver_is_classified(monkeypatch):
    def zeros(*args, **kwargs):
        raise RuntimeError("Found no NVIDIA driver on your system.")

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("12.8", zeros))
    checks = check_generation_stack(device="cuda", imports=(("torch", "torch"),))
    [failure] = [check for check in checks if not check.ok]
    assert failure.name == "driver-unavailable"
    assert "NVIDIA driver" in failure.detail


def test_too_old_driver_is_classified(monkeypatch):
    def zeros(*args, **kwargs):
        raise RuntimeError(
            "The NVIDIA driver on your system is too old (found version "
            "10010). Please update your GPU driver."
        )

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("12.8", zeros))
    checks = check_generation_stack(device="cuda", imports=(("torch", "torch"),))
    [failure] = [check for check in checks if not check.ok]
    assert failure.name == "driver-too-old"


def test_cpu_device_skips_cuda_checks(monkeypatch):
    # A CPU device asks nothing of CUDA even when torch has no CUDA build.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(None))
    checks = check_generation_stack(device="cpu", imports=(("torch", "torch"),))
    assert all(check.ok for check in checks)


def test_cuda_pass_prefers_expandable_segments(monkeypatch):
    # A CUDA pass asks the allocator for expandable segments (reserved
    # memory tracks allocated memory across the load/release sequence) —
    # unless the user configured the allocator themselves.
    settings = []
    torch = _fake_torch("12.8", lambda *a, **k: None)
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda d: types.SimpleNamespace(
            name="card", total_memory=1),
        memory=types.SimpleNamespace(_set_allocator_settings=settings.append),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        preflight_module, "GENERATION_IMPORTS", (("json", "json"),))
    monkeypatch.setattr(preflight_module, "_passed", set())
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    require_generation_stack("cuda")
    assert settings == ["expandable_segments:True"]

    monkeypatch.setattr(preflight_module, "_passed", set())
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    require_generation_stack("cuda")
    assert settings == ["expandable_segments:True"]     # user's setting kept

    monkeypatch.setattr(preflight_module, "_passed", set())
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    require_generation_stack("cpu")
    assert settings == ["expandable_segments:True"]     # cpu: nothing to set
