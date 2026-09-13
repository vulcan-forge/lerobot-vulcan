import torch

from lerobot.utils.device_utils import get_compatible_policy_dtype


def test_bfloat16_falls_back_to_float32_without_native_cuda_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 5))

    assert get_compatible_policy_dtype("bfloat16", "cuda") == "float32"


def test_bfloat16_remains_enabled_on_ampere_or_newer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))

    assert get_compatible_policy_dtype("bfloat16", "cuda") == "bfloat16"


def test_float32_is_never_changed():
    assert get_compatible_policy_dtype("float32", "cuda") == "float32"
