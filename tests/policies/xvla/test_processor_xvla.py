#!/usr/bin/env python

import pytest
import torch

from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.processor_xvla import (
    XVLAImageNetNormalizeProcessorStep,
    XVLAImageToFloatProcessorStep,
    make_xvla_pre_post_processors,
)
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_IMAGES

OBS_KEY = f"{OBS_IMAGES}.front_left"


def _transition(image: torch.Tensor) -> dict:
    return {TransitionKey.OBSERVATION: {OBS_KEY: image}}


def _imagenet_stats_like(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_STATS["mean"], dtype=image.dtype, device=image.device)
    std = torch.tensor(IMAGENET_STATS["std"], dtype=image.dtype, device=image.device)
    while mean.dim() < image.dim():
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return mean, std


def test_image_to_float_clamps_small_negative_drift() -> None:
    step = XVLAImageToFloatProcessorStep(image_keys=[OBS_KEY])
    image = torch.tensor([[[[-0.0010, 0.2], [0.3, 1.0]]]], dtype=torch.float32).repeat(1, 3, 1, 1)

    out = step(_transition(image))
    converted = out[TransitionKey.OBSERVATION][OBS_KEY]

    assert converted.min().item() == pytest.approx(0.0)
    assert converted.max().item() == pytest.approx(1.0)


def test_image_to_float_clamps_small_positive_overflow() -> None:
    step = XVLAImageToFloatProcessorStep(image_keys=[OBS_KEY])
    image = torch.tensor([[[[0.0, 1.0010], [0.5, 0.8]]]], dtype=torch.float32).repeat(1, 3, 1, 1)

    out = step(_transition(image))
    converted = out[TransitionKey.OBSERVATION][OBS_KEY]

    assert converted.max().item() == pytest.approx(1.0)
    assert converted.min().item() >= 0.0


def test_imagenet_normalize_raises_on_large_violation() -> None:
    step = XVLAImageNetNormalizeProcessorStep(image_keys=[OBS_KEY])
    image = torch.tensor([[[[-0.2, 0.2], [0.7, 1.0]]]], dtype=torch.float32).repeat(1, 3, 1, 1)

    with pytest.raises(ValueError, match="outside hard clamp range"):
        step(_transition(image))


def test_image_to_float_treats_float_1001_as_unit_scale() -> None:
    step = XVLAImageToFloatProcessorStep(image_keys=[OBS_KEY])
    image = torch.tensor([[[[0.5, 1.0010], [0.1, 0.9]]]], dtype=torch.float32).repeat(1, 3, 1, 1)

    out = step(_transition(image))
    converted = out[TransitionKey.OBSERVATION][OBS_KEY]

    # If misclassified as 0-255, values would be ~1/255 scale. We expect unit-scale values here.
    assert converted[0, 0, 0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert converted.max().item() == pytest.approx(1.0)


def test_image_to_float_converts_uint8_to_unit_range() -> None:
    step = XVLAImageToFloatProcessorStep(image_keys=[OBS_KEY])
    image = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8).repeat(1, 3, 1, 1)

    out = step(_transition(image))
    converted = out[TransitionKey.OBSERVATION][OBS_KEY]

    assert converted.dtype == torch.float32
    assert converted[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert converted[0, 0, 0, 1].item() == pytest.approx(128.0 / 255.0, abs=1e-6)
    assert converted[0, 0, 0, 2].item() == pytest.approx(1.0, abs=1e-6)


def test_end_to_end_preprocessor_accepts_regression_case(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    import lerobot.processor.tokenizer_processor as tokenizer_processor

    class DummyTokenizer:
        def __call__(
            self,
            text,
            return_tensors: str | None = None,
            padding: str | None = None,
            truncation: bool | None = None,
            max_length: int | None = None,
            **_: dict,
        ) -> dict[str, torch.Tensor]:
            batch_size = len(text)
            seq_len = max_length or 8
            return {
                "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            }

    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    config = XVLAConfig(tokenizer_name="dummy-tokenizer")
    preprocessor, _ = make_xvla_pre_post_processors(config=config, dataset_stats=None)

    # Regression-like input: tiny negative drift that used to fail strict [0,1] checks.
    image = torch.tensor(
        [
            [[-0.0010, 0.4], [0.2, 1.0]],
            [[0.0, 0.3], [0.2, 0.9]],
            [[0.1, 0.2], [0.7, 0.8]],
        ],
        dtype=torch.float32,
    )
    batch = {
        OBS_KEY: image,
        "task": "fold the shirt",
    }

    processed = preprocessor(batch)
    processed_img = processed[OBS_KEY]

    assert processed_img.shape == (1, 3, 2, 2)
    assert torch.isfinite(processed_img).all()

    # Validate that clamped-to-zero path is reflected after ImageNet normalization.
    mean, std = _imagenet_stats_like(processed_img)
    expected_clamped_zero = (torch.zeros_like(mean) - mean) / std
    assert processed_img[0, 0, 0, 0].item() == pytest.approx(expected_clamped_zero[0, 0, 0, 0].item(), abs=1e-5)
