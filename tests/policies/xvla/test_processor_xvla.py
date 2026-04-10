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
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

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


def _get_relative_and_absolute_steps(
    *,
    use_relative_actions: bool,
    relative_exclude_joints: list[str] | None = None,
    action_feature_names: list[str] | None = None,
) -> tuple[RelativeActionsProcessorStep, AbsoluteActionsProcessorStep]:
    config = XVLAConfig(
        use_relative_actions=use_relative_actions,
        relative_exclude_joints=relative_exclude_joints or ["gripper"],
        action_feature_names=action_feature_names,
    )
    preprocessor, postprocessor = make_xvla_pre_post_processors(config=config, dataset_stats=None)

    relative_step = next(
        step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)
    )
    absolute_step = next(
        step for step in postprocessor.steps if isinstance(step, AbsoluteActionsProcessorStep)
    )
    return relative_step, absolute_step


def test_xvla_relative_actions_steps_present_and_wired() -> None:
    relative_step, absolute_step = _get_relative_and_absolute_steps(use_relative_actions=True)

    assert relative_step.enabled is True
    assert absolute_step.enabled is True
    assert absolute_step.relative_step is relative_step


def test_xvla_relative_actions_roundtrip_reconstructs_absolute_actions() -> None:
    relative_step, absolute_step = _get_relative_and_absolute_steps(use_relative_actions=True)

    state = torch.tensor([[0.5, 1.0, 1.5, 2.0]], dtype=torch.float32)
    actions = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]]],
        dtype=torch.float32,
    )
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: actions.clone(),
    }

    relative_transition = relative_step(transition)
    recovered_transition = absolute_step(
        {TransitionKey.ACTION: relative_transition[TransitionKey.ACTION].clone()}
    )
    torch.testing.assert_close(recovered_transition[TransitionKey.ACTION], actions)


def test_xvla_relative_actions_excluded_gripper_dims_stay_absolute() -> None:
    relative_step, _ = _get_relative_and_absolute_steps(
        use_relative_actions=True,
        relative_exclude_joints=["gripper"],
        action_feature_names=["joint_0", "gripper", "joint_2", "right_gripper"],
    )

    state = torch.tensor([[1.0, 10.0, 2.0, 20.0]], dtype=torch.float32)
    actions = torch.tensor([[[4.0, 0.8, 9.0, 0.2]]], dtype=torch.float32)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: actions.clone(),
    }

    converted = relative_step(transition)[TransitionKey.ACTION]
    torch.testing.assert_close(converted[..., [0, 2]], torch.tensor([[[3.0, 7.0]]], dtype=torch.float32))
    torch.testing.assert_close(converted[..., [1, 3]], actions[..., [1, 3]])


def test_xvla_relative_actions_disabled_is_noop() -> None:
    relative_step, absolute_step = _get_relative_and_absolute_steps(use_relative_actions=False)

    state = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32)
    actions = torch.tensor([[[5.0, 6.0, 7.0]]], dtype=torch.float32)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: actions.clone(),
    }

    relative_transition = relative_step(transition)
    torch.testing.assert_close(relative_transition[TransitionKey.ACTION], actions)

    recovered_transition = absolute_step({TransitionKey.ACTION: actions.clone()})
    torch.testing.assert_close(recovered_transition[TransitionKey.ACTION], actions)
