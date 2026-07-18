#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.xvla.modeling_xvla import XVLAModel
from lerobot.policies.xvla_light.configuration_xvla_light import XVLALightConfig
from lerobot.policies.xvla_light.florence_cache import FlorenceFeatureCache
from lerobot.policies.xvla_light.modeling_xvla_light import XVLALightModel


def _features(offset: float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    vlm = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) + offset
    aux = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) + offset
    return vlm, aux


def test_feature_cache_roundtrip_and_misses(tmp_path: Path):
    cache = FlorenceFeatureCache(tmp_path / "florence.sqlite", signature="test-signature")
    cache.put_many({7: _features(), 11: _features(20)})

    loaded = cache.get_many([11, 99, 7])

    assert set(loaded) == {7, 11}
    assert torch.equal(loaded[7]["vlm_features"], _features()[0])
    assert torch.equal(loaded[11]["aux_visual_inputs"], _features(20)[1])
    assert cache.count() == 2


def test_feature_cache_does_not_replace_existing_rows(tmp_path: Path):
    cache = FlorenceFeatureCache(tmp_path / "florence.sqlite", signature="test-signature")
    cache.put_many({7: _features()})
    cache.put_many({7: _features(100)})

    assert torch.equal(cache.get_many([7])[7]["vlm_features"], _features()[0])


def test_feature_cache_rejects_incompatible_signature(tmp_path: Path):
    path = tmp_path / "florence.sqlite"
    FlorenceFeatureCache(path, signature="first").count()

    with pytest.raises(ValueError, match="incompatible signature"):
        FlorenceFeatureCache(path, signature="second").count()


def test_cache_config_requires_path_and_legacy_freeze_flags():
    with pytest.raises(ValueError, match="florence_cache_path"):
        XVLALightConfig(
            device="cpu",
            cache_florence_features=True,
            freeze_vision_encoder=True,
            freeze_language_encoder=True,
        )

    with pytest.raises(ValueError, match="requires both"):
        XVLALightConfig(
            device="cpu",
            cache_florence_features=True,
            florence_cache_path="cache.sqlite",
        )


def test_model_uses_cache_after_first_sample_visit(monkeypatch, tmp_path: Path):
    model = XVLALightModel.__new__(XVLALightModel)
    torch.nn.Module.__init__(model)
    model._florence_cache = FlorenceFeatureCache(tmp_path / "florence.sqlite", signature="test")
    model._cache_requests = 0
    model._cache_hits = 0
    model.config = SimpleNamespace(cache_florence_features=True)
    model.training = True
    online_call_count = 0

    def fake_forward_vlm(self, input_ids, pixel_values, image_mask):  # noqa: ARG001
        nonlocal online_call_count
        online_call_count += 1
        values = input_ids[:, :1].to(dtype=torch.float32)
        return {
            "vlm_features": values.unsqueeze(-1).expand(-1, 2, 3).clone(),
            "aux_visual_inputs": values.unsqueeze(-1).expand(-1, 4, 3).clone(),
        }

    monkeypatch.setattr(XVLAModel, "forward_vlm", fake_forward_vlm)
    input_ids = torch.tensor([[5, 0], [9, 0]])
    pixels = torch.zeros(2, 1, 3, 4, 4)
    masks = torch.ones(2, 1, dtype=torch.bool)
    cache_keys = torch.tensor([101, 202])

    first = model.forward_vlm(input_ids, pixels, masks, cache_keys)
    second = model.forward_vlm(input_ids, pixels, masks, cache_keys)

    assert online_call_count == 1
    assert model._cache_hits == 2
    assert torch.equal(first["vlm_features"], second["vlm_features"])
    assert torch.equal(first["aux_visual_inputs"], second["aux_visual_inputs"])
