from typing import Any

import pytest
from diffusers import UNet2DConditionModel
from peft import LoHaConfig, LoKrConfig, LoraConfig, OFTConfig

from diffengine.models.archs import (
    create_peft_config,
    process_ip_adapter_state_dict,
    set_unet_ip_adapter,
)
from diffengine.models.editors.ip_adapter.image_projection import ImageProjModel
from diffengine.models.editors.ip_adapter.resampler import Resampler


def test_set_unet_ip_adapter():
    unet = UNet2DConditionModel.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")
    assert not any("processor" in k for k in unet.state_dict())
    set_unet_ip_adapter(unet, num_tokens=4)
    assert any("processor.to_k_ip" in k for k in unet.state_dict())
    assert any("processor.to_v_ip" in k for k in unet.state_dict())


def test_create_peft_config():
    config: dict[str, Any] = dict(
        type="Dummy",
    )
    with pytest.raises(AssertionError, match="Unknown PEFT type"):
        create_peft_config(config)

    config = dict(
        type="LoRA",
        r=4,
    )
    config = create_peft_config(config)
    assert isinstance(config, LoraConfig)
    assert config.r == 4

    config = dict(
        type="LoHa",
        r=8,
        alpha=2,
    )
    config = create_peft_config(config)
    assert isinstance(config, LoHaConfig)
    assert config.r == 8
    assert config.alpha == 2

    config = dict(
        type="LoKr",
        r=8,
        alpha=2,
    )
    config = create_peft_config(config)
    assert isinstance(config, LoKrConfig)
    assert config.r == 8
    assert config.alpha == 2

    config = dict(
        type="OFT",
        r=8,
    )
    config = create_peft_config(config)
    assert isinstance(config, OFTConfig)
    assert config.r == 8


def test_process_ip_adapter_state_dict():
    unet = UNet2DConditionModel.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")
    set_unet_ip_adapter(unet, num_tokens=4)
    proj = ImageProjModel(
        cross_attention_dim=128,
        clip_embeddings_dim=128,
        clip_extra_context_tokens=4,
    )
    proj_state_dict = process_ip_adapter_state_dict(unet, proj)
    assert list(proj_state_dict.keys()) == ["image_proj", "ip_adapter"]
    assert "proj.weight" in proj_state_dict["image_proj"]
    assert len(proj_state_dict["ip_adapter"]) == 24

    resampler = Resampler(
        embed_dims=32,
        output_dims=32,
        hidden_dims=128,
        depth=1,
        head_dims=2,
        num_heads=2,
        num_queries=4,
        ffn_ratio=1,
    )
    proj_state_dict = process_ip_adapter_state_dict(unet, resampler)
    assert list(proj_state_dict.keys()) == ["image_proj", "ip_adapter"]
    assert "latents" in proj_state_dict["image_proj"]
    assert len(proj_state_dict["ip_adapter"]) == 24
