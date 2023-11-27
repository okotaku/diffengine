from typing import Any

import pytest
from diffusers import ControlNetModel, UNet2DConditionModel
from peft import LoHaConfig, LoKrConfig, LoraConfig, OFTConfig

from diffengine.models.archs import (
    create_peft_config,
    set_controlnet_ip_adapter,
    set_unet_ip_adapter,
    unet_attn_processors_state_dict,
)
from diffengine.models.archs.ip_adapter import CNAttnProcessor, CNAttnProcessor2_0
from diffengine.models.editors import (
    IPAdapterXL,
    IPAdapterXLDataPreprocessor,
)


def test_set_unet_ip_adapter():
    unet = UNet2DConditionModel.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")
    assert not any("processor" in k for k in unet.state_dict())
    set_unet_ip_adapter(unet)
    assert any("processor.to_k_ip" in k for k in unet.state_dict())
    assert any("processor.to_v_ip" in k for k in unet.state_dict())


def test_set_controlnet_ip_adapter():
    controlnet = ControlNetModel.from_pretrained(
        "hf-internal-testing/tiny-controlnet-sdxl")
    assert all(not isinstance(attn_processor, CNAttnProcessor)
               and not isinstance(attn_processor, CNAttnProcessor2_0)
               for attn_processor in (controlnet.attn_processors.values()))
    set_controlnet_ip_adapter(controlnet)
    assert any(
        isinstance(attn_processor, CNAttnProcessor | CNAttnProcessor2_0)
        for attn_processor in (controlnet.attn_processors.values()))


def test_unet_ip_adapter_layers_to_save():
    model = IPAdapterXL(
        "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
        image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
        data_preprocessor=IPAdapterXLDataPreprocessor())

    unet_lora_layers_to_save = unet_attn_processors_state_dict(model.unet)
    assert len(unet_lora_layers_to_save) > 0


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
        alpha=2,
    )
    config = create_peft_config(config)
    assert isinstance(config, OFTConfig)
    assert config.r == 8
    assert config.alpha == 2
