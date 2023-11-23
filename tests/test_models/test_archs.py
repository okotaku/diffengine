from typing import Any

import pytest
from diffusers import UNet2DConditionModel
from peft import LoHaConfig, LoKrConfig, LoraConfig

from diffengine.models.archs import (
    create_peft_config,
    set_unet_ip_adapter,
)


def test_set_unet_ip_adapter():
    unet = UNet2DConditionModel.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")
    assert not any("processor" in k for k in unet.state_dict())
    set_unet_ip_adapter(unet)
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
