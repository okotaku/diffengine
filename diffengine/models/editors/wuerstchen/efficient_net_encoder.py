import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torchvision.models import efficientnet_v2_l, efficientnet_v2_s


class EfficientNetEncoder(ModelMixin, ConfigMixin):
    """EfficientNet encoder for text-to-image generation.

    Copied from https://github.com/huggingface/diffusers/blob/main/examples/
    wuerstchen/text_to_image/modeling_efficient_net_encoder.py
    """

    @register_to_config
    def __init__(self, c_latent: int = 16, c_cond: int = 1280,
                 effnet: str = "efficientnet_v2_s") -> None:
        super().__init__()

        if effnet == "efficientnet_v2_s":
            self.backbone = efficientnet_v2_s(weights="DEFAULT").features
        else:
            self.backbone = efficientnet_v2_l(weights="DEFAULT").features
        self.mapper = nn.Sequential(
            nn.Conv2d(c_cond, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mapper(self.backbone(x))
