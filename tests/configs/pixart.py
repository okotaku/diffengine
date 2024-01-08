from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    Transformer2DModel,
)
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.models.editors import PixArtAlpha, PixArtAlphaDataPreprocessor
from diffengine.models.losses import L2Loss

base_model = "PixArt-alpha/PixArt-XL-2-1024-MS"
model = dict(
            type=PixArtAlpha,
             model=base_model,
             tokenizer=dict(
                 type=AutoTokenizer.from_pretrained,
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
             vae=dict(
                type=AutoencoderKL),
             transformer=dict(type=Transformer2DModel,
                               sample_size=8,
                num_layers=2,
                patch_size=2,
                attention_head_dim=8,
                num_attention_heads=3,
                caption_channels=32,
                in_channels=4,
                cross_attention_dim=24,
                out_channels=8,
                attention_bias=True,
                activation_fn="gelu-approximate",
                num_embeds_ada_norm=1000,
                norm_type="ada_norm_single",
                norm_elementwise_affine=False,
                norm_eps=1e-6),
            data_preprocessor=dict(type=PixArtAlphaDataPreprocessor),
            loss=dict(type=L2Loss))
