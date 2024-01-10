from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.editors import PixArtAlpha

base_model = "PixArt-alpha/PixArt-XL-2-1024-MS"
model = dict(type=PixArtAlpha,
             model=base_model,
             tokenizer=dict(type=T5Tokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               subfolder="text_encoder"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path="stabilityai/sd-vae-ft-ema"),
             transformer=dict(type=Transformer2DModel.from_pretrained,
                             subfolder="transformer"),
             gradient_checkpointing=True)
