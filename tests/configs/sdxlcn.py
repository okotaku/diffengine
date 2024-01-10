from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from diffengine.models.editors import (
    SDXLControlNetDataPreprocessor,
    StableDiffusionXLControlNet,
)
from diffengine.models.losses import L2Loss

base_model = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
model = dict(type=StableDiffusionXLControlNet,
             model=base_model,
             controlnet_model="hf-internal-testing/tiny-controlnet-sdxl",
             tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer",
                            use_fast=False),
             tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                            subfolder="tokenizer_2",
                            use_fast=False),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                               subfolder="text_encoder"),
             text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                               subfolder="text_encoder_2"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             subfolder="unet"),
            data_preprocessor=dict(type=SDXLControlNetDataPreprocessor),
            loss=dict(type=L2Loss))
