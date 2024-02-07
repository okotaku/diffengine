from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from diffengine.models.editors import IPAdapterXL
from diffengine.models.editors.ip_adapter.image_projection import ImageProjModel

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
model = dict(type=IPAdapterXL,
             model=base_model,
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
                pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             subfolder="unet"),
             image_encoder=dict(type=CLIPVisionModelWithProjection.from_pretrained,
                                pretrained_model_name_or_path="h94/IP-Adapter",
                                subfolder="sdxl_models/image_encoder"),
             image_projection=dict(type=ImageProjModel,
                                   clip_extra_context_tokens=4),
             gradient_checkpointing=True)
