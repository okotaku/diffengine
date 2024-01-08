from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.embeddings import IPAdapterPlusImageProjection
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from diffengine.models.editors import IPAdapterXLPlus

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
model = dict(type=IPAdapterXLPlus,
             model=base_model,
             tokenizer_one=dict(type=AutoTokenizer.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="tokenizer",
                            use_fast=False),
             tokenizer_two=dict(type=AutoTokenizer.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="tokenizer_2",
                            use_fast=False),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="scheduler"),
             text_encoder_one=dict(type=CLIPTextModel.from_pretrained,
                               pretrained_model_name_or_path=base_model,
                               subfolder="text_encoder"),
             text_encoder_two=dict(type=CLIPTextModelWithProjection.from_pretrained,
                               pretrained_model_name_or_path=base_model,
                               subfolder="text_encoder_2"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             pretrained_model_name_or_path=base_model,
                             subfolder="unet"),
             image_encoder=dict(type=CLIPVisionModelWithProjection.from_pretrained,
                                pretrained_model_name_or_path="h94/IP-Adapter",
                                subfolder="sdxl_models/image_encoder"),
             image_projection=dict(type=IPAdapterPlusImageProjection,
                                   hidden_dims=1280,
                                    depth=4,
                                    dim_head=64,
                                    heads=20,
                                    num_queries=16,
                                    ffn_ratio=4),
             gradient_checkpointing=True)
