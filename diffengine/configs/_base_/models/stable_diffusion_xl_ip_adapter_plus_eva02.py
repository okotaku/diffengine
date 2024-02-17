import timm
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.embeddings import IPAdapterPlusImageProjection
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

from diffengine.datasets.transforms import TimmImageProcessor
from diffengine.models.editors import TimmIPAdapterXLPlus

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
model = dict(type=TimmIPAdapterXLPlus,
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
            image_encoder=dict(
                type=timm.create_model,
                model_name="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
                pretrained=True,
                num_classes=0),
            image_projection=dict(type=IPAdapterPlusImageProjection,
                                hidden_dims=1280,
                                    depth=4,
                                    dim_head=64,
                                    heads=20,
                                    num_queries=16,
                                    ffn_ratio=4),
            feature_extractor=dict(
                type=TimmImageProcessor,
                pretrained="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"),
            gradient_checkpointing=True)
