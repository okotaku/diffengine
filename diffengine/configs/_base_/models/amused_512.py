from diffusers import UVit2DModel, VQModel
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from diffengine.models.editors import AMUSEd

base_model = "amused/amused-512"
model = dict(type=AMUSEd,
             model=base_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            subfolder="tokenizer"),
             text_encoder=dict(type=CLIPTextModelWithProjection.from_pretrained,
                               subfolder="text_encoder"),
             vae=dict(
                type=VQModel.from_pretrained,
                subfolder="vqvae"),
             transformer=dict(type=UVit2DModel.from_pretrained,
                       subfolder="transformer"))
