from diffusers import DDPMScheduler, Kandinsky3UNet, VQModel
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.editors import KandinskyV3

base_model = "kandinsky-community/kandinsky-3"
model = dict(type=KandinskyV3,
             model=base_model,
             tokenizer=dict(type=T5Tokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               subfolder="text_encoder",
                               variant="fp16"),
             vae=dict(
                type=VQModel.from_pretrained,
                subfolder="movq",
                variant="fp16"),
             unet=dict(type=Kandinsky3UNet.from_pretrained,
                       subfolder="unet",
                       variant="fp16"))
