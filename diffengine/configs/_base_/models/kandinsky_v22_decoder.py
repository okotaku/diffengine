from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from transformers import CLIPVisionModelWithProjection

from diffengine.models.editors import KandinskyV22Decoder

decoder_model="kandinsky-community/kandinsky-2-2-decoder"
prior_model="kandinsky-community/kandinsky-2-2-prior"
model = dict(type=KandinskyV22Decoder,
             decoder_model=decoder_model,
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             image_encoder=dict(
                 type=CLIPVisionModelWithProjection.from_pretrained,
                subfolder="image_encoder"),
             vae=dict(
                type=VQModel.from_pretrained,
                subfolder="movq"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                       subfolder="unet"))
