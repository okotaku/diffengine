from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from transformers import CLIPVisionModelWithProjection

from diffengine.models.editors import KandinskyV22Decoder

decoder_model="kandinsky-community/kandinsky-2-2-decoder"
prior_model="kandinsky-community/kandinsky-2-2-prior"
model = dict(type=KandinskyV22Decoder,
             decoder_model=decoder_model,
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=decoder_model,
                            subfolder="scheduler"),
             image_encoder=dict(
                 type=CLIPVisionModelWithProjection.from_pretrained,
                pretrained_model_name_or_path=prior_model,
                subfolder="image_encoder"),
             vae=dict(
                type=VQModel.from_pretrained,
                pretrained_model_name_or_path=decoder_model,
                subfolder="movq"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                       pretrained_model_name_or_path=decoder_model,
                       subfolder="unet"))
