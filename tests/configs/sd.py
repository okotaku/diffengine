from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import SDDataPreprocessor, StableDiffusion
from diffengine.models.losses import L2Loss

base_model = "diffusers/tiny-stable-diffusion-torch"
model = dict(
            type=StableDiffusion,
             model=base_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="scheduler"),
             text_encoder=dict(type=CLIPTextModel.from_pretrained,
                               pretrained_model_name_or_path=base_model,
                               subfolder="text_encoder"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             pretrained_model_name_or_path=base_model,
                             subfolder="unet"),
            data_preprocessor=dict(type=SDDataPreprocessor),
            loss=dict(type=L2Loss))
