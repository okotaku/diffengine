from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import StableDiffusion

base_model = "stablediffusionapi/anything-v5"
model = dict(type=StableDiffusion,
             model=base_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=CLIPTextModel.from_pretrained,
                               subfolder="text_encoder"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             subfolder="unet"),
    unet_lora_config=dict(
        type="LoRA",
        r=128,
        lora_alpha=128,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
    text_encoder_lora_config=dict(
        type="LoRA",
        r=128,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
    finetune_text_encoder=True)
