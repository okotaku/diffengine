from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.editors import DeepFloydIF

base_model = "DeepFloyd/IF-I-XL-v1.0"
model = dict(
    type=DeepFloydIF,
    model=base_model,
    tokenizer=dict(type=T5Tokenizer.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="tokenizer"),
    scheduler=dict(type=DDPMScheduler.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="scheduler"),
    text_encoder=dict(type=T5EncoderModel.from_pretrained,
                    pretrained_model_name_or_path=base_model,
                    subfolder="text_encoder"),
    unet=dict(type=UNet2DConditionModel.from_pretrained,
            pretrained_model_name_or_path=base_model,
            subfolder="unet"),
    gradient_checkpointing=True,
    unet_lora_config=dict(
        type="LoRA",
        r=8,
        lora_alpha=8,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
