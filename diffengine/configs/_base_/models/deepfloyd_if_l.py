from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.editors import DeepFloydIF

base_model = "DeepFloyd/IF-L-v1.0"
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
    gradient_checkpointing=True)
