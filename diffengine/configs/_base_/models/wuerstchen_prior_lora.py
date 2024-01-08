from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.wuerstchen import WuerstchenPrior
from transformers import CLIPTextModel, PreTrainedTokenizerFast

from diffengine.models.editors import WuerstchenPriorModel
from diffengine.models.editors.wuerstchen.efficient_net_encoder import (
    EfficientNetEncoder,
)

decoder_model="warp-ai/wuerstchen"
prior_model="warp-ai/wuerstchen-prior"
model = dict(type=WuerstchenPriorModel,
             decoder_model=decoder_model,
             tokenizer=dict(type=PreTrainedTokenizerFast.from_pretrained,
                            pretrained_model_name_or_path=prior_model,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMWuerstchenScheduler),
             text_encoder=dict(type=CLIPTextModel.from_pretrained,
                               pretrained_model_name_or_path=prior_model,
                               subfolder="text_encoder"),
             image_encoder=dict(type=EfficientNetEncoder, pretrained=True),
             prior=dict(type=WuerstchenPrior.from_pretrained,
                        pretrained_model_name_or_path=prior_model,
                        subfolder="prior"),
             prior_lora_config=dict(
                type="LoRA",
                r=8,
                lora_alpha=8,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
