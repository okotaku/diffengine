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
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMWuerstchenScheduler),
             text_encoder=dict(type=CLIPTextModel.from_pretrained,
                               subfolder="text_encoder"),
             image_encoder=dict(type=EfficientNetEncoder, pretrained=True),
             prior=dict(type=WuerstchenPrior.from_pretrained,
                        subfolder="prior"))
