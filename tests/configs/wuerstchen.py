from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.wuerstchen import WuerstchenPrior
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import SDDataPreprocessor, WuerstchenPriorModel
from diffengine.models.editors.wuerstchen.efficient_net_encoder import (
    EfficientNetEncoder,
)
from diffengine.models.losses import L2Loss

model_kwargs = {
    "c_in": 2,
    "c": 8,
    "depth": 2,
    "c_cond": 32,
    "c_r": 8,
    "nhead": 2,
}
model = dict(
    type=WuerstchenPriorModel,
    tokenizer=dict(
        type=CLIPTokenizer.from_pretrained,
        pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip"),
        scheduler=dict(type=DDPMWuerstchenScheduler),
        text_encoder=dict(type=CLIPTextModel),
        image_encoder=dict(type=EfficientNetEncoder, c_latent=2),
        prior=dict(type=WuerstchenPrior,
                **model_kwargs),
    data_preprocessor=dict(type=SDDataPreprocessor),
    loss=dict(type=L2Loss))
