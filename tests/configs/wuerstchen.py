from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.wuerstchen import WuerstchenPrior
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

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
text_config = dict(
    type=CLIPTextConfig,
    bos_token_id=0,
    eos_token_id=2,
    hidden_size=32,
    intermediate_size=37,
    layer_norm_eps=1e-05,
    num_attention_heads=4,
    num_hidden_layers=5,
    pad_token_id=1,
    vocab_size=1000,
)
model = dict(
    type=WuerstchenPriorModel,
    tokenizer=dict(
        type=CLIPTokenizer.from_pretrained,
        pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip"),
        scheduler=dict(type=DDPMWuerstchenScheduler),
        text_encoder=dict(type=CLIPTextModel,
                        config=text_config),
        image_encoder=dict(type=EfficientNetEncoder, c_latent=2),
        prior=dict(type=WuerstchenPrior,
                **model_kwargs),
    data_preprocessor=dict(type=SDDataPreprocessor),
    loss=dict(type=L2Loss))
