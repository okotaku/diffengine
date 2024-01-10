from diffusers import DDPMScheduler, PriorTransformer
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffengine.models.editors import KandinskyV22Decoder

decoder_model="kandinsky-community/kandinsky-2-2-decoder"
prior_model="kandinsky-community/kandinsky-2-2-prior"
model = dict(type=KandinskyV22Decoder,
             decoder_model=decoder_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler,
                            beta_schedule="squaredcos_cap_v2",
                            prediction_type="sample"),
             text_encoder=dict(type=CLIPTextModelWithProjection.from_pretrained,
                                 subfolder="text_encoder"),
             image_encoder=dict(
                 type=CLIPVisionModelWithProjection.from_pretrained,
                subfolder="image_encoder"),
             prior=dict(type=PriorTransformer.from_pretrained,
                        subfolder="prior"))
