# Wuerstchen

[Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.00637)

## Abstract

We introduce Würstchen, a novel architecture for text-to-image synthesis that combines competitive performance with unprecedented cost-effectiveness for large-scale text-to-image diffusion models. A key contribution of our work is to develop a latent diffusion technique in which we learn a detailed but extremely compact semantic image representation used to guide the diffusion process. This highly compressed representation of an image provides much more detailed guidance compared to latent representations of language and this significantly reduces the computational requirements to achieve state-of-the-art results. Our approach also improves the quality of text-conditioned image generation based on our user preference study. The training requirements of our approach consists of 24,602 A100-GPU hours - compared to Stable Diffusion 2.1's 200,000 GPU hours. Our approach also requires less training data to achieve these results. Furthermore, our compact latent representations allows us to perform inference over twice as fast, slashing the usual costs and carbon footprint of a state-of-the-art (SOTA) diffusion model significantly, without compromising the end performance. In a broader comparison against SOTA models our approach is substantially more efficient and compares favorably in terms of image quality. We believe that this work motivates more emphasis on the prioritization of both performance and computational accessibility.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/1abb3621-01ce-48d7-bf18-13ea820d1fc7"/>
</div>

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/wuerstchen/wuerstchen_prior_pokemon_blip.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import (
    AutoPipelineForText2Image,
)
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS, WuerstchenPrior

checkpoint = 'work_dirs/wuerstchen_prior_pokemon_blip/step10450'
prompt = 'A robot pokemon, 4k photo"'

prior = WuerstchenPrior.from_pretrained(
    checkpoint, subfolder='prior', torch_dtype=torch.float16)

pipe = AutoPipelineForText2Image.from_pretrained(
    'warp-ai/wuerstchen', prior_prior=prior, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    height=768,
    width=768,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## Results Example

#### wuerstchen_prior_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/41707bcb-3c2e-458a-9bd9-ce3bc47d2faf)
