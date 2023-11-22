# Wuerstchen Training

You can also check [`configs/wuerstchen/README.md`](../../../configs/wuerstchen/README.md) file.

## Configs

All configuration files are placed under the [`configs/wuerstchen`](../../../configs/wuerstchen/) folder.

Following is the example config fixed from the wuerstchen_prior_pokemon_blip config file in [`configs/wuerstchen/wuerstchen_prior_pokemon_blip.py`](../../../configs/wuerstchen/wuerstchen_prior_pokemon_blip.py):

```
_base_ = [
    "../_base_/models/wuerstchen_prior.py",
    "../_base_/datasets/pokemon_blip_wuerstchen.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper_cfg = dict(
    optimizer=dict(lr=1e-5),
    accumulative_counts=4)  # update every four times
```

## Run training

Run train

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example
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

You can check [`configs/wuerstchen/README.md`](../../../configs/wuerstchen/README.md#results-example) for more details.
