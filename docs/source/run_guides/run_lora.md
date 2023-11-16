# Stable Diffusion LoRA Training

You can also check [`configs/stable_diffusion_lora/README.md`](../../../configs/stable_diffusion_lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_lora`](../../../configs/stable_diffusion_lora/) folder.

Following is the example config fixed from the stable_diffusion_v15_lora_pokemon_blip config file in [`configs/stable_diffusion_lora/stable_diffusion_v15_lora_pokemon_blip.py`](../../../configs/stable_diffusion_lora/stable_diffusion_v15_lora_pokemon_blip.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_v15_lora.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]

model = dict(lora_config=dict(rank=32))  # set LoRA and rank parameter

custom_hooks = [
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='PeftSaveHook'),  # Need to change from SDCheckpointHook
]
```

#### Finetuning the text encoder and UNet with LoRA

The script also allows you to finetune the text_encoder along with the unet, [LoRA](https://arxiv.org/abs/2106.09685) parameters.

```
_base_ = [
    '../_base_/models/stable_diffusion_v15_lora.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]
model = dict(
    lora_config=dict(rank=32),  # set LoRA and rank parameter
    finetune_text_encoder=True  # fine tune text encoder
)
custom_hooks = [
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='PeftSaveHook'),  # Need to change from SDCheckpointHook
]
```

We also provide [`configs/_base_/models/stable_diffusion_v15_lora_textencoder.py`](../../../configs/_base_/models/stable_diffusion_v15_lora_textencoder.py) as a base config and [`configs/stable_diffusion/stable_diffusion_v15_lora_textencoder_pokemon_blip.py`](../../../configs/stable_diffusion/stable_diffusion_v15_lora_textencoder_pokemon_blip.py) as a whole config.

## Run LoRA training

Run LoRA training:

```
# single gpu
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE}
# Example
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion_lora/stable_diffusion_v15_lora_pokemon_blip.py

# multi gpus
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_v15_lora_pokemon_blip/step10450')
prompt = 'yoda pokemon'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")
if (checkpoint / "text_encoder").exists():
    pipe.text_encoder = PeftModel.from_pretrained(
        pipe.text_encoder, checkpoint / "text_encoder", adapter_name="default"
    )

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

We also provide inference demo scripts:

```bash
$ mim run diffengine demo_lora "yoda pokemon" work_dirs/stable_diffusion_v15_lora_pokemon_blip/step10450
```

## Results Example

#### stable_diffusion_v15_lora_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/24899409-554d-4393-88e5-f8b8d6e6b36d)

You can check [`configs/stable_diffusion_lora/README.md`](../../../configs/stable_diffusion_lora/README.md#results-example) for more details.
