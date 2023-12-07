# Stable Diffusion XL LoRA Training

You can also check [`configs/stable_diffusion_xl_lora/README.md`](../../../configs/stable_diffusion_xl_lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_xl_lora`](../../../configs/stable_diffusion_xl_lora/) folder.

Following is the example config fixed from the stable_diffusion_xl_lora_pokemon_blip config file in [`configs/stable_diffusion_xl_lora/stable_diffusion_xl_lora_pokemon_blip.py`](../../../configs/stable_diffusion_xl_lora/stable_diffusion_xl_lora_pokemon_blip.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_xl_lora.py',
    '../_base_/datasets/pokemon_blip_xl.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='PeftSaveHook'),  # Need to change from SDCheckpointHook
]
```

#### Finetuning the text encoder and UNet with LoRA

The script also allows you to finetune the text_encoder along with the unet, [LoRA](https://arxiv.org/abs/2106.09685) parameters.

```
_base_ = [
    '../_base_/models/stable_diffusion_xl_lora.py',
    '../_base_/datasets/pokemon_blip_xl.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]
model = dict(
    lora_config=dict(rank=8),  # set LoRA and rank parameter
    finetune_text_encoder=True  # fine tune text encoder
)
custom_hooks = [
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='PeftSaveHook'),  # Need to change from SDCheckpointHook
]
```

## Run LoRA training

Run LoRA training:

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# Example
$ mim train diffengine configs/stable_diffusion_xl_lora/stable_diffusion_xl_lora_pokemon_blip.py

# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_xl_lora_pokemon_blip/step20850')
prompt = 'yoda pokemon'

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")
if (checkpoint / "text_encoder_one").exists():
    pipe.text_encoder_one = PeftModel.from_pretrained(
        pipe.text_encoder_one, checkpoint / "text_encoder_one", adapter_name="default"
    )
if (checkpoint / "text_encoder_two").exists():
    pipe.text_encoder_one = PeftModel.from_pretrained(
        pipe.text_encoder_two, checkpoint / "text_encoder_two", adapter_name="default"
    )

image = pipe(
    prompt,
    num_inference_steps=50,
    height=1024,
    width=1024,
).images[0]
image.save('demo.png')
```

We also provide inference demo scripts:

```bash
$ mim run diffengine demo_lora "yoda pokemon" work_dirs/stable_diffusion_xl_lora_pokemon_blip/step20850/unet --sdmodel stabilityai/stable-diffusion-xl-base-1.0 --vaemodel madebyollin/sdxl-vae-fp16-fix --height 1024 --width 1024
```

## Results Example

#### stable_diffusion_xl_lora_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/22d1f3c0-05d8-413f-b0ac-d6bb72283945)

You can check [`configs/stable_diffusion_xl_lora/README.md`](../../../configs/stable_diffusion_xl_lora/README.md#results-example) for more details.
