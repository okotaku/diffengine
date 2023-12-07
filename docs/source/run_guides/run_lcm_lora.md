# Latent Consistency Models Training

You can also check [`configs/lcm_lora/README.md`](../../../configs/lcm_lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/lcm_lora`](../../../configs/lcm_lora/) folder.

Following is the example config fixed from the lcm_xl_lora_pokemon_blip config file in [`configs/lcm_lora/lcm_xl_lora_pokemon_blip.py`](../../../configs/lcm_lora/lcm_xl_lora_pokemon_blip.py):

```
_base_ = [
    "../_base_/models/lcm_xl_lora.py",
    "../_base_/datasets/pokemon_blip_xl_pre_compute.py",
    "../_base_/schedules/lcm_xl_50e.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(batch_size=2)

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    accumulative_counts=2)

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type="PeftSaveHook"),
]
```

## Run training

Run train

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# Example
$ mim train diffengine configs/lcm_lora/lcm_xl_lora_pokemon_blip.py

# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL, LCMScheduler
from peft import PeftModel

checkpoint = Path('work_dirs/lcm_xl_lora_pokemon_blip/step20850')
prompt = 'yoda pokemon'

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', vae=vae,
    scheduler=LCMScheduler.from_pretrained(
      'stabilityai/stable-diffusion-xl-base-1.0', subfolder="scheduler"),
    torch_dtype=torch.float16)
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
    num_inference_steps=4,
    guidance_scale=1.0,
    height=1024,
    width=1024,
).images[0]
image.save('demo.png')
```

## Results Example

#### lcm_xl_lora_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/c321c36e-ba99-42f7-ab0f-4f790253926f)

You can check [`configs/lcm_lora/README.md`](../../../configs/lcm_lora/README.md#results-example) for more details.
