# PixArt-α LoRA Training

You can also check [`configs/pixart_alpha_lora/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/pixart_alpha_lora`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_lora/) folder.

Following is the example config fixed from the pixart_alpha_1024_lora_pokemon_blip config file in [`configs/pixart_alpha_lora/pixart_alpha_1024_lora_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_lora/pixart_alpha_1024_lora_pokemon_blip.py):

```
_base_ = [
    "../_base_/models/pixart_alpha_1024_lora.py",
    "../_base_/datasets/pokemon_blip_pixart.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    dtype="bfloat16")

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type="PeftSaveHook"),
]
```

## Run LoRA training

Run LoRA training:

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train pixart_alpha_1024_lora_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/pixart_alpha_1024_lora_pokemon_blip/step20850')
prompt = 'yoda pokemon'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    vae=vae,
    torch_dtype=torch.float32,
).to("cuda")
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint / "transformer", adapter_name="default")

img = pipe(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
).images[0]
img.save("demo.png")
```

## Results Example

#### pixart_alpha_1024_lora_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/33f9c774-7896-4032-a11b-b86f4334de0a)

You can check [`configs/pixart_alpha_lora/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_lora/README.md#results-example) for more details.
