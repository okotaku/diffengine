# PixArt-α Training

You can also check [`configs/pixart_alpha/README.md`](../../../configs/pixart_alpha/README.md) file.

## Configs

All configuration files are placed under the [`configs/pixart_alpha`](../../../configs/pixart_alpha/) folder.

Following is the example config fixed from the stable_diffusion_xl_pokemon_blip config file in [`configs/pixart_alpha/pixart_alpha_1024_pokemon_blip.py`](../../../configs/pixart_alpha/pixart_alpha_1024_pokemon_blip.py):

```
_base_ = [
    "../_base_/models/pixart_alpha_1024.py",
    "../_base_/datasets/pokemon_blip_pixart.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    dtype="bfloat16",
    optimizer=dict(lr=2e-6, weight_decay=3e-2),
    clip_grad=dict(max_norm=0.01))
```

## Run training

Run train

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/pixart_alpha/pixart_alpha_1024_pokemon_blip.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/pixart_alpha/pixart_alpha_1024_pokemon_blip.py work_dirs/pixart_alpha_1024_pokemon_blip/epoch_50.pth work_dirs/pixart_alpha_1024_pokemon_blip --save-keys transformer
```

Then we can run inference.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL, Transformer2DModel
from peft import PeftModel

checkpoint = Path('work_dirs/pixart_alpha_1024_pokemon_blip')
prompt = 'yoda pokemon'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
)
transformer = Transformer2DModel.from_pretrained(checkpoint, subfolder='transformer')
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    vae=vae,
    transformer=transformer,
    torch_dtype=torch.float32,
).to("cuda")

img = pipe(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
).images[0]
img.save("demo.png")
```

## Results Example

#### pixart_alpha_1024_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/5029f450-d248-4f22-8f90-6588f1d5f91e)

You can check [`configs/pixart_alpha/README.md`](../../../configs/pixart_alpha/README.md#results-example) for more details.
