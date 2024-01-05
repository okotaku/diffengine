# PixArt-α Training

You can also check [`configs/pixart_alpha/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha/README.md) file.

## Configs

All configuration files are placed under the [`configs/pixart_alpha`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha/) folder.

Following is the example config fixed from the stable_diffusion_xl_pokemon_blip config file in [`configs/pixart_alpha/pixart_alpha_1024_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha/pixart_alpha_1024_pokemon_blip.py):

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
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train pixart_alpha_1024_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert pixart_alpha_1024_pokemon_blip work_dirs/pixart_alpha_1024_pokemon_blip/epoch_50.pth work_dirs/pixart_alpha_1024_pokemon_blip --save-keys transformer
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

![example1](https://github.com/okotaku/diffengine/assets/24734142/6b87369a-4746-4067-9a8a-5d7453fc80ce)

You can check [`configs/pixart_alpha/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha/README.md#results-example) for more details.
