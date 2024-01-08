# Latent Consistency Models Training

You can also check [`configs/lcm/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/lcm/README.md) file.

## Configs

All configuration files are placed under the [`configs/lcm`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/lcm/) folder.

Following is the example config fixed from the lcm_xl_pokemon_blip config file in [`configs/lcm/lcm_xl_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/lcm/lcm_xl_pokemon_blip.py):

```
from mmengine.config import read_base

from diffengine.engine.hooks import (
    LCMEMAUpdateHook,
    SDCheckpointHook,
    VisualizationHook,
)

with read_base():
    from .._base_.datasets.pokemon_blip_xl_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.lcm_xl import *
    from .._base_.schedules.lcm_xl_50e import *


train_dataloader.update(batch_size=2)

optim_wrapper.update(accumulative_counts=2)  # update every four times

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type=SDCheckpointHook),
    dict(type=LCMEMAUpdateHook),
]
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train lcm_xl_pokemon_blip

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert lcm_xl_pokemon_blip work_dirs/lcm_xl_pokemon_blip/epoch_50.pth work_dirs/lcm_xl_pokemon_blip --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import DiffusionPipeline, AutoencoderKL, LCMScheduler, UNet2DConditionModel

checkpoint = 'work_dirs/lcm_xl_pokemon_blip'
prompt = 'yoda pokemon'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    unet=unet,
    scheduler=LCMScheduler.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"),
    vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=4,
    height=1024,
    width=1024,
    guidance_scale=1.0,
).images[0]
image.save('demo.png')
```

## Results Example

#### lcm_xl_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/8fd9799d-11a3-4cd1-8f08-f91e9e7cef3c)

You can check [`configs/lcm/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/lcm/README.md#results-example) for more details.
