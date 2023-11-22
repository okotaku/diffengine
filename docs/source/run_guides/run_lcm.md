# Latent Consistency Models Training

You can also check [`configs/lcm/README.md`](../../../configs/lcm/README.md) file.

## Configs

All configuration files are placed under the [`configs/lcm`](../../../configs/lcm/) folder.

Following is the example config fixed from the lcm_xl_pokemon_blip config file in [`configs/lcm/lcm_xl_pokemon_blip.py`](../../../configs/lcm/lcm_xl_pokemon_blip.py):

```
_base_ = [
    "../_base_/models/lcm_xl.py",
    "../_base_/datasets/pokemon_blip_xl_pre_compute.py",
    "../_base_/schedules/lcm_xl_50e.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(batch_size=2)

optim_wrapper_cfg = dict(accumulative_counts=2)  # update every four times

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type="SDCheckpointHook"),
    dict(type="LCMEMAUpdateHook"),
]
```

## Run training

Run train

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# Example
$ mim train diffengine configs/lcm/lcm_xl_pokemon_blip.py

# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/lcm/lcm_xl_pokemon_blip.py work_dirs/lcm_xl_pokemon_blip/epoch_50.pth work_dirs/lcm_xl_pokemon_blip --save-keys unet
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

You can check [`configs/lcm/README.md`](../../../configs/lcm/README.md#results-example) for more details.
