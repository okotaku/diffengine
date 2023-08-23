# Stable Diffusion XL Training

You can also check [`configs/stable_diffusion_xl/README.md`](../../../configs/stable_diffusion_xl/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_xl`](../../../configs/stable_diffusion_xl/) folder.

Following is the example config fixed from the stable_diffusion_xl_pokemon_blip config file in [`configs/stable_diffusion_xl/stable_diffusion_xl_pokemon_blip.py`](../../../configs/stable_diffusion_xl/stable_diffusion_xl_pokemon_blip.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_xl.py',
    '../_base_/datasets/pokemon_blip_xl.py',
    '../_base_/schedules/stable_diffusion_xl_50e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)  # Because of GPU memory limit

optim_wrapper_cfg = dict(accumulative_counts=4)  # update every four times
```

#### Finetuning the text encoder and UNet

The script also allows you to finetune the text_encoder along with the unet.

```
_base_ = [
    '../_base_/models/stable_diffusion_xl.py',
    '../_base_/datasets/pokemon_blip_xl.py',
    '../_base_/schedules/stable_diffusion_xl_50e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)  # Because of GPU memory limit

optim_wrapper_cfg = dict(accumulative_counts=4)  # update every four times

model = dict(finetune_text_encoder=True)  # fine tune text encoder
```

#### Finetuning with Unet EMA

The script also allows you to finetune with Unet EMA.

```
_base_ = [
    '../_base_/models/stable_diffusion_xl.py',
    '../_base_/datasets/pokemon_blip_xl.py',
    '../_base_/schedules/stable_diffusion_xl_50e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)  # Because of GPU memory limit

optim_wrapper_cfg = dict(accumulative_counts=4)  # update every four times

custom_hooks = [  # Hook is list, we should write all custom_hooks again.
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='SDCheckpointHook'),
    dict(type='UnetEMAHook', momentum=1e-4, priority='ABOVE_NORMAL')  # setup EMA Hook
]
```

#### Finetuning with Min-SNR Weighting Strategy

The script also allows you to finetune with [Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556).

```
_base_ = [
    '../_base_/models/stable_diffusion_xl.py',
    '../_base_/datasets/pokemon_blip_xl.py',
    '../_base_/schedules/stable_diffusion_xl_50e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)  # Because of GPU memory limit

optim_wrapper_cfg = dict(accumulative_counts=4)  # update every four times

model = dict(loss=dict(type='SNRL2Loss', snr_gamma=5.0, loss_weight=1.0))  # setup Min-SNR Weighting Strategy
```

## Run training

Run train

```
# single gpu
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE}
# Example
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion_xl/stable_diffusion_xl_pokemon_blip.py

# multi gpus
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/stable_diffusion_xl/stable_diffusion_xl_pokemon_blip.py work_dirs/stable_diffusion_xl_pokemon_blip/epoch_50.pth work_dirs/stable_diffusion_xl_pokemon_blip --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/stable_diffusion_xl_pokemon_blip'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', unet=unet, vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

We also provide inference demo scripts:

```
$ mim run diffengine demo ${PROMPT} ${CHECKPOINT}
# Example
$ mim run diffengine demo "yoda pokemon" work_dirs/stable_diffusion_xl_pokemon_blip --sdmodel stabilityai/stable-diffusion-xl-base-1.0 --vaemodel madebyollin/sdxl-vae-fp16-fix
```

## Inference Text Encoder and Unet finetuned weight with diffusers

Todo

## Results Example

#### stable_diffusion_xl_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/dd04fb22-64fb-4c4f-8164-b8391d94abab)

You can check [`configs/stable_diffusion_xl/README.md`](../../../configs/stable_diffusion_xl/README.md#results-example) for more details.
