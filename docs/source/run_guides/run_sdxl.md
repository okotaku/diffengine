# Stable Diffusion XL Training

You can also check [`configs/stable_diffusion_xl/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_xl/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_xl`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion_xl) folder.

Following is the example config fixed from the stable_diffusion_xl_pokemon_blip config file in [`configs/stable_diffusion_xl/stable_diffusion_xl_pokemon_blip.py`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion_xl/stable_diffusion_xl_pokemon_blip.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


train_dataloader.update(batch_size=1)  # Because of GPU memory limit

optim_wrapper.update(accumulative_counts=4)  # update every four times
```

#### Finetuning the text encoder and UNet

The script also allows you to finetune the text_encoder along with the unet.

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


train_dataloader.update(batch_size=1)  # Because of GPU memory limit

optim_wrapper.update(accumulative_counts=4)  # update every four times

model.update(finetune_text_encoder=True)  # fine tune text encoder
```

#### Finetuning with Unet EMA

The script also allows you to finetune with Unet EMA.

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


train_dataloader.update(batch_size=1)  # Because of GPU memory limit

optim_wrapper.update(accumulative_counts=4)  # update every four times

custom_hooks = [  # Hook is list, we should write all custom_hooks again.
    dict(type=VisualizationHook, prompt=['yoda pokemon'] * 4),
    dict(type=SDCheckpointHook),
    dict(type=UnetEMAHook, momentum=1e-4, priority='ABOVE_NORMAL')  # setup EMA Hook
]
```

#### Finetuning with Min-SNR Weighting Strategy

The script also allows you to finetune with [Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556).

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


train_dataloader.update(batch_size=1)  # Because of GPU memory limit

optim_wrapper.update(accumulative_counts=4)  # update every four times

model.update(loss=dict(type='SNRL2Loss', snr_gamma=5.0, loss_weight=1.0))  # setup Min-SNR Weighting Strategy
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_xl_pokemon_blip

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert stable_diffusion_xl_pokemon_blip work_dirs/stable_diffusion_xl_pokemon_blip/epoch_50.pth work_dirs/stable_diffusion_xl_pokemon_blip --save-keys unet
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
    height=1024,
    width=1024,
).images[0]
image.save('demo.png')
```

## Inference Text Encoder and Unet finetuned weight with diffusers

Todo

## Results Example

#### stable_diffusion_xl_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/dd04fb22-64fb-4c4f-8164-b8391d94abab)

You can check [`configs/stable_diffusion_xl/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_xl/README.md#results-example) for more details.
