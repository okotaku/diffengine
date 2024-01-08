# IP-Adapter Training

You can also check [`configs/ip_adapter/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/ip_adapter/README.md) file.

## Configs

All configuration files are placed under the [`configs/ip_adapter`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/ip_adapter/) folder.

Following is the example config fixed from the stable_diffusion_xl_pokemon_blip_ip_adapter config file in [`configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_xl_ip_adapter import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_ip_adapter import *
    from .._base_.schedules.stable_diffusion_xl_50e import *


train_dataloader.update(batch_size=1)

optim_wrapper.update(accumulative_counts=4)  # update every four times
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train stable_diffusion_xl_pokemon_blip_ip_adapter
```

## Inference with diffengine

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffengine` module.

```py
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection

prompt = ''

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="sdxl_models/image_encoder",
    torch_dtype=torch.float16,
).to('cuda')
vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    image_encoder=image_encoder,
    vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')
pipe.load_ip_adapter("work_dirs/stable_diffusion_xl_pokemon_blip_ip_adapter/step41650", subfolder="", weight_name="ip_adapter.bin")

image = load_image("https://datasets-server.huggingface.co/assets/lambdalabs/pokemon-blip-captions/--/default/train/0/image/image.jpg")

image = pipe(
    prompt,
    ip_adapter_image=image,
    height=1024,
    width=1024,
).images[0]
image.save('demo.png')
```

## Results Example

#### stable_diffusion_xl_pokemon_blip_ip_adapter

![input1](https://datasets-server.huggingface.co/assets/lambdalabs/pokemon-blip-captions/--/default/train/0/image/image.jpg)

![example1](https://github.com/okotaku/diffengine/assets/24734142/6137ffb4-dff9-41de-aa6e-2910d95e6d21)
