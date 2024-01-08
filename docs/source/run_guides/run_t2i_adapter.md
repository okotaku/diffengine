# T2I Adapter Training

You can also check [`configs/t2i_adapter/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/t2i_adapter/README.md) file.

## Configs

All configuration files are placed under the [`configs/t2i_adapter`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/t2i_adapter/) folder.

Following is the example config fixed from the stable_diffusion_xl_t2i_adapter_fill50k config file in [`configs/t2i_adapter/stable_diffusion_xl_t2i_adapter_fill50k.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/t2i_adapter/stable_diffusion_xl_t2i_adapter_fill50k.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.fill50k_t2i_adapter_xl import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_xl_t2i_adapter import *
    from .._base_.schedules.stable_diffusion_3e import *


optim_wrapper.update(
    optimizer=dict(lr=1e-5),
    accumulative_counts=2,
)
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example
$ diffengine train stable_diffusion_xl_t2i_adapter_fill50k
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL
from diffusers.utils import load_image

checkpoint = 'work_dirs/stable_diffusion_xl_t2i_adapter_fill50k/step75000'
prompt = 'cyan circle with brown floral background'
condition_image = load_image(
    'https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg'
).resize((1024, 1024))

adapter = T2IAdapter.from_pretrained(
        checkpoint, subfolder='adapter', torch_dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', adapter=adapter, vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    image=condition_image,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## Results Example

#### stable_diffusion_xl_t2i_adapter_fill50k

![input1](https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg)

![example1](https://github.com/okotaku/diffengine/assets/24734142/7ea65b62-a8c4-4888-8e11-9cdb69855d3c)

You can check [`configs/t2i_adapter/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/t2i_adapter/README.md#results-example) for more details.
