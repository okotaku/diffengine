# T2I Adapter Training

You can also check [`configs/t2i_adapter/README.md`](../../../configs/t2i_adapter/README.md) file.

## Configs

All configuration files are placed under the [`configs/t2i_adapter`](../../../configs/t2i_adapter/) folder.

Following is the example config fixed from the stable_diffusion_xl_t2i_adapter_fill50k config file in [`configs/t2i_adapter/stable_diffusion_xl_t2i_adapter_fill50k.py`](../../../configs/t2i_adapter/stable_diffusion_xl_t2i_adapter_fill50k.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_xl_t2i_adapter.py',
    '../_base_/datasets/fill50k_t2i_adapter_xl.py',
    '../_base_/schedules/stable_diffusion_3e.py',
    '../_base_/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    accumulative_counts=2,
)
```

## Run training

Run train

```
# single gpu
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE}
# multi gpus
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example
$ docker compose exec diffengine mim train diffengine configs/t2i_adapter/stable_diffusion_xl_t2i_adapter_fill50k.py
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

We also provide inference demo scripts:

```
$ mim run diffengine demo_adapter ${PROMPT} ${CONDITION_IMAGE} ${CHECKPOINT} --vaemodel madebyollin/sdxl-vae-fp16-fix --height 1024 --width 1024
# Example
$ mim run diffengine demo_adapter "cyan circle with brown floral background" https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg work_dirs/stable_diffusion_xl_t2i_adapter_fill50k/step75000 --vaemodel madebyollin/sdxl-vae-fp16-fix --height 1024 --width 1024
```

## Results Example

#### stable_diffusion_xl_t2i_adapter_fill50k

![input1](https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg)

![example1](https://github.com/okotaku/diffengine/assets/24734142/7ea65b62-a8c4-4888-8e11-9cdb69855d3c)

You can check [`configs/t2i_adapter/README.md`](../../../configs/t2i_adapter/README.md#results-example) for more details.
