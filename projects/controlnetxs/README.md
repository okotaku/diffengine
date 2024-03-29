# Stable Diffusion XL ControlNetXS

[ControlNet-XS](https://vislearn.github.io/ControlNet-XS/)

## Abstract

With increasing computing capabilities, current model architectures appear to follow the trend of simply upscaling all components without validating the necessity for doing so. In this project we investigate the size and architectural design of ControlNet \[Zhang et al., 2023\] for controlling the image generation process with stable diffusion-based models. We show that a new architecture with as little as 1% of the parameters of the base model achieves state-of-the art results, considerably better than ControlNet in terms of FID score. Hence we call it ControlNet-XS. We provide the code for controlling StableDiffusion-XL \[Podell et al., 2023\] (Model B, 48M Parameters) and StableDiffusion 2.1 \[Rombach et al. 2022\] (Model B, 14M Parameters), all under openrail license. The different models are explained in the Method section below.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/d6747c67-2184-4697-9bd9-01306575c787"/>
</div>

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train projects/controlnetxs/stable_diffusion_xl_controlnetxs_fill50k.py
```

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
import torch
from diffusers import AutoencoderKL
from projects.controlnetxs.modules.controlnetxs import ControlNetXSModel
from projects.controlnetxs.modules.pipeline_controlnet_xs_sd_xl import (
    StableDiffusionXLControlNetXSPipeline,
)
from diffusers.utils import load_image

checkpoint = 'work_dirs/stable_diffusion_xl_controlnetxs_fill50k/step37500'
prompt = 'cyan circle with brown floral background'
condition_image = load_image(
    'https://github.com/okotaku/diffengine/assets/24734142/1af9dbb0-b056-435c-bc4b-62a823889191'
).resize((1024, 1024))

controlnet = ControlNetXSModel.from_pretrained(
        checkpoint, subfolder='controlnet', torch_dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    image=condition_image,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## Results Example

#### stable_diffusion_xl_controlnetxs_fill50k

![input1](https://github.com/okotaku/diffengine/assets/24734142/1af9dbb0-b056-435c-bc4b-62a823889191)

![example1](https://github.com/okotaku/diffengine/assets/24734142/c190a665-361c-4bd8-86ac-f6cd66d6a0b9)
