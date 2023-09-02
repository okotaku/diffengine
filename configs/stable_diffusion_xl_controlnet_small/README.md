# Stable Diffusion XL ControlNet Small / Mid

[ControlNet Small / Mid](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small)

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/stable_diffusion_xl_controlnet_small/stable_diffusion_xl_controlnet_small_fill50k.py
```

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image

checkpoint = 'work_dirs/stable_diffusion_xl_controlnet_small_fill50k/step25000'
prompt = 'cyan circle with brown floral background'
condition_image = load_image(
    'https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg'
)

controlnet = ControlNetModel.from_pretrained(
        checkpoint, subfolder='controlnet', torch_dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    condition_image,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

You can see more details on [`docs/source/run_guides/run_controlnet_xl.md`](../../docs/source/run_guides/run_controlnet_xl.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_xl_controlnet_small_fill50k

![input1](https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg)

![example1](<>)

#### stable_diffusion_xl_controlnet_mid_fill50k

![input1](https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg)

![example1](<>)
