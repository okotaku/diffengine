# InstructPix2Pix

[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800)

## Abstract

We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pretrained models -- a language model (GPT-3) and a text-to-image model (Stable Diffusion) -- to generate a large dataset of image editing examples. Our conditional diffusion model, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the forward pass and does not require per example fine-tuning or inversion, our model edits images quickly, in a matter of seconds. We show compelling editing results for a diverse collection of input images and written instructions.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/b9de262c-e316-4df2-88d7-690f863934e3"/>
</div>

## Citation

```
@article{brooks2022instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/instruct_pix2pix/stable_diffusion_xl_instruct_pix2pix.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/instruct_pix2pix/stable_diffusion_xl_instruct_pix2pix.py work_dirs/stable_diffusion_xl_instruct_pix2pix/epoch_3.pth work_dirs/stable_diffusion_xl_instruct_pix2pix --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import load_image

checkpoint = 'work_dirs/stable_diffusion_xl_instruct_pix2pix'
prompt = 'make the mountains snowy'
condition_image = load_image(
    'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png'
).resize((1024, 1024))

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', unet=unet, vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    image=condition_image,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## Results Example

#### stable_diffusion_xl_instruct_pix2pix

![input1](https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png)

![example1](https://github.com/okotaku/diffengine/assets/24734142/f66149fd-e375-4f85-bfbf-d4d046cd469a)

#### stable_diffusion_xl_cartoonization

![input2](https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png)

![example2](https://github.com/okotaku/diffengine/assets/24734142/1206a53c-c3f7-40a3-9f48-6d01ae176a36)

#### stable_diffusion_xl_low_level_image_process

![input3](https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/derain_the_image_1.png)

![example3](https://github.com/okotaku/diffengine/assets/24734142/55fb5c32-b77c-4667-bf81-3b7c7eb9f8a1)

## Acknowledgement

Some experiments are based on [Instruction-tuning Stable Diffusion with InstructPix2Pix](https://huggingface.co/blog/instruction-tuning-sd). Thank you for the great blog post.
