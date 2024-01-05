# Stable Diffusion XL DreamBooth

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
[Stable Diffusion XL](https://arxiv.org/abs/2307.01952)

## Abstract

Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for "personalization" of text-to-image diffusion models. Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can be used to synthesize novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while preserving the subject's key features. We also provide a new dataset and evaluation protocol for this new task of subject-driven generation.

<div align=center>
<img src="https://github.com/okotaku/dethub/assets/24734142/33b1953d-ce42-4f9a-bcbc-87050cfe4f6f"/>
</div>

## Citation

```
@inproceedings{ruiz2023dreambooth,
  title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Prepare Dataset

1. Download style data from [StyleDrop-PyTorch](https://github.com/aim-uofa/StyleDrop-PyTorch/tree/main/data).

2. Unzip the files as follows

```
data/cat_waterpainting
└── image_01_03.jpg
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/stable_diffusion_xl_dreambooth/stable_diffusion_xl_dreambooth_lora_dog.py
```

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_xl_dreambooth_lora_dog/step499')
prompt = 'A photo of sks dog in a bucket'

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")
if (checkpoint / "text_encoder_one").exists():
    pipe.text_encoder_one = PeftModel.from_pretrained(
        pipe.text_encoder_one, checkpoint / "text_encoder_one", adapter_name="default"
    )
if (checkpoint / "text_encoder_two").exists():
    pipe.text_encoder_one = PeftModel.from_pretrained(
        pipe.text_encoder_two, checkpoint / "text_encoder_two", adapter_name="default"
    )

image = pipe(
    prompt,
    num_inference_steps=50,
    height=1024,
    width=1024,
).images[0]
image.save('demo.png')
```

You can see more details on [Run  DreamBooth XL docs](../../docs/source/run_guides/run_dreambooth_xl.md#inference-with-diffusers).

## Results Example

Note that we use 2 GPUs for training all lora models.

#### stable_diffusion_xl_dreambooth_lora_dog

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/ae1e4072-d2a3-445a-b11f-23d1f178a029)

#### stable_diffusion_xl_dreambooth_lora_starbucks

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/5a334ec6-db7f-40c0-9f20-3541ab70092f)

#### stable_diffusion_xl_dreambooth_lora_potatehead

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/84c10261-64bd-4428-9c1b-41fb62b9279d)

#### stable_diffusion_xl_dreambooth_lora_keramer_face

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/17696122-f0d4-4d61-8be9-ecc4688a33cb)

#### stable_diffusion_xl_dreambooth_lora_google_backpack

![exampleback](https://github.com/okotaku/diffengine/assets/24734142/3dd06582-0eb6-4a10-a065-af5e3d452878)

#### stable_diffusion_xl_dreambooth_lora_cat_waterpainting

![examplewaterpaint](https://github.com/okotaku/diffengine/assets/24734142/d2a83569-d2da-4a24-ae2c-e89b1da93fbf)

## Acknowledgement

These experiments are based on [diffusers docs](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md). Thank you for the great experiments.
