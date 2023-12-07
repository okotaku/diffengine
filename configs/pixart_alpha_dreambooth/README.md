# PixArt-α DreamBooth

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
[PixArt-α](https://arxiv.org/abs/2310.00426)

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

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/pixart_alpha_dreambooth/pixart_alpha_1024_dreambooth_lora_dog.py
```

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/pixart_alpha_1024_dreambooth_lora_dog/step499')
prompt = 'A photo of sks dog in a bucket'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    vae=vae,
    torch_dtype=torch.float32,
).to("cuda")
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint / "transformer", adapter_name="default")

img = pipe(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
).images[0]
img.save("demo.png")
```

You can see more details on [Run PixArt-α DreamBooth docs](../../docs/source/run_guides/run_pixart_alpha_dreambooth.md#inference-with-diffusers).

## Results Example

#### pixart_alpha_512_dreambooth_lora_dog

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/2d3b59b1-2a9b-422d-adba-347504c66be2)

#### pixart_alpha_1024_dreambooth_lora_dog

![exampledog2](https://github.com/okotaku/diffengine/assets/24734142/a3fc9fcd-7cd0-4dc2-997d-3a9b303c228a)

#### pixart_alpha_1024_dreambooth_lora_cat_waterpainting

![examplestyle](https://github.com/okotaku/diffengine/assets/24734142/e48e845d-c21c-451b-97cb-2df0ec0cfd41)
