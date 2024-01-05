# SSD-1B DreamBooth

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
[SSD-1B](https://blog.segmind.com/introducing-segmind-ssd-1b/)

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
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train ssd_1b_dreambooth_lora_dog
```

## Training Speed

Environment:

- A6000 Single GPU
- nvcr.io/nvidia/pytorch:23.07-py3

Settings:

- 500 iterations training, (validation 4 images / 100 iterations)
- LoRA (rank=8) / DreamBooth

| Model  | total time |
| :----: | :--------: |
|  SDXL  | 18 m 55 s  |
| SSD-1B | 12 m 30 s  |

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/ssd_1b_dreambooth_lora_dog/step499')
prompt = 'A photo of sks dog in a bucket'

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'segmind/SSD-1B', vae=vae, torch_dtype=torch.float16)
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

#### ssd_1b_dreambooth_lora_dog

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/70a529fa-af82-4e53-951d-837c8eb88915)

## Blog post

[SSD-1B: A Leap in Efficient T2I Generation](https://medium.com/@to78314910/ssd-1b-a-leap-in-efficient-t2i-generation-138bb05fdd75)
