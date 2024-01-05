# Distill SD DreamBooth

[On Architectural Compression of Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.15798)

## Abstract

Exceptional text-to-image (T2I) generation results of Stable Diffusion models (SDMs) come with substantial computational demands. To resolve this issue, recent research on efficient SDMs has prioritized reducing the number of sampling steps and utilizing network quantization. Orthogonal to these directions, this study highlights the power of classical architectural compression for general-purpose T2I synthesis by introducing block-removed knowledge-distilled SDMs (BK-SDMs). We eliminate several residual and attention blocks from the U-Net of SDMs, obtaining over a 30% reduction in the number of parameters, MACs per sampling step, and latency. We conduct distillation-based pretraining with only 0.22M LAION pairs (fewer than 0.1% of the full training pairs) on a single A100 GPU. Despite being trained with limited resources, our compact models can imitate the original SDM by benefiting from transferred knowledge and achieve competitive results against larger multi-billion parameter models on the zero-shot MS-COCO benchmark. Moreover, we demonstrate the applicability of our lightweight pretrained models in personalized generation with DreamBooth finetuning.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/253c0dfb-fa1c-4cbf-81c0-9d6948d40413"/>
</div>

## Citation

```
@article{kim2023architectural,
  title={On Architectural Compression of Text-to-Image Diffusion Models},
  author={Kim, Bo-Kyeong and Song, Hyoung-Kyu and Castells, Thibault and Choi, Shinkook},
  journal={arXiv preprint arXiv:2305.15798},
  year={2023},
  url={https://arxiv.org/abs/2305.15798}
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
$ diffengine train small_sd_dreambooth_lora_dog
```

## Training Speed

Environment:

- A6000 Single GPU
- nvcr.io/nvidia/pytorch:23.07-py3

Settings:

- 1k iterations training, (validation 4 images / 100 iterations)
- LoRA (rank=8) / DreamBooth

|  Model   | total time |
| :------: | :--------: |
|  SDV1.5  | 16 m 39 s  |
| Small SD | 11 m 57 s  |
| Tiny SD  | 11 m 17 s  |

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel

checkpoint = Path('work_dirs/small_sd_dreambooth_lora_dog/step999')
prompt = 'A photo of sks dog in a bucket'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")
if (checkpoint / "text_encoder").exists():
    pipe.text_encoder = PeftModel.from_pretrained(
        pipe.text_encoder, checkpoint / "text_encoder", adapter_name="default"
    )

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## Results Example

#### small_sd_dreambooth_lora_dog

![example1](https://github.com/okotaku/diffengine/assets/24734142/9fa0bf8a-34f2-4e74-88b8-f33e249520e2)

We uploaded trained checkpoint on [`takuoko/small-sd-dreambooth-lora-dog`](https://huggingface.co/takuoko/small-sd-dreambooth-lora-dog).

#### tiny_sd_dreambooth_lora_dog

![example2](https://github.com/okotaku/diffengine/assets/24734142/dcb3ff5b-995e-44bc-885c-6ed050eea24d)

#### small_sd_dreambooth_lora_keramer_face

![example1](https://github.com/okotaku/diffengine/assets/24734142/ddbb2e54-f8be-46a6-a7ab-5bfb54bd29e6)

#### tiny_sd_dreambooth_lora_keramer_face

![example2](https://github.com/okotaku/diffengine/assets/24734142/45e46061-a0c2-4396-b0fd-1aab36c2e4e5)

## Reference

- [Open-sourcing Knowledge Distillation Code and Weights of SD-Small and SD-Tiny](https://huggingface.co/blog/sd_distillation)
