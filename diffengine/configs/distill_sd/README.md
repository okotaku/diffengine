# Distill SD XL

[On Architectural Compression of Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.15798)

## Abstract

Exceptional text-to-image (T2I) generation results of Stable Diffusion models (SDMs) come with substantial computational demands. To resolve this issue, recent research on efficient SDMs has prioritized reducing the number of sampling steps and utilizing network quantization. Orthogonal to these directions, this study highlights the power of classical architectural compression for general-purpose T2I synthesis by introducing block-removed knowledge-distilled SDMs (BK-SDMs). We eliminate several residual and attention blocks from the U-Net of SDMs, obtaining over a 30% reduction in the number of parameters, MACs per sampling step, and latency. We conduct distillation-based pretraining with only 0.22M LAION pairs (fewer than 0.1% of the full training pairs) on a single A100 GPU. Despite being trained with limited resources, our compact models can imitate the original SDM by benefiting from transferred knowledge and achieve competitive results against larger multi-billion parameter models on the zero-shot MS-COCO benchmark. Moreover, we demonstrate the applicability of our lightweight pretrained models in personalized generation with DreamBooth finetuning.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/253c0dfb-fa1c-4cbf-81c0-9d6948d40413"/>
</div>

## Citation

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
$ diffengine train small_sd_xl_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert small_sd_xl_pokemon_blip work_dirs/small_sd_xl_pokemon_blip/epoch_50.pth work_dirs/small_sd_xl_pokemon_blip --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/small_sd_xl_pokemon_blip'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', unet=unet, vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]
image.save('demo.png')
```

## Results Example

#### small_sd_xl_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/da9d56a5-04d7-4fba-9c88-6b13c86adb9f)

#### tiny_sd_xl_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/5ae252e7-ecb2-4af6-bf9a-e68d0f1840ce)

We uploaded trained checkpoint on [`takuoko/tiny_sd_xl_pokemon_blip`](https://huggingface.co/takuoko/tiny_sd_xl_pokemon_blip).

## Blog post

[On Architectural Compression of Text-to-Image Diffusion Models](https://medium.com/@to78314910/on-architectural-compression-of-text-to-image-diffusion-models-ce8c9cba512a)

## Acknowledgement

These implementations are based on [segmind/distill-sd](https://github.com/segmind/distill-sd). Thank you for the great open source codes.
