# Stable Diffusion XL LoRA

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
[Stable Diffusion XL](https://arxiv.org/abs/2307.01952)

## Abstract

An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/542189e1-e88f-4e80-9a51-de241b37d994"/>
</div>

## Citation

```
@inproceedings{
hu2022lora,
title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nZeVKeeFYf9}
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
$ diffengine train stable_diffusion_xl_lora_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_xl_lora_pokemon_blip/step20850')
prompt = 'yoda pokemon'

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

You can see more details on [`docs/source/run_guides/run_lora_xl.md`](../../docs/source/run_guides/run_lora_xl.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_xl_lora_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/22d1f3c0-05d8-413f-b0ac-d6bb72283945)

#### stable_diffusion_xl_lora_conv1x1_pokemon_blip

![example2](https://github.com/okotaku/diffengine/assets/24734142/9eca7f49-2d7c-4d08-90b5-045b14e32fe0)

#### stable_diffusion_xl_lora_conv3x3_pokemon_blip

![example3](https://github.com/okotaku/diffengine/assets/24734142/8ec55900-c08f-4e36-bd18-e6e81d35de6c)

#### stable_diffusion_xl_lora_waterpainting

![example4](https://github.com/okotaku/diffengine/assets/24734142/9bd797e3-07b5-452f-9f68-5bf017896c2f)
