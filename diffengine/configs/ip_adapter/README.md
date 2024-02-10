# IP-Adapter

[IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)

## Abstract

Recent years have witnessed the strong power of large text-to-image diffusion models for the impressive generative capability to create high-fidelity images. However, it is very tricky to generate desired images using only text prompt as it often involves complex prompt engineering. An alternative to text prompt is image prompt, as the saying goes: "an image is worth a thousand words". Although existing methods of direct fine-tuning from pretrained models are effective, they require large computing resources and are not compatible with other base models, text prompt, and structural controls. In this paper, we present IP-Adapter, an effective and lightweight adapter to achieve image prompt capability for the pretrained text-to-image diffusion models. The key design of our IP-Adapter is decoupled cross-attention mechanism that separates cross-attention layers for text features and image features. Despite the simplicity of our method, an IP-Adapter with only 22M parameters can achieve comparable or even better performance to a fully fine-tuned image prompt model. As we freeze the pretrained diffusion model, the proposed IP-Adapter can be generalized not only to other custom models fine-tuned from the same base model, but also to controllable generation using existing controllable tools. With the benefit of the decoupled cross-attention strategy, the image prompt can also work well with the text prompt to achieve multimodal image generation.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/5884b1ce-0550-4e08-9b10-35c501cefc99"/>
</div>

## Citation

```
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
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
$ diffengine train stable_diffusion_xl_pokemon_blip_ip_adapter
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffengine` module.

```py
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection

prompt = ''

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="sdxl_models/image_encoder",
    torch_dtype=torch.float16,
).to('cuda')
vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    image_encoder=image_encoder,
    vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')
pipe.load_ip_adapter("work_dirs/stable_diffusion_xl_pokemon_blip_ip_adapter/step41650", subfolder="", weight_name="ip_adapter.bin")

image = load_image("https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/README_files/README_2_0.png?raw=true")

image = pipe(
    prompt,
    ip_adapter_image=image,
    height=1024,
    width=1024,
).images[0]
image.save('demo.png')
```

You can see more details on [`docs/source/run_guides/run_ip_adapter.md`](../../docs/source/run_guides/run_ip_adapter.md#inference-with-diffengine).

## Results Example

#### stable_diffusion_xl_pokemon_blip_ip_adapter

![input1](https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/README_files/README_2_0.png?raw=true)

![example1](https://github.com/okotaku/diffengine/assets/24734142/6137ffb4-dff9-41de-aa6e-2910d95e6d21)

#### stable_diffusion_xl_pokemon_blip_ip_adapter_plus

![input1](https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/README_files/README_2_0.png?raw=true)

![example1](https://github.com/okotaku/diffengine/assets/24734142/723ad39d-9e0f-441b-80f7-cf9bcfd12853)

#### stable_diffusion_xl_pokemon_blip_ip_adapter_plus_pretrained

![input1](https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/README_files/README_2_0.png?raw=true)

![example1](https://github.com/okotaku/diffengine/assets/24734142/ace81220-010b-44a5-aa8f-3acdf3f54433)

#### stable_diffusion_xl_pokemon_blip_ip_adapter_plus_dinov2

![input1](https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/README_files/README_2_0.png?raw=true)

![example1](https://github.com/okotaku/diffengine/assets/24734142/5e1e2088-d00b-4909-9c64-61a7b5ac6b44)

#### stable_diffusion_xl_pokemon_blip_ip_adapter_plus_dinov2_giant

![input1](https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/README_files/README_2_0.png?raw=true)

![example1](https://github.com/okotaku/diffengine/assets/24734142/f76c33ba-c1ac-4f6f-b256-d48de5e58bf8)
