# Erasing Concepts from Diffusion Models

[Erasing Concepts from Diffusion Models](https://arxiv.org/abs/2303.07345)

## Abstract

Motivated by recent advancements in text-to-image diffusion, we study erasure of specific concepts from the model's weights. While Stable Diffusion has shown promise in producing explicit or realistic artwork, it has raised concerns regarding its potential for misuse. We propose a fine-tuning method that can erase a visual concept from a pre-trained diffusion model, given only the name of the style and using negative guidance as a teacher. We benchmark our method against previous approaches that remove sexually explicit content and demonstrate its effectiveness, performing on par with Safe Latent Diffusion and censored training. To evaluate artistic style removal, we conduct experiments erasing five modern artists from the network and conduct a user study to assess the human perception of the removed styles. Unlike previous methods, our approach can remove concepts from a diffusion model permanently rather than modifying the output at the inference time, so it cannot be circumvented even if a user has access to model weights.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/ca69f8f7-b34c-43ba-ab81-605cd29b3744"/>
</div>

## Citation

```
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Jaden Fiotto-Kaufman and David Bau},
  booktitle={Proceedings of the 2023 IEEE International Conference on Computer Vision},
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
$ mim train diffengine configs/esd/stable_diffusion_xl_gogh_esd.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/esd/stable_diffusion_xl_gogh_esd.py w
ork_dirs/stable_diffusion_xl_gogh_esd/iter_500.pth work_dirs/stable_diffusion_xl_gogh_esd --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL

prompt = 'The starry night by Van Gogh'
checkpoint = 'work_dirs/stable_diffusion_xl_gogh_esd'

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

#### stable_diffusion_xl_gogh_esd

SDXL output

![input1](https://github.com/okotaku/diffengine/assets/24734142/2cee5ced-2f2f-4e2a-9938-f36c641af104)

ESD output

![example1](https://github.com/okotaku/diffengine/assets/24734142/8860d070-db89-4faa-9c7a-6c3bba22eda5)
