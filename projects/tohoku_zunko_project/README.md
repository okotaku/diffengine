# Tohoku Zunko Zundamon Project

[Tohoku Zunko Zundamon Project](https://zunko.jp/)

# Prepare datasets

1. Download images from [Tohoku Zunko Zundamon Project](https://zunko.jp/con_illust.html).

Note that we used only 5 images for each character.

2. Unzip the files as follows

```
data
└──zunko
     ├── a1zunko46.png
     ├── a1zunko105.png
     ├── z024.png
     ├── z032.png
     └── z033.png
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine projects/tohoku_zunko_project/stable_diffusion_v15_dreambooth_lora_zunko.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import DiffusionPipeline

checkpoint = 'work_dirs/stable_diffusion_v15_dreambooth_lora_zunko/step999'
prompt = 'A photo of sks character in a bucket'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.load_lora_weights(checkpoint)

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

You can see more details on [LoRA docs](../../docs/source/run_guides/run_lora.md#inference-with-diffusers) or [`docs/source/run_guides/run_lora_xl.md`](../../docs/source/run_guides/run_lora_xl.md#inference-with-diffusers)..

## Results Example

#### stable_diffusion_v15_dreambooth_lora_zunko

Prompt is `A photo of sks character in a bucket`.

![example1](https://github.com/okotaku/diffengine/assets/24734142/951b740c-8b17-47e1-8bc9-1db87aecf6eb)

#### anythingv5_dreambooth_lora_zunko

Prompt is `1girl, sks, in a bucket`.

![example1](https://github.com/okotaku/diffengine/assets/24734142/477632a4-534e-44f7-bab0-1520cfc669d2)

#### stable_diffusion_xl_dreambooth_lora_zunko

Prompt is `A photo of sks character in a bucket`.

![example1](https://github.com/okotaku/diffengine/assets/24734142/e820e4e4-3eec-4058-9be3-7284cbd0c4db)

#### counterfeit_xl_dreambooth_lora_zunko

Prompt is `1girl, sks, in a bucket`.

![example1](https://github.com/okotaku/diffengine/assets/24734142/5c85a9f2-6eee-4891-91e3-69fc43244f38)
