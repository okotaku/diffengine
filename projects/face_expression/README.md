# Classic Anime Expressions

[Classic Anime Expressions Dataset](https://civitai.com/models/25613/classic-anime-expressions)

# Prepare datasets

1. Download images from [Classic Anime Expressions Dataset](https://civitai.com/models/25613/classic-anime-expressions).

2. Download csv from https://github.com/huggingface/diffusers/files/12074703/metadata.csv

```
wget https://github.com/huggingface/diffusers/files/12074703/metadata.csv
```

3. Unzip the files as follows

```
data/ExpressionTraining
├── 15_@ @
|    ├── 0c3a37a0ee63ea8e8b8cde3426a9b935.jpg
|    └── ...
├── ...
└── metadata.csv
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine projects/face_expression/stable_diffusion_v15_lora_face_expression.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import DiffusionPipeline

checkpoint = 'work_dirs/stable_diffusion_v15_lora_face_expression/step33000'
prompt = '1girl, >_<, blue hair'

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

#### stable_diffusion_v15_lora_face_expression

![example1](https://github.com/okotaku/diffengine/assets/24734142/2ece23bd-0e21-4ec4-a7ba-f6f39363bf01)

#### stable_diffusion_xl_lora_face_expression

![example1](https://github.com/okotaku/diffengine/assets/24734142/68c7569b-f62c-4228-a00d-997f2d963ad0)

Note that training failed. We should improve SDXL LoRA training.
