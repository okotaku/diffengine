# Kandinsky 2.2

[Kandinsky 2.2](https://habr.com/ru/companies/sberbank/articles/747446/)

## Abstract

Kandinsky 2.2 brings substantial improvements upon its predecessor, Kandinsky 2.1, by introducing a new, more powerful image encoder - CLIP-ViT-G and the ControlNet support. The switch to CLIP-ViT-G as the image encoder significantly increases the model’s capability to generate more aesthetic pictures and better understand text, thus enhancing the model’s overall performance. The addition of the ControlNet mechanism allows the model to effectively control the process of generating images. This leads to more accurate and visually appealing outputs and opens new possibilities for text-guided image manipulation.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/b07d82fb-4c2c-4216-a4b1-a64b278cee2a"/>
</div>

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train kandinsky_v22_prior_pokemon_blip
```

## Inference prior with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import AutoPipelineForText2Image, PriorTransformer

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/kandinsky_v22_prior_pokemon_blip/step10450'

prior = PriorTransformer.from_pretrained(
    checkpoint, subfolder="prior",
)
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    prior_prior=prior,
    torch_dtype=torch.float32,
)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
    width=512,
    height=512,
).images[0]
image.save('demo.png')
```

You can see more details on [`docs/source/run_guides/run_kandinsky_v22.md`](../../docs/source/run_guides/run_kandinsky_v22.md#inference-with-diffusers).

## Results Example

#### kandinsky_v22_prior_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/b709f558-5c03-4235-98d7-fe1c663182b8)

#### kandinsky_v22_decoder_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/6c9cce50-9f31-4637-9933-27697d65c830)
