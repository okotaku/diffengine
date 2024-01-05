# DeepFloyd IF

[Deepfloyd IF](https://www.deepfloyd.ai/deepfloyd-if)

## Abstract

We introduce DeepFloyd IF, a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and language understanding. DeepFloyd IF is a modular composed of a frozen text encoder and three cascaded pixel diffusion modules: a base model that generates 64x64 px image based on text prompt and two super-resolution models, each designed to generate images of increasing resolution: 256x256 px and 1024x1024 px. All stages of the model utilize a frozen text encoder based on the T5 transformer to extract text embeddings, which are then fed into a UNet architecture enhanced with cross-attention and attention pooling. The result is a highly efficient model that outperforms current state-of-the-art models, achieving a zero-shot FID score of 6.66 on the COCO dataset. Our work underscores the potential of larger UNet architectures in the first stage of cascaded diffusion models and depicts a promising future for text-to-image synthesis.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/a1b1a31f-5fb7-4a62-8502-2c8e4330f165"/>
</div>

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/deepfloyd_if/deepfloyd_if_pokemon_blip.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/deepfloyd_if/deepfloyd_if_l_pokemon_blip.py work_dirs/deepfloyd_if_l_pokemon_blip/epoch_50.pth work_dirs/deepfloyd_if_l_pokemon_blip --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/deepfloyd_if_l_pokemon_blip'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet')
pipe = DiffusionPipeline.from_pretrained(
    'DeepFloyd/IF-I-L-v1.0', unet=unet)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

We also provide inference demo scripts:

```bash
$ mim run diffengine demo_if "yoda pokemon" work_dirs/deepfloyd_if_l_pokemon_blip
```

## Results Example

#### deepfloyd_if_l_pokemon_blip

Stage1 output example
![example](https://github.com/okotaku/diffengine/assets/24734142/e1e56e8e-59ec-4256-82a1-0e2941b6ee24)

Stage3 output example
![example](https://github.com/okotaku/diffengine/assets/24734142/c71b2a64-e016-4e39-b077-457d065ee8da)
