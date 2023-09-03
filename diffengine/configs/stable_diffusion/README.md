# Stable Diffusion

[Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py work_dirs/stable_diffusion_v15_pokemon_blip/epoch_50.pth work_dirs/stable_diffusion_v15_pokemon_blip --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/stable_diffusion_v15_pokemon_blip'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', unet=unet, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

You can see more details on [`docs/source/run_guides/run_sd.md`](../../docs/source/run_guides/run_sd.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_v15_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/24d5254d-95be-46eb-8982-b38b6a11f1ba)

#### stable_diffusion_v15_textencoder_pokemon_blip

![example2](https://github.com/okotaku/diffengine/assets/24734142/5e024078-f22d-4d65-94ec-bcdeefe554be)

#### stable_diffusion_v15_ema_pokemon_blip

![example3](https://github.com/okotaku/diffengine/assets/24734142/9f3181e2-d244-4aa3-b116-4a07eec1aa5a)

#### stable_diffusion_v15_snr_pokemon_blip

![example4](https://github.com/okotaku/diffengine/assets/24734142/b98e887a-d3af-49bb-ad15-9e8250c09578)
