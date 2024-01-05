# Stable Diffusion

[Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train stable_diffusion_v15_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert stable_diffusion_v15_pokemon_blip work_dirs/stable_diffusion_v15_pokemon_blip/epoch_50.pth work_dirs/stable_diffusion_v15_pokemon_blip --save-keys unet
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
