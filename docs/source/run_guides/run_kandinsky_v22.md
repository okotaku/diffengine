# Kandinsky 2.2 Training

You can also check [`configs/kandinsky_v22/README.md`](../../../configs/kandinsky_v22/README.md) file.

## Configs

All configuration files are placed under the [`configs/kandinsky_v22`](../../../configs/kandinsky_v22/) folder.

Following is the example config fixed from the kandinsky_v22_prior_pokemon_blip config file in [`configs/kandinsky_v22/kandinsky_v22_prior_pokemon_blip.py`](../../../configs/kandinsky_v22/kandinsky_v22_prior_pokemon_blip.py):

```
_base_ = [
    "../_base_/models/kandinsky_v22_prior.py",
    "../_base_/datasets/pokemon_blip_kandinsky_prior.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]
```

## Run training

Run train

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example
$ mim train diffengine configs/kandinsky_v22/kandinsky_v22_prior_pokemon_blip.py
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

## Inference decoder with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/kandinsky_v22/kandinsky_v22_decoder_pokemon_blip.py work_dirs/kandinsky_v22_decoder_pokemon_blip/epoch_50.pth work_dirs/kandinsky_v22_decoder_pokemon_blip --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/kandinsky_v22_decoder_pokemon_blip'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet')
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    unet=unet,
    torch_dtype=torch.float32,
)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
    width=768,
    height=768,
).images[0]
image.save('demo.png')
```

## Results Example

#### kandinsky_v22_prior_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/b709f558-5c03-4235-98d7-fe1c663182b8)

You can check [`configs/kandinsky_v22/README.md`](../../../configs/kandinsky_v22/README.md#results-example) for more details.
