# Inference

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR}
# Example
$ mim run diffengine publish_model2diffusers configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py work_dirs/stable_diffusion_v15_pokemon_blip/epoch_4.pth work_dirs/stable_diffusion_v15_pokemon_blip
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

We also provide inference demo scripts:

```
$ mim run diffengine demo ${PROMPT} ${CHECKPOINT}
# Example
$ mim run diffengine demo "yoda pokemon" work_dirs/stable_diffusion_v15_snr_pokemon_blip
```

## Inference Text Encoder and Unet finetuned weight with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/stable_diffusion/stable_diffusion_v15_textencoder_pokemon_blip.py work_dirs/stable_diffusion_v15_textencoder_pokemon_blip/epoch_50.pth work_dirs/stable_diffusion_v15_textencoder_pokemon_blip --save-keys unet text_encoder
```

Then we can run inference.

```py
import torch
from transformers import CLIPTextModel
from diffusers import DiffusionPipeline, UNet2DConditionModel

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/stable_diffusion_v15_pokemon_blip'

text_encoder = CLIPTextModel.from_pretrained(
            checkpoint,
            subfolder='text_encoder',
            torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', unet=unet, text_encoder=text_encoder, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

We also provide inference demo scripts with `--text_encoder`:

```
$ mim run diffengine demo ${PROMPT} ${CHECKPOINT} --text_encoder
# Example
$ mim run diffengine demo "yoda pokemon" work_dirs/stable_diffusion_v15_textencoder_pokemon_blip --text_encoder
```

## Inference LoRA weight with diffusers

Once you have trained a LoRA model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
import torch
from diffusers import DiffusionPipeline

checkpoint = 'work_dirs/stable_diffusion_v15_dreambooth_lora_dog/step999'
prompt = 'A photo of sks dog in a bucket'

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

We also provide inference demo scripts:

```bash
$ mim run diffengine demo_lora "A photo of sks dog in a bucket" work_dirs/stable_diffusion_v15_dreambooth_lora_dog/step999
```
