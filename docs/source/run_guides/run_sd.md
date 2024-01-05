# Stable Diffusion Training

You can also check [`configs/stable_diffusion/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion) folder.

Following is the example config fixed from the stable_diffusion_v15_pokemon_blip config file in [`configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_v15.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]
```

#### Finetuning the text encoder and UNet

The script also allows you to finetune the text_encoder along with the unet.

```
_base_ = [
    '../_base_/models/stable_diffusion_v15.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]

model = dict(finetune_text_encoder=True)  # fine tune text encoder
```

We also provide [`configs/stable_diffusion/stable_diffusion_v15_textencoder_pokemon_blip.py`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion/stable_diffusion_v15_textencoder_pokemon_blip.py) as a whole config.

#### Finetuning with Unet EMA

The script also allows you to finetune with Unet EMA.

```
_base_ = [
    '../_base_/models/stable_diffusion_v15.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]

custom_hooks = [  # Hook is list, we should write all custom_hooks again.
    dict(type='VisualizationHook', prompt=['yoda pokemon'] * 4),
    dict(type='SDCheckpointHook'),
    dict(type='UnetEMAHook', momentum=1e-4, priority='ABOVE_NORMAL')  # setup EMA Hook
]
```

We also provide [`configs/stable_diffusion/stable_diffusion_v15_ema_pokemon_blip.py`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion/stable_diffusion_v15_ema_pokemon_blip.py) as a whole config.

#### Finetuning with Min-SNR Weighting Strategy

The script also allows you to finetune with [Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556).

```
_base_ = [
    '../_base_/models/stable_diffusion_v15.py',
    '../_base_/datasets/pokemon_blip.py',
    '../_base_/schedules/stable_diffusion_50e.py',
    '../_base_/default_runtime.py'
]

model = dict(loss=dict(type='SNRL2Loss', snr_gamma=5.0, loss_weight=1.0))  # setup Min-SNR Weighting Strategy
```

We also provide [`configs/min_snr_loss/stable_diffusion_v15_snr_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/min_snr_loss/stable_diffusion_v15_snr_pokemon_blip.py) as a whole config.

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_v15_pokemon_blip

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
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

## Inference Text Encoder and Unet finetuned weight with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert stable_diffusion_v15_textencoder_pokemon_blip work_dirs/stable_diffusion_v15_textencoder_pokemon_blip/epoch_50.pth work_dirs/stable_diffusion_v15_textencoder_pokemon_blip --save-keys unet text_encoder
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

## Results Example

#### stable_diffusion_v15_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/24d5254d-95be-46eb-8982-b38b6a11f1ba)

You can check [`configs/stable_diffusion/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion/README.md#results-example) for more details.
