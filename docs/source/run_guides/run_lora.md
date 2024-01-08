# Stable Diffusion LoRA Training

You can also check [`configs/stable_diffusion_lora/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_lora`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_lora/) folder.

Following is the example config fixed from the stable_diffusion_v15_lora_pokemon_blip config file in [`configs/stable_diffusion_lora/stable_diffusion_v15_lora_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_lora/stable_diffusion_v15_lora_pokemon_blip.py):

```
from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lora import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(unet_lora_config=dict(r=32,
        lora_alpha=32))

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=PeftSaveHook),  # Need to change from SDCheckpointHook
]
```

#### Finetuning the text encoder and UNet with LoRA

The script also allows you to finetune the text_encoder along with the unet, [LoRA](https://arxiv.org/abs/2106.09685) parameters.

```
from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lora_textencoder import *
    from .._base_.schedules.stable_diffusion_50e import *


model.update(
    unet_lora_config=dict(r=32,
        lora_alpha=32),
    text_encoder_lora_config=dict(r=32,
        lora_alpha=32))

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=PeftSaveHook),  # Need to change from SDCheckpointHook
]
```

We also provide [`configs/_base_/models/stable_diffusion_v15_lora_textencoder.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/_base_/models/stable_diffusion_v15_lora_textencoder.py) as a base config and [`configs/stable_diffusion/stable_diffusion_v15_lora_textencoder_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion/stable_diffusion_v15_lora_textencoder_pokemon_blip.py) as a whole config.

## Run LoRA training

Run LoRA training:

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_v15_lora_pokemon_blip

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_v15_lora_pokemon_blip/step10450')
prompt = 'yoda pokemon'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")
if (checkpoint / "text_encoder").exists():
    pipe.text_encoder = PeftModel.from_pretrained(
        pipe.text_encoder, checkpoint / "text_encoder", adapter_name="default"
    )

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## Results Example

#### stable_diffusion_v15_lora_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/24899409-554d-4393-88e5-f8b8d6e6b36d)

You can check [`configs/stable_diffusion_lora/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_lora/README.md#results-example) for more details.
