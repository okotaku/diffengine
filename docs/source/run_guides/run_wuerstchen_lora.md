# Wuerstchen LoRA Training

You can also check [`configs/wuerstchen_lora/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/wuerstchen_lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/wuerstchen_lora`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/wuerstchen_lora/) folder.

Following is the example config fixed from the wuerstchen_prior_lora_pokemon_blip config file in [`configs/wuerstchen_lora/wuerstchen_prior_lora_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/wuerstchen_lora/wuerstchen_prior_lora_pokemon_blip.py):

```
from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip_wuerstchen import *
    from .._base_.default_runtime import *
    from .._base_.models.wuerstchen_prior_lora import *
    from .._base_.schedules.stable_diffusion_50e import *


train_dataloader.update(
    batch_size=8,
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))

custom_hooks = [
    dict(type=VisualizationHook, prompt=["A robot pokemon, 4k photo"] * 4,
         height=768, width=768),
    dict(type=PeftSaveHook),
]
```

## Run LoRA training

Run LoRA training:

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train wuerstchen_prior_lora_pokemon_blip

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
```

## Results Example

#### wuerstchen_prior_lora_pokemon_blip

![example1]()

You can check [`configs/wuerstchen_lora/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/wuerstchen_lora/README.md#results-example) for more details.
