# Wuerstchen LoRA Training

You can also check [`configs/wuerstchen_lora/README.md`](../../../configs/wuerstchen_lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/wuerstchen_lora`](../../../configs/wuerstchen_lora/) folder.

Following is the example config fixed from the wuerstchen_prior_lora_pokemon_blip config file in [`configs/wuerstchen_lora/wuerstchen_prior_lora_pokemon_blip.py`](../../../configs/wuerstchen_lora/wuerstchen_prior_lora_pokemon_blip.py):

```
_base_ = [
    "../_base_/models/wuerstchen_prior_lora.py",
    "../_base_/datasets/pokemon_blip_wuerstchen.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(batch_size=8)

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))
```

## Run LoRA training

Run LoRA training:

```
# single gpu
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE}
# Example
$ docker compose exec diffengine mim train diffengine configs/wuerstchen_lora/wuerstchen_prior_lora_pokemon_blip.py

# multi gpus
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
```

## Results Example

#### wuerstchen_prior_lora_pokemon_blip

![example1]()

You can check [`configs/wuerstchen_lora/README.md`](../../../configs/wuerstchen_lora/README.md#results-example) for more details.
