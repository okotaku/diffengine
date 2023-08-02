# Stable Diffusion XL fine-tuning

[Stable Diffusion XL](https://huggingface.co/papers/2307.01952)

# Environment setup

Clone repo

```
$ git clone https://github.com/okotaku/diffengine
```

Start a docker container

```
$ docker compose up -d
```

## Run LoRA training

```
# Example
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion_xl/stable_diffusion_xl_lora_pokemon_blip.py
```

# Run inference with diffusers

## Run LoRA demo

```
$ docker compose exec diffengine mim run diffengine demo_lora ${PROMPT} ${CHECKPOINT} --sdmodel stabilityai/stable-diffusion-xl-base-1.0
# Example
$ docker compose exec diffengine mim run diffengine demo_lora "yoda pokemon" work_dirs/stable_diffusion_xl_lora_pokemon_blip/step209 --sdmodel stabilityai/stable-diffusion-xl-base-1.0
```

# Prepare configs

For basic usage of configs, see [MMPreTrain: Learn about Configs](https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html)

# Usage

# More details

See [MMEngine Docs](https://mmengine.readthedocs.io/en/latest/)
