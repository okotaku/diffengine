# Stable Diffusion XL

[Stable Diffusion XL](https://huggingface.co/papers/2307.01952)

## Run train

Start a docker container

```
$ docker compose up -d
```

Run train

```
# single gpu
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion_xl/stable_diffusion_xl_lora_pokemon_blip.py
# multi gpus
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion_xl/stable_diffusion_xl_lora_pokemon_blip.py --gpus 2 --launcher pytorch
```
