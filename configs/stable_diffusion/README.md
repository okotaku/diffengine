# Stable Diffusion

[Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## Run train

Set env variables

```
$ export DATA_DIR=/path/to/data
```

Start a docker container

```
$ docker compose up -d
```

Run train

```
# single gpu
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py
# multi gpus
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py --gpus 2 --launcher pytorch
```
