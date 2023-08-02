# Stable Diffusion fine-tuning

# Environment setup

Clone repo

```
$ git clone https://github.com/okotaku/diffengine
```

Start a docker container

```
$ docker compose up -d
```

# Run training

Run train

```
# single gpu
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE}
# Example
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py

# multi gpus
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Run LoRA training

```
# Example
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion/stable_diffusion_v15_lora_pokemon_blip.py
```

# Run inference with diffusers

1. Convert weights for diffusers

```
$ docker compose exec diffengine mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR}
# Example
$ docker compose exec diffengine mim run diffengine publish_model2diffusers configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py work_dirs/stable_diffusion_v15_pokemon_blip/epoch_4.pth work_dirs/stable_diffusion_v15_pokemon_blip
```

2. Run demo

```
$ docker compose exec diffengine mim run diffengine demo ${PROMPT} ${CHECKPOINT}
# Example
$ docker compose exec diffengine mim run diffengine demo "yoda pokemon" work_dirs/stable_diffusion_v15_snr_pokemon_blip
```

## Run LoRA demo

```
$ docker compose exec diffengine mim run diffengine demo_lora ${PROMPT} ${CHECKPOINT}
# Example
$ docker compose exec diffengine mim run diffengine demo_lora "yoda pokemon" work_dirs/stable_diffusion_v15_lora_textencoder_pokemon_blip/step209
```

# Prepare configs

For basic usage of configs, see [MMPreTrain: Learn about Configs](https://mmpretrain.readthedocs.io/en/latest/user_guides/config.html)

# Usage

# More details

See [MMEngine Docs](https://mmengine.readthedocs.io/en/latest/)
