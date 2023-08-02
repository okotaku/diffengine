# Installation

Below are quick steps for installation:

```
pip install openmim
pip install git+https://github.com/okotaku/diffengine.git
```

# Quick Run

Below are quick steps for run dreambooh training:

```
mim train diffengine stable_diffusion_v15_dreambooth_lora_dog.py
```

Outputs example is,

![img](https://github.com/okotaku/diffengine/assets/24734142/e4576779-e05f-42d0-a709-d6481eea87a9)

You can also run [Example Notebook](../../examples/example-dreambooth.ipynb). It has an example of SDV1.5 & SDV2.1 DreamBooth Training.

# Docker

Below are quick steps for installation and run dreambooh training by docker:

```
git clone https://github.com/okotaku/diffengine
cd diffengine
docker compose up -d
docker compose exec diffengine mim train diffengine stable_diffusion_v15_dreambooth_lora_dog.py
```
