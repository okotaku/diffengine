# Prerequisites

Before installing DiffEngine, please ensure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/get-started/locally/).

# Installation

Below are quick steps for installation:

```
pip install openmim
pip install git+https://github.com/okotaku/diffengine.git
```

# Get Started

DiffEngine makes training easy through its pre-defined configs. These configs provide a streamlined way to start your training process. Here's how you can get started using one of the pre-defined configs:

1. **Choose a config**: You can find various pre-defined configs in the [`configs`](configs/) directory of the DiffEngine repository. For example, if you're interested in training a model for DreamBooth using the Stable Diffusion algorithm, you can use the [`configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py`](configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py).

2. **Start Training**: Open a terminal and use the following command to start training with the chosen config:

```bash
mim train diffengine stable_diffusion_v15_dreambooth_lora_dog.py
```

3. **Monitor Progress and get resutls**: The training process will begin, and you will see output similar to the provided example. You can monitor the progress of your training as it proceeds. The outputs of training is in `work_dirs/${CONFIG_NAME}`, `work_dirs/stable_diffusion_v15_dreambooth_lora_dog` in this case.

```
work_dirs/stable_diffusion_v15_dreambooth_lora_dog
├── 20230802_033741
|   ├── 20230802_033741.log  # log file
|   └── vis_data
|         ├── 20230802_033741.json  # log json file
|         ├── config.py  # config file for each experiment
|         └── vis_image  # visualized image from each step
├── step1199
|   └── pytorch_lora_weights.bin  # weight for inferencing with diffusers.pipeline
├── iter_1200.pth  # checkpoint from each step
├── last_checkpoint  # last checkpoint, it can be used for resuming
└── stable_diffusion_v15_dreambooth_lora_dog.py  # latest config file
```

An illustrative output example is provided below:

![img](https://github.com/okotaku/diffengine/assets/24734142/e4576779-e05f-42d0-a709-d6481eea87a9)

## Example Notebook

For a more hands-on introduction to DiffEngine, you can refer to the [Example Notebook](examples/example-dreambooth.ipynb) provided in the repository. This notebook demonstrates the process of training using SDV1.5 and SDV2.1 DreamBooth configurations.

# Docker

Below are quick steps for installation and run dreambooh training by docker:

```
git clone https://github.com/okotaku/diffengine
cd diffengine
docker compose up -d
docker compose exec diffengine mim train diffengine stable_diffusion_v15_dreambooth_lora_dog.py
```
