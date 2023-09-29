# IP-Adapter Training

You can also check [`configs/ip_adapter/README.md`](../../../configs/ip_adapter/README.md) file.

## Configs

All configuration files are placed under the [`configs/ip_adapter`](../../../configs/ip_adapter/) folder.

Following is the example config fixed from the stable_diffusion_xl_pokemon_blip_ip_adapter config file in [`configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py`](../../../configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_xl_ip_adapter.py',
    '../_base_/datasets/pokemon_blip_xl_ip_adapter.py',
    '../_base_/schedules/stable_diffusion_xl_50e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=1)

optim_wrapper_cfg = dict(accumulative_counts=4)  # update every four times
```

## Run training

Run train

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py
```

## Inference with diffengine

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffengine` module.

```py
from PIL import Image
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint_to_model, _load_checkpoint

from diffengine.registry import MODELS

init_default_scope('diffengine')

prompt = ['']
example_image = ['https://datasets-server.huggingface.co/assets/lambdalabs/pokemon-blip-captions/--/default/train/0/image/image.jpg']
config = 'configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py'
checkpoint = 'work_dirs/stable_diffusion_xl_pokemon_blip_ip_adapter/epoch_50.pth'
device = 'cuda'

config = Config.fromfile(config).copy()

StableDiffuser = MODELS.build(config.model)
StableDiffuser = StableDiffuser.to(device)

checkpoint = _load_checkpoint(checkpoint, map_location='cpu')
_load_checkpoint_to_model(StableDiffuser, checkpoint['state_dict'],
                            strict=False)

image = StableDiffuser.infer(prompt, example_image=example_image, width=1024, height=1024)[0]
Image.fromarray(image).save('demo.png')
```

We also provide inference demo scripts:

```
$ mim run diffengine demo_diffengine ${PROMPT} ${CONFIG} ${CHECKPOINT} --height 1024 --width 1024 --example-image  ${EXAMPLE_IMAGE}
# Example
$ mim run diffengine demo_diffengine "" configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py work_dirs/stable_diffusion_xl_pokemon_blip_ip_adapter/epoch_50.pth --height 1024 --width 1024 --example-image https://datasets-server.huggingface.co/assets/lambdalabs/pokemon-blip-captions/--/default/train/0/image/image.jpg
```

## Results Example

#### stable_diffusion_xl_pokemon_blip_ip_adapter

![input1](https://datasets-server.huggingface.co/assets/lambdalabs/pokemon-blip-captions/--/default/train/0/image/image.jpg)

![example1](https://github.com/okotaku/diffengine/assets/24734142/6137ffb4-dff9-41de-aa6e-2910d95e6d21)
