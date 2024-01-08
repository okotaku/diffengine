# Learn about Configs

The config system has a modular and inheritance design, and more details can be found in
[mmengine docs: CONFIG](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta).

Usually, we use python files as config file. All configuration files are placed under the [`configs`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs) folder, and the directory structure is as follows:

```text
DiffEngine/diffengine/
    ├── configs/
    │   ├── _base_/                       # primitive configuration folder
    │   │   ├── datasets/                      # primitive datasets
    │   │   ├── models/                        # primitive models
    │   │   ├── schedules/                     # primitive schedules
    │   │   └── default_runtime.py             # primitive runtime setting
    │   ├── stable_diffusion/             # Stable Diffusion Algorithms Folder
    │   ├── stable_diffusion_xl/          # Stable Diffusion XL Algorithms Folder
    │   ├── ...
    └── ...
```

## Config Structure

There are four kinds of basic component files in the `configs/_base_` folders, namely：

- [models](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/_base_/models)
- [datasets](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/_base_/datasets)
- [schedules](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/_base_/schedules)
- [runtime](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/_base_/default_runtime.py)

We call all the config files in the `_base_` folder as _primitive_ config files. You can easily build your training config file by inheriting some primitive config files.

For easy understanding, we use [stable_diffusion_v15_pokemon_blip config file](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py) as an example and comment on each line.

```python
from mmengine.config import read_base

with read_base():  # This config file will inherit all config files in `_base_`.
    from .._base_.datasets.pokemon_blip import *           # model settings
    from .._base_.default_runtime import *                 # data settings
    from .._base_.models.stable_diffusion_v15 import *     # schedule settings
    from .._base_.schedules.stable_diffusion_50e import *  # runtime settings
```

We will explain the four primitive config files separately below.

### Model settings

This primitive config file includes a dict variable `model`, which mainly includes information such as network structure and loss function:

Usually, we use the **`type` field** to specify the class of the component and use other fields to pass
the initialization arguments of the class.

Following is the model primitive config of the stable_diffusion_v15 config file in [`configs/_base_/models/stable_diffusion_v15.py`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/_base_/models/stable_diffusion_v15.py):

```python
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import StableDiffusion

base_model = "runwayml/stable-diffusion-v1-5"  # pretrained model name of stable diffusion
model = dict(type=StableDiffusion,  # The type of the main model.
             model=base_model,
             tokenizer=dict(  # tokenizer settings
                type=CLIPTokenizer.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="tokenizer"),
             scheduler=dict(  # scheduler settings
                type=DDPMScheduler.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="scheduler"),
             text_encoder=dict(  # text encoder settings
                type=CLIPTextModel.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="text_encoder"),
             vae=dict(  # vae settings
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="vae"),
             unet=dict(  # unet settings
                type=UNet2DConditionModel.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="unet"))

```

### Data settings

This primitive config file includes information to construct the dataloader:

Following is the data primitive config of the stable_diffusion_v15 config in [`configs/_base_/datasets/pokemon_blip.py`]https://github.com/okotaku/diffengine/blob/main/diffengine/configs/_base_/datasets/pokemon_blip.py)：

```python
import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFDataset
from diffengine.datasets.transforms import (
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import SDCheckpointHook, VisualizationHook

train_pipeline = [  # augmentation settings
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=512, interpolation="bilinear"),
    dict(type=RandomCrop, size=512),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs),
]
train_dataloader = dict(
    batch_size=4,  # batch size
    num_workers=4,
    dataset=dict(
        type=HFDataset,  # The type of dataset
        dataset="lambdalabs/pokemon-blip-captions",  #  Dataset name or path.
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(type=VisualizationHook, prompt=['yoda pokemon'] * 4),  # validation visualize prompt
    dict(type=SDCheckpointHook)
]
```

### Schedule settings

This primitive config file mainly contains training strategy settings and the settings of training, val and
test loops:

Following is the schedule primitive config of the stable_diffusion_v15 config in [`configs/_base_/schedules/stable_diffusion_50e.py`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/_base_/schedules/stable_diffusion_50e.py)：


```python
from mmengine.hooks import CheckpointHook
from mmengine.optim import AmpOptimWrapper
from torch.optim import AdamW

optim_wrapper = dict(
    type=AmpOptimWrapper, dtype="float16",  # fp16 optimization
    optimizer=dict(type=AdamW, lr=1e-5, weight_decay=1e-2),  # Use AdamW optimizer to optimize parameters.
    clip_grad=dict(max_norm=1.0))

# Training configuration, iterate 50 epochs.
# 'by_epoch=True' means to use `EpochBaseTrainLoop`, 'by_epoch=False' means to use IterBaseTrainLoop.
train_cfg = dict(by_epoch=True, max_epochs=50)
val_cfg = None
test_cfg = None

default_hooks = dict(
    # save checkpoint per epoch and keep 3 checkpoints.
    checkpoint=dict(
        type=CheckpointHook,
        interval=1,
        max_keep_ckpts=3,
    ))
```

### Runtime settings

This part mainly includes saving the checkpoint strategy, log configuration, training parameters, breakpoint weight path, working directory, etc.

Here is the runtime primitive config file ['configs/_base_/default_runtime.py'](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/_base_/default_runtime.py) file used by almost all configs:

```
default_scope = 'diffengine'

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi-process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
```

## Inherit and Modify Config File

For easy understanding, we recommend contributors inherit from existing config files. But do not abuse the
inheritance. Usually, for all config files, we recommend the maximum inheritance level is 3.

For example, if your config file is based on ResNet with some other modification, you can first inherit the
basic stable_diffusion_v15_pokemon_blip structure, dataset and other training settings by specifying `_base_ ='./stable_diffusion_v15_pokemon_blip.py'`
(The path relative to your config file), and then modify the necessary parameters in the config file. A more
specific example, now we want to use almost all configs in `configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py`, but changing the number of training epochs from 50 to 300, modify pretrained model, modify
the learning rate schedule, and modify the dataset path, you can create a new config file
`configs/stable_diffusion/stable_diffusion_v15_pokemon_blip-300e.py` with content as below:

```python
from mmengine.config import read_base

with read_base():  # This config file will inherit all config files in `_base_`.
    from diffengine.configs.stable_diffusion.stable_diffusion_v15_pokemon_blip import * 

# trains more epochs
train_cfg.update(max_epochs=300)  # Train for 300 epochs
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=300)
]

# Use your own dataset directory
train_dataloader.update(
    dataset=dict(dataset='mydata/pokemon-blip-captions'),
)
```

## Acknowledgement

This content refers to [mmengine docs: CONFIG](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta). Thank you for the great docs.
