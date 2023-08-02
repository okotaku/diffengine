# Stable Diffusion DremBooth Training

You can also check [`configs/stable_diffusion_dreambooth/README.md`](../../../configs/stable_diffusion_dreambooth/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_dreambooth`](../../../configs/stable_diffusion_dreambooth/) folder.

#### Finetuning the text encoder and UNet

The script also allows you to finetune the text_encoder along with the unet.
Following is the example config fixed from the stable_diffusion_v15_dreambooth_lora_dog config file in [`configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py`](../../../configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_v15_lora_textencoder.py',  # fix base config to `stable_diffusion_v15_lora_textencoder.py'
    '../_base_/datasets/dog_dreambooth.py',
    '../_base_/schedules/stable_diffusion_v15_1k.py',
    '../_base_/default_runtime.py'
]
```

#### Finetuning with Full Parameters (without LoRA)

The script also allows you to finetune full parameters.
Following is the example config fixed from the stable_diffusion_v15_dreambooth_lora_dog config file in [`configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py`](../../../configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_v15.py',  # fix base config to `stable_diffusion_v15.py'
    '../_base_/datasets/dog_dreambooth.py',
    '../_base_/schedules/stable_diffusion_v15_1k.py',
    '../_base_/default_runtime.py'
]
```

#### Finetuning without prior-preserving loss

The script also allows you to finetune without prior-preserving loss.
Following is the example config fixed from the stable_diffusion_v15_dreambooth_lora_dog config file in [`configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py`](../../../configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py):

```
_base_ = [
    '../_base_/models/stable_diffusion_v15_lora.py',
    '../_base_/datasets/dog_dreambooth.py',
    '../_base_/schedules/stable_diffusion_v15_1k.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(
    dataset=dict(class_prompt=None),  # class_prompt=None means training without prior-preserving loss
)
```

## Run DreamBooth training

Run DreamBooth train

```
# single gpu
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE}
# Example
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion_dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py

# multi gpus
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`. 

```py
import torch
from diffusers import DiffusionPipeline

checkpoint = 'work_dirs/stable_diffusion_v15_dreambooth_lora_dog/step999'
prompt = 'A photo of sks dog in a bucket'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.load_lora_weights(checkpoint)

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

We also provide inference demo scripts:

```bash
$ mim run diffengine demo_lora "A photo of sks dog in a bucket" work_dirs/stable_diffusion_v15_dreambooth_lora_dog/step999
```

You can check [inference docs](inference.md) for inferencing other settings like Full Parameter Training without LoRA.

## Results Example

#### stable_diffusion_v15_dreambooth_lora_dog

![examplev15](https://github.com/okotaku/diffengine/assets/24734142/f9c2430c-cee7-43cf-868f-35c6301dc573)

You can check [stable_diffusion_dreambooth.README](../../../configs/stable_diffusion_dreambooth/README.md#results-example) for more deitals.
