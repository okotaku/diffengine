# Stable Diffusion XL DremBooth Training

You can also check [`configs/stable_diffusion_xl_dreambooth/README.md`](../../../configs/stable_diffusion_xl_dreambooth/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_xl_dreambooth`](../../../configs/stable_diffusion_xl_dreambooth/) folder.

Following is the example config fixed from the stable_diffusion_xl_dreambooth_lora_dog config file in [`configs/stable_diffusion_xl_dreambooth/stable_diffusion_xl_dreambooth_lora_dog.py`](../../../configs/stable_diffusion_xl_dreambooth/stable_diffusion_xl_dreambooth_lora_dog.py):

#### Finetuning the text encoder and UNet

The script also allows you to finetune the text_encoder along with the unet.

```
_base_ = [
    '../_base_/models/stable_diffusion_xl_lora.py',
    '../_base_/datasets/dog_dreambooth_xl.py',
    '../_base_/schedules/stable_diffusion_500.py',
    '../_base_/default_runtime.py'
]

model = dict(finetune_text_encoder=True)  # fine tune text encoder
```

#### Finetuning with Full Parameters (without LoRA)

The script also allows you to finetune full parameters.

```
_base_ = [
    '../_base_/datasets/dog_dreambooth_xl.py',
    '../_base_/schedules/stable_diffusion_500.py',
    '../_base_/default_runtime.py'
]

model = dict(  # not using lora_config
    type='StableDiffusionXL',
    model='stabilityai/stable-diffusion-xl-base-1.0',
    vae_model='madebyollin/sdxl-vae-fp16-fix')
```

#### Finetuning with prior-preserving loss

The script also allows you to finetune with prior-preserving loss.

```
_base_ = [
    '../_base_/models/stable_diffusion_xl_lora.py',
    '../_base_/datasets/dog_dreambooth_xl.py',
    '../_base_/schedules/stable_diffusion_500.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(
    dataset=dict(class_prompt='a photo of dog'),  # class_prompt=str means training with prior-preserving loss
)
```

## Run DreamBooth training

Run DreamBooth train

```
# single gpu
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE}
# Example
$ docker compose exec diffengine mim train diffengine configs/stable_diffusion_xl_dreambooth/stable_diffusion_xl_dreambooth_lora_dog.py

# multi gpus
$ docker compose exec diffengine mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch
```

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`. 

```py
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

checkpoint = 'work_dirs/stable_diffusion_xl_dreambooth_lora_dog/step499'
prompt = 'A photo of sks dog in a bucket'

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    vae=vae,
    torch_dtype=torch.float16)
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
$ mim run diffengine demo_lora "A photo of sks dog in a bucket" work_dirs/stable_diffusion_xl_dreambooth_lora_dog/step499 --sdmodel stabilityai/stable-diffusion-xl-base-1.0 --vaemodel madebyollin/sdxl-vae-fp16-fix
```

You can check [inference docs](inference.md) for inferencing other settings like Full Parameter Training without LoRA.

## Results Example

#### stable_diffusion_xl_dreambooth_lora_dog

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/ae1e4072-d2a3-445a-b11f-23d1f178a029)

You can check [stable_diffusion_xl_dreambooth.README](../../../configs/stable_diffusion_xl_dreambooth/README.md#results-example) for more deitals.
