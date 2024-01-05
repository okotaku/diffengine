# Stable Diffusion XL DremBooth Training

You can also check [`configs/stable_diffusion_xl_dreambooth/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_xl_dreambooth/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion_xl_dreambooth`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_xl_dreambooth/) folder.

Following is the example config fixed from the stable_diffusion_xl_dreambooth_lora_dog config file in [`configs/stable_diffusion_xl_dreambooth/stable_diffusion_xl_dreambooth_lora_dog.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_xl_dreambooth/stable_diffusion_xl_dreambooth_lora_dog.py):

```
_base_ = [
    "../_base_/models/stable_diffusion_xl_lora.py",
    "../_base_/datasets/dog_dreambooth_xl.py",
    "../_base_/schedules/stable_diffusion_500.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(
    dataset=dict(class_image_config=dict(model={{_base_.model.model}})))
```

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
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_xl_dreambooth_lora_dog

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_xl_dreambooth_lora_dog/step499')
prompt = 'A photo of sks dog in a bucket'

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")
if (checkpoint / "text_encoder_one").exists():
    pipe.text_encoder_one = PeftModel.from_pretrained(
        pipe.text_encoder_one, checkpoint / "text_encoder_one", adapter_name="default"
    )
if (checkpoint / "text_encoder_two").exists():
    pipe.text_encoder_one = PeftModel.from_pretrained(
        pipe.text_encoder_two, checkpoint / "text_encoder_two", adapter_name="default"
    )

image = pipe(
    prompt,
    num_inference_steps=50,
    height=1024,
    width=1024,
).images[0]
image.save('demo.png')
```

You can check [inference docs](inference.md) for inferencing other settings like Full Parameter Training without LoRA.

## Results Example

#### stable_diffusion_xl_dreambooth_lora_dog

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/ae1e4072-d2a3-445a-b11f-23d1f178a029)

You can check [`configs/stable_diffusion_xl_dreambooth/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion_xl_dreambooth/README.md#results-example) for more details.
