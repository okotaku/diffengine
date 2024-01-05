# PixArt-α DremBooth Training

You can also check [`configs/pixart_alpha_dreambooth/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_dreambooth/README.md) file.

## Configs

All configuration files are placed under the [`configs/pixart_alpha_dreambooth`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_dreambooth/) folder.

Following is the example config fixed from the pixart_alpha_1024_dreambooth_lora_dog config file in [`configs/pixart_alpha_dreambooth/pixart_alpha_1024_dreambooth_lora_dog.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_dreambooth/pixart_alpha_1024_dreambooth_lora_dog.py):

```
_base_ = [
    "../_base_/models/pixart_alpha_1024_lora.py",
    "../_base_/datasets/dog_dreambooth_pixart_1024.py",
    "../_base_/schedules/stable_diffusion_500.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(
    dataset=dict(class_image_config=dict(model={{_base_.model.model}})))

optim_wrapper = dict(
    dtype="bfloat16",
    optimizer=dict(lr=1e-4))
```

## Run DreamBooth training

Run DreamBooth train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train pixart_alpha_1024_dreambooth_lora_dog
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/pixart_alpha_1024_dreambooth_lora_dog/step499')
prompt = 'A photo of sks dog in a bucket'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    vae=vae,
    torch_dtype=torch.float32,
).to("cuda")
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint / "transformer", adapter_name="default")

img = pipe(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=50,
).images[0]
img.save("demo.png")
```

You can check [inference docs](inference.md) for inferencing other settings like Full Parameter Training without LoRA.

## Results Example

#### pixart_alpha_512_dreambooth_lora_dog

![exampledog](https://github.com/okotaku/diffengine/assets/24734142/2d3b59b1-2a9b-422d-adba-347504c66be2)

#### pixart_alpha_1024_dreambooth_lora_dog

![exampledog2](https://github.com/okotaku/diffengine/assets/24734142/a3fc9fcd-7cd0-4dc2-997d-3a9b303c228a)

You can check [`configs/pixart_alpha_dreambooth/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/pixart_alpha_dreambooth/README.md#results-example) for more details.
