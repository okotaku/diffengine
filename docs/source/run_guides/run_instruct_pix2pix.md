# InstructPix2Pix Training

You can also check [`configs/instruct_pix2pix/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/instruct_pix2pix/README.md) file.

## Configs

All configuration files are placed under the [`configs/instruct_pix2pix`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/instruct_pix2pix/) folder.

Following is the example config fixed from the stable_diffusion_xl_instruct_pix2pix config file in [`configs/instruct_pix2pix/stable_diffusion_xl_instruct_pix2pix.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/instruct_pix2pix/stable_diffusion_xl_instruct_pix2pix.py):

```
_base_ = [
    "../_base_/models/stable_diffusion_xl_instruct_pix2pix.py",
    "../_base_/datasets/instructpix2pix_xl.py",
    "../_base_/schedules/stable_diffusion_3e.py",
    "../_base_/default_runtime.py",
]

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type="Adafactor",
        lr=5e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    accumulative_counts=4)
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example
$ diffengine train stable_diffusion_xl_instruct_pix2pix
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert stable_diffusion_xl_instruct_pix2pix work_dirs/stable_diffusion_xl_instruct_pix2pix/epoch_3.pth work_dirs/stable_diffusion_xl_instruct_pix2pix --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import load_image

checkpoint = 'work_dirs/stable_diffusion_xl_instruct_pix2pix'
prompt = 'make the mountains snowy'
condition_image = load_image(
    'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png'
).resize((1024, 1024))

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', unet=unet, vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    image=condition_image,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

## Results Example

#### stable_diffusion_xl_instruct_pix2pix

![input1](https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png)

![example1](https://github.com/okotaku/diffengine/assets/24734142/f66149fd-e375-4f85-bfbf-d4d046cd469a)

You can check [`configs/instruct_pix2pix/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/instruct_pix2pix/README.md#results-example) for more details.
