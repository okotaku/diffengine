# Stable Diffusion Inpaint

[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## Abstract

By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/d874dc07-af7b-4a88-9629-f3be7c1d2d60"/>
</div>

## Citation

```
@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/stable_diffusion_inpaint/stable_diffusion_inpaint_dog.py
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/stable_diffusion_inpaint/stable_diffusion_inpaint_dog.py work_dirs/stable_diffusion_inpaint_dog/iter_1000.pth work_dirs/stable_diffusion_inpaint_dog --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.utils import load_image

prompt = 'a photo of sks dog'
img = 'https://github.com/okotaku/diffengine/assets/24734142/8e02bd0e-9dcc-49b6-94b0-86ab3b40bc2b'
mask = 'https://github.com/okotaku/diffengine/assets/24734142/d0de4fb9-9183-418a-970d-582e9324f05d'
checkpoint = 'work_dirs/stable_diffusion_inpaint_dog'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting', unet=unet, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    load_image(img).convert("RGB"),
    load_image(mask).convert("L"),
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

You can see more details on [`docs/source/run_guides/run_sd.md`](../../docs/source/run_guides/run_sd.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_inpaint_dog

![input](https://github.com/okotaku/diffengine/assets/24734142/8e02bd0e-9dcc-49b6-94b0-86ab3b40bc2b)

![mask](https://github.com/okotaku/diffengine/assets/24734142/d0de4fb9-9183-418a-970d-582e9324f05d)

![example](https://github.com/okotaku/diffengine/assets/24734142/f9ec820b-af75-4c74-8c0b-6558a0a19b95)

#### stable_diffusion_inpaint_dog_multi_mask

![input](https://github.com/okotaku/diffengine/assets/24734142/8e02bd0e-9dcc-49b6-94b0-86ab3b40bc2b)

![mask](https://github.com/okotaku/diffengine/assets/24734142/a40d1a4f-9c47-4fa0-936e-88a49c92c8d7)

![example](https://github.com/okotaku/diffengine/assets/24734142/f9766a71-0845-4dea-a037-f7dabfca200e)
