# Stable Diffusion ControlNet

[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

## Abstract

We present a neural network structure, ControlNet, to control pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small (\< 50k). Moreover, training a ControlNet is as fast as fine-tuning a diffusion model, and the model can be trained on a personal devices. Alternatively, if powerful computation clusters are available, the model can scale to large amounts (millions to billions) of data. We report that large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like edge maps, segmentation maps, keypoints, etc. This may enrich the methods to control large diffusion models and further facilitate related applications.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/97a5d6b7-90b9-4247-936c-c27e26b47cff"/>
</div>

## Citation

```
@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
  booktitle={IEEE International Conference on Computer Vision (ICCV)}
  year={2023},
}
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train stable_diffusion_v15_controlnet_fill50k
```

## Inference with diffusers

Once you have trained a model, specify the path to where the model is saved, and use it for inference with the `diffusers`.

```py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

checkpoint = 'work_dirs/stable_diffusion_v15_controlnet_fill50k/step6250'
prompt = 'cyan circle with brown floral background'
condition_image = load_image(
    'https://github.com/okotaku/diffengine/assets/24734142/1af9dbb0-b056-435c-bc4b-62a823889191'
)

controlnet = ControlNetModel.from_pretrained(
        checkpoint, subfolder='controlnet', torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    condition_image,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

You can see more details on [`docs/source/run_guides/run_controlnet.md`](../../docs/source/run_guides/run_controlnet.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_v15_controlnet_fill50k

![input1](https://github.com/okotaku/diffengine/assets/24734142/1af9dbb0-b056-435c-bc4b-62a823889191)

![example1](https://github.com/okotaku/diffengine/assets/24734142/a14cc9a6-3a40-4577-bd5a-2ddbab60970d)

#### stable_diffusion_v15_controlnet_face_spiga

![input2](https://github.com/okotaku/diffengine/assets/24734142/73da6604-e4e6-4789-a9af-7f71ae2ba750)

![example2](https://github.com/okotaku/diffengine/assets/24734142/172b7c7a-a5a0-493a-8bcf-2d6491f44f90)

## Acknowledgement

These experiments are based on [diffusers docs](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README.md) and [blog post `Train your ControlNet with diffusers 🧨`](https://huggingface.co/blog/train-your-controlnet). Thank you for the great articles.
