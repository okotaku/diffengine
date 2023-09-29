# IP-Adapter

[IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)

## Abstract

Recent years have witnessed the strong power of large text-to-image diffusion models for the impressive generative capability to create high-fidelity images. However, it is very tricky to generate desired images using only text prompt as it often involves complex prompt engineering. An alternative to text prompt is image prompt, as the saying goes: "an image is worth a thousand words". Although existing methods of direct fine-tuning from pretrained models are effective, they require large computing resources and are not compatible with other base models, text prompt, and structural controls. In this paper, we present IP-Adapter, an effective and lightweight adapter to achieve image prompt capability for the pretrained text-to-image diffusion models. The key design of our IP-Adapter is decoupled cross-attention mechanism that separates cross-attention layers for text features and image features. Despite the simplicity of our method, an IP-Adapter with only 22M parameters can achieve comparable or even better performance to a fully fine-tuned image prompt model. As we freeze the pretrained diffusion model, the proposed IP-Adapter can be generalized not only to other custom models fine-tuned from the same base model, but also to controllable generation using existing controllable tools. With the benefit of the decoupled cross-attention strategy, the image prompt can also work well with the text prompt to achieve multimodal image generation.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/5884b1ce-0550-4e08-9b10-35c501cefc99"/>
</div>

## Citation

```
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
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
$ mim train diffengine configs/ip_adapter/stable_diffusion_xl_pokemon_blip_ip_adapter.py
```

## Inference with diffusers

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

You can see more details on [`docs/source/run_guides/run_ip_adapter.md`](../../docs/source/run_guides/run_ip_adapter.md#inference-with-diffengine).

## Results Example

#### stable_diffusion_xl_pokemon_blip_ip_adapter

![input1](https://datasets-server.huggingface.co/assets/lambdalabs/pokemon-blip-captions/--/default/train/0/image/image.jpg)

![example1](https://github.com/okotaku/diffengine/assets/24734142/6137ffb4-dff9-41de-aa6e-2910d95e6d21)
