# LoHa

[FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning](https://arxiv.org/abs/2108.06098)

## Abstract

In this work, we propose a communication-efficient parameterization, FedPara, for federated learning (FL) to overcome the burdens on frequent model uploads and downloads. Our method re-parameterizes weight parameters of layers using low-rank weights followed by the Hadamard product. Compared to the conventional low-rank parameterization, our FedPara method is not restricted to low-rank constraints, and thereby it has a far larger capacity. This property enables to achieve comparable performance while requiring 3 to 10 times lower communication costs than the model with the original layers, which is not achievable by the traditional low-rank methods. The efficiency of our method can be further improved by combining with other efficient FL optimizers. In addition, we extend our method to a personalized FL application, pFedPara, which separates parameters into global and local ones. We show that pFedPara outperforms competing personalized FL methods with more than three times fewer parameters.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/0ba678b1-2adf-4b56-a972-d846ebf5738c"/>
</div>

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train stable_diffusion_xl_loha_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_xl_loha_pokemon_blip/step20850')
prompt = 'yoda pokemon'

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

## Results Example

#### stable_diffusion_xl_loha_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/3d312b70-349f-4ebb-bb11-0e8d9afaef4d)
