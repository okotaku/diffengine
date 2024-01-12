# aMUSEd

[aMUSEd: An Open MUSE Reproduction](https://arxiv.org/abs/2401.01808)

## Abstract

We present aMUSEd, an open-source, lightweight masked image model (MIM) for text-to-image generation based on MUSE. With 10 percent of MUSE's parameters, aMUSEd is focused on fast image generation. We believe MIM is under-explored compared to latent diffusion, the prevailing approach for text-to-image generation. Compared to latent diffusion, MIM requires fewer inference steps and is more interpretable. Additionally, MIM can be fine-tuned to learn additional styles with only a single image. We hope to encourage further exploration of MIM by demonstrating its effectiveness on large-scale text-to-image generation and releasing reproducible training code. We also release checkpoints for two models which directly produce images at 256x256 and 512x512 resolutions.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/d62b8007-2064-47ff-97c2-7c2377be3411"/>
</div>

## Citation

```
@misc{patil2024amused,
      title={aMUSEd: An Open MUSE Reproduction},
      author={Suraj Patil and William Berman and Robin Rombach and Patrick von Platen},
      year={2024},
      eprint={2401.01808},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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
$ diffengine train amused_512_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert amused_512_pokemon_blip work_dirs/amused_512_pokemon_blip/epoch_50.pth work_dirs/amused_512_pokemon_blip --save-keys transformer
```

Then we can run inference.

```py
from pathlib import Path

import torch
from diffusers import AmusedPipeline, UVit2DModel
from peft import PeftModel

checkpoint = Path('work_dirs/amused_512_pokemon_blip')
prompt = 'yoda pokemon'

transformer = UVit2DModel.from_pretrained(checkpoint, subfolder='transformer')
pipe = AmusedPipeline.from_pretrained(
    "amused/amused-512",
    transformer=transformer,
    torch_dtype=torch.float32,
).to("cuda")

img = pipe(
    prompt,
    width=512,
    height=512,
).images[0]
img.save("demo.png")
```

## Results Example

#### amused_512_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/a525dd2b-6663-42fb-8251-4d8767c19818)
