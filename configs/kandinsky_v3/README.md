# Kandinsky 3

[Kandinsky 3](https://ai-forever.github.io/Kandinsky-3/)

## Abstract

We present Kandinsky 3.0, a large-scale text-to-image generation model based on latent diffusion, continuing the series of text-to-image Kandinsky models and reflecting our progress to achieve higher quality and realism of image generation. Compared to previous versions of Kandinsky 2.x, Kandinsky 3.0 leverages a two times larger UNet backbone, a ten times larger text encoder and remove diffusion mapping. We describe the architecture of the model, the data collection procedure, the training technique, the production system of user interaction. We focus on the key components that, as we have identified as a result of a large number of experiments, had the most significant impact on improving the quality of our model in comparison with the other ones. By results of our side by side comparisons Kandinsky become better in text understanding and works better on specific domains.

<div align=center>
<img src=""/>
</div>

## Citation

```
@misc{arkhipkin2023kandinsky,
      title={Kandinsky 3.0 Technical Report},
      author={Vladimir Arkhipkin and Andrei Filatov and Viacheslav Vasilev and Anastasia Maltseva and Said Azizov and Igor Pavlov and Julia Agafonova and Andrey Kuznetsov and Denis Dimitrov},
      year={2023},
      eprint={2312.03511},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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
$ mim train diffengine configs/kandinsky_v3/kandinsky_v3_pokemon_blip.py
```

## Inference prior with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ mim run diffengine publish_model2diffusers ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ mim run diffengine publish_model2diffusers configs/kandinsky_v3/kandinsky_v3_pokemon_blip.py work_dirs/kandinsky_v3_pokemon_blip/epoch_50.pth work_dirs/kandinsky_v3_pokemon_blip --save-keys unet
```

Then we can run inference.

```py
import torch
from diffusers import AutoPipelineForText2Image, Kandinsky3UNet

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/kandinsky_v3_pokemon_blip'

unet = Kandinsky3UNet.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-3",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]
image.save('demo.png')
```

## Results Example

#### kandinsky_v3_pokemon_blip

![example1](<>)
