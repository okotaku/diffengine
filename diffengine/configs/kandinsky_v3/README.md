# Kandinsky 3

[Kandinsky 3](https://ai-forever.github.io/Kandinsky-3/)

## Abstract

We present Kandinsky 3.0, a large-scale text-to-image generation model based on latent diffusion, continuing the series of text-to-image Kandinsky models and reflecting our progress to achieve higher quality and realism of image generation. Compared to previous versions of Kandinsky 2.x, Kandinsky 3.0 leverages a two times larger UNet backbone, a ten times larger text encoder and remove diffusion mapping. We describe the architecture of the model, the data collection procedure, the training technique, the production system of user interaction. We focus on the key components that, as we have identified as a result of a large number of experiments, had the most significant impact on improving the quality of our model in comparison with the other ones. By results of our side by side comparisons Kandinsky become better in text understanding and works better on specific domains.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/2d670f44-9fa1-4095-be96-a82c91c9590b"/>
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
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train kandinsky_v3_pokemon_blip
```

## Inference prior with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

Before inferencing, we should convert weights for diffusers format,

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
# Note that when training colossalai, use `--colossalai` and set `INPUT_FILENAME` to index file.
$ diffengine convert kandinsky_v3_pokemon_blip work_dirs/kandinsky_v3_pokemon_blip/epoch_50.pth/model/pytorch_model.bin.index.json work_dirs/kandinsky_v3_pokemon_blip --save-keys unet --colossalai
```

Then we can run inference.

```py
from diffusers import AutoPipelineForText2Image, Kandinsky3UNet

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/kandinsky_v3_pokemon_blip'

unet = Kandinsky3UNet.from_pretrained(
    checkpoint, subfolder='unet')
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-3",
    unet=unet,
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

![example1](https://github.com/okotaku/diffengine/assets/24734142/8f078fa8-9485-40d9-8174-5996257aed88)
