# Offset Noise

[Diffusion with Offset Noise](https://www.crosslabs.org/blog/diffusion-with-offset-noise)

## Abstract

Fine-tuning against a modified noise, enables Stable Diffusion to generate very dark or light images easily.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/76038bc8-b614-49da-9751-1a9efb83995f"/>
</div>

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ mim train diffengine ${CONFIG_FILE}
# multi gpus
$ mim train diffengine ${CONFIG_FILE} --gpus 2 --launcher pytorch

# Example.
$ mim train diffengine configs/offset_noise/stable_diffusion_xl_pokemon_blip_offset_noise.py
```

## Inference with diffusers

You can see details on [`docs/source/run_guides/run_xl.md`](../../docs/source/run_guides/run_xl.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_xl_pokemon_blip_offset_noise

![example1](https://github.com/okotaku/diffengine/assets/24734142/7a3b26ff-618b-46f0-827e-32c2d47cde6f)
