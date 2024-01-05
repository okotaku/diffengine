# Pyramid Noise

[Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2)

## Abstract

This report proposes a new noising approach that adds multi-resolution noise to an image or latent image during diffusion model training. A model trained with this technique can generate stunning images with a very different aesthetic to the usual diffusion model outputs. This seems like a promising direction for future research.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/943570cf-7283-4536-ae28-cd1cce1220b7"/>
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
$ diffengine train stable_diffusion_xl_pokemon_blip_pyramid_noise
```

## Inference with diffusers

You can see details on [`docs/source/run_guides/run_xl.md`](../../docs/source/run_guides/run_xl.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_xl_pokemon_blip_pyramid_noise

![example1](https://github.com/okotaku/diffengine/assets/24734142/8ee2f0b1-6ef6-4b5e-a018-8b0acbd73ec9)
