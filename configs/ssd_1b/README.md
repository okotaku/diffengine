# SSD-1B

[SSD-1B](https://blog.segmind.com/introducing-segmind-ssd-1b/)

## Abstract

Today, Segmind is thrilled to announce the open sourcing of our new foundational model, SSD-1B, the fastest diffusion-based text-to-image model in the market, with unprecedented image generation times for a 1024x1024 image. Developed as part of our distillation series, SSD-1B is 50% smaller and 60% faster compared to the SDXL 1.0 model. This reduction in speed and size comes with a minimal impact on image quality when compared to SDXL 1.0. Furthermore, we are excited to reveal that the SSD-1B model has been licensed for commercial use, opening avenues for businesses and developers to integrate this groundbreaking technology into their services and products.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/5c5a0e65-d06d-43a0-873d-f804e1900428"/>
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
$ mim train diffengine configs/ssd_1b/ssd_1b_distill_pokemon_blip.py
```

## Inference with diffusers

You can see more details on [`docs/source/run_guides/run_xl.md`](../../docs/source/run_guides/run_xl.md#inference-with-diffusers).

## Results Example

#### ssd_1b_distill_from_sdxl_pokemon_blip

![example](https://github.com/okotaku/diffengine/assets/24734142/057a347f-4baf-443d-ac75-e8a073a43a27)

#### ssd_1b_distill_pokemon_blip

![example2](https://github.com/okotaku/diffengine/assets/24734142/304a2cf8-22a5-4c1e-a6b5-b12c1a245bf4)

## Blog post

[SSD-1B: A Leap in Efficient T2I Generation](https://medium.com/@to78314910/ssd-1b-a-leap-in-efficient-t2i-generation-138bb05fdd75)

## Acknowledgement

These implementations are based on [segmind/SSD-1B](https://github.com/segmind/SSD-1B). Thank you for the great open source project.
