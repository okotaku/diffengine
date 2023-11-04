# Time Steps Bias

[Time Steps Bias](https://github.com/huggingface/diffusers/pull/5094)

## Abstract

TBD

<div align=center>
<img src=""/>
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
$ mim train diffengine configs/timesteps_bias/stable_diffusion_xl_pokemon_blip_later_bias.py
```

## Inference with diffusers

You can see more details on [`docs/source/run_guides/run_xl.md`](../../docs/source/run_guides/run_xl.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_xl_pokemon_blip_later_bias

![example1](<>)
