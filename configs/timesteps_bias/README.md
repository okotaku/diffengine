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

![example1](https://github.com/okotaku/diffengine/assets/24734142/99f2139d-84f4-4658-8eb2-377216de2b0d)

#### stable_diffusion_xl_pokemon_blip_earlier_bias

![example2](https://github.com/okotaku/diffengine/assets/24734142/4a353bcd-44f3-4066-a095-09696cb09d6f)

#### stable_diffusion_xl_pokemon_blip_range_bias

![example3](https://github.com/okotaku/diffengine/assets/24734142/13b9b041-9103-4f04-8e19-569d59a55bcf)
