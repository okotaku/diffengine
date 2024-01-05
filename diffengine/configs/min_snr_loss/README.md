# Efficient Diffusion Training via Min-SNR Weighting Strategy

[Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)

## Abstract

Denoising diffusion models have been a mainstream approach for image generation, however, training these models often suffers from slow convergence. In this paper, we discovered that the slow convergence is partly due to conflicting optimization directions between timesteps. To address this issue, we treat the diffusion training as a multi-task learning problem, and introduce a simple yet effective approach referred to as Min-SNR-γ. This method adapts loss weights of timesteps based on clamped signal-to-noise ratios, which effectively balances the conflicts among timesteps. Our results demonstrate a significant improvement in converging speed, 3.4× faster than previous weighting strategies. It is also more effective, achieving a new record FID score of 2.06 on the ImageNet 256×256 benchmark using smaller architectures than that employed in previous state-of-the-art.

## Citation

```
@InProceedings{Hang_2023_ICCV,
    author    = {Hang, Tiankai and Gu, Shuyang and Li, Chen and Bao, Jianmin and Chen, Dong and Hu, Han and Geng, Xin and Guo, Baining},
    title     = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {7441-7451}
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
$ mim train diffengine configs/min_snr_loss/stable_diffusion_v15_snr_pokemon_blip.py
```

## Inference with diffusers

You can see more details on [`docs/source/run_guides/run_sd.md`](../../docs/source/run_guides/run_sd.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_v15_snr_pokemon_blip

![example4](https://github.com/okotaku/diffengine/assets/24734142/b98e887a-d3af-49bb-ad15-9e8250c09578)
