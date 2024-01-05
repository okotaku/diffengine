# Input Perturbation

[Input Perturbation Reduces Exposure Bias in Diffusion Models](https://arxiv.org/abs/2301.11706)

## Abstract

Denoising Diffusion Probabilistic Models have shown an impressive generation quality, although their long sampling chain leads to high computational costs. In this paper, we observe that a long sampling chain also leads to an error accumulation phenomenon, which is similar to the exposure bias problem in autoregressive text generation. Specifically, we note that there is a discrepancy between training and testing, since the former is conditioned on the ground truth samples, while the latter is conditioned on the previously generated results. To alleviate this problem, we propose a very simple but effective training regularization, consisting in perturbing the ground truth samples to simulate the inference time prediction errors. We empirically show that, without affecting the recall and precision, the proposed input perturbation leads to a significant improvement in the sample quality while reducing both the training and the inference times. For instance, on CelebA 64×64, we achieve a new state-of-the-art FID score of 1.27, while saving 37.5% of the training time.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/60b9a296-6453-4d47-9c06-f40f43766273"/>
</div>

## Citation

```
@article{ning2023input,
  title={Input Perturbation Reduces Exposure Bias in Diffusion Models},
  author={Ning, Mang and Sangineto, Enver and Porrello, Angelo and Calderara, Simone and Cucchiara, Rita},
  journal={arXiv preprint arXiv:2301.11706},
  year={2023}
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
$ diffengine train stable_diffusion_xl_pokemon_blip_input_perturbation
```

## Inference with diffusers

You can see details on [`docs/source/run_guides/run_xl.md`](../../docs/source/run_guides/run_xl.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_xl_pokemon_blip_input_perturbation

![example1](https://github.com/okotaku/diffengine/assets/24734142/b0a631e7-153c-467a-9cb6-d9155eaa7161)
