from argparse import ArgumentParser

import torch
from diffusers import AutoencoderKL, DiffusionPipeline


def null_safety(images, **kwargs):
    return images, [False] * len(images)


def main():
    parser = ArgumentParser()
    parser.add_argument('prompt', help='Prompt text')
    parser.add_argument('checkpoint', help='Prompt text')
    parser.add_argument(
        '--sdmodel',
        help='Stable Diffusion model name',
        default='runwayml/stable-diffusion-v1-5')
    parser.add_argument(
        '--vaemodel',
        type=str,
        default=None,
        help='Path to pretrained VAE model with better numerical stability. '
        'More details: https://github.com/huggingface/diffusers/pull/4038.',
    )
    parser.add_argument('--out', help='Output path', default='demo.jpg')
    parser.add_argument(
        '--device', help='Device used for inference', default='cuda')
    args = parser.parse_args()

    if args.vaemodel is not None:
        vae = AutoencoderKL.from_pretrained(
            args.vaemodel,
            torch_dtype=torch.float16,
        )
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel, vae=vae, torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel, torch_dtype=torch.float16)

    pipe.to(args.device)
    pipe.load_lora_weights(args.checkpoint)

    pipe.safety_checker = null_safety

    image = pipe(
        args.prompt,
        num_inference_steps=50,
    ).images[0]
    image.save(args.out)


if __name__ == '__main__':
    main()
