from argparse import ArgumentParser

import torch
from diffusers import AutoencoderKL, DiffusionPipeline


def main():
    parser = ArgumentParser()
    parser.add_argument('prompt', help='Prompt text.')
    parser.add_argument('checkpoint', help='Path to LoRA weight.')
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
    parser.add_argument('--out', help='Output path', default='demo.png')
    parser.add_argument(
        '--height',
        help='The height for output images.',
        default=None,
        type=int)
    parser.add_argument(
        '--width', help='The width for output images.', default=None, type=int)
    parser.add_argument(
        '--device', help='Device used for inference', default='cuda')
    args = parser.parse_args()

    if args.vaemodel is not None:
        vae = AutoencoderKL.from_pretrained(
            args.vaemodel,
            torch_dtype=torch.float16,
        )
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel, torch_dtype=torch.float16, safety_checker=None)

    pipe.to(args.device)
    pipe.load_lora_weights(args.checkpoint)

    image = pipe(
        args.prompt,
        num_inference_steps=50,
        height=args.height,
        width=args.width,
    ).images[0]
    image.save(args.out)


if __name__ == '__main__':
    main()
