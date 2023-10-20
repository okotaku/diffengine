from argparse import ArgumentParser

import torch
from diffusers import AutoencoderKL, StableDiffusionXLAdapterPipeline, T2IAdapter
from diffusers.utils import load_image


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("condition_image", help="Path to condition image.")
    parser.add_argument("checkpoint", help="Path to adapter weight.")
    parser.add_argument(
        "--sdmodel",
        help="Stable Diffusion model name",
        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument(
        "--vaemodel",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. "
        "More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument("--out", help="Output path", default="demo.png")
    parser.add_argument(
        "--height",
        help="The height for output images.",
        default=None,
        type=int)
    parser.add_argument(
        "--width", help="The width for output images.", default=None, type=int)
    parser.add_argument(
        "--device", help="Device used for inference", default="cuda")
    args = parser.parse_args()

    adapter = T2IAdapter.from_pretrained(
        args.checkpoint, subfolder="adapter", torch_dtype=torch.float16)
    if args.vaemodel is not None:
        vae = AutoencoderKL.from_pretrained(
            args.vaemodel,
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            args.sdmodel,
            adapter=adapter,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None)
    else:
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            args.sdmodel,
            adapter=adapter,
            torch_dtype=torch.float16,
            safety_checker=None)
    pipe.to(args.device)

    image = pipe(
        args.prompt,
        args.prompt,
        load_image(args.condition_image).resize((args.width, args.height)),
        num_inference_steps=50,
        height=args.height,
        width=args.width,
    ).images[0]
    image.save(args.out)


if __name__ == "__main__":
    main()
