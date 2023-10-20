from argparse import ArgumentParser

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("checkpoint", help="Path to checkpoint.")
    parser.add_argument(
        "--sdmodel",
        help="Stable Diffusion model name",
        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--text_encoder",
        action="store_true",
        help="Use trained text encoder from dir.")
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

    unet = UNet2DConditionModel.from_pretrained(
        args.checkpoint, subfolder="unet", torch_dtype=torch.float16)
    if args.vaemodel is not None and args.text_encoder:
        vae = AutoencoderKL.from_pretrained(
            args.vaemodel,
            torch_dtype=torch.float16,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.checkpoint,
            subfolder="text_encoder",
            torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            safety_checker=None)
    elif args.text_encoder:
        text_encoder = CLIPTextModel.from_pretrained(
            args.checkpoint,
            subfolder="text_encoder",
            torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel,
            unet=unet,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            safety_checker=None)
    elif args.vaemodel is not None:
        vae = AutoencoderKL.from_pretrained(
            args.vaemodel,
            torch_dtype=torch.float16,
        )
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel,
            unet=unet,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None)
    pipe.to(args.device)

    image = pipe(
        args.prompt,
        num_inference_steps=50,
        height=args.height,
        width=args.width,
    ).images[0]
    image.save(args.out)


if __name__ == "__main__":
    main()
