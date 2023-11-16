from argparse import ArgumentParser
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from peft import PeftModel


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("checkpoint", help="Path to LoRA weight.")
    parser.add_argument(
        "--sdmodel",
        help="Stable Diffusion model name",
        default="runwayml/stable-diffusion-v1-5")
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

    checkpoint = Path(args.checkpoint)

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
        pipe.to(args.device)

        pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet",
                                              adapter_name="default")
        if (checkpoint / "text_encoder_one").exists():
            pipe.text_encoder_one = PeftModel.from_pretrained(
                pipe.text_encoder_one, checkpoint / "text_encoder_one",
                adapter_name="default",
            )
        if (checkpoint / "text_encoder_two").exists():
            pipe.text_encoder_one = PeftModel.from_pretrained(
                pipe.text_encoder_two, checkpoint / "text_encoder_two",
                adapter_name="default",
            )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.sdmodel, torch_dtype=torch.float16, safety_checker=None)
        pipe.to(args.device)
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet, checkpoint / "unet", adapter_name="default")
        if (checkpoint / "text_encoder").exists():
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, checkpoint / "text_encoder", adapter_name="default",
            )

    image = pipe(
        args.prompt,
        num_inference_steps=50,
        height=args.height,
        width=args.width,
    ).images[0]
    image.save(args.out)


if __name__ == "__main__":
    main()
