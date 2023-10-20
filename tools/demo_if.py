from argparse import ArgumentParser

import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("checkpoint", help="Path to checkpoint.")
    parser.add_argument(
        "--if1-model", help="IF1 model name", default="DeepFloyd/IF-I-XL-v1.0")
    parser.add_argument(
        "--if2-model", help="IF2 model name", default="DeepFloyd/IF-II-L-v1.0")
    parser.add_argument(
        "--text_encoder",
        action="store_true",
        help="Use trained text encoder from dir.")
    parser.add_argument("--out", help="Output path", default="demo.png")
    parser.add_argument(
        "--device", help="Device used for inference", default="cuda")
    args = parser.parse_args()

    unet = UNet2DConditionModel.from_pretrained(
        args.checkpoint, subfolder="unet")
    if args.text_encoder:
        text_encoder = CLIPTextModel.from_pretrained(
            args.checkpoint,
            subfolder="text_encoder",
            torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(
            args.if1_model,
            unet=unet,
            text_encoder=text_encoder,
            torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(args.if1_model, unet=unet)

    pipe2 = DiffusionPipeline.from_pretrained(
        args.if2_model, text_encoder=None, torch_dtype=torch.float16)
    pipe3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16,
    )

    pipe.to(args.device)
    pipe2.to(args.device)
    pipe3.to(args.device)

    prompt_embeds, negative_embeds = pipe.encode_prompt(args.prompt)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        output_type="pt",
    ).images
    image = pipe2(
        image=image,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        output_type="pt",
    ).images
    image = pipe3(prompt=args.prompt, image=image, noise_level=100).images[0]
    image.save(args.out)


if __name__ == "__main__":
    main()
