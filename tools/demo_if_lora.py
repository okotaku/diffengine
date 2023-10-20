from argparse import ArgumentParser

import torch
from diffusers import DiffusionPipeline, IFPipeline


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("prompt", help="Prompt text.")
    parser.add_argument("checkpoint", help="Path to LoRA weight.")
    parser.add_argument(
        "--if1-model", help="IF1 model name", default="DeepFloyd/IF-I-XL-v1.0")
    parser.add_argument(
        "--if2-model", help="IF2 model name", default="DeepFloyd/IF-II-L-v1.0")
    parser.add_argument("--out", help="Output path", default="demo.png")
    parser.add_argument(
        "--device", help="Device used for inference", default="cuda")
    args = parser.parse_args()

    pipe = IFPipeline.from_pretrained(args.if1_model)
    pipe2 = DiffusionPipeline.from_pretrained(
        args.if2_model, text_encoder=None, torch_dtype=torch.float16)
    pipe3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16,
    )

    pipe.to(args.device)
    pipe2.to(args.device)
    pipe3.to(args.device)

    pipe.load_lora_weights(args.checkpoint)

    prompt_embeds, negative_embeds = pipe.encode_prompt(args.prompt)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        output_type="pt",
        num_inference_steps=50,
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
