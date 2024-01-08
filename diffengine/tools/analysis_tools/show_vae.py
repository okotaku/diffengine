import argparse

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a checkpoint to be published")
    parser.add_argument("image", help="Path to image")
    parser.add_argument(
        "--vaemodel",
        help="VAE model name or path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--out", help="Output path", default="demo.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vae = AutoencoderKL.from_pretrained(args.vaevaemodel).eval()
    vae_scale_factor = 2**(len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    pixel_values = image_processor.preprocess(Image.open(args.image))
    latents = vae.encode(pixel_values).latent_dist.sample()
    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image.detach(), output_type="pil")[0]
    image.save(args.out)


if __name__ == "__main__":
    main()
