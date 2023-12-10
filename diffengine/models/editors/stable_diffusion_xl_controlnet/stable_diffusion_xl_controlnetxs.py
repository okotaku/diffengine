import numpy as np
import torch
from diffusers import ControlNetXSModel, StableDiffusionXLControlNetXSPipeline
from diffusers.utils import load_image
from mmengine import print_log
from PIL import Image

from diffengine.registry import MODELS

from .stable_diffusion_xl_controlnet import StableDiffusionXLControlNet


@MODELS.register_module()
class StableDiffusionXLControlNetXS(StableDiffusionXLControlNet):
    """ControlNet XS model for Stable Diffusion XL."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.controlnet = ControlNetXSModel.init_original(
            self.unet, is_sdxl=True)

        if self.gradient_checkpointing:
            # todo[takuoko]: Support ControlNetXSModel for gradient  # noqa
            # checkpointing
            # self.controlnet.enable_gradient_checkpointing()
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        self.text_encoder_one.requires_grad_(requires_grad=False)
        self.text_encoder_two.requires_grad_(requires_grad=False)
        print_log("Set Text Encoder untrainable.", "current")
        self.unet.requires_grad_(requires_grad=False)
        print_log("Set Unet untrainable.", "current")

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              condition_image: list[str | Image.Image],
              negative_prompt: str | None = None,
              height: int | None = None,
              width: int | None = None,
              num_inference_steps: int = 50,
              output_type: str = "pil",
              **kwargs) -> list[np.ndarray]:
        """Inference function.

        Args:
        ----
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            condition_image (`List[Union[str, Image.Image]]`):
                The condition image for ControlNet.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (int, optional):
                The height in pixels of the generated image. Defaults to None.
            width (int, optional):
                The width in pixels of the generated image. Defaults to None.
            num_inference_steps (int): Number of inference steps.
                Defaults to 50.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            **kwargs: Other arguments.
        """
        assert len(prompt) == len(condition_image)
        pipeline = StableDiffusionXLControlNetXSPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            controlnet=self.controlnet,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, condition_image, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            image = pipeline(
                p,
                p,
                pil_img,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type=output_type,
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def _forward_compile(self,
                         noisy_latents: torch.Tensor,
                         timesteps: torch.Tensor,
                         prompt_embeds: torch.Tensor,
                         unet_added_conditions: dict,
                         inputs: dict) -> torch.Tensor:
        """Forward function for torch.compile."""
        return self.controlnet(
            self.unet,
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            controlnet_cond=inputs["condition_img"],
        ).sample
