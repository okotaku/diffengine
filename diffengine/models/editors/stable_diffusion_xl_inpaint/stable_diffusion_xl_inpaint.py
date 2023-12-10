from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image
from torch import nn

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusionXLInpaint(StableDiffusionXL):
    """Stable Diffusion XL Inpaint.

    Args:
    ----
        model (str): pretrained model name of stable diffusion.
            Defaults to 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDXLInpaintDataPreprocessor`.
    """

    def __init__(
        self,
        *args,
        model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        data_preprocessor: dict | nn.Module | None = None,
        **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLInpaintDataPreprocessor"}

        super().__init__(
            *args,
            model=model,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              image: list[str | Image.Image],
              mask: list[str | Image.Image],
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
            image (`List[Union[str, Image.Image]]`):
                The image for inpainting.
            mask (`List[Union[str, Image.Image]]`):
                The mask for inpainting.
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
        assert len(prompt) == len(image) == len(mask)
        pipeline = AutoPipelineForInpainting.from_pretrained(
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
        for p, img, m in zip(prompt, image, mask, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            mask_image = load_image(m) if isinstance(m, str) else m
            mask_image = mask_image.convert("RGB")
            image = pipeline(
                p,
                p,
                mask_image=mask_image,
                image=pil_img,
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

    def forward(
            self,
            inputs: dict,
            data_samples: Optional[list] = None,  # noqa
            mode: str = "loss") -> dict:
        """Forward function.

        Args:
        ----
            inputs (dict): The input dict.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.
        """
        assert mode == "loss"
        num_batches = len(inputs["img"])
        if "result_class_image" in inputs:
            # use prior_loss_weight
            weight = torch.cat([
                torch.ones((num_batches // 2, )),
                torch.ones((num_batches // 2, )) * self.prior_loss_weight,
            ]).float().reshape(-1, 1, 1, 1)
        else:
            weight = None

        latents = self.vae.encode(inputs["img"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        masked_latents = self.vae.encode(inputs["masked_images"]).latent_dist.sample()
        masked_latents = masked_latents * self.vae.config.scaling_factor

        mask = F.interpolate(inputs["mask"],
                             size=(latents.shape[2], latents.shape[3]))

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

        if not self.pre_compute_text_embeddings:
            inputs["text_one"] = self.tokenizer_one(
                inputs["text"],
                max_length=self.tokenizer_one.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt").input_ids.to(self.device)
            inputs["text_two"] = self.tokenizer_two(
                inputs["text"],
                max_length=self.tokenizer_two.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt").input_ids.to(self.device)
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                inputs["text_one"], inputs["text_two"])
        else:
            prompt_embeds = inputs["prompt_embeds"]
            pooled_prompt_embeds = inputs["pooled_prompt_embeds"]
        unet_added_conditions = {
            "time_ids": inputs["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

        model_pred = self.unet(
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        return self.loss(model_pred, noise, latents, timesteps, weight)
