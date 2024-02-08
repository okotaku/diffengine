from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from mmengine import print_log
from PIL import Image
from torch import nn

from diffengine.models.editors.stable_diffusion import StableDiffusion
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusionInpaint(StableDiffusion):
    """Stable Diffusion Inpaint.

    Args:
    ----
        model (str): pretrained model name of stable diffusion.
            Defaults to 'runwayml/stable-diffusion-v1-5'.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDInpaintDataPreprocessor`.
    """

    def __init__(self,
                 *args,
                 model: str = "runwayml/stable-diffusion-inpainting",
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDInpaintDataPreprocessor"}

        super().__init__(
            *args,
            model=model,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        # Fix input channels of Unet
        in_channels = 9
        if self.unet.in_channels != in_channels:
            out_channels = self.unet.conv_in.out_channels
            self.unet.register_to_config(in_channels=in_channels)

            with torch.no_grad():
                new_conv_in = nn.Conv2d(
                    in_channels, out_channels, self.unet.conv_in.kernel_size,
                    self.unet.conv_in.stride, self.unet.conv_in.padding,
                )
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
                self.unet.conv_in = new_conv_in

        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        if not self.finetune_text_encoder:
            self.text_encoder.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

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
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            safety_checker=None,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img, m in zip(prompt, image, mask, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            mask_image = load_image(m) if isinstance(m, str) else m
            mask_image = mask_image.convert("L")
            image = pipeline(
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
        inputs["text"] = self.tokenizer(
            inputs["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)
        num_batches = len(inputs["img"])
        if "result_class_image" in inputs:
            # use prior_loss_weight
            weight = torch.cat([
                torch.ones((num_batches // 2, )),
                torch.ones((num_batches // 2, )) * self.prior_loss_weight,
            ]).to(self.device).float().reshape(-1, 1, 1, 1)
        else:
            weight = None

        latents = self._forward_vae(inputs["img"], num_batches)
        masked_latents = self._forward_vae(inputs["masked_image"], num_batches)

        mask = F.interpolate(inputs["mask"],
                             size=(latents.shape[2], latents.shape[3]))

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

        encoder_hidden_states = self.text_encoder(inputs["text"])[0]

        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample

        return self.loss(model_pred, noise, latents, timesteps, weight)
