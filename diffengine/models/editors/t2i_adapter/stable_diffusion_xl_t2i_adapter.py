from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from diffusers.utils import load_image
from mmengine import print_log
from PIL import Image
from torch import nn

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusionXLT2IAdapter(StableDiffusionXL):
    """Stable Diffusion XL T2I Adapter.

    Args:
    ----
        adapter_model (str, optional): Path to pretrained adapter model. If
            None, use the default adapter model. Defaults to None.
        adapter_model_channels (List[int]): The channels of adapter.
            Defaults to [320, 640, 1280, 1280].
        adapter_downscale_factor (int): The downscale factor of adapter.
            Defaults to 16.
        lora_config (dict, optional): The LoRA config dict. This should be
            `None` when training ControlNet. Defaults to None.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. This should be `False` when training ControlNet.
            Defaults to False.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDControlNetDataPreprocessor`.
    """

    def __init__(self,
                 *args,
                 adapter_model: str | None = None,
                 adapter_model_channels: list[int] | None = None,
                 adapter_downscale_factor: int = 16,
                 lora_config: dict | None = None,
                 finetune_text_encoder: bool = False,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLControlNetDataPreprocessor"}
        if adapter_model_channels is None:
            adapter_model_channels = [320, 640, 1280, 1280]
        assert lora_config is None, \
            "`lora_config` should be None when training ControlNet"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training ControlNet"

        self.adapter_model = adapter_model
        self.adapter_model_channels = adapter_model_channels
        self.adapter_downscale_factor = adapter_downscale_factor

        super().__init__(
            *args,
            lora_config=lora_config,
            finetune_text_encoder=finetune_text_encoder,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def set_lora(self) -> None:
        """Set LORA for model."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.adapter_model is not None:
            self.adapter = T2IAdapter.from_pretrained(self.adapter_model)
        else:
            self.adapter = T2IAdapter(
                in_channels=3,
                channels=self.adapter_model_channels,
                num_res_blocks=2,
                downscale_factor=self.adapter_downscale_factor,
                adapter_type="full_adapter_xl",
            )

        if self.gradient_checkpointing:
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
        pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            adapter=self.adapter,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            pipeline.scheduler.register_to_config(
                prediction_type=self.prediction_type)
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

    def forward(
            self,
            inputs: torch.Tensor,
            data_samples: Optional[list] = None,  # noqa
            mode: str = "loss") -> dict:
        """Forward function.

        Args:
        ----
            inputs (torch.Tensor): The input tensor.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.
        """
        assert mode == "loss"
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

        noise = self.noise_generator(latents)

        # Cubic sampling to sample a random time step for each image.
        # For more details about why cubic sampling is used, refer to section
        # 3.4 of https://arxiv.org/abs/2302.08453
        timesteps = torch.rand((num_batches, ), device=self.device)
        timesteps = (1 -
                     timesteps**3) * self.scheduler.config.num_train_timesteps
        timesteps = timesteps.long()
        timesteps = timesteps.clamp(
            0, self.scheduler.config.num_train_timesteps - 1)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            inputs["text_one"], inputs["text_two"])
        unet_added_conditions = {
            "time_ids": inputs["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

        down_block_additional_residuals = self.adapter(inputs["condition_img"])

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            down_block_additional_residuals=down_block_additional_residuals,
        ).sample

        return self.loss(model_pred, noise, latents, timesteps, weight)
