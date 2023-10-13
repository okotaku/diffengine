from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from mmengine import print_log
from PIL import Image
from torch import nn

from diffengine.models.editors.stable_diffusion import StableDiffusion
from diffengine.models.losses.snr_l2_loss import SNRL2Loss
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusionControlNet(StableDiffusion):
    """Stable Diffusion ControlNet.

    Args:
        controlnet_model (str, optional): Path to pretrained ControlNet model.
            If None, use the default ControlNet model from Unet.
            Defaults to None.
        transformer_layers_per_block (List[int], optional):
            The number of layers per block in the transformer. More details:
            https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small.
            Defaults to None.
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
                 controlnet_model: Optional[str] = None,
                 transformer_layers_per_block: Optional[List[int]] = None,
                 lora_config: Optional[dict] = None,
                 finetune_text_encoder: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDControlNetDataPreprocessor"}
        assert lora_config is None, \
            "`lora_config` should be None when training ControlNet"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training ControlNet"

        self.controlnet_model = controlnet_model
        self.transformer_layers_per_block = transformer_layers_per_block

        super().__init__(
            *args,
            lora_config=lora_config,
            finetune_text_encoder=finetune_text_encoder,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def set_lora(self):
        """Set LORA for model."""

    def prepare_model(self):
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.controlnet_model is not None:
            pre_controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model)
        else:
            pre_controlnet = ControlNetModel.from_unet(self.unet)

        if self.transformer_layers_per_block is not None:
            down_block_types = [
                ("DownBlock2D" if i == 0 else "CrossAttnDownBlock2D")
                for i in self.transformer_layers_per_block
            ]
            self.controlnet = ControlNetModel.from_config(
                pre_controlnet.config,
                down_block_types=down_block_types,
                transformer_layers_per_block=self.transformer_layers_per_block,
            )
            self.controlnet.load_state_dict(
                pre_controlnet.state_dict(), strict=False)
            del pre_controlnet
        else:
            self.controlnet = pre_controlnet

        if self.gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        self.text_encoder.requires_grad_(requires_grad=False)
        print_log("Set Text Encoder untrainable.", "current")
        self.unet.requires_grad_(requires_grad=False)
        print_log("Set Unet untrainable.", "current")

    @torch.no_grad()
    def infer(self,
              prompt: List[str],
              condition_image: List[Union[str, Image.Image]],
              negative_prompt: Optional[str] = None,
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 50,
              output_type: str = "pil",
              **kwargs) -> List[np.ndarray]:
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            condition_image (`List[Union[str, Image.Image]]`):
                The condition image for ControlNet.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (`int`, *optional*, defaults to
                `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to
                `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (int): Number of inference steps.
                Defaults to 50.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
        """
        assert len(prompt) == len(condition_image)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, condition_image):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            image = pipeline(
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
            mode: str = "loss"):
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

        latents = self.vae.encode(inputs["img"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        if self.enable_noise_offset:
            noise = noise + self.noise_offset_weight * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=noise.device)

        num_batches = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(inputs["text"])[0]

        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=inputs["condition_img"],
            return_dict=False,
        )

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample).sample

        loss_dict = {}
        # calculate loss in FP32
        if isinstance(self.loss_module, SNRL2Loss):
            loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                weight=weight)
        else:
            loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
        loss_dict["loss"] = loss
        return loss_dict
