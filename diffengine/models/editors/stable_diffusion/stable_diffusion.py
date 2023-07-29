from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from mmengine import print_log
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.archs import set_unet_lora
from diffengine.models.losses.snr_l2_loss import SNRL2Loss
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusion(BaseModel):
    """Stable Diffusion.

    Args:
        model (str): pretrained model name of stable diffusion.
            Defaults to 'runwayml/stable-diffusion-v1-5'.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        noise_offset_weight (bool, optional):
            The weight of noise offset introduced in
            https://www.crosslabs.org/blog/diffusion-with-offset-noise
            Defaults to 0.
    """

    def __init__(
        self,
        model: str = 'runwayml/stable-diffusion-v1-5',
        loss: dict = dict(type='L2Loss', loss_weight=1.0),
        lora_config: Optional[dict] = None,
        noise_offset_weight: float = 0,
    ):
        super().__init__()
        self.model = model
        self.lora_config = deepcopy(lora_config)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

        self.enable_noise_offset = noise_offset_weight > 0
        self.noise_offset_weight = noise_offset_weight

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model, subfolder='tokenizer')
        self.scheduler = DDPMScheduler.from_pretrained(
            model, subfolder='scheduler')

        self.text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder='text_encoder')
        self.vae = AutoencoderKL.from_pretrained(model, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(
            model, subfolder='unet')
        self.prepare_model()
        self.set_lora()

    def set_lora(self):
        """Set LORA for model."""
        if self.lora_config:
            self.unet.requires_grad_(False)
            set_unet_lora(self.unet, self.lora_config)

    def prepare_model(self):
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.vae.requires_grad_(False)
        print_log('Set VAE untrainable.', 'current')
        self.text_encoder.requires_grad_(False)
        print_log('Set Text Encoder untrainable.', 'current')

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(self,
              prompt: List[str],
              height: Optional[int] = None,
              width: Optional[int] = None) -> List[np.ndarray]:
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to
                `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to
                `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
        """
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            safety_checker=None,
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p in prompt:
            image = pipeline(
                p, num_inference_steps=50, height=height,
                width=width).images[0]
            images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        data['text'] = self.tokenizer(
            data['text'],
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt').input_ids.to(self.device)
        data['pixel_values'] = torch.stack(data['pixel_values'])
        data = self.data_preprocessor(data)

        latents = self.vae.encode(data['pixel_values']).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        if self.enable_noise_offset:
            noise = noise + self.noise_offset_weight * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=noise.device)

        num_batches = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(data['text'])[0]

        if self.scheduler.config.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.config.prediction_type}')

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample

        loss_dict = dict()
        # calculate loss in FP32
        if isinstance(self.loss_module, SNRL2Loss):
            loss_mse = self.loss_module(model_pred.float(), gt.float(),
                                        timesteps,
                                        self.scheduler.alphas_cumprod)
        else:
            loss_mse = self.loss_module(model_pred.float(), gt.float())
        loss_dict['loss_mse'] = loss_mse

        parsed_loss, log_vars = self.parse_losses(loss_dict)
        optim_wrapper.update_params(parsed_loss)

        return log_vars

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')
