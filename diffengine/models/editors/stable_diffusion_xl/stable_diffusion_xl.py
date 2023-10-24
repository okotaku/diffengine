from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from mmengine import print_log
from mmengine.model import BaseModel
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig

from diffengine.models.archs import set_text_encoder_lora, set_unet_lora
from diffengine.registry import MODELS


def import_model_class_from_model_name_or_path(  # noqa
        pretrained_model_name_or_path: str,
        subfolder: str = "text_encoder"):
    """Import model class from model name or path."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder)
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":  # noqa
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        msg = f"{model_class} is not supported."
        raise ValueError(msg)


@MODELS.register_module()
class StableDiffusionXL(BaseModel):
    """`Stable Diffusion XL.

    <https://huggingface.co/papers/2307.01952>`_

    Args:
    ----
        model (str): pretrained model name of stable diffusion xl.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model (str, optional): Path to pretrained VAE model with better
            numerical stability. More details:
            https://github.com/huggingface/diffusers/pull/4038.
            Defaults to None.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        lora_config (dict, optional): The LoRA config dict.
            example. dict(rank=4). Defaults to None.
        prior_loss_weight (float): The weight of prior preservation loss.
            It works when training dreambooth with class images.
        noise_offset_weight (bool, optional):
            The weight of noise offset introduced in
            https://www.crosslabs.org/blog/diffusion-with-offset-noise
            Defaults to 0.
        prediction_type (str): The prediction_type that shall be used for
            training. Choose between 'epsilon' or 'v_prediction' or leave
            `None`. If left to `None` the default prediction type of the
            scheduler: `noise_scheduler.config.prediciton_type` is chosen.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDXLDataPreprocessor`.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
        pre_compute_text_embeddings(bool): Whether or not to pre-compute text
            embeddings to save memory. Defaults to False.
    """

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model: str | None = None,
        loss: dict | None = None,
        lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        noise_offset_weight: float = 0,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        pre_compute_text_embeddings: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLDataPreprocessor"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super().__init__(data_preprocessor=data_preprocessor)
        self.model = model
        self.lora_config = deepcopy(lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        if pre_compute_text_embeddings:
            assert not finetune_text_encoder

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

        self.enable_noise_offset = noise_offset_weight > 0
        self.noise_offset_weight = noise_offset_weight
        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        if not self.pre_compute_text_embeddings:
            self.tokenizer_one = AutoTokenizer.from_pretrained(
                model, subfolder="tokenizer", use_fast=False)
            self.tokenizer_two = AutoTokenizer.from_pretrained(
                model, subfolder="tokenizer_2", use_fast=False)

            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                model)
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                model, subfolder="text_encoder_2")
            self.text_encoder_one = text_encoder_cls_one.from_pretrained(
                model, subfolder="text_encoder")
            self.text_encoder_two = text_encoder_cls_two.from_pretrained(
                model, subfolder="text_encoder_2")

        self.scheduler = DDPMScheduler.from_pretrained(
            model, subfolder="scheduler")

        vae_path = model if vae_model is None else vae_model
        self.vae = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae" if vae_model is None else None)
        self.unet = UNet2DConditionModel.from_pretrained(
            model, subfolder="unet")
        self.prepare_model()
        self.set_lora()

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.lora_config is not None:
            if self.finetune_text_encoder:
                self.text_encoder_one.requires_grad_(requires_grad=False)
                self.text_encoder_two.requires_grad_(requires_grad=False)
                set_text_encoder_lora(self.text_encoder_one, self.lora_config)
                set_text_encoder_lora(self.text_encoder_two, self.lora_config)
            self.unet.requires_grad_(requires_grad=False)
            set_unet_lora(self.unet, self.lora_config)

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder_one.gradient_checkpointing_enable()
                self.text_encoder_two.gradient_checkpointing_enable()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        if (not self.finetune_text_encoder) and (
                not self.pre_compute_text_embeddings):
            self.text_encoder_one.requires_grad_(requires_grad=False)
            self.text_encoder_two.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

    @property
    def device(self) -> torch.device:
        """Get device information.

        Returns
        -------
            torch.device: device.
        """
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
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
        if self.pre_compute_text_embeddings:
            pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                unet=self.unet,
                safety_checker=None,
                torch_dtype=(torch.float16
                             if self.device != torch.device("cpu") else
                             torch.float32),
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                text_encoder=self.text_encoder_one,
                text_encoder_2=self.text_encoder_two,
                tokenizer=self.tokenizer_one,
                tokenizer_2=self.tokenizer_two,
                unet=self.unet,
                torch_dtype=(torch.float16
                             if self.device != torch.device("cpu") else
                             torch.float32),
            )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            pipeline.scheduler.register_to_config(
                prediction_type=self.prediction_type)
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p in prompt:
            image = pipeline(
                p,
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

    def encode_prompt(
        self,
        text_one: torch.Tensor,
        text_two: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt.

        Args:
        ----
            text_one (torch.Tensor): Token ids from tokenizer one.
            text_two (torch.Tensor): Token ids from tokenizer two.

        Returns:
        -------
            tuple[torch.Tensor, torch.Tensor]: Prompt embeddings
        """
        prompt_embeds_list = []

        text_encoders = [self.text_encoder_one, self.text_encoder_two]
        texts = [text_one, text_two]
        for text_encoder, text in zip(text_encoders, texts, strict=True):

            prompt_embeds = text_encoder(
                text,
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the
            # final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def val_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Val step."""
        msg = "val_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def test_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Test step."""
        msg = "test_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def loss(self,
             model_pred: torch.Tensor,
             noise: torch.Tensor,
             latents: torch.Tensor,
             timesteps: torch.Tensor,
             weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)

        loss_dict = {}
        # calculate loss in FP32
        if self.loss_module.use_snr:
            loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
        else:
            loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
        loss_dict["loss"] = loss
        return loss_dict

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

        noise = torch.randn_like(latents)

        if self.enable_noise_offset:
            noise = noise + self.noise_offset_weight * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=noise.device)

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

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
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        return self.loss(model_pred, noise, latents, timesteps, weight)
