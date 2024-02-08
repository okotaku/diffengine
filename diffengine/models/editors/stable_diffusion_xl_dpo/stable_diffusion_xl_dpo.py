from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusionXLDPO(StableDiffusionXL):
    """Stable Diffusion XL DPO.

    Args:
    ----
        beta_dpo (int): DPO KL Divergence penalty. Defaults to 5000.
        loss (dict, optional): The loss config. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDXLDPODataPreprocessor`.
    """

    def __init__(self,
                 *args,
                 beta_dpo: int = 5000,
                 loss: dict | None = None,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0,
                    "reduction": "none"}
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLDPODataPreprocessor"}

        super().__init__(
            *args,
            loss=loss,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

        self.beta_dpo = beta_dpo

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.orig_unet = deepcopy(
            self.unet).requires_grad_(requires_grad=False)

        super().prepare_model()

    def loss(  # type: ignore[override]
        self,
             model_pred: torch.Tensor,
             ref_pred: torch.Tensor,
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
            model_loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
            ref_loss = self.loss_module(
                ref_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
        else:
            model_loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
            ref_loss = self.loss_module(
                ref_pred.float(), gt.float(), weight=weight)
            model_loss = model_loss.mean(
                dim=list(range(1, len(model_loss.shape))))
            ref_loss = ref_loss.mean(
                dim=list(range(1, len(ref_loss.shape))))
        model_losses_w, model_losses_l = model_loss.chunk(2)
        model_diff = model_losses_w - model_losses_l

        ref_losses_w, ref_losses_l = ref_loss.chunk(2)
        ref_diff = ref_losses_w - ref_losses_l
        scale_term = -0.5 * self.beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        loss = -1 * F.logsigmoid(inside_term.mean())

        loss_dict["loss"] = loss
        return loss_dict

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
        assert "result_class_image" not in inputs, (
            "result_class_image is not supported for SDXLDPO")
        # num_batches is divided by 2 because we have two images per sample
        num_batches = len(inputs["img"]) // 2

        latents = self._forward_vae(inputs["img"], num_batches)

        noise = self.noise_generator(latents[:num_batches])
        # repeat noise for each sample set
        noise = noise.repeat(2, 1, 1, 1)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)
        # repeat timesteps for each sample set
        timesteps = timesteps.repeat(2)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

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
        # repeat text embeds for each sample set
        prompt_embeds = prompt_embeds.repeat(2, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1)
        unet_added_conditions = {
            "time_ids": inputs["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample
        with torch.no_grad():
            ref_pred = self.orig_unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
            ).sample

        return self.loss(model_pred, ref_pred, noise, latents, timesteps)
