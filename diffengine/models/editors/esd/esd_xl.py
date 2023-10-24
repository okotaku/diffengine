from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class ESDXL(StableDiffusionXL):
    """Stable Diffusion XL Erasing Concepts from Diffusion Models.

    Args:
    ----
        height (int): Image height. Defaults to 1024.
        width (int): Image width. Defaults to 1024.
        negative_guidance (float): Negative guidance for loss. Defaults to 1.0.
        train_method (str): Training method. Choice from `full`, `xattn`,
            `noxattn`, `selfattn`. Defaults to `full`
    """

    def __init__(self,
                 *args,
                 finetune_text_encoder: bool = False,
                 pre_compute_text_embeddings: bool = True,
                 height: int = 1024,
                 width: int = 1024,
                 negative_guidance: float = 1.0,
                 train_method: str = "full",
                 prediction_type: str | None = None,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "ESDXLDataPreprocessor"}
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training ESDXL"
        assert pre_compute_text_embeddings, \
            "`pre_compute_text_embeddings` should be True when training ESDXL"
        assert train_method in ["full", "xattn", "noxattn", "selfattn"]
        assert prediction_type is None, \
            "`prediction_type` should be None when training ESDXL"

        self.height = height
        self.width = width
        self.negative_guidance = negative_guidance
        self.train_method = train_method

        super().__init__(
            *args,
            finetune_text_encoder=finetune_text_encoder,
            pre_compute_text_embeddings=pre_compute_text_embeddings,
            prediction_type=prediction_type,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.lora_config is None:
            self.orig_unet = deepcopy(
                self.unet).requires_grad_(requires_grad=False)
        super().prepare_model()
        self._freeze_unet()

    def _freeze_unet(self) -> None:
        for name, module in self.unet.named_modules():
            if self.train_method == "xattn" and "attn2" not in name:  # noqa
                module.eval()
            elif self.train_method == "selfattn" and "attn1" not in name:  # noqa
                module.eval()
            elif self.train_method == "noxattn" and ("attn2" in name
                                                     or "time_embed" in name or
                                                     name.startswith("out.")):
                module.eval()

    def train(self, *, mode=True) -> None:
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_unet()

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
        timesteps = torch.randint(1, 49, (1, ), device=self.device)
        timesteps = timesteps.long()

        latents = self.infer(
            prompt=inputs["text"],
            height=self.height,
            width=self.width,
            num_inference_steps=50,
            denoising_end=timesteps[0].item() / 50,
            output_type="latent",
            guidance_scale=3)[0].unsqueeze(0)
        # train mode after inference
        self.train()

        timesteps = torch.randint(
            round(timesteps[0].item() / 50 *
                  self.scheduler.config.num_train_timesteps),
            round((timesteps[0].item() + 1) / 50 *
                  self.scheduler.config.num_train_timesteps), (1, ),
            device=self.device).long()

        prompt_embeds = inputs["prompt_embeds"]
        pooled_prompt_embeds = inputs["pooled_prompt_embeds"]
        null_prompt_embeds = inputs["null_prompt_embeds"]
        null_pooled_prompt_embeds = inputs["null_pooled_prompt_embeds"]
        time_ids = torch.Tensor(
            [[self.height, self.width, 0, 0, self.height,
              self.width]]).long().to(self.device)
        unet_added_conditions = {
            "time_ids": time_ids,
            "text_embeds": pooled_prompt_embeds,
        }
        null_unet_added_conditions = {
            "time_ids": time_ids,
            "text_embeds": null_pooled_prompt_embeds,
        }

        with torch.no_grad():
            if self.lora_config is None:
                null_model_pred = self.orig_unet(
                    latents,
                    timesteps,
                    null_prompt_embeds,
                    added_cond_kwargs=null_unet_added_conditions).sample
                orig_model_pred = self.orig_unet(
                    latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions).sample
            else:
                # scale=0 means not using the lora model.
                null_model_pred = self.unet(
                    latents,
                    timesteps,
                    null_prompt_embeds,
                    added_cond_kwargs=null_unet_added_conditions,
                    cross_attention_kwargs={
                        "scale": 0,
                    }).sample
                orig_model_pred = self.unet(
                    latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    cross_attention_kwargs={
                        "scale": 0,
                    }).sample

        model_pred = self.unet(
            latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        loss_dict = {}
        # calculate loss in FP32
        null_model_pred.requires_grad = False
        orig_model_pred.requires_grad = False
        gt = null_model_pred - self.negative_guidance * (
            orig_model_pred - null_model_pred)
        if self.loss_module.use_snr:
            loss = self.loss_module(model_pred.float(), gt.float(), timesteps,
                                    self.scheduler.alphas_cumprod,
                                    self.scheduler.config.prediction_type)
        else:
            loss = self.loss_module(model_pred.float(), gt.float())
        loss_dict["loss"] = loss
        return loss_dict
