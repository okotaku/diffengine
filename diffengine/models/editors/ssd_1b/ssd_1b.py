# flake8: noqa: PLR0915, PLR0912, C901
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from torch import nn
from transformers import AutoTokenizer

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.models.editors.stable_diffusion_xl.stable_diffusion_xl import (
    import_model_class_from_model_name_or_path,
)
from diffengine.registry import MODELS


@MODELS.register_module()
class SSD1B(StableDiffusionXL):
    """SSD1B.

    Refer to official implementation:
    https://github.com/segmind/SSD-1B/blob/main/distill_sdxl.py

    Args:
    ----
        model (str): pretrained model name of stable diffusion xl.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        student_model (str): pretrained model name of student model.
            Defaults to 'segmind/SSD-1B'.
        student_model_weight (str): pretrained model weight of student model.
            Choose between 'orig_unet' or 'unet'. 'orig_unet' load_state_dict
            from teacher model. 'unet' load_state_dict from student model.
            Defaults to 'orig_unet'.
        vae_model (str, optional): Path to pretrained VAE model with better
            numerical stability. More details:
            https://github.com/huggingface/diffusers/pull/4038.
            Defaults to None.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        unet_lora_config (dict, optional): The LoRA config dict for Unet.
            example. dict(type="LoRA", r=4). `type` is chosen from `LoRA`,
            `LoHa`, `LoKr`. Other config are same as the config of PEFT.
            https://github.com/huggingface/peft
            Defaults to None.
        text_encoder_lora_config (dict, optional): The LoRA config dict for
            Text Encoder. example. dict(type="LoRA", r=4). `type` is chosen
            from `LoRA`, `LoHa`, `LoKr`. Other config are same as the config of
            PEFT. https://github.com/huggingface/peft
            Defaults to None.
        prior_loss_weight (float): The weight of prior preservation loss.
            It works when training dreambooth with class images.
        prediction_type (str): The prediction_type that shall be used for
            training. Choose between 'epsilon' or 'v_prediction' or leave
            `None`. If left to `None` the default prediction type of the
            scheduler: `noise_scheduler.config.prediciton_type` is chosen.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDXLDataPreprocessor`.
        noise_generator (dict, optional): The noise generator config.
            Defaults to ``dict(type='WhiteNoise')``.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='TimeSteps')``.
        input_perturbation_gamma (float): The gamma of input perturbation.
            The recommended value is 0.1 for Input Perturbation.
            Defaults to 0.0.
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
        student_model: str = "segmind/SSD-1B",
        student_model_weight: str = "orig_unet",
        vae_model: str | None = None,
        loss: dict | None = None,
        unet_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        pre_compute_text_embeddings: bool = False,
    ) -> None:
        assert unet_lora_config is None, \
            "`unet_lora_config` should be None when training SSD1B"
        assert text_encoder_lora_config is None, \
            "`text_encoder_lora_config` should be None when training SSD1B"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training SSD1B"
        assert student_model_weight in ["orig_unet", "unet"], \
            "`student_model_weight` should be 'orig_unet' or 'unet'"

        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "TimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super(StableDiffusionXL, self).__init__(data_preprocessor=data_preprocessor)
        self.model = model
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        self.input_perturbation_gamma = input_perturbation_gamma

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

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
        self.orig_unet = UNet2DConditionModel.from_pretrained(
            model, subfolder="unet")

        # prepare student model
        if student_model_weight == "orig_unet":
            self.unet = UNet2DConditionModel.from_config(
                student_model, subfolder="unet")
            self.unet.load_state_dict(self.orig_unet.state_dict(),
                                      strict=False)
        elif student_model_weight == "unet":
            self.unet = UNet2DConditionModel.from_pretrained(
                student_model, subfolder="unet")
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)
        self.prepare_model()
        self.set_lora()

    def set_lora(self) -> None:
        """Set LORA for model."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.orig_unet.requires_grad_(requires_grad=False)
        super().prepare_model()
        self._cast_hook()

    def _cast_hook(self) -> None:
        self.teacher_feats: dict = {}
        self.student_feats: dict = {}

        def get_activation(activation, name, residuals_present):
            # the hook signature
            if residuals_present:

                def hook(model, input, output):  # noqa
                    activation[name] = output[0]
            else:

                def hook(model, input, output):  # noqa
                    activation[name] = output

            return hook

        # cast down blocks
        num_blocks = len(self.orig_unet.down_blocks)
        for nb in range(num_blocks):
            for i in range(2):
                self.orig_unet.down_blocks[nb].resnets[i].register_forward_hook(
                    get_activation(self.teacher_feats,f"d{nb}r{i}",
                                   residuals_present=False))
                self.unet.down_blocks[nb].resnets[i].register_forward_hook(
                    get_activation(self.student_feats,f"d{nb}r{i}",
                                   residuals_present=False))
                if hasattr(self.orig_unet.down_blocks[nb], "attentions"):
                    self.orig_unet.down_blocks[nb].attentions[i].register_forward_hook(
                        get_activation(self.teacher_feats,f"d{nb}a{i}",
                                       residuals_present=True))
                    self.unet.down_blocks[nb].attentions[i].register_forward_hook(
                        get_activation(self.student_feats,f"d{nb}a{i}",
                                       residuals_present=True))

        # cast mid blocks
        self.orig_unet.mid_block.resnets[0].register_forward_hook(
            get_activation(self.teacher_feats,f"mr{0}",
                           residuals_present=False))
        self.orig_unet.mid_block.resnets[1].register_forward_hook(
            get_activation(self.teacher_feats,f"mr{1}",
                           residuals_present=False))
        self.orig_unet.mid_block.attentions[0].register_forward_hook(
            get_activation(self.teacher_feats,f"ma{0}",
                           residuals_present=False))
        self.unet.mid_block.resnets[0].register_forward_hook(
            get_activation(self.student_feats,f"mr{0}",
                           residuals_present=False))

        # cast up blocks
        for nb in range(num_blocks):
            for i in range(3):
                self.orig_unet.up_blocks[nb].resnets[i].register_forward_hook(
                    get_activation(self.teacher_feats,f"u{nb}r{i}",
                                   residuals_present=False))
                self.unet.up_blocks[nb].resnets[i].register_forward_hook(
                    get_activation(self.student_feats,f"u{nb}r{i}",
                                   residuals_present=False))
                if hasattr(self.orig_unet.up_blocks[nb], "attentions"):
                    self.orig_unet.up_blocks[nb].attentions[i].register_forward_hook(
                        get_activation(self.teacher_feats,f"u{nb}a{i}",
                                       residuals_present=True))
                    self.unet.up_blocks[nb].attentions[i].register_forward_hook(
                        get_activation(self.student_feats,f"u{nb}a{i}",
                                       residuals_present=True))

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

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

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
        unet_added_conditions = {
            "time_ids": inputs["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

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

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        with torch.no_grad():
            teacher_pred = self.orig_unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions).sample

        loss_dict = {}
        # calculate loss in FP32
        if self.loss_module.use_snr:
            loss_features = 0
            num_blocks = len(self.orig_unet.down_blocks)
            for i in range(num_blocks):
                for j in range(2):
                    loss_features = loss_features + self.loss_module(
                        self.teacher_feats[f"d{i}r{j}"],
                        self.student_feats[f"d{i}r{j}"],
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)
                    if f"d{i}a{j}" in self.teacher_feats:
                        loss_features = loss_features + self.loss_module(
                            self.teacher_feats[f"d{i}a{j}"],
                            self.student_feats[f"d{i}a{j}"],
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)
            loss_features=loss_features + self.loss_module(
                self.teacher_feats["mr1"],self.student_feats["mr0"],
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)
            for i in range(num_blocks):
                for j in range(3):
                    loss_features += loss_features + self.loss_module(
                        self.teacher_feats[f"u{i}r{j}"],
                        self.student_feats[f"u{i}r{j}"],
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)
                    if f"u{i}a{j}" in self.teacher_feats:
                        loss_features = loss_features + self.loss_module(
                            self.teacher_feats[f"u{i}a{j}"],
                            self.student_feats[f"u{i}a{j}"],
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)

            loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                weight=weight)
            loss_kd = self.loss_module(
                model_pred.float(),
                teacher_pred.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                weight=weight)
        else:
            loss_features = 0
            num_blocks = len(self.orig_unet.down_blocks)
            for i in range(num_blocks):
                for j in range(2):
                    loss_features = loss_features + self.loss_module(
                        self.teacher_feats[f"d{i}r{j}"],
                        self.student_feats[f"d{i}r{j}"],
                    weight=weight)
                    if f"d{i}a{j}" in self.teacher_feats:
                        loss_features = loss_features + self.loss_module(
                            self.teacher_feats[f"d{i}a{j}"],
                            self.student_feats[f"d{i}a{j}"],
                    weight=weight)
            loss_features=loss_features + self.loss_module(
                self.teacher_feats["mr1"],self.student_feats["mr0"],
                    weight=weight)
            for i in range(num_blocks):
                for j in range(3):
                    loss_features += loss_features + self.loss_module(
                        self.teacher_feats[f"u{i}r{j}"],
                        self.student_feats[f"u{i}r{j}"],
                    weight=weight)
                    if f"u{i}a{j}" in self.teacher_feats:
                        loss_features = loss_features + self.loss_module(
                            self.teacher_feats[f"u{i}a{j}"],
                            self.student_feats[f"u{i}a{j}"],
                    weight=weight)

            loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
            loss_kd = self.loss_module(
                model_pred.float(), teacher_pred.float(), weight=weight)
        loss_dict["loss_sd"] = loss
        loss_dict["loss_kd"] = loss_kd * 0.5
        loss_dict["loss_features"] = loss_features * 0.5
        return loss_dict
