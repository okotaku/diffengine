# flake8: noqa: PLR0915, PLR0912, C901
import gc
from copy import deepcopy
from typing import Optional

import torch

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class DistillSDXL(StableDiffusionXL):
    """Distill Stable Diffusion XL.

    Args:
    ----
        model_type (str): The type of model to use. Choice from `sd_tiny`,
            `sd_small`.
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
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. This should be `False` when training ControlNet.
            Defaults to False.
    """

    def __init__(self,
                 *args,
                 model_type: str,
                 unet_lora_config: dict | None = None,
                 text_encoder_lora_config: dict | None = None,
                 finetune_text_encoder: bool = False,
                 **kwargs) -> None:
        assert unet_lora_config is None, \
            "`unet_lora_config` should be None when training DistillSDXL"
        assert text_encoder_lora_config is None, \
            "`text_encoder_lora_config` should be None when training DistillSDXL"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training DistillSDXL"
        assert model_type in ["sd_tiny", "sd_small"], \
            f"`model_type`={model_type} should not be supported in DistillSDXL"

        self.model_type = model_type

        super().__init__(
            *args,
            unet_lora_config=unet_lora_config,
            text_encoder_lora_config=text_encoder_lora_config,
            finetune_text_encoder=finetune_text_encoder,
            **kwargs)  # type: ignore[misc]

    def set_lora(self) -> None:
        """Set LORA for model."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.orig_unet = deepcopy(
            self.unet).requires_grad_(requires_grad=False)

        # prepare student model
        self._prepare_student()
        super().prepare_model()
        self._cast_hook()

    def _prepare_student(self) -> None:
        assert len(self.unet.up_blocks) == len(self.unet.down_blocks)
        self.num_blocks = len(self.unet.up_blocks)
        config = self.unet._internal_dict  # noqa
        config["layers_per_block"] = 1
        self.unet._internal_dict.layers_per_block = 1  # noqa
        if self.model_type == "sd_tiny":
            self.unet.mid_block = None
            config["mid_block_type"] = None

        # Commence deletion of resnets/attentions inside the U-net
        # Handle Down Blocks
        for i in range(self.num_blocks):
            delattr(self.unet.down_blocks[i].resnets, "1")
            if hasattr(self.unet.down_blocks[i], "attentions"):
                # i == 0 does not have attentions
                delattr(self.unet.down_blocks[i].attentions, "1")

        for i in range(self.num_blocks):
            self.unet.up_blocks[i].resnets[1] = self.unet.up_blocks[i].resnets[
                2]
            delattr(self.unet.up_blocks[i].resnets, "2")
            if hasattr(self.unet.up_blocks[i], "attentions"):
                self.unet.up_blocks[i].attentions[1] = self.unet.up_blocks[
                    i].attentions[2]
                delattr(self.unet.up_blocks[i].attentions, "2")

        torch.cuda.empty_cache()
        gc.collect()

    def _cast_hook(self) -> None:
        self.teacher_feats: dict = {}
        self.student_feats: dict = {}

        def get_activation(activation, name, residuals_present):  # noqa
            # the hook signature
            if residuals_present:

                def hook(model, input, output):  # noqa
                    activation[name] = output[0]
            else:

                def hook(model, input, output):  # noqa
                    activation[name] = output

            return hook

        # cast teacher
        for i in range(self.num_blocks):
            self.orig_unet.down_blocks[i].register_forward_hook(
                get_activation(
                    self.teacher_feats, "d" + str(i), residuals_present=True))
        self.orig_unet.mid_block.register_forward_hook(
            get_activation(self.teacher_feats, "m", residuals_present=False))
        for i in range(self.num_blocks):
            self.orig_unet.up_blocks[i].register_forward_hook(
                get_activation(
                    self.teacher_feats, "u" + str(i), residuals_present=False))

        # cast student
        for i in range(self.num_blocks):
            self.unet.down_blocks[i].register_forward_hook(
                get_activation(
                    self.student_feats, "d" + str(i), residuals_present=True))
        if self.model_type == "sd_small":
            self.unet.mid_block.register_forward_hook(
                get_activation(
                    self.student_feats, "m", residuals_present=False))
        for i in range(self.num_blocks):
            self.unet.up_blocks[i].register_forward_hook(
                get_activation(
                    self.student_feats, "u" + str(i), residuals_present=False))

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                self.orig_unet.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

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
            num_blocks = (
                self.num_blocks
                if self.model_type == "sd_small" else self.num_blocks - 1)
            for i in range(num_blocks):
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["d" + str(i)].float(),
                    self.student_feats["d" + str(i)].float(),
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)
            if self.model_type == "sd_small":
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["m"].float(),
                    self.student_feats["m"].float(),
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)
            elif self.model_type == "sd_tiny":
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["m"].float(),
                    self.student_feats[f"d{self.num_blocks - 1}"].float(),
                    timesteps,
                    self.scheduler.alphas_cumprod,
                    weight=weight)
            for i in range(self.num_blocks):
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["u" + str(i)].float(),
                    self.student_feats["u" + str(i)].float(),
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
            num_blocks = (
                self.num_blocks
                if self.model_type == "sd_small" else self.num_blocks - 1)
            for i in range(num_blocks):
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["d" + str(i)].float(),
                    self.student_feats["d" + str(i)].float(),
                    weight=weight)
            if self.model_type == "sd_small":
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["m"].float(),
                    self.student_feats["m"].float(),
                    weight=weight)
            elif self.model_type == "sd_tiny":
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["m"].float(),
                    self.student_feats[f"d{self.num_blocks - 1}"].float(),
                    weight=weight)
            for i in range(self.num_blocks):
                loss_features = loss_features + self.loss_module(
                    self.teacher_feats["u" + str(i)].float(),
                    self.student_feats["u" + str(i)].float(),
                    weight=weight)

            loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
            loss_kd = self.loss_module(
                model_pred.float(), teacher_pred.float(), weight=weight)
        loss_dict["loss_sd"] = loss
        loss_dict["loss_kd"] = loss_kd
        loss_dict["loss_features"] = loss_features
        return loss_dict
