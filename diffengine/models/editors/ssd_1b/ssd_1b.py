# flake8: noqa: PLR0915, PLR0912, C901
from typing import Optional

import torch
from torch import nn

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class SSD1B(StableDiffusionXL):
    """SSD1B.

    Refer to official implementation:
    https://github.com/segmind/SSD-1B/blob/main/distill_sdxl.py

    Args:
    ----
        tokenizer_one (dict): Config of tokenizer one.
        tokenizer_two (dict): Config of tokenizer two.
        scheduler (dict): Config of scheduler.
        text_encoder_one (dict): Config of text encoder one.
        text_encoder_two (dict): Config of text encoder two.
        vae (dict): Config of vae.
        teacher_unet (dict): Config of teacher unet.
        student_unet (dict): Config of student unet.
        model (str): pretrained model name of stable diffusion xl.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
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
            Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDXLDataPreprocessor`.
        noise_generator (dict, optional): The noise generator config.
            Defaults to ``dict(type='WhiteNoise')``.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='TimeSteps')``.
        input_perturbation_gamma (float): The gamma of input perturbation.
            The recommended value is 0.1 for Input Perturbation.
            Defaults to 0.0.
        vae_batch_size (int): The batch size of vae. Defaults to 8.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
        pre_compute_text_embeddings(bool): Whether or not to pre-compute text
            embeddings to save memory. Defaults to False.
        enable_xformers (bool): Whether or not to enable memory efficient
            attention. Defaults to False.
        student_weight_from_teacher (bool): Whether or not to initialize
            student model with teacher model. Defaults to False.
    """

    def __init__(
        self,
        tokenizer_one: dict,
        tokenizer_two: dict,
        scheduler: dict,
        text_encoder_one: dict,
        text_encoder_two: dict,
        vae: dict,
        teacher_unet: dict,
        student_unet: dict,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        loss: dict | None = None,
        unet_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        vae_batch_size: int = 8,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        pre_compute_text_embeddings: bool = False,
        enable_xformers: bool = False,
        student_weight_from_teacher: bool = False,
    ) -> None:
        assert unet_lora_config is None, \
            "`unet_lora_config` should be None when training SSD1B"
        assert text_encoder_lora_config is None, \
            "`text_encoder_lora_config` should be None when training SSD1B"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training SSD1B"

        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {}
        if timesteps_generator is None:
            timesteps_generator = {}
        if loss is None:
            loss = {}
        super(StableDiffusionXL, self).__init__(data_preprocessor=data_preprocessor)
        self.model = model
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers
        self.vae_batch_size = vae_batch_size

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(
                loss,
                default_args={"type": "L2Loss", "loss_weight": 1.0})
        self.loss_module: nn.Module = loss

        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        if not self.pre_compute_text_embeddings:
            self.tokenizer_one = MODELS.build(
                tokenizer_one,
                default_args={"pretrained_model_name_or_path": model})
            self.tokenizer_two = MODELS.build(
                tokenizer_two,
                default_args={"pretrained_model_name_or_path": model})

            self.text_encoder_one = MODELS.build(
                text_encoder_one,
                default_args={"pretrained_model_name_or_path": model})
            self.text_encoder_two = MODELS.build(
                text_encoder_two,
                default_args={"pretrained_model_name_or_path": model})

        self.scheduler = MODELS.build(
            scheduler,
            default_args={"pretrained_model_name_or_path": model})

        self.vae = MODELS.build(
            vae,
            default_args={"pretrained_model_name_or_path": model})
        self.orig_unet = MODELS.build(
            teacher_unet,
            default_args={"pretrained_model_name_or_path": model})
        self.unet = MODELS.build(student_unet)

        # prepare student model
        if student_weight_from_teacher:
            self.unet.load_state_dict(self.orig_unet.state_dict(),
                                      strict=False)
        self.noise_generator = MODELS.build(
            noise_generator,
            default_args={"type": "WhiteNoise"})
        self.timesteps_generator = MODELS.build(
            timesteps_generator,
            default_args={"type": "TimeSteps"})
        self.prepare_model()
        self.set_lora()
        self.set_xformers()

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

        def get_activation(activation, name, residuals_present):  # noqa
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

    def _forward_vae(self, img: torch.Tensor, num_batches: int,
                     ) -> torch.Tensor:
        """Forward vae."""
        latents = [
            self.vae.encode(
                img[i : i + self.vae_batch_size],
            ).latent_dist.sample() for i in range(
                0, num_batches, self.vae_batch_size)
        ]
        latents = torch.cat(latents, dim=0)
        return latents * self.vae.config.scaling_factor

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

        latents = self._forward_vae(inputs["img"], num_batches)

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
