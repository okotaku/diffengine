import inspect
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import PixArtAlphaPipeline
from mmengine import print_log
from mmengine.model import BaseModel
from peft import get_peft_model
from torch import nn

from diffengine.models.archs import create_peft_config
from diffengine.registry import MODELS


@MODELS.register_module()
class PixArtAlpha(BaseModel):
    """PixArt Alpha.

    Args:
    ----
        tokenizer (dict): Config of tokenizer.
        scheduler (dict): Config of scheduler.
        text_encoder (dict): Config of text encoder.
        vae (dict): Config of vae.
        transformer (dict): Config of transformer.
        model (str): pretrained model name of stable diffusion.
            Defaults to 'PixArt-alpha/PixArt-XL-2-1024-MS'.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        transformer_lora_config (dict, optional): The LoRA config dict for
            Transformer. example. dict(type="LoRA", r=4). `type` is chosen from
            `LoRA`, `LoHa`, `LoKr`. Other config are same as the config of
            PEFT. https://github.com/huggingface/peft
            Defaults to None.
        text_encoder_lora_config (dict, optional): The LoRA config dict for
            Text Encoder. example. dict(type="LoRA", r=4). `type` is chosen
            from `LoRA`, `LoHa`, `LoKr`. Other config are same as the config of
            PEFT. https://github.com/huggingface/peft
            Defaults to None.
        prior_loss_weight (float): The weight of prior preservation loss.
            It works when training dreambooth with class images.
        tokenizer_max_length (int): The max length of tokenizer.
            Defaults to 120.
        prediction_type (str): The prediction_type that shall be used for
            training. Choose between 'epsilon' or 'v_prediction' or leave
            `None`. If left to `None` the default prediction type of the
            scheduler will be used. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`PixArtAlphaDataPreprocessor`.
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
        enable_xformers (bool): Whether or not to enable memory efficient
            attention. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: dict,
        scheduler: dict,
        text_encoder: dict,
        vae: dict,
        transformer: dict,
        model: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
        loss: dict | None = None,
        transformer_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        tokenizer_max_length: int = 120,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        vae_batch_size: int = 8,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "PixArtAlphaDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {}
        if timesteps_generator is None:
            timesteps_generator = {}
        if loss is None:
            loss = {}
        super().__init__(data_preprocessor=data_preprocessor)
        if (
            transformer_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Transformer and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and transformer_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Transformer, "
                "you should set text_encoder_lora_config."
            )

        self.model = model
        self.transformer_lora_config = deepcopy(transformer_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.tokenizer_max_length = tokenizer_max_length
        self.gradient_checkpointing = gradient_checkpointing
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

        self.tokenizer = MODELS.build(
            tokenizer,
            default_args={
                "pretrained_model_name_or_path": model,
                } if not inspect.isclass(tokenizer.get("type")) else None)
        self.scheduler = MODELS.build(
            scheduler,
            default_args={
                "pretrained_model_name_or_path": model,
                } if not inspect.isclass(scheduler.get("type")) else None)

        self.text_encoder = MODELS.build(
            text_encoder,
            default_args={
                "pretrained_model_name_or_path": model,
                } if not inspect.isclass(text_encoder.get("type")) else None)
        self.vae = MODELS.build(
            vae,
            default_args={
                "pretrained_model_name_or_path": model,
                } if not inspect.isclass(vae.get("type")) else None)
        self.transformer = MODELS.build(
            transformer,
            default_args={
                "pretrained_model_name_or_path": model,
                } if not inspect.isclass(transformer.get("type")) else None)
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
        if self.text_encoder_lora_config is not None:
            text_encoder_lora_config = create_peft_config(
                self.text_encoder_lora_config)
            self.text_encoder = get_peft_model(
                self.text_encoder, text_encoder_lora_config)
            self.text_encoder.print_trainable_parameters()
        if self.transformer_lora_config is not None:
            transformer_lora_config = create_peft_config(self.transformer_lora_config)
            self.transformer = get_peft_model(self.transformer, transformer_lora_config)
            self.transformer.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        if not self.finetune_text_encoder:
            self.text_encoder.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.transformer.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

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
        pipeline = PixArtAlphaPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            torch_dtype=torch.float32,
        )
        if self.finetune_text_encoder:
            # TODO(takuoko): When parsing text_encoder directly, the  # noqa
            # results are different. So we need to parse here.
            pipeline.text_encoder = self.text_encoder
        pipeline.to(self.device)
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
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

    def _preprocess_model_input(self,
                                latents: torch.Tensor,
                                noise: torch.Tensor,
                                timesteps: torch.Tensor) -> torch.Tensor:
        """Preprocess model input."""
        if self.input_perturbation_gamma > 0:
            input_noise = noise + self.input_perturbation_gamma * torch.randn_like(
                noise)
        else:
            input_noise = noise
        return self.scheduler.add_noise(latents, input_noise, timesteps)

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
        text_inputs = self.tokenizer(
            inputs["text"],
            max_length=self.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        inputs["text"] = text_inputs.input_ids.to(self.device)
        inputs["attention_mask"] = text_inputs.attention_mask.to(self.device)
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

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(
            inputs["text"], attention_mask=inputs["attention_mask"])[0]

        if self.transformer.config.sample_size == 128:  # noqa
            added_cond_kwargs = {"resolution": inputs["resolution"],
                                 "aspect_ratio": inputs["aspect_ratio"]}
        else:
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        model_pred = self.transformer(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=inputs["attention_mask"],
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs).sample

        latent_channels = self.transformer.config.in_channels
        if self.transformer.config.out_channels // 2 == latent_channels:
            model_pred = model_pred.chunk(2, dim=1)[0]

        return self.loss(model_pred, noise, latents, timesteps, weight)
