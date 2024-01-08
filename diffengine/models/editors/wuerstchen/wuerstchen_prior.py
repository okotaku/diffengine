from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from huggingface_hub import hf_hub_download
from mmengine import print_log
from mmengine.model import BaseModel
from peft import get_peft_model
from torch import nn

from diffengine.models.archs import create_peft_config
from diffengine.registry import MODELS


@MODELS.register_module()
class WuerstchenPriorModel(BaseModel):
    """`Wuerstchen Prior.

    <https://arxiv.org/abs/2306.00637>`_

    Args:
    ----
        tokenizer (dict): Config of tokenizer.
        scheduler (dict): Config of scheduler.
        text_encoder (dict): Config of text encoder.
        image_encoder (dict): Config of image encoder.
        prior (dict): Config of prior.
        decoder_model (str): pretrained decoder model name of Wuerstchen.
            Defaults to 'warp-ai/wuerstchen'.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        prior_lora_config (dict, optional): The LoRA config dict for Prior.
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
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDDataPreprocessor`.
        noise_generator (dict, optional): The noise generator config.
            Defaults to ``dict(type='WhiteNoise')``.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='WuerstchenRandomTimeSteps')``.
        input_perturbation_gamma (float): The gamma of input perturbation.
            The recommended value is 0.1 for Input Perturbation.
            Defaults to 0.0.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: dict,
        scheduler: dict,
        text_encoder: dict,
        image_encoder: dict,
        prior: dict,
        decoder_model: str = "warp-ai/wuerstchen",
        loss: dict | None = None,
        prior_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "WuerstchenRandomTimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super().__init__(data_preprocessor=data_preprocessor)
        if (
            prior_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Prior and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and prior_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Prior, "
                "you should set text_encoder_lora_config."
            )

        self.decoder_model = decoder_model
        self.prior_lora_config = deepcopy(prior_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss
        assert not self.loss_module.use_snr, \
            "WuerstchenPriorModel does not support SNR loss."

        self.tokenizer = MODELS.build(tokenizer)
        self.text_encoder = MODELS.build(text_encoder)

        pretrained_image_encoder = image_encoder.pop("pretrained_image_encoder", False)
        self.image_encoder = MODELS.build(image_encoder)
        if pretrained_image_encoder:
            pretrained_checkpoint_file = hf_hub_download(
                "dome272/wuerstchen", filename="model_v2_stage_b.pt")
            state_dict = torch.load(pretrained_checkpoint_file, map_location="cpu")
            self.image_encoder.load_state_dict(state_dict["effnet_state_dict"])

        self.scheduler = MODELS.build(scheduler)

        self.prior = MODELS.build(prior)
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)
        self.prepare_model()
        self.set_lora()

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.text_encoder_lora_config is not None:
            text_encoder_lora_config = create_peft_config(
                self.text_encoder_lora_config)
            self.text_encoder = get_peft_model(
                self.text_encoder, text_encoder_lora_config)
            self.text_encoder.print_trainable_parameters()

        if self.prior_lora_config is not None:
            prior_lora_config = create_peft_config(self.prior_lora_config)
            self.prior = get_peft_model(self.prior, prior_lora_config)
            self.prior.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.prior.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        self.image_encoder.requires_grad_(requires_grad=False)
        print_log("Set Image Encoder untrainable.", "current")
        if not self.finetune_text_encoder:
            self.text_encoder.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

    @property
    def device(self) -> torch.device:
        """Get device information.

        Returns
        -------
            torch.device: device.
        """
        return next(self.parameters()).device

    def train(self, *, mode: bool = True) -> None:
        """Convert the model into training mode."""
        super().train(mode)
        self.image_encoder.eval()
        if not self.finetune_text_encoder:
            self.text_encoder.eval()

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
        pipeline = AutoPipelineForText2Image.from_pretrained(
            self.decoder_model,
            prior_prior=self.prior,
            prior_text_encoder=self.text_encoder,
            prior_tokenizer=self.tokenizer,
            torch_dtype=torch.float32,
        )
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p in prompt:
            image = pipeline(
                p,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
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
             timesteps: torch.Tensor,
             weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        loss_dict = {}
        # calculate loss in FP32
        if self.loss_module.use_snr:
            loss = self.loss_module(
                model_pred.float(),
                noise.float(),
                timesteps,
                self.scheduler._alpha_cumprod(timesteps, self.device),  # noqa
                "epsilon",
                weight=weight)
        else:
            loss = self.loss_module(
                model_pred.float(), noise.float(), weight=weight)
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

        image_embeds = self.image_encoder(inputs["img"])
        # scale
        image_embeds = image_embeds.add(1.0).div(42.0)

        noise = self.noise_generator(image_embeds)

        timesteps = self.timesteps_generator(num_batches,
                                            self.device)

        noisy_latents = self._preprocess_model_input(image_embeds, noise, timesteps)

        inputs_text = self.tokenizer(
            inputs["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        prompt_embeds = self.text_encoder(
            inputs_text.input_ids.to(self.device),
            attention_mask=inputs_text.attention_mask.to(self.device),
        ).last_hidden_state

        model_pred = self.prior(
            noisy_latents,
            timesteps,
            prompt_embeds)

        return self.loss(model_pred, noise, timesteps, weight)
