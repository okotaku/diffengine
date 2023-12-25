from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoPipelineForText2Image,
    DDPMScheduler,
    PriorTransformer,
)
from mmengine import print_log
from mmengine.model import BaseModel
from peft import get_peft_model
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffengine.models.archs import create_peft_config
from diffengine.registry import MODELS


@MODELS.register_module()
class KandinskyV22Prior(BaseModel):
    """KandinskyV22 Prior.

    Args:
    ----
        decoder_model (str): pretrained model name of decoder.
            Defaults to "kandinsky-community/kandinsky-2-2-decoder".
        prior_model (str): pretrained model name of prior.
            Defaults to "kandinsky-community/kandinsky-2-2-prior".
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        prior_lora_config (dict, optional): The LoRA config dict for Prior.
            example. dict(type="LoRA", r=4). `type` is chosen from `LoRA`,
            `LoHa`, `LoKr`. Other config are same as the config of PEFT.
            https://github.com/huggingface/peft
            Defaults to None.
        prior_loss_weight (float): The weight of prior preservation loss.
            It works when training dreambooth with class images.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDDataPreprocessor`.
        noise_generator (dict, optional): The noise generator config.
            Defaults to ``dict(type='WhiteNoise')``.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='TimeSteps')``.
        input_perturbation_gamma (float): The gamma of input perturbation.
            The recommended value is 0.1 for Input Perturbation.
            Defaults to 0.0.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
        enable_xformers (bool): Whether or not to enable memory efficient
            attention. Defaults to False.
    """

    def __init__(
        self,
        decoder_model: str = "kandinsky-community/kandinsky-2-2-decoder",
        prior_model: str = "kandinsky-community/kandinsky-2-2-prior",
        loss: dict | None = None,
        prior_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        gradient_checkpointing: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "TimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super().__init__(data_preprocessor=data_preprocessor)

        assert gradient_checkpointing is False, (
            "KandinskyV22Prior does not support gradient checkpointing.")

        self.decoder_model = decoder_model
        self.prior_model = prior_model
        self.prior_lora_config = deepcopy(prior_lora_config)
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

        self.tokenizer = CLIPTokenizer.from_pretrained(
            prior_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler(
            beta_schedule="squaredcos_cap_v2", prediction_type="sample")

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            prior_model, subfolder="text_encoder")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            prior_model, subfolder="image_encoder")
        self.prior = PriorTransformer.from_pretrained(
            prior_model, subfolder="prior")
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)

        self.register_buffer("clip_mean", self.prior.clip_mean.clone())
        self.register_buffer("clip_std", self.prior.clip_std.clone())
        self.prepare_model()
        self.set_lora()
        self.set_xformers()

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.prior_lora_config is not None:
            prior_lora_config = create_peft_config(self.prior_lora_config)
            self.prior = get_peft_model(self.prior, prior_lora_config)
            self.prior.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.text_encoder.requires_grad_(requires_grad=False)
        print_log("Set Text Encoder untrainable.", "current")
        self.image_encoder.requires_grad_(requires_grad=False)
        print_log("Set Image Encoder untrainable.", "current")

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.prior.enable_xformers_memory_efficient_attention()
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
        if height is None:
            height = 512
        if width is None:
            width = 512
        pipeline = AutoPipelineForText2Image.from_pretrained(
            self.decoder_model,
            prior_image_encoder=self.image_encoder,
            prior_text_encoder=self.text_encoder,
            prior_tokenizer=self.tokenizer,
            prior_prior=self.prior,
            torch_dtype=torch.float32,
        )
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
             noise: torch.Tensor,  #  noqa
             latents: torch.Tensor,
             timesteps: torch.Tensor,
             weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        gt = latents

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
        inputs_text = self.tokenizer(
            inputs["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        num_batches = len(inputs["img"])
        if "result_class_image" in inputs:
            # use prior_loss_weight
            weight = torch.cat([
                torch.ones((num_batches // 2, )),
                torch.ones((num_batches // 2, )) * self.prior_loss_weight,
            ]).to(self.device).float().reshape(-1, 1, 1, 1)
        else:
            weight = None

        with torch.no_grad():
            text_encoder_output = self.text_encoder(
                inputs_text.input_ids.to(self.device))
            prompt_embeds = text_encoder_output.text_embeds
            text_encoder_hidden_states = text_encoder_output.last_hidden_state

            image_embeds = self.image_encoder(inputs["img"]).image_embeds
            noise = self.noise_generator(image_embeds)
            timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

            image_embeds = (image_embeds - self.clip_mean) / self.clip_std
            noisy_latents = self._preprocess_model_input(image_embeds, noise, timesteps)

        model_pred = self.prior(
            noisy_latents,
            timesteps,
            proj_embedding=prompt_embeds,
            encoder_hidden_states=text_encoder_hidden_states,
            attention_mask=inputs_text.attention_mask.to(self.device)).predicted_image_embedding

        return self.loss(model_pred, noise, image_embeds, timesteps, weight)
