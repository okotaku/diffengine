from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import (
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from mmengine import print_log
from mmengine.model import BaseModel
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.archs import set_text_encoder_lora, set_unet_lora
from diffengine.models.losses.snr_l2_loss import SNRL2Loss
from diffengine.registry import MODELS


@MODELS.register_module()
class DeepFloydIF(BaseModel):
    """DeepFloyd/IF.

    Args:
    ----
        model (str): pretrained model name of stable diffusion.
            Defaults to 'DeepFloyd/IF-I-XL-v1.0'.
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
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDDataPreprocessor`.
        tokenizer_max_length (int): The max length of tokenizer.
            Defaults to 77.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
    """

    def __init__(
        self,
        model: str = "DeepFloyd/IF-I-XL-v1.0",
        loss: dict | None = None,
        lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        noise_offset_weight: float = 0,
        data_preprocessor: dict | nn.Module | None = None,
        tokenizer_max_length: int = 77,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDDataPreprocessor"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super().__init__(data_preprocessor=data_preprocessor)
        self.model = model
        self.lora_config = deepcopy(lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.tokenizer_max_length = tokenizer_max_length

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

        self.enable_noise_offset = noise_offset_weight > 0
        self.noise_offset_weight = noise_offset_weight

        self.tokenizer = T5Tokenizer.from_pretrained(
            model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(
            model, subfolder="scheduler")

        self.text_encoder = T5EncoderModel.from_pretrained(
            model, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(
            model, subfolder="unet")
        self.prepare_model()
        self.set_lora()

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.lora_config is not None:
            if self.finetune_text_encoder:
                self.text_encoder.requires_grad_(requires_grad=False)
                set_text_encoder_lora(self.text_encoder, self.lora_config)
            self.unet.requires_grad_(requires_grad=False)
            set_unet_lora(self.unet, self.lora_config)

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

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
                Choose between 'pil' and 'pt'. Defaults to 'pil'.
            **kwargs: Other arguments.
        """
        pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            safety_checker=None,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
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
            if output_type in ["latent", "pt"]:
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

        model_input = inputs["img"]

        noise = torch.randn_like(model_input)

        if self.enable_noise_offset:
            noise = noise + self.noise_offset_weight * torch.randn(
                model_input.shape[0],
                model_input.shape[1],
                1,
                1,
                device=noise.device)

        num_batches = model_input.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_model_input = self.scheduler.add_noise(model_input, noise,
                                                     timesteps)

        encoder_hidden_states = self.text_encoder(
            inputs["text"], attention_mask=inputs["attention_mask"])[0]

        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(model_input, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)

        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample

        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

        loss_dict = {}
        # calculate loss in FP32
        if isinstance(self.loss_module, SNRL2Loss):
            loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                weight=weight)
        else:
            loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
        loss_dict["loss"] = loss
        return loss_dict