from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from mmengine import print_log
from mmengine.model import BaseModel
from peft import get_peft_model
from torch import nn

from diffengine.models.archs import create_peft_config
from diffengine.registry import MODELS


@MODELS.register_module()
class KandinskyV22Decoder(BaseModel):
    """KandinskyV22 Decoder.

    Args:
    ----
        scheduler (dict): Config of scheduler.
        image_encoder (dict): Config of image encoder.
        vae (dict): Config of vae.
        unet (dict): Config of unet.
        decoder_model (str): pretrained model name of decoder.
            Defaults to "kandinsky-community/kandinsky-2-2-decoder".
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        unet_lora_config (dict, optional): The LoRA config dict for Unet.
            example. dict(type="LoRA", r=4). `type` is chosen from `LoRA`,
            `LoHa`, `LoKr`. Other config are same as the config of PEFT.
            https://github.com/huggingface/peft
            Defaults to None.
        prior_loss_weight (float): The weight of prior preservation loss.
            It works when training dreambooth with class images.
        prediction_type (str): The prediction_type that shall be used for
            training. Choose between 'epsilon' or 'v_prediction' or leave
            `None`. If left to `None` the default prediction type of the
            scheduler will be used. Defaults to None.
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
        scheduler: dict,
        image_encoder: dict,
        vae: dict,
        unet: dict,
        decoder_model: str = "kandinsky-community/kandinsky-2-2-decoder",
        loss: dict | None = None,
        unet_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        *,
        gradient_checkpointing: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "KandinskyV22DecoderDataPreprocessor"}
        if noise_generator is None:
            noise_generator = {"type": "WhiteNoise"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "TimeSteps"}
        if loss is None:
            loss = {"type": "L2Loss", "loss_weight": 1.0}
        super().__init__(data_preprocessor=data_preprocessor)

        self.decoder_model = decoder_model
        self.unet_lora_config = deepcopy(unet_lora_config)
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module: nn.Module = loss

        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        self.scheduler = MODELS.build(scheduler)

        self.image_encoder = MODELS.build(image_encoder)
        self.vae = MODELS.build(vae)
        self.unet = MODELS.build(unet)
        self.noise_generator = MODELS.build(noise_generator)
        self.timesteps_generator = MODELS.build(timesteps_generator)

        self.prepare_model()
        self.set_lora()
        self.set_xformers()

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.unet_lora_config is not None:
            unet_lora_config = create_peft_config(self.unet_lora_config)
            self.unet = get_peft_model(self.unet, unet_lora_config)
            self.unet.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
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
            height = 768
        if width is None:
            width = 768
        pipeline = AutoPipelineForText2Image.from_pretrained(
            self.decoder_model,
            movq=self.vae,
            prior_image_encoder=self.image_encoder,
            unet=self.unet,
            torch_dtype=torch.float32,
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
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
            ]).to(self.device).float().reshape(-1, 1, 1, 1)
        else:
            weight = None

        latents = self.vae.encode(inputs["img"]).latents
        image_embeds = self.image_encoder(inputs["clip_img"]).image_embeds

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                        self.device)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

        added_cond_kwargs = {"image_embeds": image_embeds}
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            None,
            added_cond_kwargs=added_cond_kwargs).sample[:, :4]

        return self.loss(model_pred, noise, image_embeds, timesteps, weight)
