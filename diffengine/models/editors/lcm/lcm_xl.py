from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from transformers import CLIPTextConfig

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS

from .lcm_modules import (
    DDIMSolver,
    extract_into_tensor,
    scalings_for_boundary_conditions,
)


@MODELS.register_module()
class LatentConsistencyModelsXL(StableDiffusionXL):
    """Stable Diffusion XL Latent Consistency Models.

    Args:
    ----
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='DDIMTimeSteps')``.
        num_ddim_timesteps (int): Number of DDIM timesteps. Defaults to 50.
        w_min (float): Minimum guidance scale. Defaults to 3.0.
        w_max (float): Maximum guidance scale. Defaults to 15.0.
        ema_type (str): The type of EMA.
            Defaults to 'ExponentialMovingAverage'.
        ema_momentum (float): The EMA momentum. Defaults to 0.05.
    """

    def __init__(self,
                 *args,
                 timesteps_generator: dict | None = None,
                 num_ddim_timesteps: int = 50,
                 w_min: float = 3.0,
                 w_max: float = 15.0,
                 ema_type: str = "ExponentialMovingAverage",
                 ema_momentum: float = 0.05,
                 **kwargs) -> None:

        if timesteps_generator is None:
            timesteps_generator = {"type": "DDIMTimeSteps",
                                   "num_ddim_timesteps": num_ddim_timesteps}
        assert timesteps_generator["type"] == "DDIMTimeSteps"

        self.ema_cfg = dict(type=ema_type, momentum=ema_momentum)

        super().__init__(*args,
                         timesteps_generator=timesteps_generator,
                         **kwargs)  # type: ignore[misc]

        self.num_ddim_timesteps = num_ddim_timesteps
        self.w_min = w_min
        self.w_max = w_max

        self.register_buffer("alpha_schedule",
                             torch.sqrt(self.scheduler.alphas_cumprod))
        self.register_buffer("sigma_schedule",
                             torch.sqrt(1 - self.scheduler.alphas_cumprod))
        self.solver = DDIMSolver(
            self.scheduler.alphas_cumprod,
            timesteps=self.scheduler.config.num_train_timesteps,
            ddim_timesteps=num_ddim_timesteps,
        )
        if self.pre_compute_text_embeddings:
            text_encoder_one_config = CLIPTextConfig.from_pretrained(
                self.model, subfolder="text_encoder")
            text_encoder_two_config = CLIPTextConfig.from_pretrained(
                self.model, subfolder="text_encoder_2")
            prompt_embeds_size = (
                text_encoder_one_config.hidden_size + \
                text_encoder_two_config.hidden_size
            )
            pooled_embeds_size = text_encoder_two_config.hidden_size
            self.register_buffer("uncond_prompt_embeds",
                                torch.zeros(1, 77, prompt_embeds_size))
            self.register_buffer("uncond_pooled_prompt_embeds",
                                torch.zeros(1, pooled_embeds_size))
        else:
            prompt_embeds_size = (
                self.text_encoder_one.config.hidden_size + \
                self.text_encoder_two.config.hidden_size
            )
            pooled_embeds_size = self.text_encoder_two.config.hidden_size
            self.register_buffer("uncond_prompt_embeds",
                                torch.zeros(1, 77, prompt_embeds_size))
            self.register_buffer("uncond_pooled_prompt_embeds",
                                torch.zeros(1, pooled_embeds_size))

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.teacher_unet = deepcopy(
            self.unet).requires_grad_(requires_grad=False)
        if self.unet_lora_config is None:
            self.target_unet = MODELS.build(
                self.ema_cfg, default_args=dict(model=self.unet))

        super().prepare_model()

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                self.teacher_unet.enable_xformers_memory_efficient_attention()
                if self.unet_lora_config is None:
                    self.target_unet.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              height: int | None = None,
              width: int | None = None,
              num_inference_steps: int = 4,
              guidance_scale: float = 1.0,
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
            guidance_scale (float): The guidance scale. Defaults to 1.0.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            **kwargs: Other arguments.
        """
        if self.pre_compute_text_embeddings:
            pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                unet=self.unet,
                scheduler=LCMScheduler.from_pretrained(
                    self.model, subfolder="scheduler"),
                safety_checker=None,
                torch_dtype=torch.float32,
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                text_encoder=self.text_encoder_one,
                text_encoder_2=self.text_encoder_two,
                tokenizer=self.tokenizer_one,
                tokenizer_2=self.tokenizer_two,
                unet=self.unet,
                scheduler=LCMScheduler.from_pretrained(
                    self.model, subfolder="scheduler"),
                torch_dtype=(torch.float16
                             if self.device != torch.device("cpu") else
                             torch.float32),
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
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type=output_type,
                guidance_scale=guidance_scale,
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def loss(  # type: ignore[override]
        self,
        model_pred: torch.Tensor,
        gt: torch.Tensor,
        timesteps: torch.Tensor,
        weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
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

        start_timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        steps = (
            self.scheduler.config.num_train_timesteps / self.num_ddim_timesteps
        )
        timesteps = start_timesteps - steps
        timesteps = torch.where(timesteps < 0, 0, timesteps).long()
        index = (start_timesteps / steps).long()

        # Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps)
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)

        noisy_latents = self._preprocess_model_input(latents, noise, start_timesteps)

        # Sample a random guidance scale w from U[w_min, w_max] and embed it
        w = (self.w_max - self.w_min) * torch.rand((num_batches,)) + self.w_min
        w = w.reshape(num_batches, 1, 1, 1)
        w = w.to(device=latents.device, dtype=latents.dtype)

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

        # Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
        noise_pred = self.unet(
            noisy_latents,
            start_timesteps,
            timestep_cond=None,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample
        pred_x_0 = self._predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_latents,
        )
        model_pred = c_skip_start * noisy_latents + c_out_start * pred_x_0

        # Use the ODE solver to predict the kth step in the augmented PF-ODE
        # trajectory after noisy_latents with both the conditioning embedding
        # c and unconditional embedding 0. Get teacher model prediction on
        # noisy_latents and conditional embedding
        with torch.no_grad():
            cond_teacher_output = self.teacher_unet(
                noisy_latents,
                start_timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions).sample
            cond_pred_x0 = self._predicted_origin(
                cond_teacher_output,
                start_timesteps,
                noisy_latents,
            )

            # Get teacher model prediction on noisy_latents and unconditional embedding
            uncond_added_conditions = deepcopy(unet_added_conditions)
            uncond_added_conditions[
                "text_embeds"
                ] = self.uncond_pooled_prompt_embeds.repeat(num_batches, 1)
            uncond_teacher_output = self.teacher_unet(
                noisy_latents,
                start_timesteps,
                encoder_hidden_states=self.uncond_prompt_embeds.repeat(
                    num_batches, 1, 1),
                added_cond_kwargs=uncond_added_conditions).sample
            uncond_pred_x0 = self._predicted_origin(
                uncond_teacher_output,
                start_timesteps,
                noisy_latents,
            )

            # Perform "CFG" to get x_prev estimate
            # (using the LCM paper's CFG formulation)
            pred_x0 = cond_pred_x0 + w * (
                cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_teacher_output + w * (
                cond_teacher_output - uncond_teacher_output)
            x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

        # Get target LCM prediction on x_prev, w, c, t_n
        with torch.no_grad():
            if self.unet_lora_config is None:
                target_noise_pred = self.target_unet(
                    x_prev,
                    timesteps,
                    timestep_cond=None,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                ).sample
            else:
                # LCM LoRA
                target_noise_pred = self.unet(
                    x_prev,
                    timesteps,
                    timestep_cond=None,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                ).sample
            pred_x_0 = self._predicted_origin(
                target_noise_pred,
                timesteps,
                x_prev,
            )
            target = c_skip * x_prev + c_out * pred_x_0

        return self.loss(model_pred, target, timesteps, weight)

    def _predicted_origin(self,
                          model_output: torch.Tensor,
                          timesteps: torch.Tensor,
                          sample: torch.Tensor) -> torch.Tensor:
        """Predict the origin of the model output.

        Args:
        ----
            model_output (torch.Tensor): The model output.
            timesteps (torch.Tensor): The timesteps.
            sample (torch.Tensor): The sample.
        """
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        sigmas = extract_into_tensor(self.sigma_schedule, timesteps)
        alphas = extract_into_tensor(self.alpha_schedule, timesteps)

        if self.scheduler.config.prediction_type == "epsilon":

            pred_x_0 = (sample - sigmas * model_output) / alphas
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_x_0 = alphas * sample - sigmas * model_output
        else:
            msg = f"Prediction type {self.prediction_type} currently not supported."
            raise ValueError(msg)

        return pred_x_0
