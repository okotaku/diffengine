from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusionXLAdapterPipeline
from diffusers.utils import load_image
from mmengine import print_log
from PIL import Image
from torch import nn

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusionXLT2IAdapter(StableDiffusionXL):
    """Stable Diffusion XL T2I Adapter.

    Args:
    ----
        adapter (dict): The adapter config.
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
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='CubicSamplingTimeSteps')``.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDControlNetDataPreprocessor`.
    """

    def __init__(self,
                 *args,
                 adapter: dict,
                 unet_lora_config: dict | None = None,
                 text_encoder_lora_config: dict | None = None,
                 finetune_text_encoder: bool = False,
                 timesteps_generator: dict | None = None,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDXLControlNetDataPreprocessor"}
        if timesteps_generator is None:
            timesteps_generator = {"type": "CubicSamplingTimeSteps"}
        assert unet_lora_config is None, \
            "`unet_lora_config` should be None when training T2IAdapter"
        assert text_encoder_lora_config is None, \
            "`text_encoder_lora_config` should be None when training T2IAdapter"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training T2IAdapter"

        self.adapter_config = adapter

        super().__init__(
            *args,
            unet_lora_config=unet_lora_config,
            text_encoder_lora_config=text_encoder_lora_config,
            finetune_text_encoder=finetune_text_encoder,
            timesteps_generator=timesteps_generator,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def set_lora(self) -> None:
        """Set LORA for model."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.adapter = MODELS.build(self.adapter_config)

        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        self.text_encoder_one.requires_grad_(requires_grad=False)
        self.text_encoder_two.requires_grad_(requires_grad=False)
        print_log("Set Text Encoder untrainable.", "current")
        self.unet.requires_grad_(requires_grad=False)
        print_log("Set Unet untrainable.", "current")

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              condition_image: list[str | Image.Image],
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
            condition_image (`List[Union[str, Image.Image]]`):
                The condition image for ControlNet.
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
        assert len(prompt) == len(condition_image)
        pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            adapter=self.adapter,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, condition_image, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            image = pipeline(
                p,
                p,
                pil_img,
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

    def _forward_compile(self,
                         noisy_latents: torch.Tensor,
                         timesteps: torch.Tensor,
                         prompt_embeds: torch.Tensor,
                         unet_added_conditions: dict,
                         inputs: dict) -> torch.Tensor:
        """Forward function for torch.compile."""
        down_block_additional_residuals = self.adapter(inputs["condition_img"])

        return self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            down_block_additional_residuals=down_block_additional_residuals,
        ).sample

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

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            inputs["text_one"], inputs["text_two"])
        unet_added_conditions = {
            "time_ids": inputs["time_ids"],
            "text_embeds": pooled_prompt_embeds,
        }

        model_pred = self._forward_compile(
            noisy_latents, timesteps, prompt_embeds, unet_added_conditions,
            inputs)

        return self.loss(model_pred, noise, latents, timesteps, weight)
