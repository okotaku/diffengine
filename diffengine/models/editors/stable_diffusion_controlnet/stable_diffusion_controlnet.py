from typing import Optional

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from mmengine import print_log
from PIL import Image
from torch import nn

from diffengine.models.editors.stable_diffusion import StableDiffusion
from diffengine.registry import MODELS


@MODELS.register_module()
class StableDiffusionControlNet(StableDiffusion):
    """Stable Diffusion ControlNet.

    Args:
    ----
        controlnet_model (str, optional): Path to pretrained ControlNet model.
            If None, use the default ControlNet model from Unet.
            Defaults to None.
        transformer_layers_per_block (List[int], optional):
            The number of layers per block in the transformer. More details:
            https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small.
            Defaults to None.
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
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDControlNetDataPreprocessor`.
    """

    def __init__(self,
                 *args,
                 controlnet_model: str | None = None,
                 transformer_layers_per_block: list[int] | None = None,
                 unet_lora_config: dict | None = None,
                 text_encoder_lora_config: dict | None = None,
                 finetune_text_encoder: bool = False,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "SDControlNetDataPreprocessor"}
        assert unet_lora_config is None, \
            "`unet_lora_config` should be None when training ControlNet"
        assert text_encoder_lora_config is None, \
            "`text_encoder_lora_config` should be None when training ControlNet"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training ControlNet"

        self.controlnet_model = controlnet_model
        self.transformer_layers_per_block = transformer_layers_per_block

        super().__init__(
            *args,
            unet_lora_config=unet_lora_config,
            text_encoder_lora_config=text_encoder_lora_config,
            finetune_text_encoder=finetune_text_encoder,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def set_lora(self) -> None:
        """Set LORA for model."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.controlnet_model is not None:
            pre_controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model)
        else:
            pre_controlnet = ControlNetModel.from_unet(self.unet)

        if self.transformer_layers_per_block is not None:
            down_block_types = [
                ("DownBlock2D" if i == 0 else "CrossAttnDownBlock2D")
                for i in self.transformer_layers_per_block
            ]
            self.controlnet = ControlNetModel.from_config(
                pre_controlnet.config,
                down_block_types=down_block_types,
                transformer_layers_per_block=self.transformer_layers_per_block,
            )
            self.controlnet.load_state_dict(
                pre_controlnet.state_dict(), strict=False)
            del pre_controlnet
        else:
            self.controlnet = pre_controlnet

        if self.gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        self.text_encoder.requires_grad_(requires_grad=False)
        print_log("Set Text Encoder untrainable.", "current")
        self.unet.requires_grad_(requires_grad=False)
        print_log("Set Unet untrainable.", "current")

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                self.controlnet.enable_xformers_memory_efficient_attention()
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
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, condition_image, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")
            image = pipeline(
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
                         encoder_hidden_states: torch.Tensor,
                         inputs: dict) -> torch.Tensor:
        """Forward function for torch.compile."""
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=inputs["condition_img"],
            return_dict=False,
        )

        return self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample).sample

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
        inputs["text"] = self.tokenizer(
            inputs["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt").input_ids.to(self.device)
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

        encoder_hidden_states = self.text_encoder(inputs["text"])[0]

        model_pred = self._forward_compile(
            noisy_latents, timesteps, encoder_hidden_states,
            inputs)

        return self.loss(model_pred, noise, latents, timesteps, weight)
