from typing import Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import load_image
from mmengine.model import BaseModel
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from diffengine.models.archs import set_controlnet_ip_adapter, set_unet_ip_adapter
from diffengine.models.editors.ip_adapter.image_projection import ImageProjModel
from diffengine.models.editors.ip_adapter.resampler import Resampler
from diffengine.registry import MODELS


@MODELS.register_module()
class IPAdapterXLPipeline(BaseModel):
    """IPAdapterXLPipeline.

    Args:
        pipeline (DiffusionPipeline): diffusers pipeline
        image_encoder (str, optional): Path to pretrained Image Encoder model.
            Defaults to 'takuoko/IP-Adapter-XL'.
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 4.
    """

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        image_encoder: str = "takuoko/IP-Adapter-XL-test",
        clip_extra_context_tokens: int = 4,
    ):
        self.image_encoder_name = image_encoder
        self.clip_extra_context_tokens = clip_extra_context_tokens

        super().__init__()
        self.pipeline = pipeline
        self.prepare_model()
        self.set_ip_adapter()

    @property
    def device(self):
        return next(self.parameters()).device

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_name, subfolder="image_encoder")
        self.image_projection = ImageProjModel(
            cross_attention_dim=self.pipeline.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens,
        )

    def set_ip_adapter(self) -> None:
        """Set IP-Adapter for model."""
        set_unet_ip_adapter(self.pipeline.unet)

        if hasattr(self.pipeline, "controlnet"):
            if isinstance(self.pipeline.controlnet, MultiControlNetModel):
                for controlnet in self.pipeline.controlnet.nets:
                    set_controlnet_ip_adapter(controlnet,
                                              self.clip_extra_context_tokens)
            else:
                set_controlnet_ip_adapter(self.pipeline.controlnet,
                                          self.clip_extra_context_tokens)

    def _encode_image(self, image, num_images_per_prompt):
        if not isinstance(image, torch.Tensor):
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(
                self.image_encoder_name, subfolder="image_processor")
            image = image_processor(image, return_tensors="pt").pixel_values

        image = image.to(device=self.device)
        image_embeddings = self.image_encoder(image).image_embeds
        image_prompt_embeds = self.image_projection(image_embeddings)
        uncond_image_prompt_embeds = self.image_projection(
            torch.zeros_like(image_embeddings))

        # duplicate image embeddings for each generation per prompt, using mps
        # friendly method
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(
            1, num_images_per_prompt, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, num_images_per_prompt, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.no_grad()
    def infer(self,
              prompt: List[str],
              example_image: List[Union[str, Image.Image]],
              negative_prompt: Optional[str] = None,
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 50,
              output_type: str = "pil",
              **kwargs) -> List[np.ndarray]:
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            example_image (`List[Union[str, Image.Image]]`):
                The image prompt or prompts to guide the image generation.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (`int`, *optional*, defaults to
                `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to
                `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (int): Number of inference steps.
                Defaults to 50.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
        """
        assert len(prompt) == len(example_image)

        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, example_image):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")

            image_embeddings, uncond_image_embeddings = self._encode_image(
                pil_img, num_images_per_prompt=1)
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = self.pipeline.encode_prompt(
                 p,
                 num_images_per_prompt=1,
                 do_classifier_free_guidance=True,
                 negative_prompt=negative_prompt)
            prompt_embeds = torch.cat([prompt_embeds, image_embeddings], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, uncond_image_embeddings], dim=1)
            image = self.pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type=output_type,
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

        return images

    def forward(
            self,
            inputs: torch.Tensor,  # noqa
            data_samples: Optional[list] = None,  # noqa
            mode: str = "tensor",  # noqa
    ) -> Union[Dict[str, torch.Tensor], list]:
        msg = "forward is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def train_step(self, data, optim_wrapper_dict):  # noqa
        msg = "train_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def val_step(self, data: Union[tuple, dict, list]) -> list:  # noqa
        msg = "val_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def test_step(self, data: Union[tuple, dict, list]) -> list:  # noqa
        msg = "test_step is not implemented now, please use infer."
        raise NotImplementedError(msg)


@MODELS.register_module()
class IPAdapterXLPlusPipeline(IPAdapterXLPipeline):
    """IPAdapterXLPlusPipeline.

    Args:
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 16.
    """

    def __init__(self, *args, clip_extra_context_tokens: int = 16, **kwargs):
        super().__init__(
            *args,
            clip_extra_context_tokens=clip_extra_context_tokens,
            **kwargs)  # type: ignore[misc]

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_name, subfolder="image_encoder")
        self.image_projection = Resampler(
            embed_dims=self.image_encoder.config.hidden_size,
            output_dims=self.pipeline.unet.config.cross_attention_dim,
            hidden_dims=1280,
            depth=4,
            head_dims=64,
            num_heads=20,
            num_queries=self.clip_extra_context_tokens,
            ffn_ratio=4)

    def _encode_image(self, image, num_images_per_prompt):
        if not isinstance(image, torch.Tensor):
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(
                self.image_encoder_name, subfolder="image_processor")
            image = image_processor(image, return_tensors="pt").pixel_values

        image = image.to(device=self.device)
        image_embeddings = self.image_encoder(
            image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_projection(image_embeddings)
        uncond_image_embeddings = self.image_encoder(
            torch.zeros_like(image),
            output_hidden_states=True).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_projection(
            uncond_image_embeddings)

        # duplicate image embeddings for each generation per prompt, using mps
        # friendly method
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(
            1, num_images_per_prompt, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, num_images_per_prompt, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        return image_prompt_embeds, uncond_image_prompt_embeds
