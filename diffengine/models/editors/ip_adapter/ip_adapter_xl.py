from typing import Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.models.embeddings import ImageProjection
from diffusers.utils import load_image
from PIL import Image
from torch import nn
from transformers import CLIPVisionModelWithProjection

from diffengine.models.archs import set_unet_ip_adapter
from diffengine.models.editors.ip_adapter.resampler import Resampler
from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class IPAdapterXL(StableDiffusionXL):
    """Stable Diffusion XL IP-Adapter.

    Args:
    ----
        image_encoder (str, optional): Path to pretrained Image Encoder model.
            Defaults to 'takuoko/IP-Adapter-XL'.
        num_image_text_embeds (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 4.
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
        zeros_image_embeddings_prob (float): The probabilities to
            generate zeros image embeddings. Defaults to 0.1.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDControlNetDataPreprocessor`.
    """

    def __init__(self,
                 *args,
                 image_encoder: str = "takuoko/IP-Adapter-XL-test",
                 num_image_text_embeds: int = 4,
                 unet_lora_config: dict | None = None,
                 text_encoder_lora_config: dict | None = None,
                 finetune_text_encoder: bool = False,
                 zeros_image_embeddings_prob: float = 0.1,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "IPAdapterXLDataPreprocessor"}
        assert unet_lora_config is None, \
            "`unet_lora_config` should be None when training IPAdapter"
        assert text_encoder_lora_config is None, \
            "`text_encoder_lora_config` should be None when training IPAdapter"
        assert not finetune_text_encoder, \
            "`finetune_text_encoder` should be False when training IPAdapter"

        self.image_encoder_name = image_encoder
        self.num_image_text_embeds = num_image_text_embeds
        self.zeros_image_embeddings_prob = zeros_image_embeddings_prob

        super().__init__(
            *args,
            unet_lora_config=unet_lora_config,
            text_encoder_lora_config=text_encoder_lora_config,
            finetune_text_encoder=finetune_text_encoder,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

        self.set_ip_adapter()

    def set_lora(self) -> None:
        """Set LORA for model."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_name, subfolder="image_encoder")
        self.image_projection = ImageProjection(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            image_embed_dim=self.image_encoder.config.projection_dim,
            num_image_text_embeds=self.num_image_text_embeds,
        )
        self.image_encoder.requires_grad_(requires_grad=False)
        super().prepare_model()

    def set_ip_adapter(self) -> None:
        """Set IP-Adapter for model."""
        self.unet.requires_grad_(requires_grad=False)
        set_unet_ip_adapter(self.unet)

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
              prompt: list[str],
              example_image: list[str | Image.Image],
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
            example_image (`List[Union[str, Image.Image]]`):
                The image prompt or prompts to guide the image generation.
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
        assert len(prompt) == len(example_image)

        pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
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
        for p, img in zip(prompt, example_image, strict=True):
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB")

            image_embeddings, uncond_image_embeddings = self._encode_image(
                pil_img, num_images_per_prompt=1)
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = pipeline.encode_prompt(
                 p,
                 num_images_per_prompt=1,
                 do_classifier_free_guidance=True,
                 negative_prompt=negative_prompt)
            prompt_embeds = torch.cat([prompt_embeds, image_embeddings], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, uncond_image_embeddings], dim=1)
            image = pipeline(
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

        del pipeline
        torch.cuda.empty_cache()

        return images

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

        # encode image
        image_embeds = self.image_encoder(inputs["clip_img"]).image_embeds
        # random zeros image embeddings
        mask = torch.multinomial(
            torch.Tensor([
                self.zeros_image_embeddings_prob,
                1 - self.zeros_image_embeddings_prob,
            ]),
            len(image_embeds),
            replacement=True).to(image_embeds)
        image_embeds = image_embeds * mask.view(-1, 1, 1, 1)

        # TODO(takuoko): drop image  # noqa
        ip_tokens = self.image_projection(image_embeds)
        prompt_embeds = torch.cat([prompt_embeds, ip_tokens], dim=1)

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        return self.loss(model_pred, noise, latents, timesteps, weight)


@MODELS.register_module()
class IPAdapterXLPlus(IPAdapterXL):
    """Stable Diffusion XL IP-Adapter Plus.

    Args:
    ----
        num_image_text_embeds (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 16.
    """

    def __init__(self,
                 *args,
                 num_image_text_embeds: int = 16,
                 **kwargs) -> None:
        super().__init__(
            *args,
            num_image_text_embeds=num_image_text_embeds,
            **kwargs)

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_name, subfolder="image_encoder")
        self.image_projection = Resampler(
            embed_dims=self.image_encoder.config.hidden_size,
            output_dims=self.unet.config.cross_attention_dim,
            hidden_dims=1280,
            depth=4,
            head_dims=64,
            num_heads=20,
            num_queries=self.num_image_text_embeds,
            ffn_ratio=4)
        self.image_encoder.requires_grad_(requires_grad=False)
        super(IPAdapterXL, self).prepare_model()

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

        # random zeros image
        clip_img = inputs["clip_img"]
        mask = torch.multinomial(
            torch.Tensor([
                self.zeros_image_embeddings_prob,
                1 - self.zeros_image_embeddings_prob,
            ]),
            len(clip_img),
            replacement=True).to(clip_img)
        clip_img = clip_img * mask.view(-1, 1, 1, 1)
        # encode image
        image_embeds = self.image_encoder(
            clip_img, output_hidden_states=True).hidden_states[-2]

        # TODO(takuoko): drop image  # noqa
        ip_tokens = self.image_projection(image_embeds)
        prompt_embeds = torch.cat([prompt_embeds, ip_tokens], dim=1)

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        return self.loss(model_pred, noise, latents, timesteps, weight)
