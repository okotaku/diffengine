from typing import Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image
from torch import nn
from transformers import CLIPImageProcessor

from diffengine.models.archs import (
    load_ip_adapter,
    process_ip_adapter_state_dict,
    set_unet_ip_adapter,
)
from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.registry import MODELS


@MODELS.register_module()
class IPAdapterXL(StableDiffusionXL):
    """Stable Diffusion XL IP-Adapter.

    Args:
    ----
        image_encoder (dict): The image encoder config.
        image_projection (dict): The image projection config.
        pretrained_adapter (str, optional): Path to pretrained IP-Adapter.
            Defaults to None.
        pretrained_adapter_subfolder (str, optional): Sub folder of pretrained
            IP-Adapter. Defaults to ''.
        pretrained_adapter_weights_name (str, optional): Weights name of
            pretrained IP-Adapter. Defaults to ''.
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
                 image_encoder: dict,
                 image_projection: dict,
                 pretrained_adapter: str | None = None,
                 pretrained_adapter_subfolder: str = "",
                 pretrained_adapter_weights_name: str = "",
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

        self.image_encoder_config = image_encoder
        self.image_projection_config = image_projection
        self.pretrained_adapter = pretrained_adapter
        self.pretrained_adapter_subfolder = pretrained_adapter_subfolder
        self.pretrained_adapter_weights_name = pretrained_adapter_weights_name
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
        self.image_encoder = MODELS.build(self.image_encoder_config)
        self.image_projection_config.update(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            image_embed_dim=self.image_encoder.config.projection_dim,
        )
        self.image_projection = MODELS.build(self.image_projection_config)
        self.image_encoder.requires_grad_(requires_grad=False)
        super().prepare_model()

    def set_ip_adapter(self) -> None:
        """Set IP-Adapter for model."""
        self.unet.requires_grad_(requires_grad=False)
        set_unet_ip_adapter(self.unet)
        if self.pretrained_adapter is not None:
            load_ip_adapter(self.unet, self.image_projection,
                            self.pretrained_adapter,
                            self.pretrained_adapter_subfolder,
                            self.pretrained_adapter_weights_name)

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

        orig_encoder_hid_proj = self.unet.encoder_hid_proj
        orig_encoder_hid_dim_type = self.unet.config.encoder_hid_dim_type

        pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            image_encoder=self.image_encoder,
            feature_extractor=CLIPImageProcessor(),
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
        adapter_state_dict = process_ip_adapter_state_dict(
            self.unet, self.image_projection)
        pipeline.load_ip_adapter(
            pretrained_model_name_or_path_or_dict=adapter_state_dict,
            subfolder="", weight_name="")
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

            image = pipeline(
                p,
                ip_adapter_image=pil_img,
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

        del pipeline, adapter_state_dict
        torch.cuda.empty_cache()

        self.unet.encoder_hid_proj = orig_encoder_hid_proj
        self.unet.config.encoder_hid_dim_type = orig_encoder_hid_dim_type

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
    """Stable Diffusion XL IP-Adapter Plus."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = MODELS.build(self.image_encoder_config)
        self.image_projection_config.update(
            embed_dims=self.image_encoder.config.hidden_size,
            output_dims=self.unet.config.cross_attention_dim,
        )
        self.image_projection = MODELS.build(self.image_projection_config)
        self.image_encoder.requires_grad_(requires_grad=False)
        super(IPAdapterXL, self).prepare_model()

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
