from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image
from torch import nn
from transformers import CLIPVisionModelWithProjection

from diffengine.models.archs import set_unet_ip_adapter
from diffengine.models.editors.ip_adapter.image_projection import \
    ImageProjModel
from diffengine.models.editors.ip_adapter.resampler import Resampler
from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.models.losses.snr_l2_loss import SNRL2Loss
from diffengine.registry import MODELS


@MODELS.register_module()
class IPAdapterXL(StableDiffusionXL):
    """Stable Diffusion XL IP-Adapter.

    Args:
        image_encoder (str, optional): Path to pretrained Image Encoder model.
            Defaults to 'takuoko/IP-Adapter-XL'.
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 4.
        lora_config (dict, optional): The LoRA config dict. This should be
            `None` when training ControlNet. Defaults to None.
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
                 image_encoder: str = 'takuoko/IP-Adapter-XL-test',
                 clip_extra_context_tokens: int = 4,
                 lora_config: Optional[dict] = None,
                 finetune_text_encoder: bool = False,
                 zeros_image_embeddings_prob: float = 0.1,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = dict(
                     type='IPAdapterXLDataPreprocessor'),
                 **kwargs):
        assert lora_config is None, \
            '`lora_config` should be None when training ControlNet'
        assert not finetune_text_encoder, \
            '`finetune_text_encoder` should be False when training ControlNet'

        self.image_encoder_name = image_encoder
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.zeros_image_embeddings_prob = zeros_image_embeddings_prob

        super().__init__(
            *args,
            lora_config=lora_config,
            finetune_text_encoder=finetune_text_encoder,
            data_preprocessor=data_preprocessor,
            **kwargs)

        self.set_ip_adapter()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_name, subfolder='image_encoder')
        self.image_projection = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens,
        )
        self.image_encoder.requires_grad_(False)
        super().prepare_model()

    def set_ip_adapter(self) -> None:
        """Set IP-Adapter for model."""
        self.unet.requires_grad_(False)
        set_unet_ip_adapter(self.unet)

    def _encode_image(self, image, num_images_per_prompt):
        if not isinstance(image, torch.Tensor):
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(
                self.image_encoder_name, subfolder='image_processor')
            image = image_processor(image, return_tensors='pt').pixel_values

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
              output_type: str = 'pil',
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

        pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            torch_dtype=(torch.float16 if self.device != torch.device('cpu')
                         else torch.float32),
        )
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p, img in zip(prompt, example_image):
            if type(img) == str:
                img = load_image(img)
            img = img.convert('RGB')

            image_embeddings, uncond_image_embeddings = self._encode_image(
                img, num_images_per_prompt=1)
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
            if output_type == 'latent':
                images.append(image)
            else:
                images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'loss'):
        assert mode == 'loss'
        inputs['text_one'] = self.tokenizer_one(
            inputs['text'],
            max_length=self.tokenizer_one.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt').input_ids.to(self.device)
        inputs['text_two'] = self.tokenizer_two(
            inputs['text'],
            max_length=self.tokenizer_two.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt').input_ids.to(self.device)
        num_batches = len(inputs['img'])
        if 'result_class_image' in inputs:
            # use prior_loss_weight
            weight = torch.cat([
                torch.ones((num_batches // 2, )),
                torch.ones((num_batches // 2, )) * self.prior_loss_weight
            ]).float().reshape(-1, 1, 1, 1)
        else:
            weight = None

        latents = self.vae.encode(inputs['img']).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        if self.enable_noise_offset:
            noise = noise + self.noise_offset_weight * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=noise.device)

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            inputs['text_one'], inputs['text_two'])
        unet_added_conditions = {
            'time_ids': inputs['time_ids'],
            'text_embeds': pooled_prompt_embeds
        }

        # encode image
        image_embeds = self.image_encoder(inputs['clip_img']).image_embeds
        # random zeros image embeddings
        mask = torch.multinomial(
            torch.Tensor([
                self.zeros_image_embeddings_prob,
                1 - self.zeros_image_embeddings_prob
            ]), len(image_embeds)).to(image_embeds)
        image_embeds = image_embeds * mask

        # todo: drop image
        ip_tokens = self.image_projection(image_embeds)
        prompt_embeds = torch.cat([prompt_embeds, ip_tokens], dim=1)

        if self.scheduler.config.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.config.prediction_type}')

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        loss_dict = dict()
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
        loss_dict['loss'] = loss
        return loss_dict


@MODELS.register_module()
class IPAdapterXLPlus(IPAdapterXL):
    """Stable Diffusion XL IP-Adapter Plus.

    Args:
        clip_extra_context_tokens (int): The number of expansion ratio of proj
            network hidden layer channels Defaults to 16.
    """

    def __init__(self, *args, clip_extra_context_tokens: int = 16, **kwargs):
        super().__init__(
            *args,
            clip_extra_context_tokens=clip_extra_context_tokens,
            **kwargs)

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_name, subfolder='image_encoder')
        self.image_projection = Resampler(
            embed_dims=self.image_encoder.config.hidden_size,
            output_dims=self.unet.config.cross_attention_dim,
            hidden_dims=1280,
            depth=4,
            head_dims=64,
            num_heads=20,
            num_queries=self.clip_extra_context_tokens,
            ffn_ratio=4)
        self.image_encoder.requires_grad_(False)
        super(IPAdapterXL, self).prepare_model()

    def _encode_image(self, image, num_images_per_prompt):
        if not isinstance(image, torch.Tensor):
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(
                self.image_encoder_name, subfolder='image_processor')
            image = image_processor(image, return_tensors='pt').pixel_values

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

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'loss'):
        assert mode == 'loss'
        inputs['text_one'] = self.tokenizer_one(
            inputs['text'],
            max_length=self.tokenizer_one.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt').input_ids.to(self.device)
        inputs['text_two'] = self.tokenizer_two(
            inputs['text'],
            max_length=self.tokenizer_two.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt').input_ids.to(self.device)
        num_batches = len(inputs['img'])
        if 'result_class_image' in inputs:
            # use prior_loss_weight
            weight = torch.cat([
                torch.ones((num_batches // 2, )),
                torch.ones((num_batches // 2, )) * self.prior_loss_weight
            ]).float().reshape(-1, 1, 1, 1)
        else:
            weight = None

        latents = self.vae.encode(inputs['img']).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        if self.enable_noise_offset:
            noise = noise + self.noise_offset_weight * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=noise.device)

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            inputs['text_one'], inputs['text_two'])
        unet_added_conditions = {
            'time_ids': inputs['time_ids'],
            'text_embeds': pooled_prompt_embeds
        }

        # random zeros image
        clip_img = inputs['clip_img']
        mask = torch.multinomial(
            torch.Tensor([
                self.zeros_image_embeddings_prob,
                1 - self.zeros_image_embeddings_prob
            ]), len(clip_img)).to(clip_img)
        clip_img = clip_img * mask
        # encode image
        image_embeds = self.image_encoder(
            clip_img, output_hidden_states=True).hidden_states[-2]

        # todo: drop image
        ip_tokens = self.image_projection(image_embeds)
        prompt_embeds = torch.cat([prompt_embeds, ip_tokens], dim=1)

        if self.scheduler.config.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.config.prediction_type}')

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        loss_dict = dict()
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
        loss_dict['loss'] = loss
        return loss_dict
