# flake8: noqa
import torch
from diffusers import StableDiffusionXLPipeline


class StableDiffusionXLPipelineCustomIPAdapter(StableDiffusionXLPipeline):
    """Custom IP Adapter for the StableDiffusionXLPipeline class.

    The difference between this class and the original
    StableDiffusionXLPipeline class is that this class uses the hidden states
    from the `hidden_states_idx` layer of the image encoder to encode the
    image.

    Args:
        *args: Variable length argument list.
        hidden_states_idx (int): Index of the hidden states to be used.
            Defaults to -2.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self,
                vae,
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2,
                unet,
                scheduler,
                image_encoder=None,
                feature_extractor=None,
                force_zeros_for_empty_prompt=True,
                add_watermarker=None,
                hidden_states_idx: int = -2):
        super().__init__(vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker)
        self.hidden_states_idx = hidden_states_idx

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        """Encodes the image.

        Args:
            image: The input image to be encoded.
            device: The device to be used for encoding.
            num_images_per_prompt: The number of images per prompt.
            output_hidden_states: Whether to output hidden states. Defaults to None.

        Returns:
            image_enc_hidden_states: Encoded hidden states of the image.
            uncond_image_enc_hidden_states: Encoded hidden states of the unconditional image.
        """
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            if self.hidden_states_idx == -1:
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).last_hidden_state
            else:
                image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[self.hidden_states_idx]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            if self.hidden_states_idx == -1:
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).last_hidden_state
            else:
                uncond_image_enc_hidden_states = self.image_encoder(
                    torch.zeros_like(image), output_hidden_states=True
                ).hidden_states[self.hidden_states_idx]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0,
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds


class StableDiffusionXLPipelineTimmIPAdapter(StableDiffusionXLPipeline):
    """Timm IP Adapter for the StableDiffusionXLPipeline class.

    The difference between this class and the original
    StableDiffusionXLPipeline class is that this class uses the timm library
    for the image encoder.

    Args:
        *args: Variable length argument list.
        hidden_states_idx (int): Index of the hidden states to be used.
            Defaults to -2.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self,
                vae,
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2,
                unet,
                scheduler,
                image_encoder=None,
                feature_extractor=None,
                force_zeros_for_empty_prompt=True,
                add_watermarker=None):
        super().__init__(vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker)

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        """Encodes the image.

        Args:
            image: The input image to be encoded.
            device: The device to be used for encoding.
            num_images_per_prompt: The number of images per prompt.
            output_hidden_states: Whether to output hidden states. Defaults to None.

        Returns:
            image_enc_hidden_states: Encoded hidden states of the image.
            uncond_image_enc_hidden_states: Encoded hidden states of the unconditional image.
        """
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image).unsqueeze(0)

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder.forward_features(image)
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder.forward_features(
                torch.zeros_like(image),
            )
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0,
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder.forward_features(image)
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device
