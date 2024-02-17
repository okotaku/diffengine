import numpy as np
import torch
from diffusers.models.embeddings import MultiIPAdapterImageProjection
from diffusers.utils import load_image
from PIL import Image

from diffengine.models.archs import process_ip_adapter_state_dict
from diffengine.models.editors.ip_adapter.ip_adapter_xl import (
    IPAdapterXL,
    IPAdapterXLPlus,
)
from diffengine.models.editors.ip_adapter.pipeline import (
    StableDiffusionXLPipelineTimmIPAdapter,
)
from diffengine.registry import MODELS


class TimmIPAdapterXLPlus(IPAdapterXLPlus):
    """Stable Diffusion XL IP-Adapter Plus."""

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.image_encoder = MODELS.build(self.image_encoder_config)
        self.image_encoder.dtype = "dummy"
        self.image_projection = MODELS.build(
            self.image_projection_config,
            default_args={
                "embed_dims": self.image_encoder.num_features,
                "output_dims": self.unet.config.cross_attention_dim})
        self.image_encoder.requires_grad_(requires_grad=False)
        super(IPAdapterXL, self).prepare_model()

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

        pipeline = StableDiffusionXLPipelineTimmIPAdapter.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor.pipeline,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
            hidden_states_idx=self.hidden_states_idx,
        )
        adapter_state_dict = process_ip_adapter_state_dict(
            self.unet, self.image_projection)

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        for state_dict in [adapter_state_dict]:
            image_projection_layer = (
                pipeline.unet._convert_ip_adapter_image_proj_to_diffusers(  # noqa
                    state_dict["image_proj"]))
            image_projection_layer.to(
                device=pipeline.unet.device, dtype=pipeline.unet.dtype)
            image_projection_layers.append(image_projection_layer)

        pipeline.unet.encoder_hid_proj = MultiIPAdapterImageProjection(
            image_projection_layers)
        pipeline.unet.config.encoder_hid_dim_type = "ip_image_proj"

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
            data_samples: list | None = None,  # noqa
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

        latents = self._forward_vae(inputs["img"], num_batches)

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
        image_embeds = self.image_encoder.forward_features(
            clip_img,
        )

        ip_tokens = self.image_projection(image_embeds)

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            (prompt_embeds, ip_tokens),
            added_cond_kwargs=unet_added_conditions).sample

        return self.loss(model_pred, noise, latents, timesteps, weight)
