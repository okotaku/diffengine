import inspect
import math
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import AmusedPipeline
from mmengine import print_log
from mmengine.model import BaseModel
from peft import get_peft_model
from torch import nn

from diffengine.models.archs import create_peft_config
from diffengine.registry import MODELS


@MODELS.register_module()
class AMUSEd(BaseModel):
    """aMUSEd.

    Args:
    ----
        tokenizer (dict): Config of tokenizer.
        text_encoder (dict): Config of text encoder.
        vae (dict): Config of vae.
        transformer (dict): Config of transformer.
        model (str): pretrained model name.
            Defaults to "amused/amused-512".
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        transformer_lora_config (dict, optional): The LoRA config dict for
            Transformer.
            example. dict(type="LoRA", r=4). `type` is chosen from `LoRA`,
            `LoHa`, `LoKr`. Other config are same as the config of PEFT.
            https://github.com/huggingface/peft
            Defaults to None.
        text_encoder_lora_config (dict, optional): The LoRA config dict for
            Text Encoder. example. dict(type="LoRA", r=4). `type` is chosen
            from `LoRA`, `LoHa`, `LoKr`. Other config are same as the config of
            PEFT. https://github.com/huggingface/peft
            Defaults to None.
        prior_loss_weight (float): The weight of prior preservation loss.
            It works when training dreambooth with class images.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`SDDataPreprocessor`.
        vae_batch_size (int): The batch size of vae. Defaults to 8.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
        enable_xformers (bool): Whether or not to enable memory efficient
            attention. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: dict,
        text_encoder: dict,
        vae: dict,
        transformer: dict,
        model: str = "amused/amused-512",
        loss: dict | None = None,
        transformer_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prior_loss_weight: float = 1.,
        data_preprocessor: dict | nn.Module | None = None,
        vae_batch_size: int = 8,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": "AMUSEdPreprocessor"}
        if loss is None:
            loss = {}
        super().__init__(data_preprocessor=data_preprocessor)
        if (
            transformer_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for Transformer and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and transformer_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA Transformer, "
                "you should set text_encoder_lora_config."
            )

        self.model = model
        self.transformer_lora_config = deepcopy(transformer_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_xformers = enable_xformers
        self.vae_batch_size = vae_batch_size

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(
                loss,
                default_args={"type": "CrossEntropyLoss", "loss_weight": 1.0})
        self.loss_module: nn.Module = loss

        self.tokenizer = MODELS.build(
            tokenizer,
            default_args={"pretrained_model_name_or_path": model,
                } if not inspect.isclass(tokenizer.get("type")) else None)

        self.text_encoder = MODELS.build(
            text_encoder,
            default_args={"pretrained_model_name_or_path": model,
                } if not inspect.isclass(text_encoder.get("type")) else None)
        self.vae = MODELS.build(
            vae,
            default_args={"pretrained_model_name_or_path": model,
                } if not inspect.isclass(vae.get("type")) else None)
        self.transformer = MODELS.build(
            transformer,
            default_args={"pretrained_model_name_or_path": model,
                } if not inspect.isclass(transformer.get("type")) else None)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.mask_id = self.transformer.config.vocab_size - 1
        self.codebook_size = self.transformer.config.codebook_size

        self.prepare_model()
        self.set_lora()
        self.set_xformers()

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.text_encoder_lora_config is not None:
            text_encoder_lora_config = create_peft_config(
                self.text_encoder_lora_config)
            self.text_encoder = get_peft_model(
                self.text_encoder, text_encoder_lora_config)
            self.text_encoder.print_trainable_parameters()
        if self.transformer_lora_config is not None:
            transformer_lora_config = create_peft_config(self.transformer_lora_config)
            self.transformer = get_peft_model(self.transformer, transformer_lora_config)
            self.transformer.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        if not self.finetune_text_encoder:
            self.text_encoder.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.transformer.enable_xformers_memory_efficient_attention()
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
              num_inference_steps: int = 12,
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
                Defaults to 12.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            **kwargs: Other arguments.
        """
        pipeline = AmusedPipeline.from_pretrained(
            self.model,
            vqvae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            torch_dtype=(torch.float16 if self.device != torch.device("cpu")
                         else torch.float32),
        )
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

    def _forward_vae(self, img: torch.Tensor, num_batches: int,
                     ) -> torch.Tensor:
        """Forward vae."""
        latents = []
        for i in range(0, num_batches, self.vae_batch_size):
            latents_ = self.vae.encode(img[i : i + self.vae_batch_size]).latents
            latents_ = self.vae.quantize(latents_)[2][2].reshape(
                num_batches, -1)
            latents.append(latents_)
        return torch.cat(latents, dim=0)

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

        timesteps = torch.rand(num_batches, device=self.device)

        # mask
        seq_len = latents.shape[1]
        mask_prob = torch.cos(timesteps * math.pi * 0.5)
        mask_prob = mask_prob.clip(0.0)
        num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
        batch_randperm = torch.rand(
            num_batches, seq_len, device=self.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        input_ids = torch.where(mask, self.mask_id, latents)
        h, w = inputs["img"].shape[-2:]
        input_ids = input_ids.reshape(
            num_batches,
            h // self.vae_scale_factor,
            w // self.vae_scale_factor)
        labels = torch.where(mask, latents, -100)

        outputs = self.text_encoder(
            inputs["text"], return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        cond_embeds = outputs[0]

        logits = self.transformer(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            micro_conds=inputs["micro_conds"],
            pooled_text_emb=cond_embeds).reshape(
                num_batches, self.codebook_size, -1).permute(
                    0, 2, 1).reshape(
                        -1, self.codebook_size)

        loss_dict = dict()
        loss = self.loss_module(
                logits, labels.view(-1), weight=weight)
        loss_dict["loss"] = loss
        return loss_dict
