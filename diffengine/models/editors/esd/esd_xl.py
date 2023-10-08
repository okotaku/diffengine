from copy import deepcopy
from typing import Optional, Union

import torch
from torch import nn

from diffengine.models.editors.stable_diffusion_xl import StableDiffusionXL
from diffengine.models.losses.snr_l2_loss import SNRL2Loss
from diffengine.registry import MODELS


@MODELS.register_module()
class ESDXL(StableDiffusionXL):
    """Stable Diffusion XL Erasing Concepts from Diffusion Models.

    Args:
        height (int): Image height. Defaults to 1024.
        width (int): Image width. Defaults to 1024.
        negative_guidance (float): Negative guidance for loss. Defaults to 1.0.
        train_method (str): Training method. Choice from `full`, `xattn`,
            `noxattn`, `selfattn`. Defaults to `full`
    """

    def __init__(self,
                 *args,
                 finetune_text_encoder: bool = False,
                 pre_compute_text_embeddings: bool = True,
                 height: int = 1024,
                 width: int = 1024,
                 negative_guidance: float = 1.0,
                 train_method: str = 'full',
                 data_preprocessor: Optional[Union[dict, nn.Module]] = dict(
                     type='ESDXLDataPreprocessor'),
                 **kwargs):
        assert not finetune_text_encoder, \
            '`finetune_text_encoder` should be False when training ESDXL'
        assert pre_compute_text_embeddings, \
            '`pre_compute_text_embeddings` should be True when training ESDXL'
        assert train_method in ['full', 'xattn', 'noxattn', 'selfattn']

        self.height = height
        self.width = width
        self.negative_guidance = negative_guidance
        self.train_method = train_method

        super().__init__(
            *args,
            finetune_text_encoder=finetune_text_encoder,
            pre_compute_text_embeddings=pre_compute_text_embeddings,
            data_preprocessor=data_preprocessor,
            **kwargs)

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.lora_config is None:
            self.orig_unet = deepcopy(self.unet).requires_grad_(False)
        super().prepare_model()
        self._freeze_unet()

    def _freeze_unet(self) -> None:
        for name, module in self.unet.named_modules():
            if self.train_method == 'xattn' and 'attn2' not in name:
                module.eval()
            elif self.train_method == 'selfattn' and 'attn1' not in name:
                module.eval()
            elif self.train_method == 'noxattn' and ('attn2' in name
                                                     or 'time_embed' in name or
                                                     name.startswith('out.')):
                module.eval()

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ESDXL, self).train(mode)
        self._freeze_unet()

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'loss'):
        assert mode == 'loss'
        timesteps = torch.randint(1, 49, (1, ), device=self.device)
        timesteps = timesteps.long()

        latents = self.infer(
            prompt=inputs['text'],
            height=self.height,
            width=self.width,
            num_inference_steps=50,
            denoising_end=timesteps[0].item() / 50,
            output_type='latent',
            guidance_scale=3)[0].unsqueeze(0)
        # train mode after inference
        self.train()

        timesteps = torch.randint(
            round(timesteps[0].item() / 50 *
                  self.scheduler.config.num_train_timesteps),
            round((timesteps[0].item() + 1) / 50 *
                  self.scheduler.config.num_train_timesteps), (1, ),
            device=self.device).long()

        prompt_embeds = inputs['prompt_embeds']
        pooled_prompt_embeds = inputs['pooled_prompt_embeds']
        null_prompt_embeds = inputs['null_prompt_embeds']
        null_pooled_prompt_embeds = inputs['null_pooled_prompt_embeds']
        time_ids = torch.Tensor(
            [[self.height, self.width, 0, 0, self.height,
              self.width]]).long().to(self.device)
        unet_added_conditions = {
            'time_ids': time_ids,
            'text_embeds': pooled_prompt_embeds
        }
        null_unet_added_conditions = {
            'time_ids': time_ids,
            'text_embeds': null_pooled_prompt_embeds
        }

        with torch.no_grad():
            if self.lora_config is None:
                null_model_pred = self.orig_unet(
                    latents,
                    timesteps,
                    null_prompt_embeds,
                    added_cond_kwargs=null_unet_added_conditions).sample
                orig_model_pred = self.orig_unet(
                    latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions).sample
            else:
                # scale=0 means not using the lora model.
                null_model_pred = self.unet(
                    latents,
                    timesteps,
                    null_prompt_embeds,
                    added_cond_kwargs=null_unet_added_conditions,
                    cross_attention_kwargs=dict(scale=0)).sample
                orig_model_pred = self.unet(
                    latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    cross_attention_kwargs=dict(scale=0)).sample

        model_pred = self.unet(
            latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions).sample

        loss_dict = dict()
        # calculate loss in FP32
        null_model_pred.requires_grad = False
        orig_model_pred.requires_grad = False
        gt = null_model_pred - self.negative_guidance * (
            orig_model_pred - null_model_pred)
        if isinstance(self.loss_module, SNRL2Loss):
            loss = self.loss_module(model_pred.float(), gt.float(), timesteps,
                                    self.scheduler.alphas_cumprod)
        else:
            loss = self.loss_module(model_pred.float(), gt.float())
        loss_dict['loss'] = loss
        return loss_dict
