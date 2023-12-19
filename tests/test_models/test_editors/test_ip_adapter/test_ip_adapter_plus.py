from unittest import TestCase

import numpy as np
import pytest
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from mmengine.optim import OptimWrapper
from PIL import Image
from torch.optim import SGD
from transformers import CLIPImageProcessor

from diffengine.models.archs import process_ip_adapter_state_dict
from diffengine.models.editors import IPAdapterXLDataPreprocessor
from diffengine.models.editors import IPAdapterXLPlus as Base
from diffengine.models.losses import L2Loss


class IPAdapterXLPlus(Base):
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
            feature_extractor=CLIPImageProcessor.from_pretrained(
                self.image_encoder_name, subfolder="image_processor"),
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


class TestIPAdapterXL(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError, match="`unet_lora_config` should be None"):
            _ = IPAdapterXLPlus(
                "hf-internal-testing/tiny-stable-diffusion-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
                image_encoder_sub_folder="image_encoder",
                data_preprocessor=IPAdapterXLDataPreprocessor(),
                unet_lora_config=dict(type="dummy"))

        with pytest.raises(
                AssertionError, match="`text_encoder_lora_config` should be None"):
            _ = IPAdapterXLPlus(
                "hf-internal-testing/tiny-stable-diffusion-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
                image_encoder_sub_folder="image_encoder",
                data_preprocessor=IPAdapterXLDataPreprocessor(),
                text_encoder_lora_config=dict(type="dummy"))

        with pytest.raises(
                AssertionError,
                match="`finetune_text_encoder` should be False"):
            _ = IPAdapterXLPlus(
                "hf-internal-testing/tiny-stable-diffusion-pipe",
                image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
                image_encoder_sub_folder="image_encoder",
                data_preprocessor=IPAdapterXLDataPreprocessor(),
                finetune_text_encoder=True)

    def test_infer(self):
        StableDiffuser = IPAdapterXLPlus(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            image_encoder_sub_folder="image_encoder",
            data_preprocessor=IPAdapterXLDataPreprocessor())

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

        # test infer with negative_prompt
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            negative_prompt="noise",
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # output_type = 'latent'
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            output_type="latent",
            height=64,
            width=64)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (4, 32, 32)

    def test_train_step(self):
        # test load with loss module
        StableDiffuser = IPAdapterXLPlus(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            image_encoder_sub_folder="image_encoder",
            loss=L2Loss(),
            data_preprocessor=IPAdapterXLDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                clip_img=[torch.zeros((3, 32, 32))],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        StableDiffuser = IPAdapterXLPlus(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            image_encoder_sub_folder="image_encoder",
            loss=L2Loss(),
            data_preprocessor=IPAdapterXLDataPreprocessor(),
            gradient_checkpointing=True)

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                text=["a dog"],
                clip_img=[torch.zeros((3, 32, 32))],
                time_ids=[torch.zeros((1, 6))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = IPAdapterXLPlus(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            image_encoder="hf-internal-testing/unidiffuser-diffusers-test",
            image_encoder_sub_folder="image_encoder",
            loss=L2Loss(),
            data_preprocessor=IPAdapterXLDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
