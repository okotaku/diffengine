from unittest import TestCase

import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import SGD

from diffengine.models.editors import ESDXL, ESDXLDataPreprocessor
from diffengine.models.losses import L2Loss


class TestESDXL(TestCase):

    def test_init(self):
        with pytest.raises(
                AssertionError,
                match="`pre_compute_text_embeddings` should be True"):
            _ = ESDXL(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                data_preprocessor=ESDXLDataPreprocessor(),
                pre_compute_text_embeddings=False)

        with pytest.raises(
                AssertionError,
                match="`finetune_text_encoder` should be False"):
            _ = ESDXL(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                data_preprocessor=ESDXLDataPreprocessor(),
                finetune_text_encoder=True)

        with pytest.raises(
                AssertionError, match="`prediction_type` should be None"):
            _ = ESDXL(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                data_preprocessor=ESDXLDataPreprocessor(),
                prediction_type="v_prediction")

    def test_infer(self):
        StableDiffuser = ESDXL(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            data_preprocessor=ESDXLDataPreprocessor())

        assert not hasattr(StableDiffuser, "tokenizer_one")
        assert not hasattr(StableDiffuser, "text_encoder_one")
        assert not hasattr(StableDiffuser, "tokenizer_two")
        assert not hasattr(StableDiffuser, "text_encoder_two")

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

    def test_train_step_with_pre_compute_embs(self):
        # test load with loss module
        StableDiffuser = ESDXL(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            train_method="xattn",
            height=64,
            width=64,
            loss=L2Loss(),
            data_preprocessor=ESDXLDataPreprocessor())

        # test train step
        data = dict(
            inputs=dict(
                text=["dog"],
                prompt_embeds=[torch.zeros((2, 64))],
                pooled_prompt_embeds=[torch.zeros(32)],
                null_prompt_embeds=[torch.zeros((2, 64))],
                null_pooled_prompt_embeds=[torch.zeros(32)]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        StableDiffuser = ESDXL(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            loss=L2Loss(),
            data_preprocessor=ESDXLDataPreprocessor())

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
