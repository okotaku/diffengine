from unittest import TestCase

import torch

from diffengine.models.utils import OffsetNoise, PyramidNoise, WhiteNoise


class TestWhiteNoise(TestCase):

    def test_init(self):
        _ = WhiteNoise()

    def test_forward(self):
        module = WhiteNoise()
        latens = torch.randn(1, 4, 16, 16)
        noise = module(latens)
        assert latens.shape == noise.shape


class TestOffsetNoise(TestCase):

    def test_init(self):
        module = OffsetNoise()
        assert module.offset_weight == 0.05

        module = OffsetNoise(offset_weight=0.2)
        assert module.offset_weight == 0.2

    def test_forward(self):
        module = OffsetNoise()
        latens = torch.randn(1, 4, 16, 16)
        noise = module(latens)
        assert latens.shape == noise.shape


class TestPyramidNoise(TestCase):

    def test_init(self):
        module = PyramidNoise()
        assert module.discount == 0.9
        assert module.random_multiplier

        module = PyramidNoise(discount=0.8, random_multiplier=False)
        assert module.discount == 0.8
        assert not module.random_multiplier

    def test_forward(self):
        module = PyramidNoise()
        latens = torch.randn(1, 4, 16, 16)
        noise = module(latens)
        assert latens.shape == noise.shape

        module = PyramidNoise(random_multiplier=False)
        latens = torch.randn(1, 4, 16, 16)
        noise = module(latens)
        assert latens.shape == noise.shape
