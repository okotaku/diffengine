from unittest import TestCase

import pytest
from diffusers import DDPMScheduler

from diffengine.models.utils import (
    CubicSamplingTimeSteps,
    EarlierTimeSteps,
    LaterTimeSteps,
    RangeTimeSteps,
    TimeSteps,
    WuerstchenRandomTimeSteps,
)


class TestTimeSteps(TestCase):

    def test_init(self):
        _ = TimeSteps()

    def test_forward(self):
        module = TimeSteps()
        scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        batch_size = 2
        timesteps = module(scheduler, batch_size, "cpu")
        assert timesteps.shape == (2,)


class TestCubicSamplingTimeSteps(TestCase):

    def test_init(self):
        _ = CubicSamplingTimeSteps()

    def test_forward(self):
        module = CubicSamplingTimeSteps()
        scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        batch_size = 2
        timesteps = module(scheduler, batch_size, "cpu")
        assert timesteps.shape == (2,)


class TestLaterTimeSteps(TestCase):

    def test_init(self):
        module = LaterTimeSteps()
        assert module.bias_multiplier == 5.
        assert module.bias_portion == 0.25

        module = LaterTimeSteps(bias_multiplier=10., bias_portion=0.5)
        assert module.bias_multiplier == 10.
        assert module.bias_portion == 0.5

        with pytest.raises(
                AssertionError, match="bias_portion must be in"):
            _ = LaterTimeSteps(bias_portion=1.1)

    def test_forward(self):
        module = LaterTimeSteps()
        scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        batch_size = 2
        timesteps = module(scheduler, batch_size, "cpu")
        assert timesteps.shape == (2,)


class TestEarlierTimeSteps(TestCase):

    def test_init(self):
        module = EarlierTimeSteps()
        assert module.bias_multiplier == 5.
        assert module.bias_portion == 0.25

        module = EarlierTimeSteps(bias_multiplier=10., bias_portion=0.5)
        assert module.bias_multiplier == 10.
        assert module.bias_portion == 0.5

        with pytest.raises(
                AssertionError, match="bias_portion must be in"):
            _ = EarlierTimeSteps(bias_portion=1.1)

    def test_forward(self):
        module = EarlierTimeSteps()
        scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        batch_size = 2
        timesteps = module(scheduler, batch_size, "cpu")
        assert timesteps.shape == (2,)


class TestRangeTimeSteps(TestCase):

    def test_init(self):
        module = RangeTimeSteps()
        assert module.bias_multiplier == 5.
        assert module.bias_begin == 0.25
        assert module.bias_end == 0.75

        module = RangeTimeSteps(bias_multiplier=10., bias_begin=0.5,
                                bias_end=1.0)
        assert module.bias_multiplier == 10.
        assert module.bias_begin == 0.5
        assert module.bias_end == 1.0

        with pytest.raises(
                AssertionError, match="bias_begin must be in"):
            _ = RangeTimeSteps(bias_begin=-1)

        with pytest.raises(
                AssertionError, match="bias_end must be in"):
            _ = RangeTimeSteps(bias_end=1.1)

        with pytest.raises(
                AssertionError, match="bias_begin must be less than bias_end"):
            _ = RangeTimeSteps(bias_begin=1, bias_end=0)

    def test_forward(self):
        module = RangeTimeSteps()
        scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        batch_size = 2
        timesteps = module(scheduler, batch_size, "cpu")
        assert timesteps.shape == (2,)


class TestWuerstchenRandomTimeSteps(TestCase):

    def test_init(self):
        _ = WuerstchenRandomTimeSteps()

    def test_forward(self):
        module = WuerstchenRandomTimeSteps()
        batch_size = 2
        timesteps = module(batch_size, "cpu")
        assert timesteps.shape == (2,)
