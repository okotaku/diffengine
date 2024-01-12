import pytest
import torch
from diffusers import DDPMScheduler

from diffengine.models.losses import (
    CrossEntropyLoss,
    DeBiasEstimationLoss,
    HuberLoss,
    L2Loss,
    SNRL2Loss,
)


def test_l2_loss():
    with pytest.raises(
            AssertionError, match="reduction should be 'mean' or 'none'"):
        _ = L2Loss(reduction="dummy")

    # test asymmetric_loss
    pred = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    gt = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.Tensor([[1], [0.1]])

    loss = L2Loss()
    assert torch.allclose(loss(pred, gt), torch.tensor(17.1667))
    assert torch.allclose(loss(pred, gt, weight=weight), torch.tensor(8.0167))

    loss = L2Loss(reduction="none")
    assert loss(pred, gt).shape == (2, 3)


def test_snr_l2_loss():
    with pytest.raises(
            AssertionError, match="reduction should be 'mean' or 'none'"):
        _ = SNRL2Loss(reduction="dummy")

    # test asymmetric_loss
    pred = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    gt = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.Tensor([[1], [0.1]])
    timesteps = (torch.ones((pred.shape[0], )) + 10)
    scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    loss = SNRL2Loss()
    assert torch.allclose(
        loss(pred, gt, timesteps.long(), scheduler.alphas_cumprod,
             scheduler.config.prediction_type),
        torch.tensor(0.9075),
        rtol=1e-04,
        atol=1e-04)

    # test with weight
    assert torch.allclose(
        loss(
            pred,
            gt,
            timesteps.long(),
            scheduler.alphas_cumprod,
            scheduler.config.prediction_type,
            weight=weight),
        torch.tensor(0.4991),
        rtol=1e-04,
        atol=1e-04)

    # test velocity objective
    assert torch.allclose(
        loss(pred, gt, timesteps.long(), scheduler.alphas_cumprod,
             "v_prediction"),
        torch.tensor(0.8980),
        rtol=1e-04,
        atol=1e-04)

    loss = SNRL2Loss(reduction="none")
    assert loss(pred, gt, timesteps.long(), scheduler.alphas_cumprod,
             scheduler.config.prediction_type).shape == (2,)


def test_debias_estimation_loss():
    with pytest.raises(
            AssertionError, match="reduction should be 'mean' or 'none'"):
        _ = DeBiasEstimationLoss(reduction="dummy")

    # test asymmetric_loss
    pred = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    gt = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.Tensor([[1], [0.1]])
    timesteps = (torch.ones((pred.shape[0], )) + 10)
    scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    loss = DeBiasEstimationLoss()
    assert torch.allclose(
        loss(pred, gt, timesteps.long(), scheduler.alphas_cumprod,
             scheduler.config.prediction_type),
        torch.tensor(1.7652),
        rtol=1e-04,
        atol=1e-04)

    # test with weight
    assert torch.allclose(
        loss(
            pred,
            gt,
            timesteps.long(),
            scheduler.alphas_cumprod,
            scheduler.config.prediction_type,
            weight=weight),
        torch.tensor(0.9708),
        rtol=1e-04,
        atol=1e-04)

    # test velocity objective
    assert torch.allclose(
        loss(pred, gt, timesteps.long(), scheduler.alphas_cumprod,
             "v_prediction"),
        torch.tensor(1.7559),
        rtol=1e-04,
        atol=1e-04)

    loss = DeBiasEstimationLoss(reduction="none")
    assert loss(pred, gt, timesteps.long(), scheduler.alphas_cumprod,
             scheduler.config.prediction_type).shape == (2,)


def test_huber_loss():
    with pytest.raises(
            AssertionError, match="reduction should be 'mean' or 'none'"):
        _ = HuberLoss(reduction="dummy")

    # test asymmetric_loss
    pred = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    gt = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.Tensor([[1], [0.1]])

    loss = HuberLoss()
    assert torch.allclose(loss(pred, gt), torch.tensor(3.0833),
        rtol=1e-04,
        atol=1e-04)
    assert torch.allclose(loss(pred, gt, weight=weight), torch.tensor(1.5833),
        rtol=1e-04,
        atol=1e-04)

    loss = HuberLoss(reduction="none")
    assert loss(pred, gt).shape == (2, 3)


def test_ce_loss():
    with pytest.raises(
            AssertionError, match="reduction should be 'mean' or 'none'"):
        _ = CrossEntropyLoss(reduction="dummy")

    # test asymmetric_loss
    pred = torch.Tensor([[-1000, 1000], [100, -100]])
    gt = torch.Tensor([0, 1]).long()
    weight = torch.tensor([0.6, 0.4])

    loss = CrossEntropyLoss()
    assert torch.allclose(loss(pred, gt), torch.tensor(1100.))
    assert torch.allclose(loss(pred, gt, weight=weight), torch.tensor(640.))

    loss = CrossEntropyLoss(reduction="none")
    assert loss(pred, gt).shape == (2,)
