import torch
from diffusers import DDPMScheduler

from diffengine.models.losses import DeBiasEstimationLoss, HuberLoss, L2Loss, SNRL2Loss


def test_l2_loss():
    # test asymmetric_loss
    pred = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    gt = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.Tensor([[1], [0.1]])

    loss = L2Loss()
    assert torch.allclose(loss(pred, gt), torch.tensor(17.1667))
    assert torch.allclose(loss(pred, gt, weight=weight), torch.tensor(8.0167))


def test_snr_l2_loss():
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


def test_debias_estimation_loss():
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


def test_huber_loss():
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
