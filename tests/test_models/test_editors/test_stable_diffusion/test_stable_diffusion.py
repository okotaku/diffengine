from diffengine.models.editors import StableDiffusion


def test_stable_diffusion():
    StableDiffuser = StableDiffusion('runwayml/stable-diffusion-v1-5')

    result = StableDiffuser.infer(
        ['an insect robot preparing a delicious meal'], height=64, width=64)
    assert len(result) == 1
    assert result[0].shape == (64, 64, 3)
