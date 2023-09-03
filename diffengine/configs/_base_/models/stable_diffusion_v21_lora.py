from diffengine.models.editors import StableDiffusion

model = dict(
    type=StableDiffusion,
    model='stabilityai/stable-diffusion-2-1',
    lora_config=dict(rank=8))
