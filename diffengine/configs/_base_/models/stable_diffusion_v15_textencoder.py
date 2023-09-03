from diffengine.models.editors import StableDiffusion

model = dict(
    type=StableDiffusion,
    model='runwayml/stable-diffusion-v1-5',
    finetune_text_encoder=True)
