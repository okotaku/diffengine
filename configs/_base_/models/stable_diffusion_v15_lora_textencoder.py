model = dict(
    type='StableDiffusion',
    model='runwayml/stable-diffusion-v1-5',
    lora_config=dict(rank=8),
    finetune_text_encoder=True)
