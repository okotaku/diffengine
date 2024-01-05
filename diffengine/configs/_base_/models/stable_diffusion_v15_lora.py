model = dict(
    type="StableDiffusion",
    model="runwayml/stable-diffusion-v1-5",
    unet_lora_config=dict(
        type="LoRA",
        r=8,
        lora_alpha=8,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
