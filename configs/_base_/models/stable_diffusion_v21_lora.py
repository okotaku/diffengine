model = dict(
    type="StableDiffusion",
    model="stabilityai/stable-diffusion-2-1",
    unet_lora_config=dict(
        type="LoRA",
        r=8,
        lora_alpha=1,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
