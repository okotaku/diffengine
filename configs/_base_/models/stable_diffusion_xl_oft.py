model = dict(
    type="StableDiffusionXL",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    unet_lora_config=dict(
        type="OFT",
        r=8,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
