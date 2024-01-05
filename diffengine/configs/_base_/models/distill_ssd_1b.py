model = dict(
    type="SSD1B",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    student_model="segmind/SSD-1B",
    student_model_weight="unet",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    gradient_checkpointing=True)
