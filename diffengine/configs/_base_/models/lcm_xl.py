model = dict(
    type="LatentConsistencyModelsXL",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    loss=dict(type="HuberLoss"),
    pre_compute_text_embeddings=True,
    gradient_checkpointing=True)
