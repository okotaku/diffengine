model = dict(type="PixArtAlpha", model="PixArt-alpha/PixArt-XL-2-1024-MS",
             vae_model="madebyollin/sdxl-vae-fp16-fix",
             gradient_checkpointing=True,
            unet_lora_config=dict(
                type="LoRA",
                r=8,
                lora_alpha=8,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
