model = dict(type="PixArtAlpha", model="PixArt-alpha/PixArt-XL-2-512x512",
             vae_model="stabilityai/sd-vae-ft-ema",
             gradient_checkpointing=True,
             transformer_lora_config=dict(
                type="LoRA",
                r=8,
                lora_alpha=8,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
