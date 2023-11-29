model = dict(
    type="WuerstchenPriorModel",
    decoder_model="warp-ai/wuerstchen",
    prior_model="warp-ai/wuerstchen-prior",
    prior_lora_config=dict(
        type="LoRA",
        r=8,
        lora_alpha=1,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]))