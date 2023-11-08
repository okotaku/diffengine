model = dict(
    type="WuerstchenPriorModel",
    decoder_model="warp-ai/wuerstchen",
    prior_model="warp-ai/wuerstchen-prior",
    lora_config=dict(rank=8))
