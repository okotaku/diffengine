model = dict(
    type="DeepFloydIF",
    model="DeepFloyd/IF-I-XL-v1.0",
    lora_config=dict(rank=8),
    gradient_checkpointing=True)
