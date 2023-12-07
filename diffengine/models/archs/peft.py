from peft import LoHaConfig, LoKrConfig, LoraConfig, OFTConfig, PeftConfig


def create_peft_config(config) -> PeftConfig:
    """Create a PEFT config from a DiffEngine config.

    Args:
    ----
        config: DiffEngine config.
    """
    peft_type = config.pop("type", "LoRA")
    assert peft_type in ["LoRA", "LoHa", "LoKr", "OFT"], \
        f"Unknown PEFT type {peft_type}"

    if peft_type == "LoRA":
        return LoraConfig(**config)
    if peft_type == "LoHa":
        return LoHaConfig(**config)
    if peft_type== "LoKr":
        return LoKrConfig(**config)
    if peft_type == "OFT":
        return OFTConfig(**config)

    return None
