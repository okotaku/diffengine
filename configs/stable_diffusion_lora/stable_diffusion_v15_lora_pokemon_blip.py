_base_ = [
    "../_base_/models/stable_diffusion_v15_lora.py",
    "../_base_/datasets/pokemon_blip.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

model = {"lora_config": {"rank": 32}}

custom_hooks = [
    {
        "type": "VisualizationHook",
        "prompt": ["yoda pokemon"] * 4,
    },
    {
        "type": "LoRASaveHook",
    },
]
