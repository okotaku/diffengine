_base_ = [
    "../_base_/models/stable_diffusion_v15_lora_textencoder.py",
    "../_base_/datasets/pokemon_blip.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

model = dict(lora_config=dict(rank=32))

custom_hooks = [
    dict(type="VisualizationHook", prompt=["yoda pokemon"] * 4),
    dict(type="LoRASaveHook"),
]
