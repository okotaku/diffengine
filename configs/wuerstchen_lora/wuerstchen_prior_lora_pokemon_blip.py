_base_ = [
    "../_base_/models/wuerstchen_prior_lora.py",
    "../_base_/datasets/pokemon_blip_wuerstchen.py",
    "../_base_/schedules/stable_diffusion_50e.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(
    batch_size=8,
)

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))

custom_hooks = [
    dict(type="VisualizationHook", prompt=["A robot pokemon, 4k photo"] * 4,
         height=768, width=768),
    dict(type="PeftSaveHook"),
]
