_base_ = [
    "../../configs/_base_/models/stable_diffusion_v15_lora.py",
    "_base_/zunko_dreambooth.py",
    "../../configs/_base_/schedules/stable_diffusion_1k.py",
    "../../configs/_base_/default_runtime.py",
]

model = dict(model="stablediffusionapi/anything-v5")

train_dataloader = dict(
    dataset=dict(
        class_image_config=dict(model={{_base_.model.model}}),
        instance_prompt="1girl, sks"))

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["1girl, sks, in a bucket"] * 4,
        by_epoch=False,
        interval=100),
    dict(type="LoRASaveHook"),
]
