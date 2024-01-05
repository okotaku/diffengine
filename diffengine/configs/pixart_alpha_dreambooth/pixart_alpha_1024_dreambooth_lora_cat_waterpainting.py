_base_ = [
    "../_base_/models/pixart_alpha_1024_lora.py",
    "../_base_/datasets/cat_waterpainting_dreambooth_pixart_1024.py",
    "../_base_/schedules/stable_diffusion_1k.py",
    "../_base_/default_runtime.py",
]

model = dict(transformer_lora_config=dict(r=64, lora_alpha=64))

train_dataloader = dict(
    dataset=dict(class_image_config=dict(model={{_base_.model.model}})))

optim_wrapper = dict(
    dtype="bfloat16",
    optimizer=dict(lr=1e-4))
