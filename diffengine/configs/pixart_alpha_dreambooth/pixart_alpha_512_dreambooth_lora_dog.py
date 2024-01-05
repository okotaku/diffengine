_base_ = [
    "../_base_/models/pixart_alpha_512_lora.py",
    "../_base_/datasets/dog_dreambooth_pixart_512.py",
    "../_base_/schedules/stable_diffusion_500.py",
    "../_base_/default_runtime.py",
]

train_dataloader = dict(
    dataset=dict(class_image_config=dict(model={{_base_.model.model}})))

optim_wrapper = dict(
    dtype="bfloat16",
    optimizer=dict(lr=1e-4))
