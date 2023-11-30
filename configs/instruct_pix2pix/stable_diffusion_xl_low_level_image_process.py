_base_ = [
    "../_base_/models/stable_diffusion_xl_instruct_pix2pix.py",
    "../_base_/datasets/low_level_image_process_xl.py",
    "../_base_/schedules/stable_diffusion_3e.py",
    "../_base_/default_runtime.py",
]

model = dict(model="diffusers/sdxl-instructpix2pix-768")

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type="Adafactor",
        lr=3e-5,
        weight_decay=1e-2,
        scale_parameter=False,
        relative_step=False),
    accumulative_counts=4)

train_cfg = dict(max_epochs=10)
