_base_ = [
    "../_base_/models/stable_diffusion_xl.py",
    "../_base_/datasets/pokemon_blip_xl.py",
    "../_base_/schedules/stable_diffusion_xl_50e.py",
    "../_base_/default_runtime.py",
]

model = dict(noise_generator=dict(type="PyramidNoise", discount=0.9))

train_dataloader = dict(batch_size=1)

optim_wrapper_cfg = dict(accumulative_counts=4)  # update every four times
