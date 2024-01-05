_base_ = [
    "../_base_/models/stable_diffusion_xl_ip_adapter_plus.py",
    "../_base_/datasets/pokemon_blip_xl_ip_adapter.py",
    "../_base_/schedules/stable_diffusion_xl_50e.py",
    "../_base_/default_runtime.py",
]

model = dict(image_encoder_sub_folder="models/image_encoder",
             pretrained_adapter="h94/IP-Adapter",
             pretrained_adapter_subfolder="sdxl_models",
             pretrained_adapter_weights_name="ip-adapter-plus_sdxl_vit-h.bin")

train_dataloader = dict(batch_size=1)

optim_wrapper = dict(accumulative_counts=4)  # update every four times

train_cfg = dict(by_epoch=True, max_epochs=10)
