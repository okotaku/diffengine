train_pipeline = [
    dict(type="SaveImageShape"),
    dict(
        type="torchvision/Resize",
        size=1024,
        interpolation="bilinear",
        keys=["img", "condition_img"]),
    dict(type="RandomCrop", size=1024, keys=["img", "condition_img"]),
    dict(type="RandomHorizontalFlip", p=0.5, keys=["img", "condition_img"]),
    dict(type="ComputeTimeIds"),
    dict(type="torchvision/ToTensor", keys=["img", "condition_img"]),
    dict(type="DumpImage", max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5],
        keys=["img", "condition_img"]),
    dict(
        type="PackInputs",
        input_keys=["img", "condition_img", "text", "time_ids"]),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type="HFControlNetDataset",
        dataset="fusing/instructpix2pix-1000-samples",
        image_column="edited_image",
        condition_column="input_image",
        caption_column="edit_prompt",
        pipeline=train_pipeline),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["make the mountains snowy"] * 4,
        condition_image=[
            'https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png'  # noqa
        ] * 4,
        height=1024,
        width=1024),
    dict(type="SDCheckpointHook"),
]
