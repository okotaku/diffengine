train_pipeline = [
    dict(type="SaveImageShape"),
    dict(
        type="torchvision/Resize",
        size=768,
        interpolation="bilinear",
        keys=["img", "condition_img"]),
    dict(type="RandomCrop", size=768, keys=["img", "condition_img"],
         force_same_size=False),
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
        dataset="instruction-tuning-sd/cartoonization",
        image_column="cartoonized_image",
        condition_column="original_image",
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
        prompt=["Generate a cartoonized version of the natural image"] * 4,
        condition_image=[
            'https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png'  # noqa
        ] * 4,
        height=768,
        width=768),
    dict(type="SDCheckpointHook"),
]
