train_pipeline = [
    dict(
        type="torchvision/Resize",
        size=512,
        interpolation="bilinear",
        keys=["img", "condition_img"]),
    dict(type="RandomCrop", size=512, keys=["img", "condition_img"]),
    dict(type="RandomHorizontalFlip", p=0.5, keys=["img", "condition_img"]),
    dict(type="torchvision/ToTensor", keys=["img", "condition_img"]),
    dict(type="DumpImage", max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="PackInputs", input_keys=["img", "condition_img", "text"]),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type="HFControlNetDataset",
        dataset="fusing/fill50k",
        condition_column="conditioning_image",
        caption_column="text",
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
        prompt=["cyan circle with brown floral background"] * 4,
        condition_image=[
            'https://datasets-server.huggingface.co/assets/fusing/fill50k/--/default/train/74/conditioning_image/image.jpg'  # noqa
        ] * 4),
    dict(type="ControlNetSaveHook"),
]
