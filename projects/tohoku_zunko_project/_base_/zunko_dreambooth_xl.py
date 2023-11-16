train_pipeline = [
    dict(type="torchvision/Resize", size=512, interpolation="bilinear"),
    dict(type="RandomCrop", size=512),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="torchvision/ToTensor"),
    dict(type="DumpImage", max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="PackInputs"),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type="HFDreamBoothDataset",
        dataset="data/zunko",
        instance_prompt="a photo of sks character",
        pipeline=train_pipeline),
    sampler=dict(type="InfiniteSampler", shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["A photo of sks character in a bucket"] * 4,
        by_epoch=False,
        interval=100),
    dict(type="PeftSaveHook"),
]
