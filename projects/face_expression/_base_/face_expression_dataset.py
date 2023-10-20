train_pipeline = [
    dict(type="torchvision/Resize", size=512, interpolation="bilinear"),
    dict(type="RandomCrop", size=512),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="torchvision/ToTensor"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="PackInputs"),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type="HFDataset",
        dataset="data/ExpressionTraining",
        pipeline=train_pipeline,
        image_column="file_name"),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=[
            "1girl, >_<, blue hair",
            "1girl, X X, blue hair",
            "1girl, @_@, blue hair",
            "1girl, =_=, blue hair",
        ]),
    dict(type="LoRASaveHook"),
]
