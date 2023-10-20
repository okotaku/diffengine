train_pipeline = [
    dict(type="SaveImageShape"),
    dict(type="torchvision/Resize", size=1024, interpolation="bilinear"),
    dict(type="CenterCrop", size=1024),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="ComputeTimeIds"),
    dict(type="torchvision/ToTensor"),
    dict(type="DumpImage", max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="PackInputs", input_keys=["img", "text", "time_ids"]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
        ],
        height=1024,
        width=1024),
    dict(type="LoRASaveHook"),
]
