train_pipeline = [
    dict(type="SaveImageShape"),
    dict(type="torchvision/Resize", size=1024, interpolation="bilinear"),
    dict(type="RandomCrop", size=1024),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="ComputePixArtImgInfo"),
    dict(type="torchvision/ToTensor"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="T5TextPreprocess"),
    dict(type="PackInputs",
         input_keys=["img", "text", "resolution", "aspect_ratio"]),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type="HFDataset",
        dataset="lambdalabs/pokemon-blip-captions",
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
        prompt=["yoda pokemon"] * 4,
        height=1024,
        width=1024),
    dict(type="PixArtCheckpointHook"),
]
