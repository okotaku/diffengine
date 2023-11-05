train_pipeline = [
    dict(type="torchvision/Resize", size=768, interpolation="bilinear"),
    dict(type="RandomCrop", size=768),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="torchvision/ToTensor"),
    dict(type="torchvision/Normalize", mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),
    dict(type="PackInputs"),
]
train_dataloader = dict(
    batch_size=4,
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
    dict(type="VisualizationHook", prompt=["A robot pokemon, 4k photo"] * 4,
         height=768, width=768),
    dict(type="WuerstchenSaveHook"),
]
