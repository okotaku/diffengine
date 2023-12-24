train_pipeline = [
    dict(type="CLIPImageProcessor",
         pretrained="kandinsky-community/kandinsky-2-2-prior"),
    dict(type="torchvision/Resize", size=768, interpolation="bicubic"),
    dict(type="RandomCrop", size=768),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="torchvision/ToTensor"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="PackInputs", input_keys=["img", "text", "clip_img"]),
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
    dict(type="VisualizationHook", prompt=["yoda pokemon"] * 4,
         height=512, width=512),
    dict(type="SDCheckpointHook"),
]
