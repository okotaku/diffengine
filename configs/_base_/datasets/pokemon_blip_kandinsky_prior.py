train_pipeline = [
    dict(type="CLIPImageProcessor", output_key="img",
         pretrained="kandinsky-community/kandinsky-2-2-prior"),
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
    dict(type="VisualizationHook", prompt=["yoda pokemon"] * 4,
         height=512, width=512),
    dict(type="PriorSaveHook"),
]
