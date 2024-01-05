train_pipeline = [
    dict(type="torchvision/Resize", size=512, interpolation="bilinear"),
    dict(type="RandomCrop", size=512),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="RandomChoice",
         transforms=[
            [dict(type="RandomChoice",
                transforms=[
                    [dict(
                        type="LoadMask",
                        mask_mode="irregular",
                        mask_config=dict(
                            num_vertices=(4, 10),
                            max_angle=6.0,
                            length_range=(20, 200),
                            brush_width=(10, 100),
                            area_ratio_range=(0.15, 0.65)))],
                    [dict(
                        type="LoadMask",
                        mask_mode="irregular",
                        mask_config=dict(
                            num_vertices=(1, 5),
                            max_angle=6.0,
                            length_range=(40, 450),
                            brush_width=(20, 250),
                            area_ratio_range=(0.15, 0.65)))],
                    [dict(
                        type="LoadMask",
                        mask_mode="irregular",
                        mask_config=dict(
                            num_vertices=(4, 70),
                            max_angle=6.0,
                            length_range=(15, 100),
                            brush_width=(5, 20),
                            area_ratio_range=(0.15, 0.65)))],
                    [dict(
                        type="LoadMask",
                        mask_mode="bbox",
                        mask_config=dict(
                            max_bbox_shape=(150, 150),
                            max_bbox_delta=50,
                            min_margin=0))],
                    [dict(
                        type="LoadMask",
                        mask_mode="bbox",
                        mask_config=dict(
                            max_bbox_shape=(300, 300),
                            max_bbox_delta=100,
                            min_margin=10))],
                ])],
         [dict(
                        type="LoadMask",
                        mask_mode="whole")]],
         prob=[0.9, 0.1],
    ),
    dict(type="torchvision/ToTensor"),
    dict(type="MaskToTensor"),
    dict(type="DumpImage", max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type="torchvision/Normalize", mean=[0.5], std=[0.5]),
    dict(type="GetMaskedImage"),
    dict(type="DumpMaskedImage", max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type="PackInputs",
         input_keys=["img", "mask", "masked_image", "text"]),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type="HFDreamBoothDataset",
        dataset="diffusers/dog-example",
        instance_prompt="a photo of sks dog",
        pipeline=train_pipeline,
        class_prompt=None),
    sampler=dict(type="InfiniteSampler", shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type="VisualizationHook",
        prompt=["a photo of sks dog"] * 4,
        image=["https://github.com/okotaku/diffengine/assets/24734142/8e02bd0e-9dcc-49b6-94b0-86ab3b40bc2b"] * 4,  # noqa
        mask=["https://github.com/okotaku/diffengine/assets/24734142/d0de4fb9-9183-418a-970d-582e9324f05d"] * 2 + [  # noqa
            "https://github.com/okotaku/diffengine/assets/24734142/a40d1a4f-9c47-4fa0-936e-88a49c92c8d7"] * 2,  # noqa
        by_epoch=False,
        width=512,
        height=512,
        interval=100),
    dict(type="SDCheckpointHook"),
]
