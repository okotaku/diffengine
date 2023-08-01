# Prepare Dataset

## Finetune 

1. Prepare `metadata.csv` and images.

The folder structure is

```
data/example
├── color.jpg
├── ...
└── metadata.csv
```

Example of `metadata.csv`.

```
file_name,text
color.jpg,"a dog"
```

2. Fix dataset config.

```
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        dataset='data/example',
        image_column='file_name'
        )
    ...
)
```

3. Run training.

## DreamBooth

1. Prepare images.

The folder structure is

```
data/example
├── dog1.jpg
├── ...
└── dog5.jpg
```

2. Fix dataset config.

```
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        dataset='data/example',
        )
    ...
)
```

3. Run training.
