# dataset settings
dataset_type = "CustomDataset"
data_preprocessor = dict(
    num_classes=7,
    mean=[35.1261, 35.1261, 35.1261],
    std=[45.1871, 45.1871, 45.1871],
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=224),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=224),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type, data_root="data/FUSAR/train", pipeline=train_pipeline
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type, data_root="data/FUSAR/test", pipeline=test_pipeline
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = [
    dict(type="AveragePrecision"),
    dict(type="MultiLabelMetric"),
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
