dataset_type = "CustomDataset"
data_preprocessor = dict(
    num_classes=3,
    mean=[13.8583, 13.8583, 13.8583],
    std=[27.1194, 27.1194, 27.1194],
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type="LoadImageFromFileMine"),
    dict(type="ResizeMine", scale=224),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFileMine"),
    dict(type="ResizeMine", scale=224),
    dict(type="PackInputs"),
]


train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type, data_root="data/OpenSARShip/format_data/train", pipeline=train_pipeline
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type, data_root="data/OpenSARShip/format_data/test", pipeline=test_pipeline
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)
val_evaluator = [
    dict(type="AveragePrecision"),
    dict(type="MultiLabelMetricWithClasses")
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
