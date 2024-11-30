"""
@ Author       : jonnyzhang 71881972+jonnyzhang02@users.noreply.github.com
@ LastEditTime : 2024-09-28 17:08
@ FilePath     : /mmpretrain/my_config/datasets/SAR-ACD-sbs32.py
@ 
@ coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
"""
# dataset settings
dataset_type = "CustomDataset"
data_preprocessor = dict(
    num_classes=6,
    mean=[13.8583, 13.8583, 13.8583],
    std=[27.1194, 27.1194, 27.1194],
)

# train_pipeline = [
#     dict(type="LoadImageFromFileMine"),
#     dict(type="ResizeMine", scale=224),
#     dict(type="PackInputs"),
# ]

# test_pipeline = [
#     dict(type="LoadImageFromFileMine"),
#     dict(type="ResizeMine", scale=224),
#     dict(type="PackInputs"),
# ]

train_pipeline = [
    dict(type="LoadImageFromFileMine"),
    dict(type="ResizeMine", scale=224),
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
        type=dataset_type, data_root="data/SAR-ACD/train", pipeline=train_pipeline
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type, data_root="data/SAR-ACD/test", pipeline=test_pipeline
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = [
    dict(type="AveragePrecision"),
    dict(type="MultiLabelMetricWithClasses"),
    dict(type='ConfusionMatrix')
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
