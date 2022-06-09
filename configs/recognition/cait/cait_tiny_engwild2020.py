_base_ = [
    '../../_base_/models/cait/cait_tiny.py', '../../_base_/default_runtime.py'
]
model=dict(backbone=dict(video_size=(32,112,112), dim=1024, patch_size=(2,14,14),
                         depth=12, heads=16, mlp_dim=2048, layer_dropout=0.05),
           cls_head=dict(type='I3DCaitHead', in_channels=1024, num_classes=1,
                         dropout_ratio=0.1),
           test_cfg=dict(average_clips='score'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/home/ai/resource/EngWild2020/Train_112_scale1_frames'
data_root_val = '/home/ai/resource/EngWild2020/validation_112_scale1_frames'
ann_file_train = 'data/engwild2020/engwild2020_train_img_list.txt'
ann_file_val = 'data/engwild2020/engwild2020_validation_img_list.txt'
ann_file_test = 'data/engwild2020/engwild2020_validation_img_list.txt'
img_norm_cfg = dict(
    mean=[97.77, 77.93, 73.06], std=[49.19, 41.64, 40.64], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, clip_window_alpha=3, downsample_ratio=5),
    dict(type='RawFrameDecode'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=32, clip_window_alpha=3, downsample_ratio=5, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=32, clip_window_alpha=3, downsample_ratio=5, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=1,
    val_dataloader=dict(
        videos_per_gpu=4,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=4,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        duplicate_times = [4, 4, 4, 4]),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['mse'])

# optimizer
optimizer = dict(
    type='Adam',
    lr=0.00001,  # this lr is used for 8 gpus
    )

lr_config = dict(policy='step', step=[40, 80])
total_epochs = 20

# runtime settings
checkpoint_config = dict(interval=1)
find_unused_parameters = False

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

