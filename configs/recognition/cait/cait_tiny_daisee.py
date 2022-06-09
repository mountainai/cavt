_base_ = [
    '../../_base_/models/cait/cait_tiny.py', '../../_base_/default_runtime.py'
]
model=dict(backbone=dict(video_size=(8,112,112), dim=1024, patch_size=(2,14,14),
                         depth=12, heads=16, mlp_dim=2048, layer_dropout=0.05),
           cls_head=dict(type='I3DCaitHead', in_channels=1024, num_classes=1,
                         dropout_ratio=0.1),
            test_cfg = dict(average_clips='score'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/home/myuser/resource/DAiSEE/DataSet'
data_root_val = '/home/myuser/resource/DAiSEE/DataSet'
data_root_test = '/home/myuser/resource/DAiSEE/DataSet'
ann_file_train = 'data/daisee/daisee_train_validation_img_list.txt'
ann_file_val = 'data/daisee/daisee_test_img_list.txt'
ann_file_test = 'data/daisee/daisee_test_img_list.txt'
img_norm_cfg = dict(
    mean=[102.21, 77.74, 72.60], std=[52.00, 40.49, 42.07], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, clip_window_alpha=5, downsample_ratio=1),
    dict(type='RawFrameDecode'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=8, clip_window_alpha=5, downsample_ratio=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=8, clip_window_alpha=5, downsample_ratio=1, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=1,
    val_dataloader=dict(
        videos_per_gpu=8,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=8,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        filename_tmpl='frame_det_00_{:06}.bmp',
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        duplicate_times = [3, 3, 3, 3]),
    val=dict(
        type=dataset_type,
        filename_tmpl='frame_det_00_{:06}.bmp',
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        filename_tmpl='frame_det_00_{:06}.bmp',
        ann_file=ann_file_test,
        data_prefix=data_root_test,
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

