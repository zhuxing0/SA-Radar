_base_ = '../rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

data_root = '' # dataset root

train_batch_size_per_gpu = 32
train_num_workers = 4

max_epochs = 100
stage2_num_epochs = 1
base_lr = 0.00008

metainfo = {
    'classes': ('person', 'bicycle', 'car', 'bus', 'truck'),
    'palette': [
        (220, 20, 60),   # person (red)
        (0, 255, 0),     # bicycle (green)
        (0, 0, 255),     # car (blue)
        (255, 165, 0),   # bus (orange)
        (128, 0, 128)    # truck (purple)
    ]
}


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=''),
        ann_file='train_rd_box_max_isim_by_RC_multi_1.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=''),
        ann_file=data_root + '/test_rd_box_max_isim_by_RC_multi_1.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/test_rd_box_max_isim_by_RC_multi_1.json')

test_evaluator = val_evaluator

model = dict(bbox_head=dict(num_classes=5))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# load COCO pre-trained weight
load_from = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])


# with open('./configs/rtmdet/rtmdet_tiny_1xb4-20e_raddet.py', 'w') as f:
#     f.write(config_balloon)
