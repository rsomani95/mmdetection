BATCH_SIZE = 32
EPOCHS = 12
WIDTH, HEIGHT = 640, 640

_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
]

# ======================= optimizer, fp16, LR ====================== #

fp16 = dict(loss_scale=512.0)  # Copied over blindly from `config/fp16/*`

# Scale linear rate as per the linear scaling rule shown in this paper:
# https://arxiv.org/abs/1706.02677
#
# For 16 images, the default LR is 0.02, so for 3 GPUs with 18 images
# per batch per GPU, that's:

DEFAULT_TOTAL_SAMPLES = 16  # 8 GPUs * 2 samples per GPU
DEFAULT_LR = 0.02

NUM_GPUS = 3
SAMPLES_PER_GPU = BATCH_SIZE
TOTAL_SAMPLES = SAMPLES_PER_GPU * NUM_GPUS
LR_SCALER = TOTAL_SAMPLES / DEFAULT_TOTAL_SAMPLES
LR = LR_SCALER * DEFAULT_LR  # == 0.0675 @18, 0.06 @16

#
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11] if EPOCHS == 12 else [16, 22]
    # step=[8, 11]  # 1x
    # step=[16, 22] # 1x
)

optimizer = dict(type="SGD", lr=LR, momentum=0.9, weight_decay=1e-4)
runner = dict(type="EpochBasedRunner", max_epochs=EPOCHS)

# ================================================================= #


# ====================`default_runtime` modified =================== #

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(project="mmdet-coco"),
            interval=10,
        ),
    ],
)
# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# ================================================================= #


# ======================== dataset settings ======================= #

dataset_type = "CocoDataset"
data_root = "/home/synopsis/datasets/coco/"
img_norm_cfg = dict(
    # These are ImageNet statistics, but unnormalised
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(WIDTH, HEIGHT), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        # keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(WIDTH, HEIGHT),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=BATCH_SIZE,  # RTX 3090 allows for batch size = 18 @ 640x640 for mnv3
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_train2017.json",
        img_prefix=data_root + "train2017/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")

# ================================================================= #
