_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    './coco_detection_custom.py',
    '../_base_/schedules/schedule_1x.py',
    # '../_base_/default_runtime.py'
    './wandb_runtime.py'
]

_default_BATCH_SIZE_PER_GPU = 2
_default_NUM_GPUS = 8
_default_TOTAL_BATCH_SIZE = _default_BATCH_SIZE_PER_GPU * _default_NUM_GPUS
_default_LR = 0.01 # !!! RETINANET ONLY !!!

BATCH_SIZE_PER_GPU = 10
NUM_GPUS = 3
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS
LR_SCALING_FACTOR = TOTAL_BATCH_SIZE / _default_TOTAL_BATCH_SIZE
SCALED_LR = _default_LR * LR_SCALING_FACTOR

# optimizer
optimizer = dict(type='SGD', lr=SCALED_LR, momentum=0.9, weight_decay=0.0001)
