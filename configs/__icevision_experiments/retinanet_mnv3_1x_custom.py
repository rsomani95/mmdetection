_base_ = [
    # Model File
    './retinanet_mnv3_timm_fpn.py',

    # Dataset File
    './coco_detection_custom.py',

    # Optimizer, LR & Momentum Schedules
    '../_base_/schedules/schedule_1x.py',

    # Runtime - Runner, Logger
    './wandb_runtime.py',
]

_default_BATCH_SIZE_PER_GPU = 2
_default_NUM_GPUS = 8
_default_TOTAL_BATCH_SIZE = _default_BATCH_SIZE_PER_GPU * _default_NUM_GPUS
_default_LR = 0.01 # !!! RETINANET ONLY !!!

BATCH_SIZE_PER_GPU = 22
NUM_GPUS = 3
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS
LR_SCALING_FACTOR = TOTAL_BATCH_SIZE / _default_TOTAL_BATCH_SIZE
SCALED_LR = _default_LR * LR_SCALING_FACTOR

# # ============ Inplace Optimizer + Scheduler ==========================
# # Use this section if doing 1cycle learning, else don't
# # If using this section, probably best to comment out '../_base_/schedules/schedule_1x.py' from the file header

# # optimizer
# optimizer_config = dict(grad_clip=None)

# # LR policy
# lr_config = dict(
#     policy="OneCycle",
#     max_lr=SCALED_LR,
#     # pct_start=0.3,
#     pct_start=0.1,
# )

# # Momentum policy
# momentum_config = dict(
#     policy="OneCycle",
#     base_momentum=0.85,
#     max_momentum=0.95,
#     pct_start=0.3,
# )
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# # ========================================================================

# optimizer
optimizer = dict(type='SGD', lr=SCALED_LR, momentum=0.9, weight_decay=0.0001)
