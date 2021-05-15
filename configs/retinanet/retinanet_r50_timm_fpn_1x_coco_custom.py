_base_ = [
    "../_base_/models/retinanet_timm_r50_fpn.py"
    "../_base_/datasets/coco_detection_custom.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
fp16 = dict(loss_scale=512.0)  # Copied over blindly from `config/fp16/*`

# Optimizer
# Scale linear rate as per the linear scaling rule shown in this paper:
# https://arxiv.org/abs/1706.02677
#
# For 16 images, the default LR is 0.02, so for 3 GPUs with 18 images
# per batch per GPU, that's:

DEFAULT_TOTAL_SAMPLES = 16  # 8 GPUs * 2 samples per GPU
DEFAULT_LR = 0.02

NUM_GPUS = 3
SAMPLES_PER_GPU = 32  #  defined in "../_base_/datasets/coco_detection_custom.py",
TOTAL_SAMPLES = SAMPLES_PER_GPU * NUM_GPUS
LR_SCALER = TOTAL_SAMPLES / DEFAULT_TOTAL_SAMPLES
LR = LR_SCALER * DEFAULT_LR  # == 0.0675 @18, 0.06 @16, 0.12 @ 32, and so on...
#
optimizer = dict(type="SGD", lr=LR, momentum=0.9, weight_decay=0.0001)
# find_unused_parameters = True  # DistributedDataParallel
