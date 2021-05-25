from timm.models.mobilenetv3 import *

import torch.nn as nn
import timm
import torch
from ..builder import BACKBONES
from typing import Optional, Collection
from torch.nn.modules.batchnorm import _BatchNorm
from abc import ABCMeta, abstractmethod
from typing import Tuple, Collection, List
from torch import Tensor


class BaseMobileNetV3(nn.Module, metaclass=ABCMeta):
    """
    Base class that implements model freezing and forward methods
    """

    @abstractmethod
    def test_pretrained_weights(self):
        pass

    def post_init_setup(self):
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        self.test_pretrained_weights()

    def freeze(self, freeze_stem: bool = True, freeze_blocks: int = 1):
        "Optionally freeze the stem and/or Inverted Residual blocks of the model"
        assert 0 <= freeze_blocks <= 7, f"Can freeze 0-7 blocks only"
        m = self.model

        # Stem freezing logic
        if freeze_stem:
            for l in [m.conv_stem, m.bn1]:
                l.eval()
                for param in l.parameters():
                    param.requires_grad = False

        # `freeze_blocks=1` freezes the first block, and so on
        for i, block in enumerate(m.blocks, start=1):
            if i > freeze_blocks:
                break
            else:
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        "Convert the model to training mode while optionally freezing BatchNorm"
        super(BaseMobileNetV3, self).train(mode)
        self.freeze(
            freeze_stem=self.frozen_stem,
            freeze_blocks=self.frozen_stages,
        )
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x) -> Tuple[Tensor]:  # should return a tuple
        return tuple(self.model(x))


@BACKBONES.register_module(force=True)
class MobileNetV3Large100(BaseMobileNetV3):
    def __init__(
        self,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (1, 2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "MobileNetV3 Large with hardcoded `pretrained=True`"
        super(MobileNetV3Large100, self).__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.frozen_stem = frozen_stem
        self.model = mobilenetv3_large_100(
            pretrained=True, features_only=True, out_indices=out_indices
        )
        self.post_init_setup()

    def test_pretrained_weights(self):
        model = mobilenetv3_large_100(pretrained=True)
        assert torch.equal(self.model.conv_stem.weight, model.conv_stem.weight)


@BACKBONES.register_module(force=True)
class MobileNetV3Large100_MIIL_IN21K(BaseMobileNetV3):
    def __init__(
        self,
        pretrained: bool = True,  # doesn't matter
        out_indices: Collection[int] = (1, 2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        frozen_stem: bool = True,
    ):
        "MobileNetV3 Large (MIIL ImageNet 21k) with hardcoded `pretrained=True`"
        super(MobileNetV3Large100_MIIL_IN21K, self).__init__()
        self.norm_eval = norm_eval
        self.frozen_stem = frozen_stem
        self.frozen_stages = frozen_stages
        self.model = mobilenetv3_large_100_miil_in21k(
            pretrained=True, features_only=True, out_indices=out_indices
        )
        self.post_init_setup()

    def test_pretrained_weights(self):
        model = mobilenetv3_large_100_miil_in21k(pretrained=True)
        assert torch.equal(self.model.conv_stem.weight, model.conv_stem.weight)
