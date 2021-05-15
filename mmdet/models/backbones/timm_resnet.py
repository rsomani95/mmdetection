import torch.nn as nn
import timm
from ..builder import BACKBONES
from typing import Optional, Collection
from torch.nn.modules.batchnorm import _BatchNorm


@BACKBONES.register_module
class TIMMResNet50(nn.Module):
    """
    Wrapper for timm's "resnet50"
    """

    def __init__(
        self,
        # model_name: str = model_name,
        pretrained: bool = True,
        checkpoint_path="",
        scriptable: bool = None,
        exportable: bool = None,
        no_jit: bool = None,
        out_indices: Optional[Collection[int]] = (1, 2, 3, 4),
        # norm_eval: bool = True,
        # frozen_stages: int = 1,
        **timm_kwargs,
    ):
        super().__init__()
        timm_kwargs.update(dict(features_only=True))
        if out_indices is not None:
            timm_kwargs.update(dict(out_indices=out_indices))

        self.pretrained = pretrained
        # self.norm_eval = norm_eval
        # self.frozen_stages = frozen_stages
        self.model = timm.create_model(
            model_name="resnet50",
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            scriptable=scriptable,
            exportable=exportable,
            no_jit=no_jit,
            **timm_kwargs,
        )

    def init_weights(self, pretrained=None):
        pass

    def _freeze_stages(self):
        pass

    # def train(self):
    #     pass

    def forward(self, x):
        return tuple(self.model(x))