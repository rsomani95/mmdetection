import torch.nn as nn
import timm
from ..builder import BACKBONES
from typing import Optional, Collection
from torch.nn.modules.batchnorm import _BatchNorm


@BACKBONES.register_module
class MobileNetV3AABlocks(nn.Module):
    """
    Wrapper for timm's "mobilenetv3_large_100_aa". Current (as of 14 May 2021),
    this is under a PR and thus you need to install `timm` from this fork:
      https://github.com/rsomani95/pytorch-image-models/tree/aa-effnets
    """

    def __init__(
        self,
        # model_name: str = model_name,
        pretrained: bool = True,
        checkpoint_path="",
        scriptable: bool = None,
        exportable: bool = None,
        no_jit: bool = None,
        out_indices: Optional[Collection[int]] = (0, 1, 2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        **timm_kwargs,
    ):
        super().__init__()
        timm_kwargs.update(dict(features_only=True))
        if out_indices is not None:
            timm_kwargs.update(dict(out_indices=out_indices))

        self.pretrained = pretrained
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.model = timm.create_model(
            model_name="mobilenetv3_large_100_aa",
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
        m = self.model

        # Freeze the stem
        if self.frozen_stages > 0:
            for layer in [m.conv_stem, m.bn1, m.act1]:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze bottleneck blocks
        for i in range(1, self.frozen_stages - 1):
            layer = m.blocks[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        "Convert the model into training mode"
        super(MobileNetV3AABlocks, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.model.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        return self.model(x)
