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
        # -> timm.create_model args
        # model_name: str = model_name,
        pretrained: bool = True,
        out_indices: Optional[Collection[int]] = (1, 2, 3, 4),
        # -> freeze layer args
        norm_eval: bool = True,
        frozen_stages: int = 1,
        **timm_kwargs,
    ):
        super().__init__()
        # model freezing args
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        # timm args
        self.pretrained = pretrained
        self.model = timm.create_model(
            model_name="resnet50",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self._freeze_stages()

    def init_weights(self, pretrained=None):
        pass

    def _freeze_stages(self):
        # Freeze stem regardless of `freeze_stages` value
        # `freeze_stages` only refer to the bottleneck blocks
        # ACRONYMS: m => model; l => layer
        m = self.model
        for l in [m.conv1, m.bn1]:
            l.eval()
            for param in l.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            l = getattr(m, f"layer{i}")
            l.eval()
            for param in l.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(TIMMResNet50, self).train(mode)
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        return tuple(self.model(x))
