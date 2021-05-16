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
        # Freeze stem regardless of `freeze_stages` value
        # `freeze_stages` only refer to the bottleneck blocks
        # ACRONYMS: m => model; l => layer
        m = self.model
        for l in [m.conv1, m.bn1, m.act1]:
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
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        return tuple(self.model(x))