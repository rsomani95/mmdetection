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
        # out_indices: Optional[Collection[int]] = (1, 2, 3, 4),
        norm_eval: bool = True,
        frozen_stages: int = 1,
        **timm_kwargs,
    ):
        super().__init__()
        # timm_kwargs.update(dict(features_only=True))
        # if out_indices is not None:
        #     timm_kwargs.update(dict(out_indices=out_indices))

        self.pretrained = pretrained
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        model = timm.create_model(
            model_name="resnet50",
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            scriptable=scriptable,
            exportable=exportable,
            no_jit=no_jit,
            **timm_kwargs,
        )
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return tuple(features)  # List/tuple of 4 feature maps

    def init_weights(self, pretrained=None):
        pass

    def _freeze_stages(self):
        # Freeze stem regardless of `freeze_stages` value
        # `freeze_stages` only refer to the bottleneck blocks
        # ACRONYMS: m => model; l => layer
        # m = self.model
        m = self
        m.conv1.eval()
        m.bn1.eval()
        for l in [m.conv1, m.bn1]:
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
