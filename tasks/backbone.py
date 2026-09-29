"""Shared dense-prediction adapter for the hierarchical VisionHOPE backbone."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from visionhope.models import create_model
from visionhope.models.layers import LayerNorm2d
from visionhope.tools.checkpoint import read_weights


class VisionHOPEBackbone(nn.Module):
    """Return normalized stage features for Mask R-CNN and UPerNet.

    ImageNet initialization excludes the classification head and output norms.
    Load complete task checkpoints through the task model, including its neck and heads.
    """

    def __init__(
        self,
        arch: str = "tiny",
        in_chans: int = 3,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        drop_path_rate: float | None = None,
        use_checkpoint: bool = False,
        out_norm: bool = True,
        channels_last: bool = True,
        frozen_stages: int = -1,
        norm_eval: bool = False,
        pretrained: str | None = None,
        init_cfg: Mapping | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if arch not in {"tiny", "small", "base"}:
            raise ValueError("arch must be 'tiny', 'small', or 'base'")
        indices = tuple(int(i) for i in out_indices)
        if not indices or len(set(indices)) != len(indices) or any(i not in range(4) for i in indices):
            raise ValueError("out_indices must contain unique stage indices in [0, 3]")
        if indices != tuple(sorted(indices)):
            raise ValueError("out_indices must be in increasing stage order")
        if frozen_stages not in range(-1, 4):
            raise ValueError("frozen_stages must be between -1 and 3")
        if init_cfg is not None and init_cfg.get("type") != "Pretrained":
            raise ValueError("Only Pretrained initialization is supported")
        if drop_path_rate is not None:
            kwargs["drop_path_rate"] = drop_path_rate
        model = create_model(f"visionhope_{arch}", in_chans=in_chans, num_classes=1000, **kwargs)
        self.arch = arch
        self.embed_dims = list(model.embed_dims)
        self.downsample_layers = model.downsample_layers
        self.stages = model.stages
        self.out_norms = nn.ModuleList(
            LayerNorm2d(dim) if out_norm else nn.Identity() for dim in self.embed_dims
        )
        self.out_indices = indices
        self.num_features = [self.embed_dims[i] for i in indices]
        self.use_checkpoint = bool(use_checkpoint)
        self.channels_last = bool(channels_last)
        self.frozen_stages = frozen_stages
        self.norm_eval = bool(norm_eval)
        self.init_cfg = init_cfg
        self.pretrained = pretrained or (init_cfg or {}).get("checkpoint")
        self._pretrained_loaded = False
        if self.pretrained:
            self.init_weights()
        self._freeze_stages()

    def init_weights(self) -> None:
        if not self.pretrained or self._pretrained_loaded:
            return
        path = Path(self.pretrained).expanduser()
        state = read_weights(path)
        backbone = OrderedDict((key, value) for key, value in state.items()
                               if key.startswith(("downsample_layers.", "stages.")))
        if hasattr(state, "_metadata"):
            backbone._metadata = OrderedDict(
                (key, value) for key, value in state._metadata.items()
                if key in {"", "downsample_layers", "stages"}
                or key.startswith(("downsample_layers.", "stages."))
            )
        incompatible = self.load_state_dict(backbone, strict=False)
        unexpected = incompatible.unexpected_keys
        missing = [name for name in incompatible.missing_keys if not name.startswith("out_norms.")]
        if missing or unexpected:
            raise RuntimeError(f"Incomplete backbone initialization: missing={missing}, unexpected={unexpected}")
        self._pretrained_loaded = True
        logging.getLogger(__name__).info("Loaded ImageNet initialization from %s", path)

    def _freeze_stages(self) -> None:
        for index in range(self.frozen_stages + 1):
            for module in (self.downsample_layers[index], self.stages[index], self.out_norms[index]):
                module.eval()
                module.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        outputs = []
        for index, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = downsample(x)
            if self.use_checkpoint and self.training:
                for block in stage:
                    x = checkpoint(block, x, use_reentrant=False)
            else:
                x = stage(x)
            if index in self.out_indices:
                outputs.append(self.out_norms[index](x))
        return tuple(outputs)
