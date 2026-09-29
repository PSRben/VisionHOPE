"""A reusable VisionHOPE block with its two residual branches."""

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_

from .layers import FeedForwardNetwork, build_norm
from .operator import VisionHOPEOperator
from visionhope.utils.residual import can_fuse_residual_add, visionhope_residual_add


class VisionHOPEBlock(nn.Module):
    """VisionHOPE block accepting and returning NCHW features.

    ``mixer_dim`` sets the operator width and defaults to ``dim``.
    """

    def __init__(self, dim, head_dim=16, chunk_size=None, mlp_ratio=4.0,
                 drop_path=0.0, mixer_dim=None, local_mode="gated_dwconv7",
                 norm_layer="layernorm2d", layer_scale_init_value=1e-6,
                 use_grn=True, use_mlp_dwconv=True, use_direction_scale=True,
                 direction_merge="sum", use_block_cpe=True, use_output_lepe=True,
                 share_direction_skip=True):
        super().__init__()
        mixer_dim = int(mixer_dim) if mixer_dim is not None else dim
        if local_mode not in {"gated_dwconv7", "residual_dwconv", "silu_dwconv"}:
            raise ValueError("Unsupported local_mode")
        self.dim = dim
        self.mixer_dim = mixer_dim
        self.local_mode = local_mode
        self._inference_prepared = False
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) if use_block_cpe else None
        self.norm1 = build_norm(dim, norm_layer)

        self.proj_in = nn.Conv2d(dim, mixer_dim, 1)
        if use_direction_scale and direction_merge == "sum":
            direction_merge = "scale"
        self.operator = VisionHOPEOperator(mixer_dim, head_dim, chunk_size, direction_merge,
                                          share_direction_skip, _initialize=False)
        local_kernel = 7 if local_mode == "gated_dwconv7" else 3
        self.local_conv = nn.Conv2d(mixer_dim, mixer_dim, local_kernel,
                                    padding=local_kernel // 2, groups=mixer_dim)
        self.local_gate = nn.Conv2d(mixer_dim, mixer_dim, 1) if local_mode == "gated_dwconv7" else None
        self.local_act = nn.SiLU() if local_mode == "silu_dwconv" else nn.Identity()
        self.local_position = nn.Conv2d(mixer_dim, mixer_dim, 5, padding=2, groups=mixer_dim) if use_output_lepe else None
        self.proj_out = nn.Linear(mixer_dim, dim)
        self.operator.reset_parameters()
        trunc_normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm(dim, norm_layer)
        self.ffn = FeedForwardNetwork(dim, int(dim * mlp_ratio), use_grn=use_grn,
                                     use_dwconv=use_mlp_dwconv)
        if layer_scale_init_value is not None and layer_scale_init_value > 0.0:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1))
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def _local_features(self, x):
        x_inner = self.proj_in(x)
        if self.local_mode == "gated_dwconv7":
            return x_inner + torch.sigmoid(self.local_gate(x_inner)) * self.local_conv(x_inner)
        x_conv = self.local_act(self.local_conv(x_inner))
        return x_conv if self.local_mode == "silu_dwconv" else x_inner + x_conv

    def spatial_branch(self, x):
        channels_last = x.is_contiguous(memory_format=torch.channels_last)
        x_local = self._local_features(x)
        if self._inference_prepared:
            path, _ = self.operator._inference_path(x_local)
            if path != "ordinary":
                out = self.operator._forward_fast(x_local, path)
                if self.local_position is not None:
                    out = out + self.local_position(x_local).permute(0, 2, 3, 1)
                out = self.proj_out(out)
                return out.permute(0, 3, 1, 2).contiguous()

        out = self.operator(x_local, channels_last=channels_last)
        if self.local_position is not None:
            out = out + self.local_position(x_local)
        out = F.conv2d(out, self.proj_out.weight[:, :, None, None], self.proj_out.bias)
        if channels_last:
            return out.contiguous(memory_format=torch.channels_last)
        return out.contiguous()

    def _residual_add(self, x, branch, gamma):
        if gamma is None:
            return x + self.drop_path(branch)
        if can_fuse_residual_add(x, branch, gamma):
            return visionhope_residual_add(x, branch, gamma,
                getattr(self.drop_path, "drop_prob", 0.0), self.training,
                getattr(self.drop_path, "scale_by_keep", True))
        return x + self.drop_path(gamma * branch)

    def forward(self, x):
        if self.training and self._inference_prepared:
            raise RuntimeError("A block with folded LayerScale weights cannot be trained")
        if self.cpe is not None:
            x = x + self.cpe(x)
        branch = self.spatial_branch(self.norm1(x))
        x = self._residual_add(x, branch, self.gamma1)
        branch = self.ffn(self.norm2(x))
        return self._residual_add(x, branch, self.gamma2)
