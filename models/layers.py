"""Normalization, feed-forward, embedding, and downsampling layers."""

import torch
from torch import nn
from torch.nn import functional as F
from visionhope.utils.grn import GlobalResponseNorm
from visionhope.utils.layer_norm_cuda import can_fuse_layernorm2d, visionhope_layernorm2d

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.inference_layout = False

    def forward(self, x):
        if self.inference_layout:
            return F.layer_norm(x.permute(0, 2, 3, 1), (self.weight.numel(),),
                                self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        channels_last = x.is_contiguous(memory_format=torch.channels_last)
        channels = x.shape[1]
        if can_fuse_layernorm2d(x, self.weight, self.bias):
            return visionhope_layernorm2d(x, self.weight, self.bias, self.eps)
        x = F.layer_norm(
            x.permute(0, 2, 3, 1),
            (channels,),
            self.weight,
            self.bias,
            self.eps,
        )
        x = x.permute(0, 3, 1, 2)
        if channels_last:
            return x.contiguous(memory_format=torch.channels_last)
        return x.contiguous()


def build_norm(dim, norm_layer="group"):
    if norm_layer == "group":
        return nn.GroupNorm(1, dim)
    if norm_layer == "layernorm2d":
        return LayerNorm2d(dim)
    raise ValueError(f"Unsupported norm_layer={norm_layer!r}.")


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        use_grn=False,
        use_dwconv=False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.use_dwconv = use_dwconv
        self.dwconv = (
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
            if use_dwconv
            else nn.Identity()
        )
        self.dwconv_act = act_layer() if use_dwconv else nn.Identity()
        self.grn = GlobalResponseNorm(hidden_features) if use_grn else nn.Identity()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.use_dwconv:
            x = self.dwconv_act(x + self.dwconv(x))
        x = self.grn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    """Stride-four convolutional stem for VisionHOPE."""

    def __init__(self, in_chans, embed_dim, patch_size=4):
        super().__init__()
        if patch_size != 4:
            raise ValueError("The VisionHOPE stem requires patch_size=4")
        widths = [embed_dim // 2, embed_dim // 2, embed_dim, embed_dim]
        strides = [2, 1, 2, 1]
        if min(widths) <= 0:
            raise ValueError("embed_dim is too small for the convolutional stem")
        layers = []
        for i, (width, stride) in enumerate(zip(widths, strides)):
            layers.extend([nn.Conv2d(in_chans, width, 3, stride=stride, padding=1),
                           nn.BatchNorm2d(width)])
            if i < 3:
                layers.append(nn.GELU())
            in_chans = width
        self.proj = nn.Sequential(*layers)
        self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class Downsample(nn.Module):
    """Overlapping stride-two convolution followed by batch normalization."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))
