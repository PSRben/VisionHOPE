"""The four-stage hierarchical VisionHOPE family."""

import torch
from torch import nn

from .block import VisionHOPEBlock
from .common import ClassificationModel, initialize_weights
from .layers import Downsample, PatchEmbedding, build_norm


class HierarchicalVisionHOPE(ClassificationModel):
    """Hierarchical VisionHOPE with a projection classifier head.

    ``chunk_sizes=None`` uses row/column lengths at runtime, including rectangular
    inputs. Explicit four-element ``chunk_sizes`` select fixed chunks per stage.
    """

    def __init__(self, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256, 512),
                 mixer_dims=(32, 64, 128, 256), depths=(3, 4, 18, 4), chunk_sizes=None,
                 mlp_ratios=(4.0, 4.0, 4.0, 4.0), head_dim=16, img_size=None,
                 drop_path_rate=0.1, local_mode="gated_dwconv7", norm_layer="layernorm2d",
                 layer_scale_init_value=1e-6, use_grn=True, use_mlp_dwconv=True,
                 use_direction_scale=True, direction_merge="sum", use_block_cpe=True,
                 use_output_lepe=True, projection_dim=1024, share_direction_skip=True):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = list(embed_dims)
        self.mixer_dims = list(mixer_dims) if mixer_dims is not None else list(embed_dims)
        self.depths = list(depths)
        self.chunk_sizes = [None] * 4 if chunk_sizes is None else list(chunk_sizes)
        self.head_dim = int(head_dim)
        self.img_size = img_size
        self.share_direction_skip = share_direction_skip
        if isinstance(mlp_ratios, (int, float)):
            mlp_ratios = [float(mlp_ratios)] * 4
        if any(len(items) != 4 for items in (self.embed_dims, self.mixer_dims, self.depths,
                                            self.chunk_sizes, mlp_ratios)):
            raise ValueError("Hierarchical VisionHOPE requires four stage configurations")
        if min(self.depths) < 1 or min(self.embed_dims) < 1:
            raise ValueError("Stage depths and widths must be positive")

        self.downsample_layers = nn.ModuleList([PatchEmbedding(in_chans, embed_dims[0])])
        for i in range(3):
            self.downsample_layers.append(Downsample(embed_dims[i], embed_dims[i + 1]))
        rates = [value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        offset = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                VisionHOPEBlock(
                    dim=embed_dims[i], head_dim=head_dim, chunk_size=self.chunk_sizes[i],
                    mlp_ratio=mlp_ratios[i], drop_path=rates[offset + j],
                    mixer_dim=self.mixer_dims[i], local_mode=local_mode, norm_layer=norm_layer,
                    layer_scale_init_value=layer_scale_init_value, use_grn=use_grn,
                    use_mlp_dwconv=use_mlp_dwconv, use_direction_scale=use_direction_scale,
                    direction_merge=direction_merge, use_block_cpe=use_block_cpe,
                    use_output_lepe=use_output_lepe, share_direction_skip=share_direction_skip,
                ) for j in range(depths[i])
            ]))
            offset += depths[i]
        self.norm = build_norm(embed_dims[-1], norm_layer)
        self.head_pre_logits = nn.Conv2d(embed_dims[-1], projection_dim, 1)
        self.head_norm = nn.BatchNorm2d(projection_dim)
        self.head_act = nn.SiLU()
        self.num_features = int(projection_dim)
        self.head = nn.Linear(self.num_features, num_classes)
        self.grad_checkpointing = False
        self.apply(initialize_weights)

    def forward_intermediates(self, x):
        """Return the four stage features for detection and segmentation heads."""
        outputs = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = self._run_blocks(stage, downsample(x))
            outputs.append(x)
        return tuple(outputs)

    def forward_features(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = self._run_blocks(stage, downsample(x))
        x = self.norm(x)
        x = self.head_pre_logits(x)
        x = self.head_norm(x)
        x = self.head_act(x)
        return x.mean(dim=[2, 3])
