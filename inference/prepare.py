"""Enable CUDA inference for SRNL components and VisionHOPE models."""

import torch

from visionhope.models.block import VisionHOPEBlock
from visionhope.models.layers import LayerNorm2d
from visionhope.models.operator import VisionHOPEOperator
from visionhope.models.srnl import SRNL
from visionhope.utils.grn import GlobalResponseNorm


def prepare_for_inference(model):
    """Set evaluation mode and enable fast inference.

    Blocks fold LayerScale into projection weights in place. Use an eval-only copy.
    Standalone SRNL and operators enable fast inference without changing weights.
    Returns preparation counts by component; ``line_mixer`` counts prepared blocks.
    """
    blocks = [module for module in model.modules() if isinstance(module, VisionHOPEBlock)]
    for block in blocks:
        if type(block.operator) is not VisionHOPEOperator or type(block.operator.srnl) is not SRNL:
            raise ValueError("Production fusion requires the unmodified VisionHOPE operator and SRNL")
    model.eval()
    counts = dict(line_mixer=0, srnl=0, operator=0, layernorm2d=0, grn=0, layerscale=0)
    for module in model.modules():
        if isinstance(module, SRNL):
            module.fast_inference = True
            counts["srnl"] += 1
        elif isinstance(module, VisionHOPEOperator):
            counts["operator"] += 1
        elif isinstance(module, LayerNorm2d):
            module.inference_layout = True
            counts["layernorm2d"] += 1
        elif isinstance(module, GlobalResponseNorm):
            counts["grn"] += 1
    with torch.no_grad():
        for block in blocks:
            counts["line_mixer"] += 1
            for gamma_name, projection in (("gamma1", block.proj_out), ("gamma2", block.ffn.fc2)):
                gamma = getattr(block, gamma_name)
                if gamma is None:
                    continue
                scale = gamma.detach().view(-1).to(dtype=projection.weight.dtype,
                                                   device=projection.weight.device)
                shape = (-1,) + (1,) * (projection.weight.ndim - 1)
                projection.weight.mul_(scale.view(shape))
                if projection.bias is not None:
                    projection.bias.mul_(scale)
                setattr(block, gamma_name, None)
                counts["layerscale"] += 1
            block._inference_prepared = True
    if blocks:
        model._inference_prepared = True
    return counts
