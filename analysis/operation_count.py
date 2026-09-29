"""Count inference MACs during eager execution."""

from collections import Counter
import math

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from visionhope.models import SRNL
from visionhope.models.layers import LayerNorm2d
from visionhope.utils.grn import GlobalResponseNorm


def recurrence_macs(module, x, spatial_shape=None):
    """Inference arithmetic, including padded tokens and every boundary refresh.

    The 20d term groups scalar and vector operations into a fixed per-token cost.
    """
    if x.ndim == 5:
        batch, directions, heads, length, dim = x.shape
    elif x.ndim == 3:
        batch, length, _ = x.shape
        directions, heads, dim = 1, module.num_heads, module.head_dim
    else:
        raise ValueError("SRNL profiling requires its public packed or sequence input")
    if module.chunk_size is not None:
        chunks = [module.chunk_size] * directions
    elif spatial_shape is None:
        chunks = [64] * directions
    else:
        height, width = spatial_shape
        if height * width != length:
            raise ValueError("SRNL profile shape does not match the serialized token count")
        chunks = [width if route < 2 else height for route in range(directions)]
    token_cost = 6 * dim**2 + 20 * dim
    boundary_cost = 3 * dim**3 + 2 * dim**2
    per_direction = []
    for chunk in chunks:
        boundaries = math.ceil(length / chunk)
        processed = boundaries * chunk
        per_direction.append(dict(chunk_size=chunk, real_tokens=length,
                                  processed_tokens=processed, boundaries=boundaries,
                                  macs=batch * heads * (processed * token_cost + boundaries * boundary_cost)))
    return dict(input_shape=list(x.shape), head_dim=dim, heads=heads,
                directions=per_direction,
                macs=sum(row["macs"] for row in per_direction))


def convolution_macs(x, weight, output, transposed=False):
    spatial = x.shape[2:] if transposed else output.shape[2:]
    return x.shape[0] * weight.numel() * math.prod(spatial)


class OperationCount(TorchDispatchMode):
    """Collect module and functional MAC counts during eager inference."""

    _CONVOLUTIONS = (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                     nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    _NORMS = (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
              nn.InstanceNorm2d, nn.InstanceNorm3d, LayerNorm2d)
    _ATOMIC = _CONVOLUTIONS + _NORMS + (
        nn.Linear, nn.modules.batchnorm._BatchNorm, nn.AdaptiveAvgPool2d,
        GlobalResponseNorm, SRNL,
    )

    def __init__(self, model):
        super().__init__()
        if model.training:
            raise ValueError("Inference MACs require model.eval()")
        if getattr(model, "_inference_prepared", False):
            raise ValueError("Profile an ordinary model before inference preparation")
        self.model = model
        self.counts, self.unsupported = Counter(), Counter()
        self.recurrences, self.handles = [], []
        self.atomic_depth = 0

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self._ATOMIC):
                self.handles.append(module.register_forward_pre_hook(self._before))
                self.handles.append(module.register_forward_hook(self._after(name), with_kwargs=True))
        return super().__enter__()

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            self.atomic_depth = 0

    def _before(self, module, args):
        self.atomic_depth += 1

    def _after(self, name):
        def count(module, args, kwargs, output):
            self.atomic_depth -= 1
            if self.atomic_depth:
                return
            x = args[0]
            if isinstance(module, SRNL):
                row = recurrence_macs(module, x, kwargs.get("spatial_shape"))
                self.recurrences.append(dict(name=name, **row))
                self.counts["srnl"] += row["macs"]
            elif isinstance(module, self._CONVOLUTIONS):
                self.counts["conv"] += convolution_macs(x, module.weight, output, module.transposed)
            elif isinstance(module, nn.Linear):
                self.counts["linear"] += x.numel() * module.out_features
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.counts["batch_norm"] += x.numel() * (2 if module.affine else 1)
            elif isinstance(module, self._NORMS):
                kind = "group_norm" if isinstance(module, nn.GroupNorm) else "layer_norm"
                if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    kind = "instance_norm"
                self.counts[kind] += x.numel() * (5 if module.weight is not None else 4)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                self.counts["adaptive_avg_pool2d"] += x.numel()
            elif isinstance(module, GlobalResponseNorm):
                self.unsupported["global_response_norm (reductions and elementwise arithmetic excluded)"] += 1
        return count

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        if self.atomic_depth:
            return output
        name = func._schema.name.split("::")[-1]
        result = output[0] if isinstance(output, (tuple, list)) else output
        if name in {"convolution", "_convolution"}:
            self.counts["conv"] += convolution_macs(args[0], args[1], result, args[6])
        elif name in {"mm", "bmm", "matmul"}:
            self.counts[name] += result.numel() * args[0].shape[-1]
        elif name == "addmm":
            self.counts["addmm"] += result.numel() * args[1].shape[-1]
        elif name in {"native_layer_norm", "layer_norm", "native_group_norm", "group_norm"}:
            weight_index = 2 if "layer_norm" in name else (1 if name == "native_group_norm" else 2)
            kind = "layer_norm" if "layer_norm" in name else "group_norm"
            self.counts[kind] += args[0].numel() * (5 if args[weight_index] is not None else 4)
        elif name in {"native_batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training"}:
            self.counts["batch_norm"] += args[0].numel() * (2 if args[1] is not None else 1)
        elif name in {"upsample_nearest2d", "upsample_bilinear2d", "grid_sampler_2d"}:
            kind = "grid_sampler" if name == "grid_sampler_2d" else name
            self.counts[kind] += result.numel() * (1 if name == "upsample_nearest2d" else 4)
        elif name in {"adaptive_avg_pool2d", "_adaptive_avg_pool2d"}:
            self.counts["adaptive_avg_pool2d"] += args[0].numel()
        elif name not in {
            "view", "_unsafe_view", "reshape", "permute", "transpose", "t", "slice", "select",
            "as_strided", "expand", "clone", "contiguous", "detach", "alias", "empty", "empty_like",
            "empty_strided", "new_empty", "new_zeros", "zeros", "zeros_like", "ones", "ones_like",
            "copy_", "_to_copy", "to", "lift_fresh", "lift_fresh_copy", "cat", "stack", "split",
            "split_with_sizes", "unbind", "unsqueeze", "squeeze", "index", "index_select",
            "index_put_", "nonzero", "arange", "scalar_tensor", "full", "full_like", "fill_",
            "is_pinned", "record_stream", "_local_scalar_dense", "relu", "relu_", "dropout",
        }:
            self.unsupported[f"aten::{name}"] += 1
        return output

    def report(self):
        total = sum(self.counts.values())
        return dict(macs=total, standard_ops_macs=total - self.counts["srnl"],
                    srnl_macs=self.counts["srnl"], by_operator=dict(self.counts),
                    unsupported_ops=dict(self.unsupported), srnl_components=self.recurrences,
                    convention="one multiply-add = one MAC; fvcore standard-op convention plus analytical SRNL",
                    excluded_custom_ops=["NMS", "RoIAlign", "GRN reductions", "direction fusion elementwise arithmetic"],
                    execution="actual eager outputs; no tracing, no dummy activations")
