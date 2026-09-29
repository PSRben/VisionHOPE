"""The four-direction VisionHOPE operator on a spatial feature map."""

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from .srnl import SRNL
from visionhope.utils.cuda import on_tensor_device
from visionhope.utils.direction_fusion import (
    can_fuse_scale_postprocess,
    visionhope_postprocess_scale,
)


class VisionHOPEOperator(nn.Module):
    """Serialize four scans, apply SRNL, and restore/fuse the spatial outputs.

    Input/output use logical NCHW with ``dim`` channels. ``W_q`` projects every
    spatial token exactly once; all four scans reuse those queries. The learned
    skip is shared across directions by default. Setting ``share_direction_skip``
    to False gives each direction a separate channel-wise scale.
    ``fast_inference`` selects spatial or sequence fusion in eval mode with
    gradients disabled; the operator and its SRNL share this setting.
    """

    def __init__(self, dim, head_dim=16, chunk_size=None, direction_merge="scale",
                 share_direction_skip=True, *, fast_inference=False, _initialize=True):
        super().__init__()
        if not isinstance(share_direction_skip, bool):
            raise TypeError("share_direction_skip must be a bool")
        if direction_merge not in {"sum", "scale", "softmax"}:
            raise ValueError("direction_merge must be sum, scale, or softmax")
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.share_direction_skip = share_direction_skip
        self.direction_merge = direction_merge
        self.srnl = SRNL(dim, head_dim, chunk_size, directions=4,
                         fast_inference=fast_inference, _initialize=False)
        self.num_heads = self.srnl.num_heads
        self.W_q = nn.Linear(dim, dim)
        skip_shape = (self.num_heads, head_dim) if share_direction_skip else (4, self.num_heads, head_dim)
        self.d_skip = nn.Parameter(torch.zeros(skip_shape))
        self.direction_scale = nn.Parameter(torch.ones(4, dim)) if direction_merge == "scale" else None
        self.direction_logits = nn.Parameter(torch.zeros(4, dim)) if direction_merge == "softmax" else None
        if _initialize:
            self.reset_parameters()

    @property
    def fast_inference(self):
        return self.srnl.fast_inference

    @fast_inference.setter
    def fast_inference(self, enabled):
        self.srnl.fast_inference = enabled

    def reset_parameters(self):
        self.srnl.reset_parameters()
        trunc_normal_(self.W_q.weight, std=0.02)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.d_skip)
        if self.direction_scale is not None:
            nn.init.ones_(self.direction_scale)
        if self.direction_logits is not None:
            nn.init.zeros_(self.direction_logits)

    def project_queries(self, seq_hw, height, width):
        q_hw = self.W_q(seq_hw)
        q_wh = q_hw.reshape(q_hw.shape[0], height, width, self.dim)
        q_wh = q_wh.transpose(1, 2).reshape_as(q_hw)
        return torch.stack((q_hw, q_hw.flip(1), q_wh, q_wh.flip(1)), dim=1)

    def direction_weights(self):
        if self.direction_merge == "scale":
            return self.direction_scale
        if self.direction_merge == "softmax":
            return 4.0 * torch.softmax(self.direction_logits, dim=0)
        return self.d_skip.new_ones((4, self.dim))

    @staticmethod
    def serialize(x, seq_hw=None):
        if seq_hw is None:
            seq_hw = x.flatten(2).transpose(1, 2)
        seq_wh = x.transpose(2, 3).flatten(2).transpose(1, 2)
        return torch.stack([
            seq_hw, torch.flip(seq_hw, dims=[1]), seq_wh, torch.flip(seq_wh, dims=[1])
        ], dim=1)

    def forward(self, x, *, channels_last=None):
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"VisionHOPEOperator expects [B, {self.dim}, H, W]")
        if channels_last is None:
            channels_last = x.is_contiguous(memory_format=torch.channels_last)
        path, _ = self._inference_path(x)
        if path != "ordinary":
            out = self._forward_fast(x, path).permute(0, 3, 1, 2)
            memory_format = torch.channels_last if channels_last else torch.contiguous_format
            return out.contiguous(memory_format=memory_format)
        return self._sequence_forward(x, channels_last)

    def _sequence_forward(self, x, channels_last, *, memories=None):
        batch_size, _, height, width = x.shape
        # Share the flattened view between inputs and queries.
        seq_hw = x.flatten(2).transpose(1, 2)
        xs = self.serialize(x, seq_hw)
        xc_multi = xs.view(batch_size, 4, xs.shape[2], self.num_heads, self.head_dim)
        xc_multi = xc_multi.transpose(2, 3).contiguous()
        q_raw = self.project_queries(seq_hw, height, width)
        if memories is None:
            y = self.srnl(xc_multi, q_raw, spatial_shape=(height, width))
        else:
            from .srnl_core.fused import srnl_sequence_inference

            y = srnl_sequence_inference(memories, xc_multi, q_raw, [self.srnl.chunk_size] * 4)

        weights = self.direction_weights()
        if can_fuse_scale_postprocess(y, xc_multi, self.d_skip, weights, height, width):
            return visionhope_postprocess_scale(
                y, xc_multi, self.d_skip, weights, height, width, channels_last
            )

        skip_directions = 1 if self.share_direction_skip else 4
        y = y + self.d_skip.view(1, skip_directions, self.num_heads, 1, self.head_dim) * xc_multi
        y = y.transpose(2, 3).flatten(3)
        out_hw_fwd = y[:, 0].view(batch_size, height, width, self.dim).permute(0, 3, 1, 2)
        out_hw_bwd = y[:, 1].flip(1).view(batch_size, height, width, self.dim).permute(0, 3, 1, 2)
        out_wh_fwd = y[:, 2].view(batch_size, width, height, self.dim).permute(0, 3, 2, 1)
        out_wh_bwd = y[:, 3].flip(1).view(batch_size, width, height, self.dim).permute(0, 3, 2, 1)
        if self.direction_merge in {"scale", "softmax"}:
        # Accumulate the four weighted outputs in scan order.
            scales = self.direction_weights().view(4, self.dim, 1, 1)
            out = scales[0] * out_hw_fwd
            out.add_(scales[1] * out_hw_bwd)
            out.add_(scales[2] * out_wh_fwd)
            out.add_(scales[3] * out_wh_bwd)
        else:
            out = out_hw_fwd + out_hw_bwd
            out.add_(out_wh_fwd)
            out.add_(out_wh_bwd)
        return out

    def _inference_path(self, x):
        if self.training:
            return "ordinary", "The operator is in training mode"
        query_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        path, reason = self.srnl._inference_path(x, query_dtype)
        if path == "ordinary":
            return path, reason
        height, width = x.shape[-2:]
        chunk_size = self.srnl.chunk_size
        if chunk_size is None or chunk_size == height == width:
            return "spatial", None
        return "sequence", None

    def inference_status(self, x):
        """Report path selection for a valid NCHW input in the current context."""
        path, reason = self._inference_path(x)
        return {"path": path, "reason": reason}

    @on_tensor_device
    def _forward_fast(self, x, path):
        """Inference-only fusion, returning contiguous NHWC for output projection.

        Row/column chunks use spatial fusion. Other fixed chunks use the
        sequence kernel followed by directional restoration and fusion.
        """
        if torch.jit.is_tracing():
            raise RuntimeError("SRNL TorchScript tracing is unsupported; use eager or CUDA Graph inference.")
        if x.ndim != 4 or not x.is_cuda or x.shape[1] != self.dim or min(x.shape) <= 0:
            raise ValueError(f"Fused VisionHOPE expects nonempty CUDA [B, {self.dim}, H, W]")
        height, width = x.shape[-2:]
        memories = self.srnl._validate_memories(x.device)
        skip_shape = (self.num_heads, self.head_dim) if self.share_direction_skip else (4, self.num_heads, self.head_dim)
        if self.d_skip.shape != skip_shape or self.d_skip.device != x.device:
            raise ValueError("Directional skip has an invalid shape or device")
        for scale in (self.direction_scale, self.direction_logits):
            if scale is not None and (scale.shape != (4, self.dim) or scale.device != x.device):
                raise ValueError("Directional fusion weights have an invalid shape or device")
        if path == "sequence":
            return self._sequence_forward(x, True, memories=memories).permute(0, 2, 3, 1).contiguous()
        from .srnl_core.fused import load_fused_cuda

        xs = self.serialize(x)
        q_raw = self.project_queries(x.flatten(2).transpose(1, 2), height, width)
        states = tuple(state.contiguous().float() for state in memories)
        if self.direction_merge == "softmax":
            weights = 4.0 * torch.softmax(self.direction_logits.detach().float(), dim=0)
        else:
            weights = self.direction_weights().detach().float()
        weights = weights.contiguous()
        return load_fused_cuda(xs.dtype).srnl_line_forward_seq_noatomic(
            *states, xs.contiguous(), q_raw.contiguous(), self.d_skip.contiguous().float(),
            weights, int(height), int(width), self.head_dim,
        )
