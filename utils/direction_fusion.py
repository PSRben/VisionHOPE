"""Restore and fuse directional SRNL outputs for FP16, BF16, and FP32 inputs.

After restoring each scan to image order, fusion combines content reads,
channel-wise skips, and direction weights:

    out = sum_d scale[d] * (y[d] + d_skip[d] * x[d])

Skip scales are shared across directions by default. A [4, num_heads, head_dim]
tensor supplies independent skip scales for each direction.
"""

from pathlib import Path

from functools import partial

from visionhope.utils.cuda import on_tensor_device

import torch
from torch.utils.cpp_extension import load_inline

try:
    from torch.amp import custom_bwd, custom_fwd

    custom_fwd_cuda = partial(custom_fwd, device_type="cuda")
    custom_bwd_cuda = partial(custom_bwd, device_type="cuda")
except ImportError:
    from torch.cuda.amp import custom_bwd as custom_bwd_cuda
    from torch.cuda.amp import custom_fwd as custom_fwd_cuda

from visionhope.utils.cuda_source import _cuda_source_io_variant


_EXTENSION_NAME = "visionhope_postprocess_scale_io"
_postprocess_cuda = None
_postprocess_cuda_fp16 = None
_postprocess_cuda_fp32 = None


cuda_source = (Path(__file__).parent / "csrc" / "direction_fusion.cu").read_text()
cuda_source_fp32 = _cuda_source_io_variant(cuda_source, torch.float32)
cuda_source_fp16 = _cuda_source_io_variant(cuda_source, torch.float16)

cpp_source = (Path(__file__).parent / "csrc" / "direction_fusion.cpp").read_text()


def load_postprocess_cuda(io_dtype=torch.bfloat16):
    global _postprocess_cuda, _postprocess_cuda_fp16, _postprocess_cuda_fp32
    if io_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"SRNL postprocess supports fp16/bf16/fp32 y/x, got {io_dtype}.")
    if io_dtype == torch.float16:
        if _postprocess_cuda_fp16 is None:
            _postprocess_cuda_fp16 = load_inline(
                name=_EXTENSION_NAME + "_fp16io",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source_fp16,
                functions=["visionhope_postprocess_scale_forward", "visionhope_postprocess_scale_backward"],
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
        return _postprocess_cuda_fp16
    if io_dtype == torch.float32:
        if _postprocess_cuda_fp32 is None:
            _postprocess_cuda_fp32 = load_inline(
                name=_EXTENSION_NAME + "_fp32io",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source_fp32,
                functions=["visionhope_postprocess_scale_forward", "visionhope_postprocess_scale_backward"],
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
        return _postprocess_cuda_fp32
    if _postprocess_cuda is None:
        _postprocess_cuda = load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["visionhope_postprocess_scale_forward", "visionhope_postprocess_scale_backward"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _postprocess_cuda


class VisionHOPEPostprocessScaleFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd_cuda
    @on_tensor_device
    def forward(ctx, y, x, d_skip, direction_scale, height, width, channels_last):
        if not y.is_cuda:
            raise RuntimeError("Fused SRNL postprocess requires CUDA tensors.")
        y_c = y.contiguous()
        x_c = x.contiguous()
        d_skip_c = d_skip.contiguous()
        scale_c = direction_scale.contiguous()
        out = load_postprocess_cuda(y_c.dtype).visionhope_postprocess_scale_forward(
            y_c, x_c, d_skip_c, scale_c, int(height), int(width),
            bool(channels_last), False
        )
        ctx.save_for_backward(y_c, x_c, d_skip_c, scale_c)
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.channels_last = bool(channels_last)
        ctx.d_skip_shape = d_skip.shape
        ctx.scale_shape = direction_scale.shape
        return out

    @staticmethod
    @custom_bwd_cuda
    @on_tensor_device
    def backward(ctx, d_out):
        y, x, d_skip, direction_scale = ctx.saved_tensors
        d_out_c = d_out
        if d_out_c.dtype != torch.float32:
            d_out_c = d_out_c.float()
        dy, dx, d_dskip, d_scale = load_postprocess_cuda(y.dtype).visionhope_postprocess_scale_backward(
            d_out_c, y, x, d_skip, direction_scale, ctx.height, ctx.width,
            ctx.channels_last, False, 128
        )
        return dy, dx, d_dskip.view(ctx.d_skip_shape), d_scale.view(ctx.scale_shape), None, None, None


def can_fuse_scale_postprocess(y, x, d_skip, direction_scale, height, width):
    if direction_scale is None:
        return False
    if not (y.is_cuda and x.is_cuda and d_skip.is_cuda and direction_scale.is_cuda):
        return False
    if any(tensor.device != y.device for tensor in (x, d_skip, direction_scale)):
        return False
    if y.dtype not in (torch.float16, torch.bfloat16, torch.float32) or x.dtype != y.dtype:
        return False
    if d_skip.dtype != torch.float32 or direction_scale.dtype != torch.float32:
        return False
    if y.dim() != 5 or x.shape != y.shape or y.shape[1] != 4:
        return False
    if tuple(d_skip.shape) not in ((y.shape[2], y.shape[4]), (4, y.shape[2], y.shape[4])):
        return False
    if tuple(direction_scale.shape) != (4, y.shape[2] * y.shape[4]):
        return False
    return int(height) * int(width) == int(y.shape[3])


def visionhope_postprocess_scale(y, x, d_skip, direction_scale, height, width, channels_last=False):
    return VisionHOPEPostprocessScaleFunction.apply(
        y, x, d_skip, direction_scale, int(height), int(width), bool(channels_last)
    )
