"""Fused query normalization and layout conversion for SRNL."""

from pathlib import Path

from functools import partial

from visionhope.utils.cuda import on_tensor_device

import torch
from torch.utils.cpp_extension import load_inline

from visionhope.utils.cuda_source import _check_io_dtype, _cuda_source_io_variant

try:
    from torch.amp import custom_bwd, custom_fwd

    custom_fwd_cuda = partial(custom_fwd, device_type="cuda")
    custom_bwd_cuda = partial(custom_bwd, device_type="cuda")
except ImportError:
    from torch.cuda.amp import custom_bwd as custom_bwd_cuda
    from torch.cuda.amp import custom_fwd as custom_fwd_cuda


_EXTENSION_NAME = "visionhope_qnorm_io"
_qnorm_cuda = {}


cuda_source = (Path(__file__).parent / "csrc" / "query_normalization.cu").read_text()

cpp_source = (Path(__file__).parent / "csrc" / "query_normalization.cpp").read_text()

cuda_source_fp32 = _cuda_source_io_variant(cuda_source, torch.float32)
cuda_source_fp16 = _cuda_source_io_variant(cuda_source, torch.float16)


def load_qnorm_cuda(io_dtype=torch.bfloat16):
    global _qnorm_cuda
    io_dtype = _check_io_dtype(io_dtype)
    if io_dtype not in _qnorm_cuda:
        suffix = "_fp32io" if io_dtype == torch.float32 else ("_fp16io" if io_dtype == torch.float16 else "")
        source = cuda_source_fp32 if io_dtype == torch.float32 else (cuda_source_fp16 if io_dtype == torch.float16 else cuda_source)
        _qnorm_cuda[io_dtype] = load_inline(
            name=_EXTENSION_NAME + suffix,
            cpp_sources=cpp_source,
            cuda_sources=source,
            functions=["visionhope_qnorm_forward", "visionhope_qnorm_backward"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _qnorm_cuda[io_dtype]


class VisionHOPEQNormFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd_cuda
    @on_tensor_device
    def forward(ctx, q_raw, num_heads, head_dim):
        q_raw_c = q_raw.contiguous()
        q_out, inv_norm = load_qnorm_cuda(q_raw_c.dtype).visionhope_qnorm_forward(
            q_raw_c, int(num_heads), int(head_dim), True
        )
        ctx.save_for_backward(q_out, inv_norm)
        ctx.num_heads = int(num_heads)
        ctx.head_dim = int(head_dim)
        ctx.input_dtype = q_raw_c.dtype
        return q_out

    @staticmethod
    @custom_bwd_cuda
    @on_tensor_device
    def backward(ctx, d_q_out):
        q_out, inv_norm = ctx.saved_tensors
        d_q_raw = load_qnorm_cuda(ctx.input_dtype).visionhope_qnorm_backward(
            d_q_out.contiguous().float(), q_out, inv_norm, ctx.num_heads, ctx.head_dim, True
        )
        return d_q_raw, None, None


def can_fuse_qnorm(q_raw, num_heads, head_dim):
    if not q_raw.is_cuda or q_raw.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if q_raw.dim() != 4 or q_raw.shape[1] != 4:
        return False
    if int(head_dim) not in {4, 8, 16, 32, 64}:
        return False
    return int(q_raw.shape[-1]) == int(num_heads) * int(head_dim)


def visionhope_qnorm(q_raw, num_heads, head_dim):
    return VisionHOPEQNormFunction.apply(q_raw, int(num_heads), int(head_dim))
