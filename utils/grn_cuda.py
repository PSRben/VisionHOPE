"""Shared CUDA GRN backend with first-order autograd and FP32 output.

Use :mod:`visionhope.utils.grn` for the public module/functional API and CPU/layout fallback.
The CUDA extension is compiled lazily on the first eligible call, never on import.
"""
from pathlib import Path

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

from visionhope.utils.cuda import on_tensor_device

_CUDA_EXTENSION = None
_CUDA_EXTENSION_NAME = "visionhope_grn_cuda"


def _load_cuda_extension():
    global _CUDA_EXTENSION
    if _CUDA_EXTENSION is None:
        source_dir = Path(__file__).resolve().parent / "csrc"
        _CUDA_EXTENSION = load(
            name=_CUDA_EXTENSION_NAME,
            sources=[str(source_dir / "grn.cpp"), str(source_dir / "grn_cuda.cu")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _CUDA_EXTENSION


class _VisionHOPEGRNCudaFunction(Function):
    @staticmethod
    @on_tensor_device
    def forward(ctx, x, gamma, beta, eps):
        batch, channels, height, width = x.shape
        # Allocate FP32 outputs and statistics.
        y = torch.empty_like(x, dtype=torch.float32, memory_format=torch.channels_last)
        gx = torch.empty((batch, channels), device=x.device, dtype=torch.float32)
        inv_den = torch.empty((batch,), device=x.device, dtype=torch.float32)
        _load_cuda_extension().grn_forward_cuda(x, gamma, beta, y, gx, inv_den, float(eps))
        ctx.save_for_backward(x, gamma, gx, inv_den)
        ctx.shape = (batch, channels, height, width)
        ctx.beta_shape = beta.shape
        ctx.beta_stride = beta.stride()
        return y

    @staticmethod
    @on_tensor_device
    @once_differentiable
    def backward(ctx, dout):
        x, gamma, gx, inv_den = ctx.saved_tensors
        dout = dout.to(dtype=torch.float32, memory_format=torch.channels_last)
        batch, channels, _, _ = ctx.shape
        s = torch.empty((batch, channels), device=x.device, dtype=torch.float32)
        dgamma_bc = torch.empty((batch, channels), device=x.device, dtype=torch.float32)
        dbeta_bc = torch.empty((batch, channels), device=x.device, dtype=torch.float32)
        sum_sg = torch.empty((batch,), device=x.device, dtype=torch.float32)
        dx = torch.empty_like(x, memory_format=torch.channels_last)
        # Match parameter gradient shapes and strides.
        dgamma = torch.empty_strided(gamma.shape, gamma.stride(), device=gamma.device,
                                    dtype=torch.float32)
        dbeta = torch.empty_strided(ctx.beta_shape, ctx.beta_stride, device=gamma.device,
                                   dtype=torch.float32)
        _load_cuda_extension().grn_backward_cuda(
            dout,
            x,
            gamma,
            gx,
            inv_den,
            s,
            dgamma_bc,
            dbeta_bc,
            sum_sg,
            dx,
            dgamma,
            dbeta,
        )
        return dx, dgamma, dbeta, None


def can_fuse_grn(x, gamma, beta):
    if x.ndim != 4 or not x.is_cuda or not x.is_contiguous(memory_format=torch.channels_last):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3]
    # Check CUDA grid and index limits.
    if x.shape[0] < 1 or x.shape[0] > 65535 or channels < 1 or spatial < 1:
        return False
    if x.numel() > 2**31 - 1 - 4096 * 256:
        return False
    return (gamma.device == x.device and beta.device == x.device
            and gamma.dtype == torch.float32 and beta.dtype == torch.float32
            and gamma.is_contiguous() and beta.is_contiguous()
            and gamma.numel() == channels and beta.numel() == channels)


def visionhope_grn(x, gamma, beta, eps):
    return _VisionHOPEGRNCudaFunction.apply(x, gamma, beta, eps)
