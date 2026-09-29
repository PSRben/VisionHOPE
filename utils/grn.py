"""Global Response Normalization shared by training and inference.

CUDA channels-last tensors use one fused forward/backward implementation.
Statistics, affine arithmetic and output are FP32 for FP32/BF16/FP16 inputs.
CPU and other layouts use the same formula through PyTorch in FP32.
"""
import torch
from torch import nn

from visionhope.utils.grn_cuda import can_fuse_grn, visionhope_grn

__all__ = ["GlobalResponseNorm", "global_response_norm", "grn_reference"]


def _check_shape(x, gamma, beta):
    if x.ndim != 4:
        raise ValueError("GRN expects logical NCHW input with four dimensions")
    if not x.is_floating_point() or not gamma.is_floating_point() or not beta.is_floating_point():
        raise TypeError("GRN expects floating-point input and channel parameters")
    if gamma.numel() != x.shape[1] or beta.numel() != x.shape[1]:
        raise ValueError("GRN gamma and beta must each contain one value per channel")
    if gamma.device != x.device or beta.device != x.device:
        raise ValueError("GRN input, gamma and beta must be on the same device")


def grn_reference(x, gamma, beta, eps=1e-6):
    """Evaluate the GRN formula with PyTorch FP32 operations and autograd.

    Gamma and beta may be shaped [C] or [1, C, 1, 1]; their gradients have
    matching shapes and dtypes. Input gradients follow PyTorch's input dtype.
    """
    _check_shape(x, gamma, beta)
    x = x.float()
    gamma = gamma.float().reshape(1, x.shape[1], 1, 1)
    beta = beta.float().reshape(1, x.shape[1], 1, 1)
    gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
    nx = gx / (gx.mean(dim=1, keepdim=True) + float(eps))
    return x * (1.0 + gamma * nx) + beta


def global_response_norm(x, gamma, beta, eps=1e-6):
    """Return FP32 GRN output; automatically use the shared CUDA backend.

    The fused backend requires CUDA channels-last input, FP32 contiguous
    parameters and the documented CUDA grid/index limits. It is identical for
    train/eval/no_grad/inference_mode and does not require a prepare step.
    """
    _check_shape(x, gamma, beta)
    if can_fuse_grn(x, gamma, beta):
        return visionhope_grn(x, gamma, beta, float(eps))
    return grn_reference(x, gamma, beta, eps)


class GlobalResponseNorm(nn.Module):
    """GRN for logical NCHW tensors, with FP32 output in training and inference.

    The affine parameters gamma and beta have shape [1, C, 1, 1].
    Use AMP around the model while keeping its master parameters in FP32.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x):
        return global_response_norm(x, self.gamma, self.beta, self.eps)

    def extra_repr(self):
        return f"dim={self.gamma.numel()}, eps={self.eps}, output_dtype=float32"
