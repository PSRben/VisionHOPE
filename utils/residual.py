from visionhope.utils.cuda import on_tensor_device

import torch
from functools import lru_cache
from torch.autograd import Function

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@lru_cache(maxsize=None)
def _row_capacity(device):
    properties = triton.runtime.driver.utils.get_device_properties(device)
    return properties["max_shared_mem"] // (8 * 4)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit
    def _residual_forward_kernel(
        residual_ptr,
        branch_ptr,
        gamma_ptr,
        mask_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < total
        channel = offsets % channels
        row = offsets // channels
        batch = row // spatial
        residual = tl.load(residual_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + channel, mask=valid, other=0.0).to(tl.float32)
        drop_scale = tl.load(mask_ptr + batch, mask=valid, other=0.0).to(tl.float32)
        out = residual + branch * gamma * drop_scale
        tl.store(out_ptr + offsets, out, mask=valid)

    @triton.jit
    def _residual_backward_kernel(
        dout_ptr,
        branch_ptr,
        gamma_ptr,
        mask_ptr,
        dbranch_ptr,
        dgamma_part_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        block_m: tl.constexpr,
        block_c: tl.constexpr,
    ):
        group = tl.program_id(0)
        row_offsets = group * block_m + tl.arange(0, block_m)[:, None]
        cols = tl.arange(0, block_c)[None, :]
        row_mask = row_offsets < rows
        col_mask = cols < channels
        mask = row_mask & col_mask
        offsets = row_offsets * channels + cols
        batch = row_offsets // spatial

        dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        drop_scale = tl.load(mask_ptr + batch, mask=row_mask, other=0.0).to(tl.float32)
        branch_scale = gamma * drop_scale

        tl.store(dbranch_ptr + offsets, dout * branch_scale, mask=mask)
        dgamma = tl.sum(tl.where(row_mask, dout * branch * drop_scale, 0.0), axis=0)
        part_offsets = group * channels + tl.arange(0, block_c)
        tl.store(dgamma_part_ptr + part_offsets, dgamma, mask=tl.arange(0, block_c) < channels)

    @triton.jit
    def _residual_reduce_kernel(
        dgamma_part_ptr,
        dgamma_ptr,
        groups: tl.constexpr,
        channels: tl.constexpr,
        block_g: tl.constexpr,
    ):
        channel = tl.program_id(0)
        group_offsets = tl.arange(0, block_g)
        valid = group_offsets < groups
        offsets = group_offsets * channels + channel
        dgamma = tl.load(dgamma_part_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        tl.store(dgamma_ptr + channel, tl.sum(dgamma, axis=0))


class _VisionHOPEResidualFunction(Function):
    @staticmethod
    @on_tensor_device
    def forward(ctx, residual, branch, gamma, mask):
        batch, channels, height, width = branch.shape
        spatial = height * width
        total = batch * spatial * channels
        out = torch.empty_like(branch, dtype=torch.float32, memory_format=torch.channels_last)
        block = 256
        grid = (triton.cdiv(total, block),)
        _residual_forward_kernel[grid](
            residual,
            branch,
            gamma,
            mask,
            out,
            total,
            channels,
            spatial,
            block,
            num_warps=4,
        )
        ctx.save_for_backward(branch, gamma, mask)
        ctx.shape = (batch, channels, height, width)
        return out

    @staticmethod
    @on_tensor_device
    def backward(ctx, dout):
        branch, gamma, mask = ctx.saved_tensors
        if not dout.is_contiguous(memory_format=torch.channels_last):
            dout = dout.contiguous(memory_format=torch.channels_last)
        batch, channels, height, width = ctx.shape
        rows = batch * height * width
        spatial = height * width
        block_c = _next_power_of_2(channels)
        block_m = min(32, max(1, 8192 // block_c))
        groups = triton.cdiv(rows, block_m)
        dresidual = dout
        dbranch = torch.empty_like(branch, memory_format=torch.channels_last)
        dgamma_part = torch.empty((groups, channels), device=branch.device, dtype=torch.float32)
        _residual_backward_kernel[(groups,)](
            dout,
            branch,
            gamma,
            mask,
            dbranch,
            dgamma_part,
            rows,
            channels,
            spatial,
            block_m,
            block_c,
            num_warps=4,
        )
        dgamma = torch.empty((channels,), device=branch.device, dtype=torch.float32)
        block_g = _next_power_of_2(groups)
        _residual_reduce_kernel[(channels,)](
            dgamma_part,
            dgamma,
            groups,
            channels,
            block_g,
            num_warps=8,
        )
        return dresidual, dbranch, dgamma.view(channels, 1, 1), None


def can_fuse_residual_add(residual, branch, gamma):
    if not _TRITON_AVAILABLE:
        return False
    if residual.ndim != 4 or branch.ndim != 4 or residual.shape != branch.shape:
        return False
    if not residual.is_cuda or not branch.is_cuda or not gamma.is_cuda:
        return False
    if residual.device != branch.device or gamma.device != branch.device:
        return False
    if not gamma.is_contiguous() or not gamma.is_floating_point():
        return False
    if not residual.is_contiguous(memory_format=torch.channels_last):
        return False
    if not branch.is_contiguous(memory_format=torch.channels_last):
        return False
    if (branch.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or residual.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return False
    channels = branch.shape[1]
    return (gamma.numel() == channels and channels > 0
            and _next_power_of_2(channels) <= _row_capacity(branch.device.index))


def visionhope_residual_add(residual, branch, gamma, drop_prob, training, scale_by_keep=True):
    batch = branch.shape[0]
    if training and drop_prob > 0.0:
        keep_prob = 1.0 - float(drop_prob)
        mask = torch.empty((batch,), device=branch.device, dtype=torch.float32).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            mask.div_(keep_prob)
    else:
        mask = torch.ones((batch,), device=branch.device, dtype=torch.float32)
    return _VisionHOPEResidualFunction.apply(residual, branch, gamma, mask)
