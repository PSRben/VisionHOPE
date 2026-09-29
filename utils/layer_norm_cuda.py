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
    # Limit the FP32 elements per block.
    return properties["max_shared_mem"] // (8 * 4)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit
    def _layernorm2d_forward_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        y_ptr,
        mean_ptr,
        rstd_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        block_c: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_c)
        mask = cols < channels
        x = tl.load(x_ptr + row * channels + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / channels
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / channels
        rstd = 1.0 / tl.sqrt(var + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_centered * rstd * weight + bias
        tl.store(y_ptr + row * channels + cols, y, mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _layernorm2d_backward_kernel(
        dy_ptr,
        x_ptr,
        weight_ptr,
        mean_ptr,
        rstd_ptr,
        dx_ptr,
        dweight_part_ptr,
        dbias_part_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
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

        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)

        x_hat = (x - mean) * rstd
        weighted_dy = dy * weight
        sum_weighted_dy = tl.sum(tl.where(col_mask, weighted_dy, 0.0), axis=1)[:, None]
        sum_weighted_dy_xhat = tl.sum(tl.where(col_mask, weighted_dy * x_hat, 0.0), axis=1)[:, None]
        inv_channels = 1.0 / channels
        dx = (weighted_dy - sum_weighted_dy * inv_channels - x_hat * sum_weighted_dy_xhat * inv_channels) * rstd
        tl.store(dx_ptr + offsets, dx, mask=mask)

        dweight = tl.sum(tl.where(row_mask, dy * x_hat, 0.0), axis=0)
        dbias = tl.sum(tl.where(row_mask, dy, 0.0), axis=0)
        part_offsets = group * channels + tl.arange(0, block_c)
        tl.store(dweight_part_ptr + part_offsets, dweight, mask=tl.arange(0, block_c) < channels)
        tl.store(dbias_part_ptr + part_offsets, dbias, mask=tl.arange(0, block_c) < channels)

    @triton.jit
    def _layernorm2d_reduce_kernel(
        dweight_part_ptr,
        dbias_part_ptr,
        dweight_ptr,
        dbias_ptr,
        groups: tl.constexpr,
        channels: tl.constexpr,
        block_g: tl.constexpr,
    ):
        channel = tl.program_id(0)
        group_offsets = tl.arange(0, block_g)
        mask = group_offsets < groups
        offsets = group_offsets * channels + channel
        dweight = tl.load(dweight_part_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dbias = tl.load(dbias_part_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(dweight_ptr + channel, tl.sum(dweight, axis=0))
        tl.store(dbias_ptr + channel, tl.sum(dbias, axis=0))


class _VisionHOPELayerNorm2dFunction(Function):
    @staticmethod
    @on_tensor_device
    def forward(ctx, x, weight, bias, eps):
        batch, channels, height, width = x.shape
        rows = batch * height * width
        block_c = _next_power_of_2(channels)
        y = torch.empty_like(x, dtype=torch.float32, memory_format=torch.channels_last)
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
        _layernorm2d_forward_kernel[(rows,)](
            x,
            weight,
            bias,
            y,
            mean,
            rstd,
            rows,
            channels,
            float(eps),
            block_c,
            num_warps=4,
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.shape = (batch, channels, height, width)
        return y

    @staticmethod
    @on_tensor_device
    def backward(ctx, dy):
        # Convert upstream gradients to contiguous channels-last layout.
        dy = dy.contiguous(memory_format=torch.channels_last)
        x, weight, mean, rstd = ctx.saved_tensors
        batch, channels, height, width = ctx.shape
        rows = batch * height * width
        block_c = _next_power_of_2(channels)
        # Set the row tile size from the channel width.
        block_m = min(32, max(1, 8192 // block_c))
        groups = triton.cdiv(rows, block_m)
        dx = torch.empty_like(x, memory_format=torch.channels_last)
        dweight_part = torch.empty((groups, channels), device=x.device, dtype=torch.float32)
        dbias_part = torch.empty((groups, channels), device=x.device, dtype=torch.float32)
        _layernorm2d_backward_kernel[(groups,)](
            dy,
            x,
            weight,
            mean,
            rstd,
            dx,
            dweight_part,
            dbias_part,
            rows,
            channels,
            block_m,
            block_c,
            num_warps=4,
        )
        dweight = torch.empty_like(weight, dtype=torch.float32)
        dbias = torch.empty_like(weight, dtype=torch.float32)
        block_g = _next_power_of_2(groups)
        _layernorm2d_reduce_kernel[(channels,)](
            dweight_part,
            dbias_part,
            dweight,
            dbias,
            groups,
            channels,
            block_g,
            num_warps=8,
        )
        return dx, dweight, dbias, None


def can_fuse_layernorm2d(x, weight, bias):
    if not _TRITON_AVAILABLE:
        return False
    if x.ndim != 4 or not x.is_cuda:
        return False
    if not x.is_contiguous(memory_format=torch.channels_last):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    channels = x.shape[1]
    # Check the row capacity for this device.
    if channels < 1 or _next_power_of_2(channels) > _row_capacity(x.device.index):
        return False
    if weight is None or bias is None:
        return False
    return (weight.device == x.device and bias.device == x.device
            and weight.is_contiguous() and bias.is_contiguous()
            and weight.is_floating_point() and bias.is_floating_point()
            and weight.numel() == channels and bias.numel() == channels)


def visionhope_layernorm2d(x, weight, bias, eps):
    return _VisionHOPELayerNorm2dFunction.apply(x, weight, bias, eps)
