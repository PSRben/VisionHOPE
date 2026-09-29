"""CUDA autograd for long SRNL chunks and row/column scans.

Logical chunk lengths determine memory refresh boundaries; internal tiles
bound temporary storage. The kernels accept FP16, BF16, or FP32 inputs and
keep recurrence state in FP32.
"""

from __future__ import annotations

from pathlib import Path

from functools import partial

from visionhope.utils.cuda import on_tensor_device

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from . import chunk as _base
from .source import line_source

try:
    from torch.amp import custom_bwd, custom_fwd
    custom_fwd_cuda = partial(custom_fwd, device_type="cuda")
    custom_bwd_cuda = partial(custom_bwd, device_type="cuda")
except ImportError:
    from torch.cuda.amp import custom_bwd as custom_bwd_cuda
    from torch.cuda.amp import custom_fwd as custom_fwd_cuda



MAX_LOGICAL_CHUNK = None
RECOMP_TILE = 8
SUPPORTED_HEAD_DIMS = (4, 8, 16, 32, 64)


def _tag_float(value):
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


_EXTENSION_NAME = (
    "srnl_lines"
    f"_eta{_tag_float(_base.STEP_SCALE)}"
    f"_amin{_tag_float(_base.MIN_RETENTION)}"
    f"_abias{_tag_float(_base.RETENTION_BIAS)}"
    f"_mu{_tag_float(_base.INJECTION_MARGIN)}"
    f"_e2a{int(_base.SPECTRAL_CLAMP)}"
    f"_e2am{_tag_float(_base.SPECTRAL_MARGIN)}"
)
_srnl_cuda = None
_srnl_cuda_d64 = None
_srnl_cuda_fp16 = None
_srnl_cuda_d64_fp16 = None
_srnl_cuda_fp32 = None
_srnl_cuda_d64_fp32 = None


large_cuda_source = line_source()
varline_cuda_source = line_source(variable=True)
large_cuda_source_d64 = line_source(head64=True)
varline_cuda_source_d64 = line_source(variable=True, head64=True)
large_cuda_source_fp16 = _base._cuda_source_io_variant(large_cuda_source, torch.float16)
varline_cuda_source_fp16 = _base._cuda_source_io_variant(varline_cuda_source, torch.float16)
large_cuda_source_fp32 = _base._cuda_source_io_variant(large_cuda_source, torch.float32)
varline_cuda_source_fp32 = _base._cuda_source_io_variant(varline_cuda_source, torch.float32)
large_cuda_source_d64_fp16 = _base._cuda_source_io_variant(large_cuda_source_d64, torch.float16)
varline_cuda_source_d64_fp16 = _base._cuda_source_io_variant(varline_cuda_source_d64, torch.float16)
large_cuda_source_d64_fp32 = _base._cuda_source_io_variant(large_cuda_source_d64, torch.float32)
varline_cuda_source_d64_fp32 = _base._cuda_source_io_variant(varline_cuda_source_d64, torch.float32)


cpp_source = (
    _base.cpp_source
    + (Path(__file__).parent / "csrc" / "lines.cpp").read_text()
)


def load_cuda(io_dtype=torch.bfloat16):
    global _srnl_cuda, _srnl_cuda_fp16, _srnl_cuda_fp32
    io_dtype = _base._check_io_dtype(io_dtype)
    if io_dtype == torch.float16:
        if _srnl_cuda_fp16 is None:
            _srnl_cuda_fp16 = load_inline(
                name=_EXTENSION_NAME + "_fp16io",
                cpp_sources=cpp_source,
                cuda_sources=_base.cuda_helper_source + large_cuda_source_fp16 + varline_cuda_source_fp16,
                functions=[
                    "srnl_forward_qraw_large",
                    "srnl_backward_qraw_large",
                    "srnl_forward_qraw_varline",
                    "srnl_backward_qraw_varline",
                ],
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
        return _srnl_cuda_fp16
    if io_dtype == torch.float32:
        if _srnl_cuda_fp32 is None:
            _srnl_cuda_fp32 = load_inline(
                name=_EXTENSION_NAME + "_fp32io",
                cpp_sources=cpp_source,
                cuda_sources=_base.cuda_helper_source + large_cuda_source_fp32 + varline_cuda_source_fp32,
                functions=[
                    "srnl_forward_qraw_large",
                    "srnl_backward_qraw_large",
                    "srnl_forward_qraw_varline",
                    "srnl_backward_qraw_varline",
                ],
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )
        return _srnl_cuda_fp32
    if _srnl_cuda is None:
        _srnl_cuda = load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=cpp_source,
            cuda_sources=_base.cuda_helper_source + large_cuda_source + varline_cuda_source,
            functions=[
                "srnl_forward_qraw_large",
                "srnl_backward_qraw_large",
                "srnl_forward_qraw_varline",
                "srnl_backward_qraw_varline",
            ],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _srnl_cuda


def load_cuda_head64(io_dtype=torch.bfloat16):
    global _srnl_cuda_d64, _srnl_cuda_d64_fp16, _srnl_cuda_d64_fp32
    io_dtype = _base._check_io_dtype(io_dtype)
    if io_dtype == torch.float16:
        if _srnl_cuda_d64_fp16 is None:
            _srnl_cuda_d64_fp16 = load_inline(
                name=_EXTENSION_NAME + "_d64only_fp16io",
                cpp_sources=cpp_source,
                cuda_sources=_base.cuda_helper_source + large_cuda_source_d64_fp16 + varline_cuda_source_d64_fp16,
                functions=[
                    "srnl_forward_qraw_large",
                    "srnl_backward_qraw_large",
                    "srnl_forward_qraw_varline",
                    "srnl_backward_qraw_varline",
                ],
                verbose=False,
                extra_cuda_cflags=["-O2", "--use_fast_math", "--threads=4"],
            )
        return _srnl_cuda_d64_fp16
    if io_dtype == torch.float32:
        if _srnl_cuda_d64_fp32 is None:
            _srnl_cuda_d64_fp32 = load_inline(
                name=_EXTENSION_NAME + "_d64only_fp32io",
                cpp_sources=cpp_source,
                cuda_sources=_base.cuda_helper_source + large_cuda_source_d64_fp32 + varline_cuda_source_d64_fp32,
                functions=[
                    "srnl_forward_qraw_large",
                    "srnl_backward_qraw_large",
                    "srnl_forward_qraw_varline",
                    "srnl_backward_qraw_varline",
                ],
                verbose=False,
                extra_cuda_cflags=["-O2", "--use_fast_math", "--threads=4"],
            )
        return _srnl_cuda_d64_fp32
    if _srnl_cuda_d64 is None:
        _srnl_cuda_d64 = load_inline(
            name=_EXTENSION_NAME + "_d64only",
            cpp_sources=cpp_source,
            cuda_sources=_base.cuda_helper_source + large_cuda_source_d64 + varline_cuda_source_d64,
            functions=[
                "srnl_forward_qraw_large",
                "srnl_backward_qraw_large",
                "srnl_forward_qraw_varline",
                "srnl_backward_qraw_varline",
            ],
            verbose=False,
            extra_cuda_cflags=["-O2", "--use_fast_math", "--threads=4"],
        )
    return _srnl_cuda_d64


class SRNLLongChunkFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd_cuda
    @on_tensor_device
    def forward(ctx, M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q_raw, chunk_size, num_heads, head_dim):
        if torch.jit.is_tracing():
            raise RuntimeError("SRNL TorchScript tracing is unsupported; use eager or CUDA Graph inference.")
        if not x.is_cuda:
            raise RuntimeError("SRNL large-line qraw CUDA kernel requires CUDA tensors.")

        state_dtypes = (M_m_0.dtype, M_v_0.dtype, M_k_0.dtype, M_eta_0.dtype, M_alpha_0.dtype)
        input_dtypes = (x.dtype, q_raw.dtype)
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"SRNL large-line qraw CUDA kernel supports fp16/bf16/fp32 x, got {x.dtype}.")
        if q_raw.dtype != x.dtype:
            raise RuntimeError(
                f"SRNL large-line qraw CUDA kernel expects q_raw dtype to match x dtype, got {q_raw.dtype} vs {x.dtype}."
            )
        B, directions, x_num_heads, seq_len, x_head_dim = x.shape
        num_heads = int(num_heads)
        head_dim = int(head_dim)
        chunk_size = int(chunk_size)
        if directions not in (1, 2, 4) or x_num_heads != num_heads or x_head_dim != head_dim:
            raise AssertionError("Invalid x shape for large-line qraw kernel.")
        if q_raw.dim() != 4 or q_raw.shape[0] != B or q_raw.shape[1] != directions or q_raw.shape[2] != seq_len:
            raise AssertionError("Invalid q_raw shape for large-line qraw kernel.")
        if q_raw.shape[3] != num_heads * head_dim:
            raise AssertionError("q_raw channel count does not match num_heads * head_dim.")
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise AssertionError(f"large-line CUDA kernel requires head_dim in {SUPPORTED_HEAD_DIMS}, got {head_dim}.")
        if chunk_size <= _base.MAX_CHUNK:
            raise ValueError("large-line CUDA kernel is only intended for chunk_size > base MAX_CHUNK.")

        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            q_raw = F.pad(q_raw, (0, 0, 0, pad_len))
            seq_len = x.shape[-2]

        n_flat = B * directions * num_heads
        if M_m_0.dim() == 4:
            state_count = directions * num_heads
        elif M_m_0.dim() == 5:
            state_count = n_flat
        else:
            raise AssertionError(f"Expected memory tensors with 4 or 5 dims, got shape {tuple(M_m_0.shape)}.")

        num_chunks = seq_len // chunk_size
        max_subchunks = (chunk_size + RECOMP_TILE - 1) // RECOMP_TILE
        M_m_0_f = M_m_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_v_0_f = M_v_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_k_0_f = M_k_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_eta_0_f = M_eta_0.contiguous().view(state_count, head_dim).to(dtype=torch.float32)
        M_alpha_0_f = M_alpha_0.contiguous().view(state_count, head_dim).to(dtype=torch.float32)
        x_b = x.contiguous().view(n_flat, seq_len, head_dim)
        q_raw_b = q_raw.contiguous()

        G_prefix_chunk = torch.empty(
            (n_flat, max(num_chunks - 1, 0), head_dim, head_dim), device=x.device, dtype=torch.float32
        )
        K_saved = torch.empty((n_flat, seq_len, head_dim), device=x.device, dtype=torch.float32)
        V_saved = torch.empty((n_flat, seq_len, head_dim), device=x.device, dtype=torch.float32)
        Eta_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        Alpha_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        EtaGrad_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        AlphaGrad_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        InvK_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        save_qinv = 1
        QInv_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        G_checkpoint_saved = torch.empty(
            (n_flat, num_chunks * max_subchunks, head_dim, head_dim), device=x.device, dtype=torch.float32
        )
        G_final_chunk = torch.empty((n_flat, num_chunks, head_dim, head_dim), device=x.device, dtype=torch.float32)

        srnl_cuda = load_cuda_head64(x_b.dtype) if head_dim == 64 else load_cuda(x_b.dtype)
        y_f = srnl_cuda.srnl_forward_qraw_large(
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_raw_b,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
            chunk_size,
            head_dim,
            state_count,
            num_heads,
            save_qinv,
            max_subchunks,
        )

        ctx.save_for_backward(
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_raw_b,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
        )
        ctx.constants = (chunk_size, head_dim, pad_len, state_count, num_heads, save_qinv, max_subchunks)
        ctx.orig_shapes = (M_m_0.shape, M_eta_0.shape, x.shape, q_raw.shape)
        ctx.orig_dtypes = state_dtypes + input_dtypes

        y = y_f.view(x.shape)
        if pad_len > 0:
            y = y[:, :, :, :-pad_len, :]
        return y

    @staticmethod
    @custom_bwd_cuda
    @on_tensor_device
    def backward(ctx, dY):
        (
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_raw_b,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
        ) = ctx.saved_tensors
        chunk_size, head_dim, pad_len, state_count, num_heads, save_qinv, max_subchunks = ctx.constants
        M_shape, scalar_shape, x_shape, q_raw_shape = ctx.orig_shapes
        M_m_dtype, M_v_dtype, M_k_dtype, M_eta_dtype, M_alpha_dtype, x_dtype, q_raw_dtype = ctx.orig_dtypes
        n_flat, seq_len = x_b.shape[0], x_b.shape[1]

        if pad_len > 0:
            dY = F.pad(dY, (0, 0, 0, pad_len))
        dY_b = dY.contiguous().view(n_flat, seq_len, head_dim).to(dtype=x_b.dtype)

        srnl_cuda = load_cuda_head64(x_b.dtype) if head_dim == 64 else load_cuda(x_b.dtype)
        dM_m_0, dM_v_0, dM_k_0, dM_eta_0, dM_alpha_0, dX, dQRaw = srnl_cuda.srnl_backward_qraw_large(
            dY_b,
            x_b,
            q_raw_b,
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
            chunk_size,
            head_dim,
            state_count,
            num_heads,
            save_qinv,
            max_subchunks,
        )

        dX = dX.view(x_shape)
        dQRaw = dQRaw.view(q_raw_shape)
        if pad_len > 0:
            dX = dX[:, :, :, :-pad_len, :]
            dQRaw = dQRaw[:, :, :-pad_len, :]

        return (
            dM_m_0.view(M_shape).to(dtype=M_m_dtype),
            dM_v_0.view(M_shape).to(dtype=M_v_dtype),
            dM_k_0.view(M_shape).to(dtype=M_k_dtype),
            dM_eta_0.view(scalar_shape).to(dtype=M_eta_dtype),
            dM_alpha_0.view(scalar_shape).to(dtype=M_alpha_dtype),
            dX.to(dtype=x_dtype),
            dQRaw.to(dtype=q_raw_dtype),
            None,
            None,
            None,
        )


def srnl_long_chunk(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q_raw, chunk_size, num_heads, head_dim):
    return SRNLLongChunkFunction.apply(
        M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q_raw, int(chunk_size), int(num_heads), int(head_dim)
    )


class SRNLVariableLineFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd_cuda
    @on_tensor_device
    def forward(
        ctx,
        M_m_0,
        M_v_0,
        M_k_0,
        M_eta_0,
        M_alpha_0,
        x,
        q_raw,
        row_chunk_size,
        col_chunk_size,
        num_heads,
        head_dim,
    ):
        if torch.jit.is_tracing():
            raise RuntimeError("SRNL TorchScript tracing is unsupported; use eager or CUDA Graph inference.")
        if not x.is_cuda:
            raise RuntimeError("SRNL varline qraw CUDA kernel requires CUDA tensors.")

        state_dtypes = (M_m_0.dtype, M_v_0.dtype, M_k_0.dtype, M_eta_0.dtype, M_alpha_0.dtype)
        input_dtypes = (x.dtype, q_raw.dtype)
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"SRNL varline qraw CUDA kernel supports fp16/bf16/fp32 x, got {x.dtype}.")
        if q_raw.dtype != x.dtype:
            raise RuntimeError(
                f"SRNL varline qraw CUDA kernel expects q_raw dtype to match x dtype, got {q_raw.dtype} vs {x.dtype}."
            )
        B, directions, x_num_heads, seq_len, x_head_dim = x.shape
        row_chunk_size = int(row_chunk_size)
        col_chunk_size = int(col_chunk_size)
        num_heads = int(num_heads)
        head_dim = int(head_dim)
        if directions != 4 or x_num_heads != num_heads or x_head_dim != head_dim:
            raise AssertionError("Invalid x shape for varline qraw kernel.")
        if q_raw.dim() != 4 or q_raw.shape[0] != B or q_raw.shape[1] != 4 or q_raw.shape[2] != seq_len:
            raise AssertionError("Invalid q_raw shape for varline qraw kernel.")
        if q_raw.shape[3] != num_heads * head_dim:
            raise AssertionError("q_raw channel count does not match num_heads * head_dim.")
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise AssertionError(f"varline CUDA kernel requires head_dim in {SUPPORTED_HEAD_DIMS}, got {head_dim}.")
        if row_chunk_size <= 0 or col_chunk_size <= 0:
            raise ValueError("row_chunk_size and col_chunk_size must be positive.")
        if seq_len % row_chunk_size != 0 or seq_len % col_chunk_size != 0:
            raise ValueError(
                f"seq_len={seq_len} must be divisible by row_chunk_size={row_chunk_size} and col_chunk_size={col_chunk_size}."
            )

        n_flat = B * directions * num_heads
        if M_m_0.dim() == 4:
            state_count = directions * num_heads
        elif M_m_0.dim() == 5:
            state_count = n_flat
        else:
            raise AssertionError(f"Expected memory tensors with 4 or 5 dims, got shape {tuple(M_m_0.shape)}.")

        row_num_chunks = seq_len // row_chunk_size
        col_num_chunks = seq_len // col_chunk_size
        max_num_chunks = max(row_num_chunks, col_num_chunks)
        max_chunk_size = max(row_chunk_size, col_chunk_size)
        max_subchunks = (max_chunk_size + RECOMP_TILE - 1) // RECOMP_TILE
        line_groups = B * num_heads
        row_prefix_chunks = max(row_num_chunks - 1, 0)
        col_prefix_chunks = max(col_num_chunks - 1, 0)
        packed_num_chunks = line_groups * (2 * row_num_chunks + 2 * col_num_chunks)
        packed_prefix_chunks = line_groups * (2 * row_prefix_chunks + 2 * col_prefix_chunks)

        M_m_0_f = M_m_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_v_0_f = M_v_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_k_0_f = M_k_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_eta_0_f = M_eta_0.contiguous().view(state_count, head_dim).to(dtype=torch.float32)
        M_alpha_0_f = M_alpha_0.contiguous().view(state_count, head_dim).to(dtype=torch.float32)
        x_b = x.contiguous().view(n_flat, seq_len, head_dim)
        q_raw_b = q_raw.contiguous()

        G_prefix_chunk = torch.empty(
            (max(packed_prefix_chunks, 1), head_dim, head_dim), device=x.device, dtype=torch.float32
        )
        K_saved = torch.empty((n_flat, seq_len, head_dim), device=x.device, dtype=torch.float32)
        V_saved = torch.empty((n_flat, seq_len, head_dim), device=x.device, dtype=torch.float32)
        Eta_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        Alpha_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        EtaGrad_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        AlphaGrad_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        InvK_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        save_qinv = 1
        QInv_saved = torch.empty((n_flat, seq_len), device=x.device, dtype=torch.float32)
        G_checkpoint_saved = torch.empty(
            (packed_num_chunks * max_subchunks, head_dim, head_dim), device=x.device, dtype=torch.float32
        )
        G_final_chunk = torch.empty((packed_num_chunks, head_dim, head_dim), device=x.device, dtype=torch.float32)

        srnl_cuda = load_cuda_head64(x_b.dtype) if head_dim == 64 else load_cuda(x_b.dtype)
        y_f = srnl_cuda.srnl_forward_qraw_varline(
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_raw_b,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
            row_chunk_size,
            col_chunk_size,
            head_dim,
            state_count,
            num_heads,
            save_qinv,
            max_num_chunks,
            max_subchunks,
        )

        ctx.save_for_backward(
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_raw_b,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
        )
        ctx.constants = (
            row_chunk_size,
            col_chunk_size,
            head_dim,
            state_count,
            num_heads,
            save_qinv,
            max_num_chunks,
            max_subchunks,
        )
        ctx.orig_shapes = (M_m_0.shape, M_eta_0.shape, x.shape, q_raw.shape)
        ctx.orig_dtypes = state_dtypes + input_dtypes
        return y_f.view(x.shape)

    @staticmethod
    @custom_bwd_cuda
    @on_tensor_device
    def backward(ctx, dY):
        (
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_raw_b,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
        ) = ctx.saved_tensors
        (
            row_chunk_size,
            col_chunk_size,
            head_dim,
            state_count,
            num_heads,
            save_qinv,
            max_num_chunks,
            max_subchunks,
        ) = ctx.constants
        M_shape, scalar_shape, x_shape, q_raw_shape = ctx.orig_shapes
        M_m_dtype, M_v_dtype, M_k_dtype, M_eta_dtype, M_alpha_dtype, x_dtype, q_raw_dtype = ctx.orig_dtypes
        n_flat, seq_len = x_b.shape[0], x_b.shape[1]
        dY_b = dY.contiguous().view(n_flat, seq_len, head_dim).to(dtype=x_b.dtype)

        srnl_cuda = load_cuda_head64(x_b.dtype) if head_dim == 64 else load_cuda(x_b.dtype)
        dM_m_0, dM_v_0, dM_k_0, dM_eta_0, dM_alpha_0, dX, dQRaw = srnl_cuda.srnl_backward_qraw_varline(
            dY_b,
            x_b,
            q_raw_b,
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            QInv_saved,
            G_checkpoint_saved,
            G_final_chunk,
            row_chunk_size,
            col_chunk_size,
            head_dim,
            state_count,
            num_heads,
            save_qinv,
            max_num_chunks,
            max_subchunks,
        )

        return (
            dM_m_0.view(M_shape).to(dtype=M_m_dtype),
            dM_v_0.view(M_shape).to(dtype=M_v_dtype),
            dM_k_0.view(M_shape).to(dtype=M_k_dtype),
            dM_eta_0.view(scalar_shape).to(dtype=M_eta_dtype),
            dM_alpha_0.view(scalar_shape).to(dtype=M_alpha_dtype),
            dX.view(x_shape).to(dtype=x_dtype),
            dQRaw.view(q_raw_shape).to(dtype=q_raw_dtype),
            None,
            None,
            None,
            None,
        )


def srnl_variable_lines(
    M_m_0,
    M_v_0,
    M_k_0,
    M_eta_0,
    M_alpha_0,
    x,
    q_raw,
    row_chunk_size,
    col_chunk_size,
    num_heads,
    head_dim,
):
    return SRNLVariableLineFunction.apply(
        M_m_0,
        M_v_0,
        M_k_0,
        M_eta_0,
        M_alpha_0,
        x,
        q_raw,
        int(row_chunk_size),
        int(col_chunk_size),
        int(num_heads),
        int(head_dim),
    )
