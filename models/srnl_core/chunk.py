"""CUDA autograd for SRNL with chunks of up to 64 tokens.

Accepts FP16, BF16, or FP32 inputs with FP32 recurrence buffers.
Backward reconstructs chunk-start memories from saved prefix gains.
"""

from pathlib import Path

import os

from .source import chunk_source, memory_helpers

from visionhope.utils.cuda import on_tensor_device

import torch
from visionhope.utils.cuda_source import _check_io_dtype, _cuda_source_io_variant
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from functools import partial

try:
    from torch.amp import custom_bwd, custom_fwd
    custom_fwd_cuda = partial(custom_fwd, device_type="cuda")
    custom_bwd_cuda = partial(custom_bwd, device_type="cuda")
except ImportError:
    from torch.cuda.amp import custom_bwd as custom_bwd_cuda
    from torch.cuda.amp import custom_fwd as custom_fwd_cuda


MAX_CHUNK = 64
SUPPORTED_HEAD_DIMS = (4, 8, 16, 32, 64)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


STEP_SCALE = _env_float("VISIONHOPE_STEP_SCALE", 0.025)
MIN_RETENTION = _env_float("VISIONHOPE_MIN_RETENTION", 0.0)
RETENTION_BIAS = _env_float("VISIONHOPE_RETENTION_BIAS", 2.197224577)
INJECTION_MARGIN = _env_float("VISIONHOPE_INJECTION_MARGIN", 0.999)
SPECTRAL_CLAMP = True
SPECTRAL_MARGIN = _env_float("VISIONHOPE_SPECTRAL_MARGIN", 0.999999)

if not (0.0 <= MIN_RETENTION < 1.0):
    raise ValueError(f"MIN_RETENTION must be in [0, 1), got {MIN_RETENTION}")
if not (0.0 < INJECTION_MARGIN < 1.0):
    raise ValueError(f"INJECTION_MARGIN must be in (0, 1), got {INJECTION_MARGIN}")
if not (STEP_SCALE > 0.0):
    raise ValueError(f"STEP_SCALE must be positive, got {STEP_SCALE}")
if not (0.0 < SPECTRAL_MARGIN <= 1.0):
    raise ValueError(f"SPECTRAL_MARGIN must be in (0, 1], got {SPECTRAL_MARGIN}")


def _tag_float(value):
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _cuda_float_literal(value):
    text = f"{float(value):.9g}"
    if "e" not in text and "E" not in text and "." not in text:
        text += ".0"
    return f"{text}f"


_EXTENSION_NAME = (
    "srnl_core"
    f"_eta{_tag_float(STEP_SCALE)}"
    f"_amin{_tag_float(MIN_RETENTION)}"
    f"_abias{_tag_float(RETENTION_BIAS)}"
    f"_mu{_tag_float(INJECTION_MARGIN)}"
    f"_e2a{int(SPECTRAL_CLAMP)}"
    f"_e2am{_tag_float(SPECTRAL_MARGIN)}"
)
_srnl_cuda = None
_srnl_cuda_d64 = {}
_srnl_cuda_fp16 = None
_srnl_cuda_d64_fp16 = {}
_srnl_cuda_fp32 = None
_srnl_cuda_d64_fp32 = {}


cuda_helper_source = (
    memory_helpers()
    .replace("__ETA_SCALE__", _cuda_float_literal(STEP_SCALE))
    .replace("__ALPHA_MIN__", _cuda_float_literal(MIN_RETENTION))
    .replace("__ALPHA_BIAS__", _cuda_float_literal(RETENTION_BIAS))
    .replace("__STAB_MU__", _cuda_float_literal(INJECTION_MARGIN))
    .replace("__ETA2ALPHA_CLAMP__", "1" if SPECTRAL_CLAMP else "0")
    .replace("__ETA2ALPHA_FACTOR__", _cuda_float_literal(2.0 * SPECTRAL_MARGIN))
)
cuda_source = cuda_helper_source + chunk_source()
d64_cuda_source = (cuda_helper_source + chunk_source(head64=True)).replace(
    "#pragma unroll", "#pragma unroll 1"
)
cuda_source_fp16 = _cuda_source_io_variant(cuda_source, torch.float16)
cuda_source_fp32 = _cuda_source_io_variant(cuda_source, torch.float32)
d64_cuda_source_fp16 = _cuda_source_io_variant(d64_cuda_source, torch.float16)
d64_cuda_source_fp32 = _cuda_source_io_variant(d64_cuda_source, torch.float32)

cpp_source = (Path(__file__).parent / "csrc" / "chunk.cpp").read_text()


def load_cuda(io_dtype=torch.bfloat16):
    global _srnl_cuda, _srnl_cuda_fp16, _srnl_cuda_fp32
    io_dtype = _check_io_dtype(io_dtype)
    if io_dtype == torch.float16:
        if _srnl_cuda_fp16 is None:
            _srnl_cuda_fp16 = load_inline(
                name=_EXTENSION_NAME + "_fp16io",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source_fp16,
                functions=["srnl_forward", "srnl_forward_qraw", "srnl_backward", "srnl_backward_qraw"],
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math", "--split-compile=8", "--threads=4"],
            )
        return _srnl_cuda_fp16
    if io_dtype == torch.float32:
        if _srnl_cuda_fp32 is None:
            _srnl_cuda_fp32 = load_inline(
                name=_EXTENSION_NAME + "_fp32io",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source_fp32,
                functions=["srnl_forward", "srnl_forward_qraw", "srnl_backward", "srnl_backward_qraw"],
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math", "--split-compile=8", "--threads=4"],
            )
        return _srnl_cuda_fp32
    if _srnl_cuda is None:
        _srnl_cuda = load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["srnl_forward", "srnl_forward_qraw", "srnl_backward", "srnl_backward_qraw"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "--split-compile=8", "--threads=4"],
        )
    return _srnl_cuda


def load_cuda_head64(cmax=MAX_CHUNK, io_dtype=torch.bfloat16):
    global _srnl_cuda_d64, _srnl_cuda_d64_fp16, _srnl_cuda_d64_fp32
    io_dtype = _check_io_dtype(io_dtype)
    cmax = int(cmax)
    if cmax not in {8, 16, 32, 64}:
        cmax = MAX_CHUNK
    cache = _srnl_cuda_d64_fp32 if io_dtype == torch.float32 else (_srnl_cuda_d64_fp16 if io_dtype == torch.float16 else _srnl_cuda_d64)
    if cmax not in cache:
        suffix = "_fp32io" if io_dtype == torch.float32 else ("_fp16io" if io_dtype == torch.float16 else "")
        source = d64_cuda_source_fp32 if io_dtype == torch.float32 else (d64_cuda_source_fp16 if io_dtype == torch.float16 else d64_cuda_source)
        cache[cmax] = load_inline(
            name=_EXTENSION_NAME + f"_d64full_nounroll_o2c{cmax}" + suffix,
            cpp_sources=cpp_source,
            cuda_sources=source.replace("__D64_CMAX__", str(cmax)),
            functions=["srnl_forward", "srnl_forward_qraw", "srnl_backward", "srnl_backward_qraw"],
            verbose=False,
            extra_cuda_cflags=["-O2", "--use_fast_math", "--threads=4"],
        )
    return cache[cmax]


class SRNLNormalizedQueryFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd_cuda
    @on_tensor_device
    def forward(ctx, M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q, chunk_size):
        if torch.jit.is_tracing():
            raise RuntimeError("SRNL TorchScript tracing is unsupported; use eager or CUDA Graph inference.")
        if not x.is_cuda:
            raise RuntimeError("SRNL CUDA kernel requires CUDA tensors.")

        state_dtypes = (M_m_0.dtype, M_v_0.dtype, M_k_0.dtype, M_eta_0.dtype, M_alpha_0.dtype)
        input_dtypes = (x.dtype, q.dtype)
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"SRNL CUDA kernel supports fp16/bf16/fp32 x, got {x.dtype}.")
        if q.dtype != torch.float32:
            raise RuntimeError(f"SRNL normalized q path expects fp32 q, got {q.dtype}.")
        B, directions, num_heads, seq_len, head_dim = x.shape
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise AssertionError(
                f"SRNL CUDA kernel requires head_dim in {SUPPORTED_HEAD_DIMS}, got {head_dim}."
            )
        if chunk_size > MAX_CHUNK:
            raise ValueError(f"chunk_size={chunk_size} exceeds MAX_CHUNK={MAX_CHUNK} in CUDA kernel")

        n_flat = B * directions * num_heads
        if M_m_0.dim() == 4:
            state_count = directions * num_heads
        elif M_m_0.dim() == 5:
            state_count = n_flat
        else:
            raise AssertionError(
                f"Expected memory tensors with 4 or 5 dims, got shape {tuple(M_m_0.shape)}."
            )

        pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            q = F.pad(q, (0, 0, 0, pad_len))
            seq_len = x.shape[-2]

        num_chunks = seq_len // chunk_size
        chunk_size_int, head_dim_int = int(chunk_size), int(head_dim)

        M_m_0_f = M_m_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_v_0_f = M_v_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_k_0_f = M_k_0.contiguous().view(state_count, head_dim, head_dim).to(dtype=torch.float32)
        M_eta_0_f = M_eta_0.contiguous().view(state_count, head_dim).to(dtype=torch.float32)
        M_alpha_0_f = M_alpha_0.contiguous().view(state_count, head_dim).to(dtype=torch.float32)
        x_b = x.contiguous().view(n_flat, seq_len, head_dim)
        q_f = q.contiguous().view(n_flat, seq_len, head_dim)

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
        save_g = 0
        G_token_saved = torch.empty((1,), device=x.device, dtype=torch.float32)
        G_final_chunk = torch.empty((1,), device=x.device, dtype=torch.float32)

        srnl_cuda = load_cuda_head64(MAX_CHUNK, x_b.dtype) if head_dim_int == 64 else load_cuda(x_b.dtype)
        y_f = srnl_cuda.srnl_forward(
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_f,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            G_token_saved,
            G_final_chunk,
            chunk_size_int,
            head_dim_int,
            state_count,
            save_g,
        )

        ctx.save_for_backward(
            M_m_0_f,
            M_v_0_f,
            M_k_0_f,
            M_eta_0_f,
            M_alpha_0_f,
            x_b,
            q_f,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            G_token_saved,
            G_final_chunk,
        )
        ctx.constants = (chunk_size_int, head_dim_int, pad_len, state_count, save_g)
        ctx.orig_shapes = (M_m_0.shape, M_eta_0.shape, x.shape)
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
            q_f,
            G_prefix_chunk,
            K_saved,
            V_saved,
            Eta_saved,
            Alpha_saved,
            EtaGrad_saved,
            AlphaGrad_saved,
            InvK_saved,
            G_token_saved,
            G_final_chunk,
        ) = ctx.saved_tensors
        chunk_size_int, head_dim_int, pad_len, state_count, save_g = ctx.constants
        M_shape, scalar_shape, x_shape = ctx.orig_shapes
        M_m_dtype, M_v_dtype, M_k_dtype, M_eta_dtype, M_alpha_dtype, x_dtype, q_dtype = ctx.orig_dtypes
        n_flat, seq_len = x_b.shape[0], x_b.shape[1]

        if pad_len > 0:
            dY = F.pad(dY, (0, 0, 0, pad_len))
        dY_b = dY.contiguous().view(n_flat, seq_len, head_dim_int).to(dtype=x_b.dtype)

        srnl_cuda = load_cuda_head64(MAX_CHUNK, x_b.dtype) if head_dim_int == 64 else load_cuda(x_b.dtype)
        dM_m_0, dM_v_0, dM_k_0, dM_eta_0, dM_alpha_0, dX, dQ = srnl_cuda.srnl_backward(
            dY_b,
            x_b,
            q_f,
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
            G_token_saved,
            G_final_chunk,
            chunk_size_int,
            head_dim_int,
            state_count,
            save_g,
        )

        dX = dX.view(x_shape)
        dQ = dQ.view(x_shape)
        if pad_len > 0:
            dX = dX[:, :, :, :-pad_len, :]
            dQ = dQ[:, :, :, :-pad_len, :]

        return (
            dM_m_0.view(M_shape).to(dtype=M_m_dtype),
            dM_v_0.view(M_shape).to(dtype=M_v_dtype),
            dM_k_0.view(M_shape).to(dtype=M_k_dtype),
            dM_eta_0.view(scalar_shape).to(dtype=M_eta_dtype),
            dM_alpha_0.view(scalar_shape).to(dtype=M_alpha_dtype),
            dX.to(dtype=x_dtype),
            dQ.to(dtype=q_dtype),
            None,
        )


class SRNLRawQueryFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd_cuda
    @on_tensor_device
    def forward(ctx, M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q_raw, chunk_size, num_heads, head_dim):
        if torch.jit.is_tracing():
            raise RuntimeError("SRNL TorchScript tracing is unsupported; use eager or CUDA Graph inference.")
        if not x.is_cuda:
            raise RuntimeError("SRNL qraw CUDA kernel requires CUDA tensors.")

        state_dtypes = (M_m_0.dtype, M_v_0.dtype, M_k_0.dtype, M_eta_0.dtype, M_alpha_0.dtype)
        input_dtypes = (x.dtype, q_raw.dtype)
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"SRNL qraw CUDA kernel supports fp16/bf16/fp32 x, got {x.dtype}.")
        if q_raw.dtype != x.dtype:
            raise RuntimeError(f"SRNL qraw CUDA kernel expects q_raw dtype to match x dtype, got {q_raw.dtype} vs {x.dtype}.")
        B, directions, x_num_heads, seq_len, x_head_dim = x.shape
        num_heads = int(num_heads)
        head_dim = int(head_dim)
        if directions != 4 or x_num_heads != num_heads or x_head_dim != head_dim:
            raise AssertionError("Invalid x shape for SRNL qraw kernel.")
        if q_raw.dim() != 4 or q_raw.shape[0] != B or q_raw.shape[1] != 4 or q_raw.shape[2] != seq_len:
            raise AssertionError("Invalid q_raw shape for SRNL qraw kernel.")
        if q_raw.shape[3] != num_heads * head_dim:
            raise AssertionError("q_raw channel count does not match num_heads * head_dim.")
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise AssertionError(
                f"SRNL CUDA kernel requires head_dim in {SUPPORTED_HEAD_DIMS}, got {head_dim}."
            )
        if chunk_size > MAX_CHUNK:
            raise ValueError(f"chunk_size={chunk_size} exceeds MAX_CHUNK={MAX_CHUNK} in CUDA kernel")

        n_flat = B * directions * num_heads
        if M_m_0.dim() == 4:
            state_count = directions * num_heads
        elif M_m_0.dim() == 5:
            state_count = n_flat
        else:
            raise AssertionError(
                f"Expected memory tensors with 4 or 5 dims, got shape {tuple(M_m_0.shape)}."
            )

        pad_len = (int(chunk_size) - seq_len % int(chunk_size)) % int(chunk_size)
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            q_raw = F.pad(q_raw, (0, 0, 0, pad_len))
            seq_len = x.shape[-2]

        num_chunks = seq_len // int(chunk_size)
        chunk_size_int, head_dim_int = int(chunk_size), int(head_dim)

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
        save_g = 0
        G_token_saved = torch.empty((1,), device=x.device, dtype=torch.float32)
        G_final_chunk = torch.empty((1,), device=x.device, dtype=torch.float32)

        use_cmax = 1
        d64_cmax = d64_cmax_for_chunk(chunk_size_int, use_cmax)
        srnl_cuda = load_cuda_head64(d64_cmax, x_b.dtype) if head_dim_int == 64 else load_cuda(x_b.dtype)
        y_f = srnl_cuda.srnl_forward_qraw(
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
            G_token_saved,
            G_final_chunk,
            QInv_saved,
            chunk_size_int,
            head_dim_int,
            state_count,
            save_g,
            num_heads,
            save_qinv,
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
            G_token_saved,
            G_final_chunk,
        )
        ctx.constants = (
            chunk_size_int,
            head_dim_int,
            pad_len,
            state_count,
            save_g,
            num_heads,
            save_qinv,
            bwd_groups_for_head_dim(head_dim_int),
            use_cmax,
            d64_cmax,
        )
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
            G_token_saved,
            G_final_chunk,
        ) = ctx.saved_tensors
        chunk_size_int, head_dim_int, pad_len, state_count, save_g, num_heads, save_qinv, bwd_groups, use_cmax, d64_cmax = ctx.constants
        M_shape, scalar_shape, x_shape, q_raw_shape = ctx.orig_shapes
        M_m_dtype, M_v_dtype, M_k_dtype, M_eta_dtype, M_alpha_dtype, x_dtype, q_raw_dtype = ctx.orig_dtypes
        n_flat, seq_len = x_b.shape[0], x_b.shape[1]

        if pad_len > 0:
            dY = F.pad(dY, (0, 0, 0, pad_len))
        dY_b = dY.contiguous().view(n_flat, seq_len, head_dim_int).to(dtype=x_b.dtype)

        srnl_cuda = load_cuda_head64(d64_cmax, x_b.dtype) if head_dim_int == 64 else load_cuda(x_b.dtype)
        dM_m_0, dM_v_0, dM_k_0, dM_eta_0, dM_alpha_0, dX, dQRaw = srnl_cuda.srnl_backward_qraw(
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
            G_token_saved,
            G_final_chunk,
            QInv_saved,
            chunk_size_int,
            head_dim_int,
            state_count,
            save_g,
            num_heads,
            save_qinv,
            bwd_groups,
            use_cmax,
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


def bwd_groups_for_head_dim(head_dim):
    # Pack independent heads into one warp.
    return max(1, 32 // int(head_dim))


def d64_cmax_for_chunk(chunk_size, use_cmax):
    if not use_cmax:
        return MAX_CHUNK
    return min(MAX_CHUNK, max(8, 1 << (int(chunk_size) - 1).bit_length()))


def can_fuse_raw_query(x, q_raw, num_heads, head_dim):
    if not (x.is_cuda and q_raw.is_cuda):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or q_raw.dtype != x.dtype:
        return False
    if x.dim() != 5 or q_raw.dim() != 4:
        return False
    if x.shape[1] != 4 or q_raw.shape[1] != 4:
        return False
    if int(head_dim) not in SUPPORTED_HEAD_DIMS:
        return False
    return (
        int(x.shape[0]) == int(q_raw.shape[0])
        and int(x.shape[3]) == int(q_raw.shape[2])
        and int(x.shape[2]) == int(num_heads)
        and int(x.shape[4]) == int(head_dim)
        and int(q_raw.shape[3]) == int(num_heads) * int(head_dim)
    )


def srnl_raw_query(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q_raw, chunk_size, num_heads, head_dim):
    return SRNLRawQueryFunction.apply(
        M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q_raw, chunk_size, int(num_heads), int(head_dim)
    )


def srnl_normalized_query(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q, chunk_size):
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32) and q.dtype == torch.float32:
        return SRNLNormalizedQueryFunction.apply(M_m_0, M_v_0, M_k_0, M_eta_0, M_alpha_0, x, q, chunk_size)
    raise RuntimeError("The release HOPE CUDA path expects fp16/bf16/fp32 x and fp32 normalized q.")


srnl_normalized_query.supports_broadcast_state = True
