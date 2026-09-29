"""Inference-only CUDA execution for spatial scans and general SRNL sequences."""

from __future__ import annotations

from pathlib import Path

from visionhope.utils.cuda import CUDA_GROUP_SOURCE, on_tensor_device

import torch
from torch.utils.cpp_extension import load_inline

from .chunk import (
    RETENTION_BIAS,
    MIN_RETENTION,
    SPECTRAL_CLAMP,
    SPECTRAL_MARGIN,
    STEP_SCALE,
    INJECTION_MARGIN,
    _cuda_float_literal,
    _cuda_source_io_variant,
    _tag_float,
)


_EXTENSION_NAME = (
    "srnl_inference"
    f"_eta{_tag_float(STEP_SCALE)}"
    f"_amin{_tag_float(MIN_RETENTION)}"
    f"_abias{_tag_float(RETENTION_BIAS)}"
    f"_mu{_tag_float(INJECTION_MARGIN)}"
    f"_e2a{int(SPECTRAL_CLAMP)}"
    f"_e2am{_tag_float(SPECTRAL_MARGIN)}"
)
_extensions = {}

cuda_source = (Path(__file__).parent / "csrc" / "fused.cu").read_text()

cuda_source = (
    cuda_source
    .replace("__SRNL_INFERENCE_SOURCE__", CUDA_GROUP_SOURCE
             + (Path(__file__).parent / "csrc" / "inference_recurrence.cuh").read_text()
             + (Path(__file__).parent / "csrc" / "parallel_line.cuh").read_text())
    .replace("__ETA_SCALE__", _cuda_float_literal(STEP_SCALE))
    .replace("__ALPHA_MIN__", _cuda_float_literal(MIN_RETENTION))
    .replace("__ALPHA_BIAS__", _cuda_float_literal(RETENTION_BIAS))
    .replace("__STAB_MU__", _cuda_float_literal(INJECTION_MARGIN))
    .replace("__ETA2ALPHA_CLAMP__", "1" if SPECTRAL_CLAMP else "0")
    .replace("__ETA2ALPHA_FACTOR__", _cuda_float_literal(2.0 * SPECTRAL_MARGIN))
)
cuda_source_fp16 = _cuda_source_io_variant(cuda_source, torch.float16)
cuda_source_fp32 = _cuda_source_io_variant(cuda_source, torch.float32)

cpp_source = (Path(__file__).parent / "csrc" / "fused.cpp").read_text()


def load_fused_cuda(io_dtype=torch.bfloat16):
    variants = {
        torch.bfloat16: ("", cuda_source),
        torch.float16: ("_fp16io", cuda_source_fp16),
        torch.float32: ("_fp32io", cuda_source_fp32),
    }
    if io_dtype not in variants:
        raise TypeError(f"SRNL inference supports FP16, BF16, and FP32, got {io_dtype}")
    if io_dtype not in _extensions:
        suffix, source = variants[io_dtype]
        _extensions[io_dtype] = load_inline(
            name=_EXTENSION_NAME + suffix,
            cpp_sources=cpp_source,
            cuda_sources=source,
            functions=[
                "srnl_line_forward",
                "srnl_line_forward_noatomic",
                "srnl_line_forward_seq",
                "srnl_line_forward_seq_noatomic",
                "srnl_sequence_forward",
            ],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _extensions[io_dtype]


@on_tensor_device
def srnl_sequence_inference(states, x, queries, chunk_sizes):
    """Read a packed sequence without allocating backward intermediates."""
    if torch.jit.is_tracing():
        raise RuntimeError("SRNL TorchScript tracing is unsupported; use eager or CUDA Graph inference")
    if torch.is_grad_enabled():
        raise RuntimeError("SRNL inference requires disabled gradients")
    memories = tuple(state.contiguous().float() for state in states)
    return load_fused_cuda(x.dtype).srnl_sequence_forward(
        *memories, x.contiguous(), queries.contiguous(), chunk_sizes,
    )
