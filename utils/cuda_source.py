"""CUDA source specialization for input and output dtypes."""

import torch

_FP32_IO_REPLACEMENTS = (
    ("data_ptr<at::BFloat16>()", "data_ptr<float>()"),
    ("at::ScalarType::BFloat16", "at::ScalarType::Float"),
    ("torch::kBFloat16", "torch::kFloat32"),
    ("__bfloat162float(", "("),
    ("__float2bfloat16(", "("),
    ("__nv_bfloat16", "float"),
    ("bf16 X", "fp32 X"),
    ("bf16 dY", "fp32 dY"),
    ("bf16 Y", "fp32 Y"),
    ("bf16 QRaw", "fp32 QRaw"),
    ("bf16 Q_raw", "fp32 Q_raw"),
    ("bf16 dY/X/QRaw", "fp32 dY/X/QRaw"),
    ("bf16 dY/X", "fp32 dY/X"),
)
_FP32_IO_REQUIRED_MARKERS = (
    "__nv_bfloat16",
    "at::ScalarType::BFloat16",
    "data_ptr<at::BFloat16>()",
)
_FP32_IO_FORBIDDEN_MARKERS = (
    "__nv_bfloat16",
    "__bfloat162float",
    "__float2bfloat16",
    "at::ScalarType::BFloat16",
    "data_ptr<at::BFloat16>()",
    "torch::kBFloat16",
)
_FP16_IO_REPLACEMENTS = (
    ("data_ptr<at::BFloat16>()", "data_ptr<at::Half>()"),
    ("at::ScalarType::BFloat16", "at::ScalarType::Half"),
    ("torch::kBFloat16", "torch::kFloat16"),
    ("__bfloat162float(", "__half2float("),
    ("__float2bfloat16(", "__float2half_rn("),
    ("__nv_bfloat16", "__half"),
    ("bf16 X", "fp16 X"),
    ("bf16 dY", "fp16 dY"),
    ("bf16 Y", "fp16 Y"),
    ("bf16 QRaw", "fp16 QRaw"),
    ("bf16 Q_raw", "fp16 Q_raw"),
    ("bf16 dY/X/QRaw", "fp16 dY/X/QRaw"),
    ("bf16 dY/X", "fp16 dY/X"),
)
_FP16_IO_REQUIRED_MARKERS = _FP32_IO_REQUIRED_MARKERS
_FP16_IO_FORBIDDEN_MARKERS = _FP32_IO_FORBIDDEN_MARKERS


def _cuda_source_io_variant(source, io_dtype):
    """Build CUDA source for the requested input and output dtype."""
    if io_dtype == torch.bfloat16:
        return source
    if io_dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"Unsupported CUDA I/O source dtype: {io_dtype}.")

    if io_dtype == torch.float16:
        replacements = _FP16_IO_REPLACEMENTS
        required_markers = _FP16_IO_REQUIRED_MARKERS
        forbidden_markers = _FP16_IO_FORBIDDEN_MARKERS
        dtype_name = "fp16"
    else:
        replacements = _FP32_IO_REPLACEMENTS
        required_markers = _FP32_IO_REQUIRED_MARKERS
        forbidden_markers = _FP32_IO_FORBIDDEN_MARKERS
        dtype_name = "fp32"

    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise RuntimeError(
            f"Cannot build {dtype_name} I/O CUDA source: source is missing expected bf16 markers "
            f"{missing}. This usually means the source was already converted or changed."
        )

    converted = source
    if io_dtype == torch.float16 and "#include <cuda_fp16.h>" not in converted:
        converted = "#include <cuda_fp16.h>\n" + converted
    applied = []
    for old, new in replacements:
        count = converted.count(old)
        if count:
            converted = converted.replace(old, new)
            applied.append(old)

    remaining = [marker for marker in forbidden_markers if marker in converted]
    if remaining:
        raise RuntimeError(
            f"Incomplete {dtype_name} I/O CUDA source conversion; remaining bf16 markers: "
            f"{remaining}."
        )
    if len(applied) < 6:
        raise RuntimeError(
            f"Suspicious {dtype_name} I/O CUDA source conversion: too few marker classes changed "
            f"({applied})."
        )
    return converted


def _check_io_dtype(io_dtype):
    if io_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"HOPE CUDA release path supports fp16, bf16, and fp32 I/O, got {io_dtype}.")
    return io_dtype

