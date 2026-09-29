"""FP32 SRNL CUDA recurrence with first-order autograd.

The public trainable module is :class:`visionhope.models.SRNL`. These functions
expose the packed recurrence used by that module and its directional operator.
"""

from .chunk import MAX_CHUNK, SUPPORTED_HEAD_DIMS, srnl_normalized_query, srnl_raw_query
from .lines import srnl_long_chunk, srnl_variable_lines

__all__ = [
    "MAX_CHUNK", "SUPPORTED_HEAD_DIMS", "srnl_normalized_query", "srnl_raw_query",
    "srnl_long_chunk", "srnl_variable_lines",
]
