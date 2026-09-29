"""Shared numerical utilities for VisionHOPE training and inference."""

from .grn import GlobalResponseNorm, global_response_norm, grn_reference

__all__ = ["GlobalResponseNorm", "global_response_norm", "grn_reference"]
