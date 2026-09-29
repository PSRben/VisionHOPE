"""Self-referential nested learning (SRNL), including its five learned memories."""

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

from .srnl_core import (
    MAX_CHUNK,
    SUPPORTED_HEAD_DIMS,
    srnl_long_chunk,
    srnl_normalized_query,
    srnl_raw_query,
    srnl_variable_lines,
)
from .srnl_core.chunk import can_fuse_raw_query
from visionhope.utils.query_normalization import can_fuse_qnorm, visionhope_qnorm


class SRNL(nn.Module):
    """Chunkwise five-memory recurrence with FP32 state and CUDA autograd.

    A standalone instance accepts ``x, queries`` of shape ``[B, N, dim]`` and
    returns that shape. Queries are unnormalized; omitting them uses ``x``.
    ``chunk_size`` is a positive integer. With ``None``, standalone sequences
    use chunks of 64 tokens, while image scans use their actual row/column
    lengths. Padding affects only the final chunk and is removed from output.

    The four-direction operator uses the packed interface: inputs of shape
    ``[B, directions, heads, N, head_dim]`` and queries ``[B, directions, N, dim]``.
    Initial memories are learned; recurrent trajectories are local to each
    forward and never cached between images. First-order gradients are supported.
    ``fast_inference`` enables the inference kernel in eval mode with gradients
    disabled. Training and gradient-enabled calls retain the ordinary path.
    """

    def __init__(self, dim, head_dim=16, chunk_size=None, directions=1, *,
                 fast_inference=False, _initialize=True):
        super().__init__()
        if head_dim not in SUPPORTED_HEAD_DIMS or dim <= 0 or dim % head_dim:
            raise ValueError(f"dim must be positive and divisible by head_dim in {SUPPORTED_HEAD_DIMS}")
        if directions not in (1, 2, 4):
            raise ValueError("directions must be 1, 2, or 4")
        if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0):
            raise ValueError("chunk_size must be a positive integer or None")
        if not isinstance(fast_inference, bool):
            raise TypeError("fast_inference must be a bool")
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.num_heads = self.dim // self.head_dim
        self.directions = int(directions)
        self.chunk_size = chunk_size
        self.fast_inference = fast_inference
        self.memory_init_std = 0.01
        matrix_shape = (directions, self.num_heads, head_dim, head_dim)
        vector_shape = (directions, self.num_heads, head_dim)
        self.M_m_0 = nn.Parameter(torch.empty(matrix_shape))
        self.M_v_0 = nn.Parameter(torch.empty(matrix_shape))
        self.M_k_0 = nn.Parameter(torch.empty(matrix_shape))
        self.M_eta_0 = nn.Parameter(torch.empty(vector_shape))
        self.M_alpha_0 = nn.Parameter(torch.empty(vector_shape))
        if _initialize:
            self.reset_parameters()

    @property
    def initial_memories(self):
        """Content, value, key, learning-rate, and retention memories, in core order."""
        return self.M_m_0, self.M_v_0, self.M_k_0, self.M_eta_0, self.M_alpha_0

    def reset_parameters(self):
        for memory in self.initial_memories[:3]:
            trunc_normal_(memory, std=self.memory_init_std)
        nn.init.zeros_(self.M_eta_0)
        nn.init.zeros_(self.M_alpha_0)

    def _inference_path(self, x, query_dtype):
        if not self.fast_inference:
            return "ordinary", "Fast inference is disabled"
        if self.training:
            return "ordinary", "The module is in training mode"
        if torch.is_grad_enabled():
            return "ordinary", "Gradients are enabled"
        if not x.is_cuda:
            return "ordinary", "SRNL requires CUDA inputs"
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return "ordinary", "SRNL requires FP16, BF16, or FP32 inputs"
        if query_dtype != x.dtype:
            return "ordinary", "Fast inference requires matching input and query dtypes"
        return "sequence", None

    def inference_status(self, x, queries=None):
        """Report path selection for valid inputs in the current eval/grad context."""
        path, reason = self._inference_path(x, x.dtype if queries is None else queries.dtype)
        return {"path": path, "reason": reason}

    def _validate_memories(self, device):
        states = self.initial_memories
        matrix_shape = (self.directions, self.num_heads, self.head_dim, self.head_dim)
        vector_shape = (self.directions, self.num_heads, self.head_dim)
        for i, state in enumerate(states):
            expected = matrix_shape if i < 3 else vector_shape
            if (not isinstance(state, torch.Tensor) or tuple(state.shape) != expected
                    or not state.is_floating_point() or state.device != device):
                raise ValueError(f"SRNL memory {i} must be floating point, shape {expected}, on {device}")
        return states

    def _run(self, x, queries, spatial_shape):
        batch, directions, heads, length, head_dim = x.shape
        if (directions, heads, head_dim) != (self.directions, self.num_heads, self.head_dim):
            raise ValueError("Packed input dimensions do not match this SRNL instance")
        if queries.shape != (batch, directions, length, self.dim):
            raise ValueError("Packed queries must have shape [B, directions, N, dim]")
        if not x.is_cuda or x.device != queries.device:
            raise ValueError("SRNL inputs and memories must be on the same CUDA device")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("SRNL input must be FP16, BF16, or FP32")
        if queries.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("SRNL queries must be FP16, BF16, or FP32")
        states = self._validate_memories(x.device)
        if batch <= 0 or length <= 0:
            raise ValueError("SRNL requires nonempty batches and sequences")
        if spatial_shape is not None:
            height, width = spatial_shape
            if min(height, width) <= 0 or height * width != length or directions != 4:
                raise ValueError("spatial_shape must match the four serialized image scans")
        else:
            height = width = None
        chunk_size = self.chunk_size
        if chunk_size is None:
            chunk_size = width if width is not None else MAX_CHUNK

        path, _ = self._inference_path(x, queries.dtype)
        if path == "sequence":
            from .srnl_core.fused import srnl_sequence_inference

            chunks = [chunk_size] * directions
            if self.chunk_size is None and spatial_shape is not None:
                chunks = [width, width, height, height]
            return srnl_sequence_inference(states, x, queries, chunks)

        if can_fuse_raw_query(x, queries, heads, head_dim):
            if self.chunk_size is None and height != width:
                return srnl_variable_lines(*states, x, queries, width, height, heads, head_dim)
            kernel = srnl_raw_query if chunk_size <= MAX_CHUNK else srnl_long_chunk
            return kernel(*states, x, queries, chunk_size, heads, head_dim)

        if chunk_size > MAX_CHUNK:
            # Normalize queries within the long-chunk kernel.
            if x.dtype != queries.dtype:
                raise TypeError("Long chunks require queries and input to have the same dtype")
            return srnl_long_chunk(*states, x, queries, chunk_size, heads, head_dim)

        if can_fuse_qnorm(queries, heads, head_dim):
            normalized = visionhope_qnorm(queries, heads, head_dim)
        else:
            normalized = queries.view(batch, directions, length, heads, head_dim)
            normalized = F.normalize(normalized.transpose(2, 3).float(), p=2, dim=-1).contiguous()
        if self.chunk_size is None and spatial_shape is not None and height != width:
            # Process row and column scans with their respective chunk lengths.
            outputs = []
            for start, size in ((0, width), (2, height)):
                if size > MAX_CHUNK:
                    raise TypeError("Long rectangular scans require matching input/query dtypes")
                sliced = tuple(state[start:start + 2] for state in states)
                outputs.append(srnl_normalized_query(
                    *sliced, x[:, start:start + 2], normalized[:, start:start + 2], size
                ))
            return torch.cat(outputs, dim=1)
        return srnl_normalized_query(*states, x, normalized, chunk_size)

    def forward(self, x, queries=None, *, spatial_shape=None):
        if x.ndim == 5:
            if queries is None:
                queries = x.transpose(2, 3).flatten(3)
            return self._run(x, queries, spatial_shape)
        if x.ndim != 3 or self.directions != 1 or x.shape[-1] != self.dim:
            raise ValueError("Standalone SRNL expects [B, N, dim] and directions=1")
        if queries is None:
            queries = x
        if queries.shape != x.shape:
            raise ValueError("Standalone queries must have the same shape as input")
        batch, length, _ = x.shape
        packed = x.view(batch, 1, length, self.num_heads, self.head_dim)
        packed = packed.transpose(2, 3).contiguous()
        output = self._run(packed, queries.unsqueeze(1), spatial_shape)
        return output[:, 0].transpose(1, 2).reshape(batch, length, self.dim)

    def extra_repr(self):
        return (f"dim={self.dim}, head_dim={self.head_dim}, directions={self.directions}, "
                f"chunk_size={self.chunk_size}, fast_inference={self.fast_inference}")
