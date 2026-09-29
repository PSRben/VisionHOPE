"""CUDA device guards and thread-group source."""

from pathlib import Path

from functools import wraps

import torch


def on_tensor_device(function):
    """Run custom launches on the tensor's device, preserving its current stream."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        tensor = next((value for value in (*args, *kwargs.values())
                       if isinstance(value, torch.Tensor)), None)
        if tensor is None or not tensor.is_cuda:
            return function(*args, **kwargs)
        if tensor.device.index == torch.cuda.current_device():
            return function(*args, **kwargs)
        with torch.cuda.device(tensor.device):
            return function(*args, **kwargs)
    return wrapped

CUDA_GROUP_SOURCE = (Path(__file__).parent / "csrc" / "thread_groups.cuh").read_text()
