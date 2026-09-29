"""Read and write clean weights using the model's native parameter names."""

from collections import OrderedDict
from collections.abc import Mapping

import torch


def _validate_state_dict(state):
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state_dict must be a nonempty mapping of tensor names to tensors")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
           for key, value in state.items()):
        raise TypeError("state_dict must contain only named tensors")


def read_weights(path, *, map_location="cpu"):
    """Read the sole supported weight container: ``{'state_dict': model.state_dict()}``."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"state_dict"}:
        raise ValueError("A VisionHOPE weight file must contain only 'state_dict'")
    state = payload["state_dict"]
    _validate_state_dict(state)
    return state


def save_weights(path, state_dict):
    """Save parameters and buffers on CPU, preserving names and module metadata."""
    _validate_state_dict(state_dict)
    state = OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items())
    if hasattr(state_dict, "_metadata"):
        state._metadata = state_dict._metadata
    torch.save({"state_dict": state}, path)


def load_checkpoint(model, path, *, strict=True, map_location="cpu"):
    """Load native model keys directly; no field selection or key conversion."""
    return model.load_state_dict(read_weights(path, map_location=map_location), strict=strict)
