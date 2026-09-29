"""Training checkpoints with explicit optimizer parameter identities."""

from copy import deepcopy

import torch


def restore_optimizer(optimizer, model, payload, current_names):
    saved = deepcopy(payload["optimizer"])
    saved_names = payload["optimizer_param_names"]
    if len(saved_names) != len(current_names) or len(saved["param_groups"]) != len(saved_names):
        raise ValueError("Optimizer group count differs from the checkpoint")
    for group, previous, current in zip(saved["param_groups"], saved_names, current_names):
        if len(previous) != len(group["params"]) or set(previous) != set(current):
            raise ValueError("Optimizer parameter identities differ from the checkpoint")
        identifiers = dict(zip(previous, group["params"]))
        group["params"] = [identifiers[name] for name in current]
    optimizer.load_state_dict(saved)


def restore_training(model, optimizer, scaler, ema, scheduler, path, parameter_names):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "optimizer", "optimizer_param_names", "amp_scaler",
                "scheduler", "epoch", "config", "checkpoint_queue", "history"}
    if required - payload.keys():
        raise ValueError("Resume requires a complete training checkpoint; use --initial-checkpoint for weights")
    model.load_state_dict(payload["state_dict"], strict=True)
    restore_optimizer(optimizer, model, payload, parameter_names)
    scaler.load_state_dict(payload["amp_scaler"])
    if ema is not None:
        if "state_dict_ema" not in payload:
            raise ValueError("EMA training resume requires state_dict_ema")
        ema.module.load_state_dict(payload["state_dict_ema"], strict=True)
    scheduler.load_state_dict(payload["scheduler"])
    return payload
