"""AdamW parameter groups for VisionHOPE training."""

def parameter_group(name, parameter):
    """Assign a parameter to its optimizer group."""
    if any(key in name for key in ("M_eta_0", "M_alpha_0")):
        return "gate"
    if any(key in name for key in ("M_m_0", "M_v_0", "M_k_0")):
        return "memory"
    if (parameter.ndim <= 1 or name.endswith(".bias") or "d_skip" in name
            or "direction_scale" in name or "gamma" in name or "grn.beta" in name):
        return "no_decay"
    return "decay"


def parameter_groups(model, config):
    grouped = {name: [] for name in ("no_decay", "decay", "memory", "gate")}
    names = {name: [] for name in grouped}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = parameter_group(name, parameter)
            grouped[group].append(parameter)
            names[group].append(name)
    settings = {
        "no_decay": (0.0, config.lr),
        "decay": (config.weight_decay, config.lr),
        "memory": (config.memory_weight_decay, config.lr * config.memory_lr_scale),
        "gate": (config.gate_weight_decay, config.lr * config.gate_lr_scale),
    }
    groups = []
    for name, parameters in grouped.items():
        decay, lr = settings[name]
        groups.append({"params": parameters, "weight_decay": decay, "lr": lr})
    return groups, [names[name] for name in grouped]
