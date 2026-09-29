"""MESA distillation from an EMA teacher."""

import torch
import torch.nn.functional as F


def active(config, epoch):
    return (config.mesa and epoch >= config.mesa_start_epoch
            and (config.mesa_end_epoch < 0 or epoch < config.mesa_end_epoch))


def distillation_loss(student, teacher, kind="softce", temperature=5.0):
    if kind == "softce":
        probability = F.softmax(teacher.float(), dim=-1).detach()
        log_probability = F.log_softmax(student.float(), dim=-1)
        return torch.sum(-probability * log_probability, dim=-1).mean()
    if kind == "kl":
        log_probability = F.log_softmax(student.float() / temperature, dim=-1)
        probability = F.softmax(teacher.float() / temperature, dim=-1).detach()
        return F.kl_div(log_probability, probability, reduction="batchmean") * (temperature * temperature)
    raise ValueError(f"Unknown MESA loss {kind!r}")
