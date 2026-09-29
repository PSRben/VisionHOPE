"""Learning-rate scaling and scheduler restoration."""

from copy import deepcopy
import math
from numbers import Integral, Real


_BASELINES = {
    "classification": (1e-3, 1024),
    "detection": (4e-4, 64),
    "segmentation": (6e-5, 16),
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is too large") from error
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def resolve_learning_rate(task, *, batch_size, world_size, grad_accum_steps=1, lr=None):
    """Return a JSON-compatible learning-rate configuration.

    ``lr`` is an optional absolute target, not a reference rate to scale again.
    Retain the resolved rate in normal checkpoint settings for continuation.
    Warmup and minimum rates are separate recipe settings and are unchanged.
    """
    if not isinstance(task, str) or task not in _BASELINES:
        raise ValueError(f"Unknown training task: {task!r}")
    batch_size = _integer(batch_size, "batch_size")
    world_size = _integer(world_size, "world_size")
    grad_accum_steps = _integer(grad_accum_steps, "grad_accum_steps")
    effective_batch = batch_size * world_size * grad_accum_steps
    base_lr, base_batch_size = _BASELINES[task]
    explicit_lr = None if lr is None else _number(lr, "lr")
    try:
        target = base_lr * effective_batch / base_batch_size if lr is None else explicit_lr
    except OverflowError as error:
        raise ValueError("effective_batch is too large") from error
    return {
        "task": task, "batch_size": batch_size, "world_size": world_size,
        "grad_accum_steps": grad_accum_steps, "effective_batch": effective_batch,
        "base_lr": base_lr, "base_batch_size": base_batch_size,
        "lr": _number(target, "resolved lr"), "explicit_lr": explicit_lr,
    }


def prepare_iteration_schedule(task, config, grad_accum_steps):
    """Convert one unscaled dense recipe from update attempts to microbatches.

    Call once on the unscaled recipe, before constructing MMEngine's runner.
    ADE20K's iteration budget, validation/checkpoint periods and schedulers use
    update-attempt units. COCO keeps its epoch schedule and converts only its
    iteration-based warmup. The input is unchanged, including when an error is
    raised. Accumulation one returns an equal deep copy.
    """
    accumulation = _integer(grad_accum_steps, "grad_accum_steps")
    if not isinstance(task, str) or task not in {"detection", "segmentation"}:
        raise ValueError("Iteration conversion is for detection or segmentation")
    result = deepcopy(config)
    if accumulation == 1:
        return result
    loop = result["train_cfg"]
    expected = "EpochBasedTrainLoop" if task == "detection" else "IterBasedTrainLoop"
    if loop.get("type") != expected:
        raise ValueError(f"Expected {expected} before converting accumulation")
    for scheduler in result["param_scheduler"]:
        if not scheduler.get("by_epoch", True):
            allowed = {"LinearLR"} if task == "detection" else {"LinearLR", "PolyLR"}
            if scheduler.get("type") not in allowed:
                raise ValueError("Unsupported iteration scheduler for gradient accumulation")
            scheduler["begin"] = _integer(scheduler.get("begin", 0), "scheduler begin", 0) * accumulation
            scheduler["end"] = _integer(scheduler["end"], "scheduler end") * accumulation
    if task == "segmentation":
        loop["max_iters"] = _integer(loop["max_iters"], "max_iters") * accumulation
        loop["val_interval"] = _integer(loop["val_interval"], "val_interval") * accumulation
        checkpoint = result.get("default_hooks", {}).get("checkpoint")
        if checkpoint and not checkpoint.get("by_epoch", True) and checkpoint.get("interval", -1) > 0:
            checkpoint["interval"] = _integer(checkpoint["interval"], "checkpoint interval") * accumulation
    return result


def _optimizer(value):
    return getattr(value, "optimizer", value)


def rebase_learning_rate(optimizer, schedulers, *, previous_lr, target_lr, epoch=None):
    """Adjust restored optimizer/schedulers to a new absolute target rate.

    Invoke *after* restoring both optimizer and scheduler state, and only when
    changing the training rate; strict continuation should skip this function.
    ``previous_lr`` is the saved checkpoint's resolved target, not its decayed
    current rate. For timm cosine schedules, pass the next training ``epoch``.
    Their absolute warmup/minimum rates are preserved and the current rate is
    recomputed at that epoch. MMEngine LinearLR/MultiStepLR/zero-floor PolyLR
    retain their counters and scale their rate-valued state together.

    Each changed parameter group saves ``lr_reference`` for repeat calls and
    subsequent checkpoint loads. Group multipliers and optimizer moments are
    untouched. Scheduler timelines are not converted during restoration.
    """
    previous_lr = _number(previous_lr, "previous_lr")
    target_lr = _number(target_lr, "target_lr")
    optimizer = _optimizer(optimizer)
    if not optimizer.param_groups:
        raise ValueError("Optimizer has no parameter groups")
    schedulers = list(schedulers) if isinstance(schedulers, (list, tuple)) else [schedulers]
    if not schedulers or len({id(scheduler) for scheduler in schedulers}) != len(schedulers):
        raise ValueError("Provide each scheduler exactly once")
    timm = (len(schedulers) == 1 and type(schedulers[0]).__module__ == "timm.scheduler.cosine_lr"
            and type(schedulers[0]).__name__ == "CosineLRScheduler")
    groups = None
    for scheduler in schedulers:
        if _optimizer(getattr(scheduler, "optimizer", None)) is not optimizer:
            raise ValueError("Scheduler belongs to a different optimizer")
        if timm:
            if not scheduler.t_in_epochs:
                raise ValueError("Only epoch-based timm cosine schedules are supported")
        elif (type(scheduler).__module__ != "mmengine.optim.scheduler.lr_scheduler"
              or type(scheduler).__name__ not in {"LinearLR", "MultiStepLR", "PolyLR"}
              or scheduler.param_name != "lr"):
            raise ValueError("Unsupported scheduler for learning-rate continuation")
        if not timm and getattr(scheduler, "eta_min", 0) != 0:
            raise ValueError("Only zero-floor MMEngine polynomial schedules are supported")
        # MMEngine wrappers can append a virtual group for the base rate.
        scheduler_groups = scheduler.optimizer.param_groups
        if groups is None:
            groups = scheduler_groups
        elif (len(scheduler_groups) != len(groups)
              or any(left is not right for left, right in zip(scheduler_groups, groups))):
            raise ValueError("Schedulers use different optimizer parameter groups")
        if len(scheduler.base_values) != len(groups):
            raise ValueError("Scheduler and optimizer parameter groups differ")
        if not timm and len(scheduler._last_value) != len(groups):
            raise ValueError("Scheduler last values do not match the parameter groups")
    references = [_number(group.get("lr_reference", previous_lr), "lr_reference") for group in groups]
    if any(reference != references[0] for reference in references):
        raise ValueError("Optimizer groups have inconsistent learning-rate references")
    current_reference = references[0]
    ratio = _number(target_lr / current_reference, "learning-rate ratio")
    if ratio == 1:
        return {"previous_lr": current_reference, "lr": target_lr, "ratio": 1.0, "changed": False}
    if timm:
        epoch = _integer(epoch, "epoch", 0)

    # Validate every value before changing any optimizer or scheduler state.
    scaled_groups = [{key: _number(_number(group[key], key, True) * ratio, key, True)
                      for key in ("lr", "initial_lr")} for group in groups]
    scaled_bases = [[_number(_number(value, "base value", True) * ratio, "base value", True)
                     for value in scheduler.base_values] for scheduler in schedulers]
    scaled_last = [] if timm else [
        [_number(_number(value, "last value", True) * ratio, "last value", True)
         for value in scheduler._last_value] for scheduler in schedulers
    ]
    for group, values in zip(groups, scaled_groups):
        group.update(values, lr_reference=target_lr)
    optimizer.defaults["lr"] = target_lr
    for index, scheduler in enumerate(schedulers):
        scheduler.base_values = scaled_bases[index]
        if timm:
            if scheduler.warmup_t:
                scheduler.warmup_steps = [(value - scheduler.warmup_lr_init) / scheduler.warmup_t
                                          for value in scheduler.base_values]
            scheduler.step(epoch, scheduler.metric)
        else:
            scheduler._last_value = scaled_last[index]
    return {"previous_lr": current_reference, "lr": target_lr, "ratio": ratio, "changed": True}
