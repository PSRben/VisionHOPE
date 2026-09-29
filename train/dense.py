"""MMEngine training for Mask R-CNN and UPerNet."""

from copy import deepcopy
from pathlib import Path
import os

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.loops import EpochBasedTrainLoop

from visionhope.inference.provenance import write_json
from visionhope.tasks.config import dense_config
from visionhope.tools.checkpoint import read_weights
from .dense_checkpoint import RankedCheckpointHook
from .learning_rate import prepare_iteration_schedule, rebase_learning_rate, resolve_learning_rate


class VisionHOPERunner(Runner):
    """Restore full model and optimizer state by parameter identity."""

    def load_checkpoint(self, filename, map_location="cpu", strict=True, revise_keys=None):
        if not strict or revise_keys is not None:
            raise ValueError("Task checkpoints require strict loading with native parameter names")
        if self._resume:
            payload = torch.load(filename, map_location=map_location, weights_only=False)
            required = {"state_dict", "meta", "message_hub", "optimizer", "param_schedulers", "optimizer_param_names",
                        "checkpoint_queue", "training_settings", "gradients"}
            if required - payload.keys():
                raise ValueError("Resume requires a complete training checkpoint, not inference-only weights")
            if payload["training_settings"]["grad_accum_steps"] != self.cfg.optim_wrapper.accumulative_counts:
                raise ValueError("Resume must keep grad_accum_steps; start a new schedule to change accumulation")
            self._resumed_learning_rate = payload["training_settings"]["lr"]
        else:
            payload = {"state_dict": read_weights(filename, map_location=map_location)}
        self.call_hook("after_load_checkpoint", checkpoint=payload)
        model = self.model.module if is_model_wrapper(self.model) else self.model
        model.load_state_dict(payload["state_dict"], strict=True)
        if self._resume:
            parameters = dict(model.named_parameters())
            identities = {id(parameter): name for name, parameter in parameters.items()}
            current = [[identities[id(parameter)] for parameter in group["params"]]
                       for group in self.optim_wrapper.optimizer.param_groups]
            previous = payload["optimizer_param_names"]
            saved = deepcopy(payload["optimizer"])
            if len(previous) != len(saved["param_groups"]) or len(current) != len(previous):
                raise ValueError("Dense optimizer checkpoint has incompatible parameter groups")
            previous_groups = {}
            for names, group in zip(previous, saved["param_groups"]):
                if len(names) != len(group["params"]) or tuple(names) in previous_groups:
                    raise ValueError("Invalid saved optimizer parameter identities")
                previous_groups[tuple(names)] = group
            if set(previous_groups) != {tuple(names) for names in current}:
                raise ValueError("Dense optimizer parameters differ from the checkpoint")
            saved["param_groups"] = [previous_groups[tuple(names)] for names in current]
            payload["optimizer"] = saved
        self._has_loaded = True
        return payload


class OptimizerIdentityHook(Hook):
    def before_save_checkpoint(self, runner, checkpoint):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        checkpoint["optimizer_param_names"] = [
            [names[id(parameter)] for parameter in group["params"]]
            for group in runner.optim_wrapper.optimizer.param_groups
        ]

    def before_train(self, runner):
        loop = runner.train_loop
        if isinstance(loop, EpochBasedTrainLoop):
            remaining = max(0, loop.max_epochs - loop.epoch) * len(loop.dataloader)
            total = loop.iter + remaining
            if total != loop.max_iters:
                # A changed batch/world size changes only the remaining epoch lengths.
                loop._max_iters = total
                runner.optim_wrapper.initialize_count_status(runner.model, loop.iter, total)
        previous = getattr(runner, "_resumed_learning_rate", None)
        target = runner.cfg.optim_wrapper.optimizer.lr
        if previous is not None and previous != target:
            rebase_learning_rate(runner.optim_wrapper, runner.param_schedulers,
                                 previous_lr=previous, target_lr=target)


class TrainingAuditHook(Hook):
    """Check gradients and optimizer updates during short training runs."""

    def __init__(self, attempted_steps, accumulation):
        self.step_handle = None
        self.parameters = {}
        self.gradients = {}
        self.iterations = []
        self.attempted_steps = attempted_steps
        self.accumulation = accumulation
        self.previous_step = 0
        self.scale_before = None

    def before_train(self, runner):
        self.parameters = {
            name: parameter for name, parameter in runner.model.named_parameters()
            if parameter.requires_grad and any(token in name for token in (".srnl.M_", ".W_q.", ".grn."))
        }
        self.step_handle = runner.optim_wrapper.optimizer.register_step_pre_hook(self._record_gradients)

    def _record_gradients(self, optimizer, args, kwargs):
        self.gradients = {
            name: bool(torch.isfinite(parameter.grad).all())
            for name, parameter in self.parameters.items() if parameter.grad is not None
        }

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        scaler = getattr(runner.optim_wrapper, "loss_scaler", None)
        self.scale_before = scaler.get_scale() if scaler is not None else None

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        optimizer = runner.optim_wrapper.optimizer
        steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
        step = max(steps, default=0)
        increment = step - self.previous_step
        attempted = (int(runner.iter) + 1) % self.accumulation == 0
        missing = sorted(self.parameters.keys() - self.gradients.keys()) if increment else []
        nonfinite = sorted(name for name, finite in self.gradients.items() if not finite)
        scaler = getattr(runner.optim_wrapper, "loss_scaler", None)
        scale_after = scaler.get_scale() if scaler is not None else None
        record = {
            "iteration": int(runner.iter), "optimizer_step": step, "optimizer_updates": increment,
            "update_attempted": attempted, "amp_skipped_update": attempted and increment == 0,
            "gradient_parameters": len(self.gradients), "missing_gradients": missing,
            "nonfinite_gradients": nonfinite, "scale_before": self.scale_before,
            "scale_after": scale_after,
            "losses": {key: float(value) for key, value in (outputs or {}).items()},
        }
        self.iterations.append(record)
        if runner.rank == 0:
            write_json(Path(runner.work_dir) / "training_attempts.json", self.iterations)
        if increment not in {0, 1} or (increment and (missing or nonfinite)):
            raise FloatingPointError("An optimizer update used incomplete or nonfinite SRNL/GRN gradients")
        if attempted and not increment and not (
            scale_after is not None and self.scale_before is not None and scale_after < self.scale_before
        ):
            raise RuntimeError("An optimizer update was skipped without AMP overflow recovery")
        self.previous_step = step
        if attempted:
            self.gradients.clear()

    def after_train(self, runner):
        self.step_handle.remove()
        completed = sum(record["optimizer_updates"] for record in self.iterations)
        result = {"task": runner.cfg.default_scope, "iterations": self.iterations,
                  "attempted_optimizer_steps": self.attempted_steps,
                  "completed_optimizer_steps": completed,
                  "amp_skipped_updates": sum(record["amp_skipped_update"] for record in self.iterations),
                  "successful_updates_have_finite_gradients": completed > 0,
                  "complete": completed > 0, "world_size": runner.world_size}
        if runner.rank == 0:
            write_json(Path(runner.work_dir) / "smoke.json", result)
        if not completed:
            raise RuntimeError("No optimizer update completed; increase smoke steps for AMP scale recovery")


def train(task, arch, data_root, output, *, schedule="1x", pretrained=None, resume=None,
          max_steps=0, batch_size=None, workers=None, precision=None, grad_accum_steps=1,
          activation_checkpointing=None, share_direction_skip=None, seed=42, lr=None, save_top_k=5,
          drop_path_rate=None, collect_device=None):
    cfg = dense_config(task, arch, data_root, output, schedule, pretrained,
                       drop_path_rate=drop_path_rate, collect_device=collect_device)
    if type(save_top_k) is not int or save_top_k < 1:
        raise ValueError("save_top_k must be a positive integer")
    cfg = prepare_iteration_schedule(task, cfg, grad_accum_steps)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else int(
        os.environ.get("WORLD_SIZE", "1"))
    cfg.launcher = "pytorch" if world_size > 1 else "none"
    cfg.randomness = {"seed": seed}
    if resume:
        cfg.load_from = str(Path(resume).expanduser().resolve())
        cfg.resume = True
    if batch_size is not None:
        cfg.train_dataloader.batch_size = batch_size
    if workers is not None:
        if workers < 0:
            raise ValueError("workers must be nonnegative")
        for split in ("train", "val", "test"):
            cfg[f"{split}_dataloader"].num_workers = workers
            cfg[f"{split}_dataloader"].persistent_workers = workers > 0
            if workers == 0:
                cfg[f"{split}_dataloader"].pop("prefetch_factor", None)
    if activation_checkpointing is not None:
        cfg.model.backbone.use_checkpoint = activation_checkpointing
    if share_direction_skip is not None:
        cfg.model.backbone.share_direction_skip = share_direction_skip
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    cfg.optim_wrapper.accumulative_counts = grad_accum_steps
    cfg.optim_wrapper.optimizer.lr = resolve_learning_rate(
        task, batch_size=cfg.train_dataloader.batch_size, world_size=world_size,
        grad_accum_steps=grad_accum_steps, lr=lr,
    )["lr"]
    # Scaling is resolved here once, including accumulation, before optimizer construction.
    cfg.auto_scale_lr = dict(enable=False)
    checkpoint = cfg.default_hooks.checkpoint
    cfg.default_hooks.checkpoint = dict(
        type="mmengine.RankedCheckpointHook", interval=checkpoint["interval"],
        by_epoch=checkpoint.get("by_epoch", True), metric=checkpoint["save_best"], keep=save_top_k,
    )
    if precision is not None:
        if precision == "fp32":
            cfg.optim_wrapper.type = "OptimWrapper"
            cfg.optim_wrapper.pop("dtype", None)
            cfg.optim_wrapper.pop("loss_scale", None)
        elif precision in {"fp16", "bf16"}:
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.dtype = "float16" if precision == "fp16" else "bfloat16"
        else:
            raise ValueError("precision must be fp32, bf16, or fp16")
    if max_steps:
        if max_steps < 0 or resume:
            raise ValueError("Smoke steps must be positive and start without --resume")
        cfg.train_cfg = {"type": "IterBasedTrainLoop", "max_iters": max_steps * grad_accum_steps,
                         "val_interval": max_steps * grad_accum_steps + 1}
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
        cfg.default_hooks.checkpoint = None
    Path(output).mkdir(parents=True, exist_ok=True)
    cfg.dump(str(Path(output) / "task_config.py"))
    runner = VisionHOPERunner.from_cfg(cfg)
    runner.register_hook(OptimizerIdentityHook())
    if max_steps:
        runner.register_hook(TrainingAuditHook(max_steps, grad_accum_steps))
    runner.train()
    return runner
