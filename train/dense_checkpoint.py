"""Validation-ranked clean weights and complete MMEngine training state."""

from pathlib import Path

import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

from .checkpoints import CheckpointManager


@HOOKS.register_module()
class RankedCheckpointHook(Hook):
    priority = "VERY_LOW"

    def __init__(self, interval, metric, keep=5, by_epoch=True):
        self.interval = int(interval)
        self.metric = metric
        self.keep = keep
        self.by_epoch = by_epoch
        self.restored = None
        self.gradients = None
        self.pending = None

    def after_load_checkpoint(self, runner, checkpoint):
        self.restored = checkpoint.get("checkpoint_queue")
        self.gradients = checkpoint.get("gradients")

    def before_train(self, runner):
        if self.gradients is not None:
            if len(self.gradients) != runner.world_size:
                raise ValueError("A checkpoint with pending accumulated gradients must keep world size")
            model = runner.model.module if is_model_wrapper(runner.model) else runner.model
            parameters = dict(model.named_parameters())
            for name, gradient in self.gradients[runner.rank].items():
                parameter = parameters[name]
                parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)
            self.gradients = None
        if runner.rank != 0:
            return
        self.manager = CheckpointManager(runner.work_dir, self.keep, self.metric)
        if self.restored is not None:
            self.manager.restore(self.restored, Path(runner.cfg.load_from).parent)

    def before_save_checkpoint(self, runner, checkpoint):
        if self.pending is not None:
            checkpoint.update(self.pending)
        checkpoint["training_settings"] = {
            "lr": runner.cfg.optim_wrapper.optimizer.lr,
            "grad_accum_steps": runner.cfg.optim_wrapper.get("accumulative_counts", 1),
        }

    def _save(self, runner, epoch, iteration, score=None):
        gradients = None
        if not runner.optim_wrapper.should_update():
            # DDP no_sync leaves distinct partial gradients on each rank.
            model = runner.model.module if is_model_wrapper(runner.model) else runner.model
            local = {name: parameter.grad.detach().cpu() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
            gradients = [None] * runner.world_size if runner.rank == 0 else None
            if runner.world_size > 1:
                dist.gather_object(local, gradients, dst=0)
            else:
                gradients[0] = local
        if runner.rank != 0:
            return
        def writer(path, metadata):
            self.pending = {**metadata, "gradients": gradients}
            try:
                runner.save_checkpoint(
                    str(path.parent), path.name, save_optimizer=True, save_param_scheduler=True,
                    meta={"epoch": epoch, "iter": iteration}, by_epoch=self.by_epoch,
                )
            finally:
                self.pending = None
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        self.manager.save(writer, weights=model.state_dict(),
                          step=epoch if self.by_epoch else iteration, score=score)

    def after_train_epoch(self, runner):
        if self.by_epoch and (self.every_n_epochs(runner, self.interval) or self.is_last_train_epoch(runner)):
            self._save(runner, runner.epoch + 1, runner.iter)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.by_epoch and (self.every_n_train_iters(runner, self.interval)
                                  or self.is_last_train_iter(runner)):
            self._save(runner, runner.epoch, runner.iter + 1)

    def after_val_epoch(self, runner, metrics=None):
        if metrics is None or self.metric not in metrics:
            raise ValueError(f"Validation did not return checkpoint metric {self.metric!r}")
        # Validation runs after MMEngine advances the completed epoch/iteration.
        self._save(runner, runner.epoch, runner.iter, float(metrics[self.metric]))
