"""ImageNet training with CUDA autograd, native AMP, DDP, MESA and accumulation.

Training pipeline adapted from timm 0.6.11 (Copyright 2020 Ross Wightman).
"""

from contextlib import nullcontext
from pathlib import Path
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import ModelEmaV2, distribute_bn, random_seed

from visionhope.models import create_model
from visionhope.tools.checkpoint import load_checkpoint
from visionhope.inference.provenance import write_json
from .data import classification_loaders
from .checkpoints import CheckpointManager, classification_score
from .learning_rate import rebase_learning_rate, resolve_learning_rate
from .mesa import active as mesa_active, distillation_loss
from .optim import parameter_groups
from .state import restore_training


def autocast(precision):
    if precision == "fp32":
        return nullcontext()
    if precision not in {"bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def prepare_batch(images, targets, args, mixup=None):
    if not args.prefetcher:
        images, targets = images.cuda(), targets.cuda()
        if mixup is not None:
            images, targets = mixup(images, targets)
    if args.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, targets


@torch.no_grad()
def validate(model, loader, args):
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device="cuda")
    for index, (images, targets) in enumerate(loader):
        images, targets = prepare_batch(images, targets, args)
        with autocast(args.precision):
            logits = model(images)
        # Compute validation cross-entropy outside autocast.
        loss = F.cross_entropy(logits, targets)
        hits = logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(targets[:, None])
        totals += torch.stack((loss.double() * len(targets), hits[:, 0].sum(),
                               hits.any(1).sum(), torch.tensor(len(targets), device="cuda")))
        if args.validation_steps and index + 1 >= args.validation_steps:
            break
    if dist.is_initialized():
        dist.all_reduce(totals)
    return {"loss": float(totals[0] / totals[3]), "top1": float(100 * totals[1] / totals[3]),
            "top5": float(100 * totals[2] / totals[3]), "samples": int(totals[3])}


def gradient_evidence(model):
    families = {"memory": [], "query": [], "grn": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "memory" if ".srnl.M_" in name else (
            "query" if ".operator.W_q." in name else ("grn" if ".grn." in name else None)
        )
        if family:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Missing or nonfinite CUDA training gradient: {name}")
            families[family].append(name)
    if any(not names for names in families.values()):
        raise RuntimeError("Training did not cover the SRNL, query and GRN parameters")
    return {name: {"parameters": len(parameters), "all_present_and_finite": True}
            for name, parameters in families.items()}


def train_epoch(epoch, model, loader, optimizer, scaler, scheduler, loss_fn, ema, mixup, args,
                steps_per_epoch, world_size, remaining_steps=0):
    model.train()
    mixing = not args.mixup_off_epoch or epoch < args.mixup_off_epoch
    if mixup is not None:
        mixup.mixup_enabled = mixing
    elif hasattr(loader, "mixup_enabled"):
        loader.mixup_enabled = mixing
    optimizer.zero_grad()
    totals = torch.zeros(4, dtype=torch.float64, device="cuda")
    steps = attempts = 0
    evidence = None
    attempt_records = []
    use_mesa = mesa_active(args, epoch)
    for index, (images, targets) in enumerate(loader):
        # Discard the final incomplete gradient-accumulation group.
        if index // args.grad_accum_steps >= steps_per_epoch:
            break
        images, targets = prepare_batch(images, targets, args, mixup)
        teacher = None
        if use_mesa:
            ema.module.eval()
            with torch.no_grad(), autocast(args.precision):
                teacher = ema.module(images).detach()
        with autocast(args.precision):
            logits = model(images)
            base_loss = loss_fn(logits, targets)
            mesa_loss = distillation_loss(logits, teacher, args.mesa_loss, args.mesa_temperature)\
                if use_mesa else base_loss.new_zeros(())
            loss = base_loss + args.mesa_weight * mesa_loss if use_mesa else base_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {index}")
        scaler.scale(loss / args.grad_accum_steps).backward()
        update = (index + 1) % args.grad_accum_steps == 0
        if update:
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= scale_before
            if args.max_steps and updated:
                evidence = gradient_evidence(model)
            if args.max_steps:
                attempt_records.append({"attempt": attempts + 1, "updated": updated,
                                        "scale_before": scale_before, "scale_after": scaler.get_scale()})
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            attempts += 1
            steps += int(updated)
        totals += torch.stack((loss.detach().double(), base_loss.detach().double(),
                               mesa_loss.detach().double(), loss.new_tensor(1.0).double())) * len(images)
        scheduler.step_update(epoch * len(loader) + index + 1)
        if index % args.log_interval == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"Epoch {epoch} batch {index}/{len(loader)} loss={float(loss):.5f} "
                  f"lr={optimizer.param_groups[0]['lr']:.7g}", flush=True)
        if remaining_steps and attempts >= remaining_steps:
            break
    if dist.is_initialized():
        dist.all_reduce(totals)
    if not attempts:
        raise ValueError("Dataset is smaller than one global batch including accumulation")
    return {"loss": float(totals[0] / totals[3]), "classification_loss": float(totals[1] / totals[3]),
            "mesa_loss": float(totals[2] / totals[3]), "mesa_active": use_mesa,
            "optimizer_steps": steps, "optimizer_attempts": attempts,
            "amp_skipped_updates": attempts - steps,
            "attempts": attempt_records if args.max_steps else None,
            "images": int(totals[3]), "gradient_audit": evidence}


def train(args):
    world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    initialized_here = world_size > 1 and not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl", init_method="env://")
    torch.backends.cudnn.benchmark = True
    random_seed(args.seed, rank)
    args.lr = resolve_learning_rate(
        "classification", batch_size=args.batch_size, world_size=world_size,
        grad_accum_steps=args.grad_accum_steps, lr=args.lr,
    )["lr"]
    model_arguments = dict(num_classes=args.num_classes, img_size=args.image_size,
                           drop_path_rate=args.drop_path_rate,
                           share_direction_skip=args.share_direction_skip)
    model = create_model(args.model, **model_arguments)
    if args.initial_checkpoint:
        load_checkpoint(model, args.initial_checkpoint, strict=True)
    if args.activation_checkpointing:
        model.set_grad_checkpointing(True)
    model.cuda()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.sync_bn and world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    bare_model = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True,
                                        find_unused_parameters=False)
    groups, parameter_names = parameter_groups(bare_model, args)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")
    ema = ModelEmaV2(bare_model, decay=args.ema_decay) if args.ema else None
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.min_lr,
                                  warmup_lr_init=args.warmup_lr, warmup_t=args.warmup_epochs,
                                  cycle_mul=1.0, cycle_decay=0.5, cycle_limit=1, k_decay=1.0)
    start_epoch, restored = 0, None
    if args.resume:
        restored = restore_training(bare_model, optimizer, scaler, ema, scheduler,
                                    args.resume, parameter_names)
        if restored["config"]["grad_accum_steps"] != args.grad_accum_steps:
            raise ValueError("Resume must keep grad_accum_steps; use --initial-checkpoint to start a new schedule")
        start_epoch = int(restored["epoch"]) + 1
        previous_lr = restored["config"]["lr"]
        if previous_lr != args.lr:
            rebase_learning_rate(optimizer, scheduler, previous_lr=previous_lr,
                                 target_lr=args.lr, epoch=start_epoch)
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
        scheduler.step(start_epoch)
    train_loader, validation_loader, mixup, data = classification_loaders(args, bare_model, world_size > 1)
    steps_per_epoch = len(train_loader.dataset) // (args.batch_size * args.grad_accum_steps * world_size)
    mixing = args.mixup > 0 or args.cutmix > 0
    loss_fn = SoftTargetCrossEntropy() if mixing else (
        LabelSmoothingCrossEntropy(args.smoothing) if args.smoothing else torch.nn.CrossEntropyLoss()
    )
    output = Path(args.output)
    checkpoints = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        checkpoints = CheckpointManager(output, args.save_top_k, f"top1:{args.checkpoint_metric}")
        if restored is not None:
            checkpoints.restore(restored["checkpoint_queue"], Path(args.resume).parent)
        write_json(output / "config.json", vars(args).copy())
        write_json(output / "optimizer_groups.json", [
            {"names": names, "weight_decay": group["weight_decay"], "lr": group["initial_lr"]}
            for names, group in zip(parameter_names, optimizer.param_groups)
        ])
    started = time.time()
    completed_steps = 0
    attempted_steps = 0
    history = restored["history"] if restored is not None else []
    try:
        for epoch in range(start_epoch, args.epochs + args.cooldown_epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            remaining = args.max_steps - attempted_steps if args.max_steps else 0
            metrics = train_epoch(epoch, model, train_loader, optimizer, scaler, scheduler, loss_fn,
                                  ema, mixup, args, steps_per_epoch, world_size, remaining)
            completed_steps += metrics["optimizer_steps"]
            attempted_steps += metrics["optimizer_attempts"]
            if world_size > 1:
                distribute_bn(model, world_size, True)
            validation = None if args.skip_validation else validate(model, validation_loader, args)
            ema_validation = None
            if ema is not None and not args.skip_validation:
                if world_size > 1:
                    distribute_bn(ema, world_size, True)
                ema_validation = validate(ema.module, validation_loader, args)
            scheduler.step(epoch + 1, validation["top1"] if validation else None)
            history.append({"epoch": epoch, "train": metrics, "validation": validation,
                            "ema_validation": ema_validation})
            if rank == 0:
                write_json(output / "history.json", history)
                # Runs limited by --max-steps do not save resumable checkpoints.
                if not args.max_steps:
                    score, field = classification_score(validation, ema_validation, args.checkpoint_metric)
                    checkpoint = {
                        "epoch": epoch, "model": args.model, "state_dict": bare_model.state_dict(),
                        "optimizer": optimizer.state_dict(), "optimizer_param_names": parameter_names,
                        "amp_scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
                        "config": vars(args).copy(), "history": history,
                        "validation": validation, "ema_validation": ema_validation,
                    }
                    if ema is not None:
                        checkpoint["state_dict_ema"] = ema.module.state_dict()
                    def save(path, metadata):
                        torch.save({**checkpoint, **metadata}, path)
                    checkpoints.save(save, weights=checkpoint[field], step=epoch + 1,
                                     score=score, field=field)
            if args.max_steps and attempted_steps >= args.max_steps:
                break
        result = {"optimizer_steps": completed_steps, "world_size": world_size,
                  "optimizer_attempts": attempted_steps, "complete": completed_steps > 0,
                  "elapsed_seconds": time.time() - started, "data_config": data,
                  "smoke": bool(args.max_steps), "history": history}
        if rank == 0:
            write_json(output / ("smoke.json" if args.max_steps else "result.json"), result)
        if args.max_steps and completed_steps == 0:
            raise RuntimeError("No optimizer update completed; increase smoke steps for AMP scale recovery")
        return result
    finally:
        if initialized_here:
            dist.destroy_process_group()
