"""Image classification evaluation with Top-k counts and saved predictions."""

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from timm.data import create_dataset, create_loader, resolve_data_config

from visionhope.models import create_model
from visionhope.tools.checkpoint import load_checkpoint
from .prepare import prepare_for_inference
from .provenance import write_json


def evaluate(entry, mode, output, *, limit=0, workers=4, batch_size=None):
    arguments = dict(num_classes=entry["num_classes"], img_size=tuple(entry["input_size"][-2:]),
                     share_direction_skip=entry["share_direction_skip"])
    model = create_model(entry["model"], **arguments)
    load_checkpoint(model, entry["checkpoint"], strict=True)
    model.eval().cuda()
    torch.backends.cudnn.benchmark = entry["cudnn_benchmark"]
    if entry["channels_last"]:
        model.to(memory_format=torch.channels_last)
    result = {"strict_load": True, "params": sum(p.numel() for p in model.parameters())}
    if mode == "fused":
        result["prepare"] = prepare_for_inference(model)
    config = resolve_data_config({"input_size": tuple(entry["input_size"]),
                                  "crop_pct": entry["crop_pct"],
                                  "interpolation": entry["interpolation"]}, model=model)
    dataset = create_dataset(root=entry["dataset"], name="", split="validation")
    if len(dataset) != entry["samples"]:
        raise ValueError(f"Dataset has {len(dataset)} images; protocol requires {entry['samples']}")
    batch = batch_size or entry.get("batch_size", 32)
    loader = create_loader(
        dataset, input_size=config["input_size"], batch_size=batch, use_prefetcher=True,
        interpolation=config["interpolation"], mean=config["mean"], std=config["std"],
        num_workers=workers, distributed=False, crop_pct=entry["crop_pct"],
        pin_memory=True, persistent_workers=workers > 0,
    )
    precision = entry["precision"]
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported precision {precision}")
    context = nullcontext if precision == "fp32" else lambda: torch.autocast(
        "cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )
    total = correct1 = correct5 = 0
    loss_sum = 0.0
    predictions = []
    with torch.no_grad():
        for index, (images, targets) in enumerate(loader):
            if entry["channels_last"]:
                images = images.contiguous(memory_format=torch.channels_last)
            with context():
                logits = model(images)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite classification logits")
            predicted = logits.topk(min(5, logits.shape[1]), dim=1).indices
            hits = predicted.eq(targets[:, None])
            correct1 += int(hits[:, 0].sum())
            correct5 += int(hits.any(1).sum())
            total += len(targets)
            loss_sum += float(F.cross_entropy(logits.float(), targets, reduction="sum"))
            predictions.append(torch.cat([targets[:, None], predicted], dim=1).cpu())
            if index % 50 == 0:
                print(f"{entry['id']} {mode}: {total}/{len(dataset)} Top-1 {100 * correct1 / total:.5f}",
                      flush=True)
            if limit and total >= limit:
                break
    if not limit and total != len(dataset):
        raise RuntimeError("Evaluation did not visit the full validation dataset")
    torch.save(torch.cat(predictions), Path(output).with_suffix(".predictions.pt"))
    sample_ids = dataset.filenames(basename=False, absolute=False)[:total]
    if len(sample_ids) != total or len(set(sample_ids)) != total:
        raise RuntimeError("Validation filenames contain missing or duplicate sample identifiers")
    write_json(Path(output).with_suffix(".sample_ids.json"), sample_ids)
    result.update(samples=total, top1=100 * correct1 / total, top5=100 * correct5 / total,
                  correct_top1=correct1, correct_top5=correct5, loss=loss_sum / total,
                  data_config=config, batch_size=batch, channels_last=entry["channels_last"],
                  unique_samples=len(set(sample_ids)), cudnn_benchmark=entry["cudnn_benchmark"])
    return result
