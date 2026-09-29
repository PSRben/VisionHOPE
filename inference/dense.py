"""Strict full-model evaluation for Mask R-CNN and UPerNet."""

from contextlib import nullcontext
from pathlib import Path

import torch

from visionhope.tasks.config import dense_config
from visionhope.tools.checkpoint import load_checkpoint
from .metrics import capture_raw_metrics
from .prepare import prepare_for_inference
from .provenance import write_json


def evaluate(entry, mode, output, *, limit=0, workers=None, batch_size=None, shard=None):
    from mmengine.hooks import Hook
    from mmengine.runner import Runner

    precision = entry["precision"]
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Evaluation precision must be fp32, bf16, or fp16")
    output = Path(output)
    cfg = dense_config(entry["task"], entry["arch"], entry["dataset"], output.parent / output.stem,
                       schedule=entry.get("schedule", "1x"),
                       collect_device=entry.get("collect_device"))
    cfg.model.backbone.share_direction_skip = entry["share_direction_skip"]
    cfg.default_hooks.logger.interval = 100
    if entry["test_scale"] is not None:
        for transform in cfg.test_dataloader.dataset.pipeline:
            if transform.type == "Resize":
                transform.scale = tuple(entry["test_scale"])
                break
        else:
            raise ValueError("Test pipeline has no Resize transform")
    if precision != "fp32":
        cfg.test_cfg.fp16 = True
    if workers is not None:
        cfg.test_dataloader.num_workers = workers
        cfg.test_dataloader.persistent_workers = workers > 0
    cfg.test_dataloader.batch_size = batch_size or entry["batch_size"]
    if limit:
        cfg.test_dataloader.dataset.indices = min(limit, entry["samples"])
    if shard:
        cfg.test_dataloader.dataset.indices = list(range(shard["start"], shard["stop"]))
    cfg.dump(str(output.with_suffix(".config.py")))
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, entry["checkpoint"], strict=True)
    runner.model.eval()
    result = {"strict_load": True, "params": sum(p.numel() for p in runner.model.parameters()),
              "test_precision": precision, "batch_size": cfg.test_dataloader.batch_size,
              "cudnn_benchmark": cfg.env_cfg.cudnn_benchmark}
    if mode == "fused":
        result["prepare"] = prepare_for_inference(runner.model)
    seen = []

    class SampleTrackingHook(Hook):
        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            for sample in outputs:
                seen.append(str(sample.metainfo.get("img_id", sample.metainfo.get("img_path"))))

    runner.register_hook(SampleTrackingHook())
    # Accessing test_loop initializes the dataset and evaluator.
    dataset_samples = len(runner.test_loop.dataloader.dataset)
    expected = shard["stop"] - shard["start"] if shard else entry["samples"]
    if not limit and dataset_samples != expected:
        raise ValueError(f"Dataset has {dataset_samples} samples; expected {expected}")
    if shard:
        metric = runner.test_loop.evaluator.metrics[0]
        if metric.__class__.__name__ not in {"CocoMetric", "IoUMetric"}:
            raise TypeError("Only the original COCO/IoU evaluators support prediction sharding")
        result["metric_dataset_meta"] = metric.dataset_meta

        def save_predictions(records):
            if len(records) != expected:
                raise RuntimeError("Incomplete prediction shard")
            torch.save(records, output.with_suffix(".metric_results.pt"))
            return {"prediction_shard_samples": len(records)}

        metric.compute_metrics = save_predictions
    context = nullcontext() if precision == "fp32" else torch.autocast(
        "cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )
    with capture_raw_metrics(entry["task"]) as raw_metrics, context:
        metrics = runner.test()
    if len(set(seen)) != len(seen) or (not limit and len(seen) != expected):
        raise RuntimeError("Missing or duplicate validation samples")
    write_json(output.with_suffix(".sample_ids.json"), seen)
    result.update(metrics={key: float(value) for key, value in metrics.items()},
                  raw_metrics=raw_metrics, samples=len(seen), unique_samples=len(set(seen)))
    return result
