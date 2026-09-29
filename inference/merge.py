"""Concatenate prediction shards before computing dataset-level AP or IoU."""

import json
import time
from pathlib import Path

import torch

from visionhope.tasks.config import dense_config
from .evaluate import normalize_entry, output_path
from .metrics import capture_raw_metrics
from .provenance import sha256, write_json


def merge(entry, mode, output_dir, num_shards):
    import fcntl
    entry = normalize_entry(entry)
    output = output_path(output_dir, entry, mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _merge(entry, mode, output_dir, num_shards)


def _merge(entry, mode, output_dir, num_shards):
    entry = normalize_entry(entry)
    if num_shards <= 0 or num_shards > entry["samples"]:
        raise ValueError("num_shards must be between 1 and the dataset size")
    expected_protocol = json.loads(json.dumps(entry, default=str))
    expected_checkpoint = sha256(Path(entry["checkpoint"]).expanduser())
    if entry["task"] == "detection":
        from mmdet.utils import register_all_modules
        from mmdet.registry import METRICS
    elif entry["task"] == "segmentation":
        from mmseg.utils import register_all_modules
        from mmseg.registry import METRICS
    else:
        raise ValueError("Only dense prediction tasks use prediction sharding")
    from mmengine.logging import MMLogger
    register_all_modules()
    MMLogger.get_instance("visionhope_evaluation", log_level="INFO")
    output = output_path(output_dir, entry, mode)
    if output.exists():
        raise FileExistsError(output)
    payloads, predictions, identifiers, provenance = [], [], [], []
    for index in range(num_shards):
        path = output_path(output_dir, entry, mode, index, num_shards)
        payload = json.loads(path.read_text())
        payload["protocol"] = normalize_entry(payload["protocol"])
        for key, expected in {"id": entry["id"], "task": entry["task"], "model": entry["model"],
                              "mode": mode, "protocol": expected_protocol,
                              "checkpoint_sha256": expected_checkpoint}.items():
            if payload.get(key) != expected:
                raise ValueError(f"Shard {path} does not match the requested {key}")
        shard = {"index": index, "count": num_shards,
                 "start": entry["samples"] * index // num_shards,
                 "stop": entry["samples"] * (index + 1) // num_shards}
        if payload.get("prediction_shard") != shard or not payload.get("complete"):
            raise ValueError(f"Incomplete or incorrect shard: {path}")
        if not payload.get("strict_load"):
            raise ValueError(f"Shard was not evaluated with strict checkpoint loading: {path}")
        records = torch.load(path.with_suffix(".metric_results.pt"), map_location="cpu")
        sample_ids = json.loads(path.with_suffix(".sample_ids.json").read_text())
        expected = shard["stop"] - shard["start"]
        if len(records) != expected or len(sample_ids) != expected or payload["samples"] != expected:
            raise ValueError(f"Incomplete records in shard {index}")
        if entry["task"] == "detection":
            if [str(record[1]["img_id"]) for record in records] != sample_ids:
                raise ValueError("COCO records do not match sample identifiers")
            if any(str(gt["img_id"]) != str(pred["img_id"]) for gt, pred in records):
                raise ValueError("COCO prediction and ground-truth identifiers differ")
        if payloads:
            for key in ("checkpoint_sha256", "protocol", "params",
                        "metric_dataset_meta", "prepare"):
                if payload.get(key) != payloads[0].get(key):
                    raise ValueError(f"Inconsistent {key} across shards")
        payloads.append(payload)
        predictions.extend(records)
        identifiers.extend(sample_ids)
        provenance.append({"path": str(path), "sha256": sha256(path), "samples": expected,
                           "prediction_sha256": sha256(path.with_suffix(".metric_results.pt"))})
    if len(identifiers) != entry["samples"] or len(set(identifiers)) != len(identifiers):
        raise ValueError("Missing or duplicate samples in merged predictions")
    cfg = dense_config(entry["task"], entry["arch"], entry["dataset"], output.parent,
                       schedule=entry.get("schedule", "1x"),
                       collect_device=entry.get("collect_device"))
    metric = METRICS.build(cfg.test_evaluator)
    metric.dataset_meta = payloads[0]["metric_dataset_meta"]
    metric.results = predictions
    start = time.time()
    with capture_raw_metrics(entry["task"]) as raw_metrics:
        metrics = metric.evaluate(entry["samples"])
    payload = {key: value for key, value in payloads[0].items()
               if key not in {"prediction_shard", "metric_dataset_meta", "visible_device"}}
    payload.update(metrics={key: float(value) for key, value in metrics.items()}, raw_metrics=raw_metrics,
                   samples=len(identifiers), unique_samples=len(identifiers), shards=provenance,
                   aggregation_seconds=time.time() - start,
                   evaluation_aggregation="Original evaluator once over ordered concatenated predictions/statistics",
                   elapsed_seconds=sum(p["elapsed_seconds"] for p in payloads) + time.time() - start,
                   complete=True)
    write_json(output.with_suffix(".sample_ids.json"), identifiers)
    write_json(output, payload)
    return output
