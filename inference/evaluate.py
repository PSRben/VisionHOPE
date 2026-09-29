"""Run one evaluation, with optional disjoint dense-prediction sharding."""

from collections.abc import Mapping
import json
import time
from pathlib import Path

from visionhope.tasks.specs import DATASET_SAMPLES, MODEL_NAMES, classification_protocol
from .provenance import provenance, write_json


def normalize_entry(entry):
    if not isinstance(entry, Mapping):
        raise TypeError("Each evaluation entry must be a mapping")
    entry = dict(entry)
    task = entry.get("task")
    if not isinstance(task, str) or task not in DATASET_SAMPLES:
        raise ValueError(f"Unknown task {task!r}")
    allowed = {"id", "task", "model", "checkpoint", "dataset", "python", "schedule",
               "precision", "batch_size", "samples", "share_direction_skip"}
    allowed.update({"num_classes", "input_size", "crop_pct", "interpolation", "channels_last",
                    "cudnn_benchmark"}
                   if task == "classification" else {"arch", "collect_device", "test_scale"})
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(f"Unknown {task} evaluation fields: {sorted(unknown, key=str)}")
    for name in ("id", "model", "python"):
        if name == "python" and name not in entry:
            continue
        value = entry.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Evaluation field {name!r} must be a nonempty string")
    for name in ("checkpoint", "dataset"):
        value = entry.get(name)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"Evaluation field {name!r} must be a nonempty path")
    if "schedule" in entry:
        schedules = ("1x",) if task == "segmentation" else ("1x", "3x")
        if entry["schedule"] not in schedules:
            raise ValueError(f"{task} schedule must be one of {schedules}")
    if task == "classification":
        entry = {**classification_protocol(entry["model"]), **entry}
        if type(entry["num_classes"]) is not int or entry["num_classes"] <= 0:
            raise ValueError("num_classes must be a positive integer")
        if entry["precision"] not in ("fp32", "bf16", "fp16"):
            raise ValueError("Classification precision must be fp32, bf16, or fp16")
        crop = entry["crop_pct"]
        if isinstance(crop, bool) or not isinstance(crop, (int, float)) or not 0 < crop <= 1:
            raise ValueError("crop_pct must lie in (0, 1]")
        size = entry["input_size"]
        if (not isinstance(size, (list, tuple)) or len(size) != 3 or size[0] != 3
                or any(type(value) is not int or value <= 0 for value in size)):
            raise ValueError("input_size must contain three positive integers: 3 H W")
        if entry["interpolation"] not in ("bicubic", "bilinear", "nearest"):
            raise ValueError("interpolation must be bicubic, bilinear, or nearest")
        for name in ("channels_last", "cudnn_benchmark"):
            if type(entry[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
    else:
        if entry["model"] not in MODEL_NAMES:
            raise ValueError("Dense prediction requires a hierarchical VisionHOPE model")
        arch = entry["model"].removeprefix("visionhope_")
        entry.setdefault("precision", "fp32")
        entry.setdefault("samples", DATASET_SAMPLES[task])
        entry.setdefault("batch_size", 1)
        entry.setdefault("arch", arch)
        entry.setdefault("collect_device", "cpu")
        entry.setdefault("test_scale", None)
        scale = entry["test_scale"]
        if scale is not None and (not isinstance(scale, (list, tuple)) or len(scale) != 2
                                  or any(type(value) is not int or value <= 0 for value in scale)):
            raise ValueError("test_scale must contain two positive integers: W H")
        if entry["arch"] != arch:
            raise ValueError("arch must match the size specified by model")
        if entry["precision"] not in ("fp32", "bf16", "fp16"):
            raise ValueError("Evaluation precision must be fp32, bf16, or fp16")
        if entry["collect_device"] not in ("cpu", "gpu"):
            raise ValueError("collect_device must be cpu or gpu")
    entry.setdefault("share_direction_skip", True)
    if type(entry["share_direction_skip"]) is not bool:
        raise ValueError("share_direction_skip must be a boolean")
    for name in ("batch_size", "samples"):
        value = entry[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return entry


def output_path(output_dir, entry, mode, shard_index=None, num_shards=1, limit=0):
    output = Path(output_dir) / entry["id"] / f"{mode}.json"
    if shard_index is not None:
        output = output.parent / "shards" / f"{mode}.shard_{shard_index:02d}_of_{num_shards:02d}.json"
    if limit:
        output = output.with_stem(output.stem + ".smoke")
    return output


def evaluate(entry, mode, output_dir, *, limit=0, workers=None, batch_size=None,
             shard_index=None, num_shards=1):
    entry = normalize_entry(entry)
    if limit < 0 or (workers is not None and workers < 0) or (batch_size is not None and batch_size <= 0):
        raise ValueError("limit/workers must be nonnegative and batch_size must be positive")
    if batch_size is not None:
        entry["batch_size"] = batch_size
    if mode not in {"ordinary", "fused"}:
        raise ValueError("mode must be ordinary or fused")
    shard = None
    if shard_index is not None:
        if entry["task"] == "classification" or limit:
            raise ValueError("Sharding is available only for full COCO/ADE20K evaluations")
        if not 0 <= shard_index < num_shards <= entry["samples"]:
            raise ValueError("Invalid prediction shard interval")
        shard = {"index": shard_index, "count": num_shards,
                 "start": entry["samples"] * shard_index // num_shards,
                 "stop": entry["samples"] * (shard_index + 1) // num_shards}
    output = output_path(output_dir, entry, mode, shard_index, num_shards, limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The lock protects both metadata and companion predictions from duplicate workers.
    import fcntl
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            raise FileExistsError(f"Result already exists: {output}. Choose another output directory.")
        payload = provenance(entry, mode)
        payload["protocol"] = entry
        if shard:
            payload["prediction_shard"] = shard
        start = time.time()
        if entry["task"] == "classification":
            from .classification import evaluate as run
            result = run(entry, mode, output, limit=limit, workers=4 if workers is None else workers,
                         batch_size=batch_size)
        else:
            from .dense import evaluate as run
            result = run(entry, mode, output, limit=limit, workers=workers,
                         batch_size=batch_size, shard=shard)
        payload.update(result)
        payload["elapsed_seconds"] = time.time() - start
        payload["complete"] = not bool(limit)
        write_json(output, payload)
        print(json.dumps({key: value for key, value in payload.items()
                          if key in {"id", "mode", "samples", "top1", "top5", "metrics", "complete"}},
                         indent=2), flush=True)
    return output
