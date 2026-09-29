"""Evaluate a resource manifest and assemble full-precision result tables."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys

import numpy as np

from .evaluate import normalize_entry, output_path
from .provenance import sha256, write_json


def load_manifest(path):
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list):
        raise TypeError("The evaluation manifest must be a list of task entries")
    entries = [normalize_entry(entry) for entry in entries]
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Evaluation manifest IDs must be unique")
    return entries


def metric_row(payload):
    row = {key: payload[key] for key in ("id", "task", "model", "mode", "samples")}
    if payload["task"] == "classification":
        row.update({key: payload[key] for key in ("top1", "top5", "correct_top1", "correct_top5", "loss")})
    elif payload["task"] == "detection":
        for result in payload["raw_metrics"]:
            prefix = "bbox" if result["type"] == "bbox" else "mask"
            for index, metric in enumerate(("AP", "AP50", "AP75")):
                row[f"{prefix}_{metric}"] = 100 * result["stats"][index]
    else:
        raw = payload["raw_metrics"][0]
        row.update(mIoU=100 * float(np.nanmean(raw["IoU"])),
                   mAcc=100 * float(np.nanmean(raw["Acc"])), aAcc=100 * float(raw["aAcc"]))
    return row


def summarize(entries, output_dir, modes):
    output = Path(output_dir)
    rows = []
    for entry in entries:
        entry = normalize_entry(entry)
        checkpoint_hash = sha256(Path(entry["checkpoint"]).expanduser())
        for mode in modes:
            path = output_path(output, entry, mode)
            payload = json.loads(path.read_text())
            payload["protocol"] = normalize_entry(payload["protocol"])
            if not payload.get("complete") or payload.get("prediction_shard"):
                raise ValueError(f"The table requires a full evaluation: {path}")
            if payload["samples"] != entry["samples"] or not payload.get("strict_load"):
                raise ValueError(f"Incomplete validation protocol: {path}")
            if payload["protocol"] != json.loads(json.dumps(entry, default=str)) or payload["mode"] != mode:
                raise ValueError(f"Result does not match the requested experiment: {path}")
            if payload["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError(f"Result does not match the checkpoint: {path}")
            rows.append(metric_row(payload))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    write_json(output / "accuracy_summary.json", rows)
    with (output / "accuracy_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    (output / "accuracy_summary.md").write_text("\n".join(lines) + "\n")
    return rows


def run_table(manifest, output_dir, *, ids=None, modes=("ordinary", "fused"), devices=("0",),
              summarize_only=False, workers=None):
    entries = load_manifest(manifest)
    if ids:
        unknown = set(ids) - {entry["id"] for entry in entries}
        if unknown:
            raise ValueError(f"Unknown manifest IDs: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["id"] in ids]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if summarize_only:
        return summarize(entries, output, modes)
    queue = Queue()
    for entry in entries:
        for mode in modes:
            queue.put((entry, mode))

    def work(device):
        while True:
            try:
                entry, mode = queue.get_nowait()
            except Empty:
                return
            executable = entry.get("python", sys.executable)
            command = [executable, "-m", "visionhope.tasks.cli", "inference", "evaluate",
                       "--manifest", str(Path(manifest).resolve()), "--id", entry["id"],
                       "--mode", mode, "--output", str(output)]
            if workers is not None:
                command.extend(["--workers", str(workers)])
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
            environment.setdefault("OMP_NUM_THREADS", "4")
            # Use a separate extension cache for each interpreter.
            identifier = hashlib.sha256(executable.encode()).hexdigest()[:12]
            environment.setdefault("TORCH_EXTENSIONS_DIR", str(output / "extensions" / identifier))
            log = output / entry["id"] / f"{mode}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            print(f"Completed {entry['id']} {mode} on CUDA device {device}", flush=True)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a nonempty list of unique CUDA device identifiers")
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(work, device) for device in devices]
        for future in futures:
            future.result()
    return summarize(entries, output, modes)
