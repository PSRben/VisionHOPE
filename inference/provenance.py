"""Record the checkpoint and runtime settings used for evaluation."""

import hashlib
import json
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(entry, mode):
    from visionhope.models.srnl_core import chunk
    checkpoint = Path(entry["checkpoint"]).expanduser().resolve()
    packages = {}
    for package in ("torchvision", "timm", "numpy", "Pillow", "mmengine", "mmcv", "mmdet",
                    "mmsegmentation", "pycocotools"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return {
        "id": entry["id"], "task": entry["task"], "model": entry["model"], "mode": mode,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "visible_device": os.getenv("CUDA_VISIBLE_DEVICES"),
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn": torch.backends.cudnn.allow_tf32},
        "precision": entry["precision"],
        "packages": packages,
        "visionhope_environment": {key: value for key, value in os.environ.items()
                                   if key.startswith("VISIONHOPE_")},
        "stability": {name: getattr(chunk, name) for name in (
            "STEP_SCALE", "MIN_RETENTION", "RETENTION_BIAS", "INJECTION_MARGIN",
            "SPECTRAL_CLAMP", "SPECTRAL_MARGIN",
        )},
        "grn_policy": "FP32 statistics, affine and output; shared training/inference implementation",
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, default=str) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
