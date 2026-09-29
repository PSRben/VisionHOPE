"""Build task configurations with caller-provided paths and settings."""

from pathlib import Path
from importlib import import_module


def dense_config(task, arch, data_root, output, schedule="1x", pretrained=None, *,
                 drop_path_rate=None, collect_device=None):
    from mmengine.config import Config

    if task == "detection":
        from .detection.config import build_config
        cfg = build_config(arch, schedule)
    elif task == "segmentation":
        if schedule != "1x":
            raise ValueError("UPerNet uses its 160k-iteration schedule; COCO 3x does not apply")
        from .segmentation.config import build_config
        cfg = build_config(arch)
    else:
        raise ValueError(f"Unknown dense prediction task: {task}")
    if drop_path_rate is not None:
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must lie in [0, 1)")
        cfg["model"]["backbone"]["drop_path_rate"] = drop_path_rate
    if collect_device is not None:
        if collect_device not in {"cpu", "gpu"}:
            raise ValueError("collect_device must be cpu or gpu")
        for split in ("val", "test"):
            cfg[f"{split}_evaluator"]["collect_device"] = collect_device
    import_module(f"visionhope.tasks.{task}.registry")
    root = str(Path(data_root).expanduser().resolve())
    for split in ("train", "val", "test"):
        cfg[f"{split}_dataloader"]["dataset"]["data_root"] = root
    if task == "detection":
        for split in ("val", "test"):
            cfg[f"{split}_evaluator"]["ann_file"] = str(
                Path(root) / "annotations/instances_val2017.json"
            )
    if pretrained:
        cfg["model"]["backbone"]["init_cfg"] = {
            "type": "Pretrained", "checkpoint": str(Path(pretrained).expanduser().resolve())
        }
    cfg["work_dir"] = str(Path(output).expanduser().resolve())
    return Config(cfg)
