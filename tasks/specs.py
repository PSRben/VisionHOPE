"""Supported models and shared evaluation defaults; recipes live in scripts/test."""

MODEL_NAMES = (
    "visionhope_tiny", "visionhope_small", "visionhope_base",
)

DATASET_SAMPLES = {"classification": 50000, "detection": 5000, "segmentation": 2000}


def classification_protocol(model):
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; choose one of {MODEL_NAMES}")
    return {
        "num_classes": 1000,
        "input_size": [3, 224, 224],
        "crop_pct": 1.0,
        "precision": "fp32",
        "channels_last": False,
        "cudnn_benchmark": True,
        "interpolation": "bicubic",
        "batch_size": 32,
        "samples": 50000,
    }
