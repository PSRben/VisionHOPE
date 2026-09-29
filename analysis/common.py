"""Shared arguments and reporting for VisionHOPE efficiency measurements."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform

import torch

from visionhope.tasks.specs import MODEL_NAMES


TASK_SHAPES = {
    "classification": (224, 224),
    "detection": (1280, 800),
    "segmentation": (512, 2048),
}


def add_model_arguments(parser):
    parser.add_argument("--task", choices=tuple(TASK_SHAPES), default="classification",
                        help="Classifier, COCO Mask R-CNN, or ADE20K UPerNet")
    parser.add_argument("--model", choices=MODEL_NAMES, default="visionhope_small",
                        help="VisionHOPE backbone (default: visionhope_small)")
    parser.add_argument("--share-direction-skip", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Share skip scales across directions; must match the checkpoint "
                             "(default: enabled)")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional state_dict checkpoint for the complete task model")
    parser.add_argument("--shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"),
                        help="Input size; defaults: classification 224 224, detection 1280 800, "
                             "segmentation 512 2048")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0 or 0")
    parser.add_argument("--seed", type=int, default=1234, help="Model and synthetic-input seed")
    parser.add_argument("--output", type=Path, help="Optional JSON file; also print the report")


def validate_model_arguments(parser, args):
    args.shape = tuple(args.shape or TASK_SHAPES[args.task])
    if min(args.shape) < 32:
        parser.error("input height and width must be at least 32")
    if args.task != "classification":
        if any(side % 32 for side in args.shape):
            parser.error("dense-task tensor inputs must have height and width divisible by 32")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if device.type != "cuda":
        parser.error("VisionHOPE SRNL requires a CUDA device")


def configure_runtime(args, *, tf32=False, cudnn_benchmark=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VisionHOPE measurements")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return device


def precision_context(precision):
    if precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast("cuda", dtype=dtype)


def output_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from output_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from output_tensors(item)


def check_output(output):
    tensors = list(output_tensors(output))
    if not tensors:
        raise RuntimeError("The model returned no output tensors")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise FloatingPointError("Model output contains NaN or Inf")


def runtime_info(device):
    return {
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def write_report(report, output):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write(text)
    print(text, end="", flush=True)
