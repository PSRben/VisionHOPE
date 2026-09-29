"""ImageNet command-line arguments and validation; recipes live in scripts/train."""

import argparse
import math
from pathlib import Path

from visionhope.tasks.specs import MODEL_NAMES


TRAINING_CHOICES = {
    "model": MODEL_NAMES,
    "precision": ("fp32", "bf16", "fp16"),
    "mesa_loss": ("softce", "kl"),
    "checkpoint_metric": ("auto", "raw", "ema"),
    "train_interpolation": (
        "random", "nearest", "bilinear", "bicubic", "box", "hamming", "lanczos",
    ),
}


def add_arguments(parser, defaults=None):
    """Register training options and apply optional configuration-file defaults."""
    inputs = parser.add_argument_group("Model and data")
    inputs.add_argument(
        "--model", choices=TRAINING_CHOICES["model"], metavar="MODEL", default="visionhope_tiny",
        help="Model architecture; choices: %(choices)s",
    )
    inputs.add_argument(
        "--data-root", default="data/imagenet",
        help="ImageNet root containing training and validation images",
    )
    inputs.add_argument(
        "--output", default="outputs/classification",
        help="Directory for weights, training state, configuration and logs",
    )
    inputs.add_argument(
        "--batch-size", type=int, default=128,
        help="Training images per GPU, before accumulation",
    )
    inputs.add_argument(
        "--validation-batch-size", type=int, default=128,
        help="Validation images per GPU",
    )
    inputs.add_argument(
        "--num-classes", type=int, default=1000,
        help="Number of classes",
    )
    inputs.add_argument(
        "--image-size", type=int, default=224,
        help="Square input size in pixels for training and validation",
    )
    inputs.add_argument(
        "--config",
        help="YAML/JSON training settings; explicit CLI values override the file",
    )

    optimizer = parser.add_argument_group("Schedule and optimizer")
    optimizer.add_argument(
        "--epochs", type=int, default=300,
        help="Cosine schedule length, including warmup but excluding cooldown",
    )
    optimizer.add_argument(
        "--cooldown-epochs", type=int, default=0,
        help="Additional training epochs at the minimum LR; 0 disables cooldown",
    )
    optimizer.add_argument(
        "--warmup-epochs", type=int, default=20,
        help="Linear warmup duration within the main schedule",
    )
    optimizer.add_argument(
        "--lr", type=float, default=None,
        help="Absolute target LR; omitted values scale 0.001 by effective global batch / 1024",
    )
    optimizer.add_argument(
        "--warmup-lr", type=float, default=1e-06,
        help="Initial warmup LR; not scaled with batch size",
    )
    optimizer.add_argument(
        "--min-lr", type=float, default=1e-05,
        help="Cosine LR floor and cooldown LR; not scaled with batch size",
    )
    optimizer.add_argument(
        "--weight-decay", type=float, default=0.05,
        help="AdamW decay for ordinary weight parameters; biases and normalization scales are "
             "exempt",
    )
    optimizer.add_argument(
        "--memory-weight-decay", type=float, default=0.01,
        help="AdamW decay for the learned content, key and value initial memories",
    )
    optimizer.add_argument(
        "--memory-lr-scale", type=float, default=1.0,
        help="Target LR multiplier for content, key and value initial memories",
    )
    optimizer.add_argument(
        "--gate-weight-decay", type=float, default=0.0,
        help="AdamW decay for learning-rate and retention initial memories",
    )
    optimizer.add_argument(
        "--gate-lr-scale", type=float, default=1.0,
        help="Target LR multiplier for learning-rate and retention initial memories",
    )
    optimizer.add_argument(
        "--clip-grad", type=float, default=5.0,
        help="Maximum global L2 gradient norm; 0 clips gradients to zero",
    )
    optimizer.add_argument(
        "--drop-path-rate", type=float, default=0.2,
        help="Maximum stochastic-depth probability, increasing with block depth; 0 disables it",
    )

    execution = parser.add_argument_group("Execution")
    execution.add_argument(
        "--precision", choices=TRAINING_CHOICES["precision"], default="bf16",
        help="Training and in-training validation precision; model parameters stay FP32",
    )
    execution.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=False,
        help="Use channels-last tensor memory layout",
    )
    execution.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Microbatches per optimizer update attempt; 1 disables accumulation",
    )
    execution.add_argument(
        "--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False,
        help="Recompute block activations during backward",
    )
    execution.add_argument(
        "--share-direction-skip", action=argparse.BooleanOptionalAction, default=True,
        help="Share learned channel-wise skip scales across directions; must match loaded weights",
    )
    execution.add_argument(
        "--workers", type=int, default=8,
        help="Data-loader workers per process; 0 loads in the main process",
    )
    execution.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed; distributed ranks receive distinct training seeds",
    )
    execution.add_argument(
        "--sync-bn", action=argparse.BooleanOptionalAction, default=False,
        help="Synchronize BatchNorm statistics across training GPUs",
    )
    execution.add_argument(
        "--log-interval", type=int, default=50,
        help="Training log interval in minibatches; positive integer",
    )

    teacher = parser.add_argument_group("EMA and MESA")
    teacher.add_argument(
        "--ema", action=argparse.BooleanOptionalAction, default=True,
        help="Maintain an exponential moving average of model weights",
    )
    teacher.add_argument(
        "--ema-decay", type=float, default=0.9998,
        help="EMA decay coefficient in [0, 1]",
    )
    teacher.add_argument(
        "--mesa", action=argparse.BooleanOptionalAction, default=False,
        help="Add MESA distillation from the EMA teacher; requires --ema",
    )
    teacher.add_argument(
        "--mesa-weight", type=float, default=1.0,
        help="Multiplier on the MESA loss added to classification loss",
    )
    teacher.add_argument(
        "--mesa-start-epoch", type=int, default=75,
        help="First zero-based epoch using MESA, inclusive",
    )
    teacher.add_argument(
        "--mesa-end-epoch", type=int, default=-1,
        help="Zero-based epoch at which MESA stops, exclusive; -1 means no scheduled stop",
    )
    teacher.add_argument(
        "--mesa-loss", choices=TRAINING_CHOICES["mesa_loss"], default="softce",
        help="softce uses teacher probabilities; kl uses temperature-scaled KL divergence",
    )
    teacher.add_argument(
        "--mesa-temperature", type=float, default=5.0,
        help="Positive distillation temperature for mesa-loss=kl; unused by softce",
    )

    augmentation = parser.add_argument_group("Augmentation and preprocessing")
    augmentation.add_argument(
        "--mixup", type=float, default=0.8,
        help="Mixup Beta-distribution alpha; 0 disables Mixup",
    )
    augmentation.add_argument(
        "--cutmix", type=float, default=1.0,
        help="CutMix Beta-distribution alpha; 0 disables CutMix",
    )
    augmentation.add_argument(
        "--mixup-off-epoch", type=int, default=0,
        help="Zero-based epoch at which Mixup/CutMix stops; 0 means no scheduled stop",
    )
    augmentation.add_argument(
        "--smoothing", type=float, default=0.1,
        help="Label-smoothing strength; 0 disables label smoothing",
    )
    augmentation.add_argument(
        "--prefetcher", action=argparse.BooleanOptionalAction, default=True,
        help="Use the CUDA prefetch loader and perform mixing in the data collator",
    )
    augmentation.add_argument(
        "--crop-pct", type=float, default=0.875,
        help="Center-crop ratio for validation during training",
    )
    augmentation.add_argument(
        "--train-interpolation", choices=TRAINING_CHOICES["train_interpolation"], default="random",
        help="Training resize interpolation; random samples bilinear or bicubic",
    )

    checkpoints = parser.add_argument_group("Checkpoints and short checks")
    checkpoints.add_argument(
        "--initial-checkpoint", default=None,
        help="Model weights for initialization; starts a new optimizer and schedule",
    )
    checkpoints.add_argument(
        "--resume", default=None,
        help="training_state.pth to restore model, optimizer, schedule and available "
             "EMA/AMP state",
    )
    checkpoints.add_argument(
        "--start-epoch", type=int, default=None,
        help="Zero-based starting epoch; does not restore training state",
    )
    checkpoints.add_argument(
        "--max-steps", type=int, default=0,
        help="Maximum optimizer update attempts; 0 uses the full schedule",
    )
    checkpoints.add_argument(
        "--validation-steps", type=int, default=0,
        help="Limit minibatches per validation pass; 0 evaluates the full loader",
    )
    checkpoints.add_argument(
        "--skip-validation", action=argparse.BooleanOptionalAction, default=False,
        help="Skip raw/EMA validation; disables validation-based top-k ranking",
    )
    checkpoints.add_argument(
        "--save-top-k", type=int, default=20,
        help="Number of best validation checkpoints to retain; positive integer",
    )
    checkpoints.add_argument(
        "--checkpoint-metric", choices=TRAINING_CHOICES["checkpoint_metric"], default="auto",
        help="Rank by better raw/EMA Top-1 (auto), raw only, or EMA only; ema requires EMA "
             "training",
    )

    allowed = {action.dest for action in parser._actions
               if action.dest not in {"help", "config"}}
    for name in defaults or {}:
        if name not in allowed:
            parser.error(f"Unknown training setting {name!r}")
    parser.set_defaults(**{name: value for name, value in (defaults or {}).items()
                           if value is not None})


def read_config(argv):
    """Read optional file defaults before parsing the complete command line."""
    if "--help" in argv or "-h" in argv:
        return {}
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    options, _ = config_parser.parse_known_args(argv)
    if options.config is None:
        return {}
    import yaml
    defaults = yaml.safe_load(Path(options.config).read_text())
    if not isinstance(defaults, dict):
        raise TypeError("Training settings must be a mapping")
    return defaults


def validate_args(args):
    """Check supported values and dependent training settings."""
    for name, choices in TRAINING_CHOICES.items():
        value = getattr(args, name)
        if value not in choices:
            raise ValueError(f"Invalid {name}={value!r}; choose from {choices}")
    for name in ("channels_last", "activation_checkpointing", "share_direction_skip", "sync_bn",
                 "ema", "mesa", "prefetcher", "skip_validation"):
        if type(getattr(args, name)) is not bool:
            raise ValueError(f"{name} must be a boolean, not a string or number")
    for names, minimum in (
        (("batch_size", "validation_batch_size", "num_classes", "image_size", "epochs",
          "grad_accum_steps", "log_interval", "save_top_k"), 1),
        (("cooldown_epochs", "warmup_epochs", "workers", "seed", "mesa_start_epoch",
          "mixup_off_epoch", "max_steps", "validation_steps", "start_epoch"), 0),
        (("mesa_end_epoch",), -1),
    ):
        for name in names:
            value = getattr(args, name)
            if name == "start_epoch" and value is None:
                continue
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("lr", "warmup_lr", "min_lr", "weight_decay", "memory_weight_decay",
                 "memory_lr_scale", "gate_weight_decay", "gate_lr_scale", "clip_grad",
                 "drop_path_rate", "ema_decay", "mesa_weight", "mesa_temperature",
                 "mixup", "cutmix", "smoothing", "crop_pct"):
        value = getattr(args, name)
        if name == "lr" and value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number")
    if args.lr == 0:
        raise ValueError("lr must be positive when specified")
    if args.mesa_temperature == 0:
        raise ValueError("mesa_temperature must be positive")
    for name in ("drop_path_rate", "smoothing"):
        if getattr(args, name) >= 1:
            raise ValueError(f"{name} must lie in [0, 1)")
    if args.ema_decay > 1:
        raise ValueError("ema_decay must lie in [0, 1]")
    if not 0 < args.crop_pct <= 1:
        raise ValueError("crop_pct must lie in (0, 1]")
    for name in ("data_root", "output", "initial_checkpoint", "resume"):
        value = getattr(args, name)
        if name in {"initial_checkpoint", "resume"} and value is None:
            continue
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{name} must be a nonempty path")
    if args.mesa and not args.ema:
        raise ValueError("MESA requires the EMA teacher")
    if args.checkpoint_metric == "ema" and not args.ema:
        raise ValueError("EMA checkpoint ranking requires EMA training")
